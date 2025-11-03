"""
Separete module for gas recovery and routing logic. Gets gas from coke and blast furnace
and routes it according to scenario settings, applying credits and adjusting emission factors.
"""
import logging 
import pandas as pd
from types import SimpleNamespace
from typing import Dict

from .forge.steel_model_core import (
    build_route_mask,
    build_pre_for_route,
    build_routes_interactive,
    calculate_balance_matrix,
    expand_energy_tables_for_active,
    calculate_energy_balance,
    adjust_energy_balance,
    STAGE_MATS,
)


def compute_inside_elec_reference_for_share(
    recipes, energy_int, energy_shares, energy_content, params,
    route_key: str, demand_qty: float, stage_ref: str = "IP3"
) -> float:
    inside_elec_ref, _f_internal, _ef_internal = compute_fixed_plant_elec_model(
        recipes, energy_int, energy_shares, energy_content, params,
        route_key=route_key, demand_qty=demand_qty, stage_ref=stage_ref
    )
    return float(inside_elec_ref)


def compute_inside_gas_reference_for_share(
    recipes,
    energy_int,
    energy_shares,
    energy_content,
    params,
    route_key: str,
    demand_qty: float,
    stage_ref: str = "IP3",
    stage_lookup=None,
    gas_carrier=None,
    fallback_materials=None,
    **_,
):
    """
    Compute total plant-level gas consumption (MJ) for a fixed reference chain
    (same idea as compute_inside_elec_reference_for_share but for process gas).
    Returns total gas consumption in MJ for the deterministic reference chain.
    """
    # Reuse the fixed-plant model to compute reference runs and gas volumes
    inside_elec_ref, f_internal, ef_internal = compute_fixed_plant_elec_model(
        recipes, energy_int, energy_shares, energy_content, params,
        route_key=route_key, demand_qty=demand_qty, stage_ref=stage_ref
    )

    # Reconstruct reference production levels (similar to compute_fixed_plant_elec_model)
    import copy as _copy
    recipes_ref = _copy.deepcopy(recipes)
    energy_int_ref = dict(energy_int)
    energy_shares_ref = {k: dict(v) for k, v in energy_shares.items()}

    pre_mask_ref = build_route_mask(route_key, recipes_ref)
    try:
        pre_select_ref, pre_mask_from_prebuilder, _ = build_pre_for_route(route_key)
        if pre_mask_from_prebuilder:
            pre_mask_ref.update(pre_mask_from_prebuilder)
    except Exception:
        pre_select_ref = {}

    # Deterministic route for reference
    demand_mat_ref = STAGE_MATS[stage_ref]
    final_demand_ref = {demand_mat_ref: float(demand_qty)}
    production_routes_ref = build_routes_interactive(
        recipes_ref, demand_mat_ref, pre_select=pre_select_ref, pre_mask=pre_mask_ref, interactive=False
    )

    balance_ref, prod_ref = calculate_balance_matrix(recipes_ref, final_demand_ref, production_routes_ref)
    if balance_ref is None:
        return 0.0

    # Ensure energy tables for active procs
    active_procs_ref = [p for p, r in prod_ref.items() if r > 1e-9]
    expand_energy_tables_for_active(active_procs_ref, energy_shares_ref, energy_int_ref)

    # Calculate total gas consumption from energy balance
    energy_ref = calculate_energy_balance(prod_ref, energy_int_ref, energy_shares_ref)
    total_gas = 0.0
    if 'Gas' in energy_ref.columns:
        # exclude Utility Plant row
        rows = [r for r in energy_ref.index if r not in ("TOTAL", "Utility Plant")]
        total_gas = float(energy_ref.loc[rows, 'Gas'].clip(lower=0).sum())

    return total_gas


def apply_gas_routing_and_credits(
    energy_balance: pd.DataFrame,
    recipes: list,
    prod_levels: dict,
    params: SimpleNamespace,
    energy_shares: dict,
    energy_int: dict,
    energy_content: dict,
    e_efs: dict,
    scenario: dict,
    credit_on: bool,
    compute_inside_gas_reference_fn=None,
):
    """
    Centralize gas routing, process-gas splitting, EF blending and credit application.

    Modifies and returns:
      - energy_balance (possibly adjusted)
      - updated e_efs (Gas and Process Gas keys)
      - meta dict with gas routing diagnostics

    If compute_inside_gas_reference_fn is provided it will be used to compute
    the plant-level reference gas consumption; otherwise a fallback of 0.0 is used.
    """
    # Prepare recipes dict
    recipes_dict = {r.name: r for r in recipes}

    gas_config = scenario.get('gas_config', {}) or {}
    process_roles = scenario.get('process_roles', {}) or {}
    fallback_materials = set(scenario.get('fallback_materials', []))

    process_gas_carrier = gas_config.get('process_gas_carrier') or 'Process Gas'
    natural_gas_carrier = gas_config.get('natural_gas_carrier') or 'Gas'
    utility_process_name = gas_config.get('utility_process') or 'Utility Plant'

    def _roles_for(proc_name: str) -> set:
        roles = process_roles.get(proc_name, set())
        if isinstance(roles, dict):
            iterable = roles.keys()
        elif isinstance(roles, (list, tuple, set)):
            iterable = roles
        elif roles:
            iterable = [roles]
        else:
            iterable = []
        return {str(r).lower() for r in iterable}

    # Compute process gas volumes (legacy steel logic)
    try:
        gas_coke_MJ = prod_levels.get('Coke Production', 0.0) * recipes_dict.get('Coke Production', Process('',{},{})).outputs.get(process_gas_carrier, 0.0)
    except Exception:
        gas_coke_MJ = 0.0
    try:
        bf_adj = float(getattr(params, 'bf_adj_intensity', 0.0))
        bf_base = float(getattr(params, 'bf_base_intensity', 0.0))
        gas_bf_MJ = (bf_adj - bf_base) * prod_levels.get('Blast Furnace', 0.0)
    except Exception:
        gas_bf_MJ = 0.0
    total_gas_MJ = float(gas_coke_MJ + gas_bf_MJ)

    # Blend EF helper
    def _blend_EF(shares: Dict[str, float], efs: Dict[str, float]) -> float:
        fuels = [(c, s) for c, s in shares.items() if c != 'Electricity' and s > 0]
        if not fuels:
            return 0.0
        denom = sum(s for _, s in fuels) or 1e-12
        return sum(s * float(efs.get(c, 0.0)) for c, s in fuels) / denom

    EF_coke_gas = _blend_EF(energy_shares.get('Coke Production', {}), e_efs)
    EF_bf_gas = _blend_EF(energy_shares.get('Blast Furnace', {}), e_efs)
    EF_process_gas = EF_coke_gas if total_gas_MJ <= 1e-9 else (
        (EF_coke_gas * (gas_coke_MJ / max(1e-12, total_gas_MJ))) + (EF_bf_gas * (gas_bf_MJ / max(1e-12, total_gas_MJ)))
    )

    gas_source_names = gas_config.get('gas_sources')
    if not gas_source_names:
        gas_source_names = [
            name for name in recipes_dict.keys()
            if 'gas_source' in _roles_for(name)
        ]
    gas_source_names = list(dict.fromkeys(gas_source_names))

    gas_sources_MJ = 0.0
    ef_weighted = 0.0
    weight_sum = 0.0
    gas_source_details: Dict[str, float] = {}
    for src in gas_source_names:
        proc = recipes_dict.get(src)
        if not proc:
            continue
        gas_output = float(proc.outputs.get(process_gas_carrier, 0.0) or 0.0)
        if gas_output <= 0:
            continue
        runs = float(prod_levels.get(src, 0.0) or 0.0)
        contribution = runs * gas_output
        if contribution <= 1e-12:
            continue
        gas_sources_MJ += contribution
        gas_source_details[src] = contribution
        shares = energy_shares.get(src, {})
        ef_source = _blend_EF(shares, e_efs)
        if ef_source <= 0:
            ef_source = float(e_efs.get(process_gas_carrier, 0.0))
        ef_weighted += ef_source * contribution
        weight_sum += contribution

    use_descriptor_sources = (
        gas_sources_MJ > 0 and abs(gas_sources_MJ - total_gas_MJ) <= 1e-6
    )
    if use_descriptor_sources:
        total_gas_MJ = float(gas_sources_MJ)
        if weight_sum > 0:
            EF_process_gas = ef_weighted / weight_sum
        else:
            EF_process_gas = float(e_efs.get(process_gas_carrier, 0.0))
        # keep per-source values if provided; fall back to legacy values otherwise
        gas_coke_MJ = float(gas_source_details.get('Coke Production', gas_coke_MJ))
        gas_bf_MJ = float(gas_source_details.get('Blast Furnace', gas_bf_MJ))
    else:
        if gas_sources_MJ <= 0 and total_gas_MJ <= 1e-9:
            EF_process_gas = float(e_efs.get(process_gas_carrier, 0.0))

    try:
        util_eff = recipes_dict.get(utility_process_name, Process('',{},{})).outputs.get('Electricity', 0.0)
    except Exception:
        util_eff = 0.0
    if util_eff <= 0 and utility_process_name != 'Utility Plant':
        try:
            util_eff = recipes_dict.get('Utility Plant', Process('',{},{})).outputs.get('Electricity', 0.0)
        except Exception:
            util_eff = 0.0

    # Read gas routing from scenario
    gas_routing = scenario.get('gas_routing', {})
    default_direct = gas_config.get('default_direct_use_fraction')
    if default_direct is None:
        default_direct = 0.5
    direct_use_fraction = gas_routing.get('direct_use_fraction', default_direct)
    if direct_use_fraction is None:
        direct_use_fraction = default_direct
    direct_use_fraction = max(0.0, min(1.0, float(direct_use_fraction)))
    electricity_fraction = gas_routing.get('electricity_fraction')
    if electricity_fraction is None:
        electricity_fraction = max(0.0, 1.0 - direct_use_fraction)
    else:
        electricity_fraction = max(0.0, min(1.0, float(electricity_fraction)))
    if direct_use_fraction + electricity_fraction > 1.0:
        electricity_fraction = max(0.0, 1.0 - direct_use_fraction)

    direct_use_gas_MJ = total_gas_MJ * direct_use_fraction
    electricity_gas_MJ = total_gas_MJ * electricity_fraction

    internal_elec = electricity_gas_MJ * util_eff

    # Compute plant-level total gas consumption
    total_gas_consumption_plant = 0.0
    if compute_inside_gas_reference_fn:
        try:
            total_gas_consumption_plant = 0.0
            if compute_inside_gas_reference_fn:
                total_gas_consumption_plant = float(compute_inside_gas_reference_fn(
                    recipes=recipes,
                    energy_int=energy_int,
                    energy_shares=energy_shares,
                    energy_content=energy_content,
                    params=params,
                    route_key=scenario.get('route_preset', None) or '',
                    demand_qty=float(scenario.get('demand_qty', 1000.0)),
                    stage_ref=scenario.get('stage_ref', 'IP3'),
                ))
        except Exception:
            total_gas_consumption_plant = 0.0

    f_internal_gas = (min(1.0, direct_use_gas_MJ / total_gas_consumption_plant)
                      if total_gas_consumption_plant > 1e-9 else 0.0)

    ef_natural_gas = e_efs.get(natural_gas_carrier, 0.0)
    ef_gas_blended = (f_internal_gas * EF_process_gas + (1 - f_internal_gas) * ef_natural_gas)

    # update emission factors
    e_efs = dict(e_efs)
    e_efs[natural_gas_carrier] = ef_gas_blended
    e_efs[process_gas_carrier] = EF_process_gas

    # Internal electricity fraction at plant-level
    # compute a inside reference for electricity if provided in scenario or compute fallback 0
    inside_elec_ref = float(scenario.get('inside_elec_ref', 0.0))
    if inside_elec_ref <= 0.0 and 'compute_inside_elec_reference_for_share' in globals():
        try:
            inside_elec_ref = compute_inside_elec_reference_for_share(
                recipes=recipes,
                energy_int=energy_int,
                energy_shares=energy_shares,
                energy_content=energy_content if 'energy_content' in globals() else {},
                params=params,
                route_key=scenario.get('route_preset', None) or '',
                demand_qty=float(scenario.get('demand_qty', 1000.0)),
                stage_ref=scenario.get('stage_ref', 'IP3'),
            )
        except Exception:
            inside_elec_ref = 0.0

    f_internal = min(1.0, internal_elec / inside_elec_ref) if inside_elec_ref > 1e-9 else 0.0
    ef_internal_electricity = (EF_process_gas / util_eff) if util_eff > 1e-9 else 0.0

    # Apply credits (modify energy_balance in place copy)
    eb = energy_balance.copy()
    if credit_on:
        eb = adjust_energy_balance(eb, internal_elec)

        if direct_use_gas_MJ > 0 and total_gas_consumption_plant > 1e-9:
            for process_name in eb.index:
                if natural_gas_carrier in eb.columns:
                    current_gas = eb.loc[process_name, natural_gas_carrier]
                    if current_gas > 0:
                        reduction = current_gas * f_internal_gas
                        eb.loc[process_name, natural_gas_carrier] = current_gas - reduction
                        if process_gas_carrier not in eb.columns:
                            eb[process_gas_carrier] = 0.0
                        eb.loc[process_name, process_gas_carrier] += reduction
    else:
        internal_elec = 0.0
        total_gas_MJ = 0.0
        direct_use_gas_MJ = 0.0
        electricity_gas_MJ = 0.0
        gas_sources_MJ = 0.0
        EF_process_gas = 0.0

    meta = {
        'total_process_gas_MJ': total_gas_MJ,
        'gas_coke_MJ': gas_coke_MJ,
        'gas_bf_MJ': gas_bf_MJ,
        'gas_sources_MJ': gas_sources_MJ,
        'gas_source_details': gas_source_details,
        'direct_use_gas_MJ': direct_use_gas_MJ,
        'electricity_gas_MJ': electricity_gas_MJ,
        'total_gas_consumption_plant': total_gas_consumption_plant,
        'f_internal_gas': f_internal_gas,
        'ef_gas_blended': ef_gas_blended,
        'EF_coke_gas': EF_coke_gas,
        'EF_bf_gas': EF_bf_gas,
        'EF_process_gas': EF_process_gas,
        'util_eff': util_eff,
        'direct_use_fraction': direct_use_fraction,
        'electricity_fraction': electricity_fraction,
        'f_internal': f_internal,
        'ef_internal_electricity': ef_internal_electricity,
        'process_gas_carrier': process_gas_carrier,
        'natural_gas_carrier': natural_gas_carrier,
        'utility_process': utility_process_name,
        'fallback_materials': list(fallback_materials),
    }

    return eb, e_efs, meta