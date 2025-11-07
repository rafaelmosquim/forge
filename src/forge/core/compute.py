"""Core computations and transforms.

This module starts as wrappers around the legacy monolith and incrementally
duplicates selected functions so callers can depend on `forge.core.compute`
without the monolith. Where we provide local implementations, they take
precedence; otherwise we delegate to `forge.steel_model_core`.
"""
from __future__ import annotations

from forge import steel_model_core as _core
from typing import Dict, List
import pandas as pd
import os
from .models import Process

# -----------------------
# Local implementations
# -----------------------

def _env_flag_truthy(var_name: str) -> bool:
    try:
        raw = os.environ.get(var_name, "")
    except Exception:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _debug_print(*args, **kwargs) -> None:
    if _env_flag_truthy("FORGE_DEBUG_IO"):
        print(*args, **kwargs)


def expand_energy_tables_for_active(active_names, energy_shares, energy_int):
    """Let variant names like 'Continuous Casting (R)' reuse base rows.

    Mutates energy_shares/energy_int in-place by copying entries for each
    active process variant from its base name (text before " (").
    """
    def base(n: str) -> str:
        return n.split(" (")[0]

    for n in list(active_names or []):
        b = base(n)
        if n not in energy_shares and b in energy_shares:
            energy_shares[n] = dict(energy_shares[b])
        if n not in energy_int and b in energy_int:
            energy_int[n] = energy_int[b]


def calculate_internal_electricity(prod_level: Dict[str, float], recipes_dict: Dict[str, Process], params) -> float:
    """Compute internal electricity from recovered gases (BF delta + Coke gas).

    Utility Plant efficiency is taken as the Electricity output per MJ gas
    from the 'Utility Plant' recipe.
    """
    util_eff = 0.0
    if 'Utility Plant' in recipes_dict:
        util_eff = recipes_dict['Utility Plant'].outputs.get('Electricity', 0.0)

    internal_elec = 0.0

    # BF top-gas delta between adjusted and base intensities
    bf_runs = float(prod_level.get('Blast Furnace', 0.0))
    if bf_runs > 0 and hasattr(params, 'bf_base_intensity') and hasattr(params, 'bf_adj_intensity'):
        bf_delta = params.bf_adj_intensity - params.bf_base_intensity
        gf = bf_runs * bf_delta
        _debug_print(f"DBG gas BF: runs={bf_runs:.3f}, delta={bf_delta:.2f} MJ/run → {gf:.1f} MJ")
        internal_elec += gf * util_eff

    # Coke-oven gas
    cp_runs = float(prod_level.get('Coke Production', 0.0))
    if 'Coke Production' in recipes_dict:
        gas_per_run_cp = recipes_dict['Coke Production'].outputs.get('Process Gas', 0.0)
        if cp_runs > 0 and gas_per_run_cp > 0:
            gf_cp = cp_runs * gas_per_run_cp
            _debug_print(f"DBG gas Coke: runs={cp_runs:.3f}, gas_per_run={gas_per_run_cp:.2f} MJ/run → {gf_cp:.1f} MJ")
            internal_elec += gf_cp * util_eff

    return internal_elec


def adjust_energy_balance(energy_df, internal_elec):
    """Apply internal electricity credit to energy balance DataFrame.

    - Subtract internal_elec from TOTAL's Electricity
    - Add/overwrite a 'Utility Plant' row with negative Electricity equal to
      the internal_elec (export from recovered gases)
    """
    df = energy_df.copy()
    if 'Electricity' not in df.columns:
        df['Electricity'] = 0.0
    # TOTAL row may not exist; defensively create if missing
    if 'TOTAL' not in df.index:
        try:
            df.loc['TOTAL'] = df.sum(numeric_only=True)
        except Exception:
            df.loc['TOTAL'] = 0.0
    df.loc['TOTAL', 'Electricity'] = float(df.loc['TOTAL'].get('Electricity', 0.0)) - float(internal_elec)
    df.loc['Utility Plant'] = 0.0
    df.loc['Utility Plant', 'Electricity'] = -float(internal_elec)
    return df


def derive_energy_shares(recipes: List[Process], energy_content: Dict[str, float]):
    """Derive energy carrier shares per process from recipe inputs.

    Electricity is taken directly (MJ units). Fuels are converted using
    lower heating values provided in `energy_content`.
    Returns a mapping: { process_name: { carrier: share, ... }, ... }
    """
    shares: Dict[str, Dict[str, float]] = {}
    for proc in recipes:
        MJ_by_carrier: Dict[str, float] = {}
        for c, amt in proc.inputs.items():
            if c == 'Electricity':
                MJ_by_carrier[c] = float(amt)
            elif c in energy_content:
                MJ_by_carrier[c] = float(amt) * float(energy_content[c])
        total = sum(MJ_by_carrier.values())
        if total > 1e-12:
            shares[proc.name] = {c: mj / total for c, mj in MJ_by_carrier.items()}
        else:
            shares[proc.name] = {}
    return shares


def apply_gas_routing_and_credits(
    energy_balance: pd.DataFrame,
    recipes: list,
    prod_levels: dict,
    params,
    energy_shares: dict,
    energy_int: dict,
    energy_content: dict,
    e_efs: dict,
    scenario: dict,
    credit_on: bool,
    compute_inside_gas_reference_fn=None,
):
    """Gas routing, EF blending and electricity credits.

    Returns a tuple (energy_balance: DataFrame, e_efs: dict, meta: dict).
    """
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

    def _blend_EF(shares: Dict[str, float], efs: Dict[str, float]) -> float:
        fuels = [(c, s) for c, s in (shares or {}).items() if c != 'Electricity' and s > 0]
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

    total_gas_consumption_plant = 0.0
    if compute_inside_gas_reference_fn:
        try:
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

    e_efs = dict(e_efs)
    e_efs[natural_gas_carrier] = ef_gas_blended
    e_efs[process_gas_carrier] = EF_process_gas

    inside_elec_ref = float(scenario.get('inside_elec_ref', 0.0))
    if inside_elec_ref <= 0.0:
        try:
            inside_elec_ref = compute_inside_elec_reference_for_share(
                recipes=recipes,
                energy_int=energy_int,
                energy_shares=energy_shares,
                energy_content=energy_content,
                params=params,
                route_key=scenario.get('route_preset', None) or '',
                demand_qty=float(scenario.get('demand_qty', 1000.0)),
                stage_ref=scenario.get('stage_ref', 'IP3'),
            )
        except Exception:
            inside_elec_ref = 0.0

    f_internal = min(1.0, internal_elec / inside_elec_ref) if inside_elec_ref > 1e-9 else 0.0
    ef_internal_electricity = (EF_process_gas / util_eff) if util_eff > 1e-9 else 0.0

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

# Calculations
calculate_balance_matrix = _core.calculate_balance_matrix
# Use our local implementations (names shadow the getattr fallbacks)
# expand_energy_tables_for_active defined above
# calculate_internal_electricity defined above
calculate_energy_balance = _core.calculate_energy_balance
from .costs import analyze_energy_costs, analyze_material_costs
from .lci import calculate_lci
from .transforms import (
    apply_fuel_substitutions,
    apply_dict_overrides,
    apply_recipe_overrides,
    adjust_blast_furnace_intensity,
    adjust_process_gas_intensity,
)
calculate_emissions = _core.calculate_emissions
calculate_lci = getattr(_core, "calculate_lci", lambda *a, **k: None)

# Transforms/overrides (gas routing implemented locally)

# Reference helpers
compute_inside_elec_reference_for_share = getattr(_core, "compute_inside_elec_reference_for_share", lambda *a, **k: 0.0)
compute_inside_gas_reference_for_share = getattr(_core, "compute_inside_gas_reference_for_share", lambda *a, **k: 0.0)

__all__ = [
    # calcs
    "calculate_balance_matrix",
    "expand_energy_tables_for_active",
    "calculate_internal_electricity",
    "calculate_energy_balance",
    "adjust_energy_balance",
    "derive_energy_shares",
    "analyze_energy_costs",
    "analyze_material_costs",
    "calculate_emissions",
    "calculate_lci",
    # transforms
    "apply_fuel_substitutions",
    "apply_dict_overrides",
    "apply_recipe_overrides",
    "adjust_blast_furnace_intensity",
    "adjust_process_gas_intensity",
    "apply_gas_routing_and_credits",
    # refs
    "compute_inside_elec_reference_for_share",
    "compute_inside_gas_reference_for_share",
]
