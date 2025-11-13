"""Core computations and transforms.

This module starts as wrappers around the legacy monolith and incrementally
duplicates selected functions so callers can depend on `forge.core.compute`
without the monolith. Where we provide local implementations, they take
precedence; otherwise we delegate to `forge.steel_model_core`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import pandas as pd
from .models import Process
from .routing import STAGE_MATS, build_route_mask
from .engine import calculate_balance_matrix, calculate_energy_balance

# -----------------------
# Local implementations
# -----------------------


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

    gas_meta = getattr(params, 'process_gases', {}) or {}
    internal_elec = 0.0

    if gas_meta:
        for carrier, meta in gas_meta.items():
            try:
                proc_name = str(meta.get('source_process') or '').strip()
            except Exception:
                proc_name = ''
            if not proc_name or proc_name not in recipes_dict:
                continue
            runs = float(prod_level.get(proc_name, 0.0) or 0.0)
            if runs <= 0:
                continue
            qty = float(recipes_dict[proc_name].outputs.get(carrier, 0.0) or 0.0)
            if qty <= 0:
                continue
            try:
                energy_per_unit = float(meta.get('energy_mj_per_unit', 0.0))
            except (TypeError, ValueError):
                energy_per_unit = 0.0
            if energy_per_unit <= 0:
                continue
            energy_MJ = runs * qty * energy_per_unit
            internal_elec += energy_MJ * util_eff

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

    def _blend_EF(shares: Dict[str, float], efs: Dict[str, float]) -> float:
        # Exclude Electricity and the process-gas carrier from the EF blend.
        # The EF for process gas should reflect the primary thermal inputs
        # (e.g., Coal, Coke, NG, Charcoal), not recycled process gas itself.
        pg_name = process_gas_carrier if 'process_gas_carrier' in locals() else 'Process Gas'
        fuels = [
            (c, s)
            for c, s in (shares or {}).items()
            if c not in {'Electricity', pg_name} and s > 0
        ]
        if not fuels:
            return 0.0
        denom = sum(s for _, s in fuels) or 1e-12
        return sum(s * float(efs.get(c, 0.0)) for c, s in fuels) / denom

    def _collect_sources_from_specs(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for spec in specs or []:
            proc_name = str(spec.get('process') or '').strip()
            if not proc_name:
                continue
            carrier_name = str(spec.get('carrier') or process_gas_carrier).strip()
            proc = recipes_dict.get(proc_name)
            if not proc:
                continue
            gas_output = float(proc.outputs.get(carrier_name, 0.0) or 0.0)
            if gas_output <= 0:
                continue
            runs = float(prod_levels.get(proc_name, 0.0) or 0.0)
            if runs <= 0:
                continue
            energy_per_unit = spec.get('energy_per_unit')
            try:
                energy_per_unit = float(energy_content.get(carrier_name, energy_per_unit))
            except Exception:
                try:
                    energy_per_unit = float(energy_per_unit)
                except Exception:
                    energy_per_unit = None
            if spec.get('outputs_in_MJ'):
                contribution = runs * gas_output
            else:
                if energy_per_unit is None or energy_per_unit <= 0:
                    continue
                contribution = runs * gas_output * energy_per_unit
            if contribution <= 1e-12:
                continue
            rows.append({'process': proc_name, 'carrier': carrier_name, 'mj': contribution})
        return rows

    EF_coke_gas = _blend_EF(energy_shares.get('Coke Production', {}), e_efs)
    EF_bf_gas = _blend_EF(energy_shares.get('Blast Furnace', {}), e_efs)
    EF_process_gas = float(e_efs.get(process_gas_carrier, 0.0))

    process_gas_specs = gas_config.get('process_gas_sources') or []
    gas_source_details: Dict[str, float] = {}
    gas_credit_details: Dict[str, Dict[str, float]] = {}
    gas_coke_MJ = 0.0
    gas_bf_MJ = 0.0
    gas_sources_MJ = 0.0
    total_gas_MJ = 0.0

    if process_gas_specs:
        contributions = _collect_sources_from_specs(process_gas_specs)
        for row in contributions:
            proc_name = row['process']
            gas_source_details[proc_name] = gas_source_details.get(proc_name, 0.0) + row['mj']
            gas_credit_details.setdefault(proc_name, {})['mj'] = gas_source_details[proc_name]
        gas_sources_MJ = sum(gas_source_details.values())
        gas_coke_MJ = gas_source_details.get('Coke Production', 0.0)
        gas_bf_MJ = gas_source_details.get('Blast Furnace', 0.0)
        total_gas_MJ = float(gas_sources_MJ)
        if total_gas_MJ > 1e-9:
            ef_weighted = 0.0
            for row in contributions:
                shares = energy_shares.get(row['process'], {})
                ef_source = _blend_EF(shares, e_efs)
                if ef_source <= 0:
                    ef_source = float(e_efs.get(row['carrier'], e_efs.get(process_gas_carrier, 0.0)))
                ef_weighted += ef_source * row['mj']
            EF_process_gas = ef_weighted / total_gas_MJ if ef_weighted > 0 else float(e_efs.get(process_gas_carrier, 0.0))
        else:
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

    total_gas_consumption_plant = 0.0
    carrier_list = [natural_gas_carrier]
    if process_gas_carrier and process_gas_carrier != natural_gas_carrier:
        carrier_list.append(process_gas_carrier)
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
                gas_carriers=carrier_list,
            ))
        except Exception:
            total_gas_consumption_plant = 0.0

    direct_use_gas_MJ = max(0.0, min(total_gas_MJ, total_gas_MJ * direct_use_fraction))
    remaining_gas = max(0.0, total_gas_MJ - direct_use_gas_MJ)
    electricity_potential = total_gas_MJ * electricity_fraction
    electricity_gas_MJ = max(0.0, min(remaining_gas, electricity_potential))

    internal_elec = electricity_gas_MJ * util_eff

    f_internal_gas = (min(1.0, direct_use_gas_MJ / total_gas_consumption_plant)
                      if total_gas_consumption_plant > 1e-9 else 0.0)

    if total_gas_MJ > 1e-9:
        for proc_name, mj in gas_source_details.items():
            share = mj / total_gas_MJ if total_gas_MJ > 0 else 0.0
            detail = gas_credit_details.setdefault(proc_name, {})
            detail['mj'] = mj
            detail['share'] = share
            detail['direct_use_MJ'] = share * direct_use_gas_MJ
            detail['electricity_MJ'] = share * electricity_gas_MJ

    # Capture grid/baseline factors before any in-place updates
    ef_natural_gas_grid = float(e_efs.get(natural_gas_carrier, 0.0) or 0.0)
    ef_electricity_grid = float(e_efs.get('Electricity', 0.0) or 0.0)

    ef_gas_blended = (f_internal_gas * EF_process_gas + (1 - f_internal_gas) * ef_natural_gas_grid)

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
    # Blended electricity EF actually used at plant level
    ef_electricity_used = (f_internal * ef_internal_electricity) + ((1.0 - f_internal) * ef_electricity_grid)

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
        'gas_credit_details': gas_credit_details,
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
        # Additional diagnostics for debugging EFs (units: g CO2e/MJ)
        'ef_nat_gas_grid': ef_natural_gas_grid,
        'ef_electricity_grid': ef_electricity_grid,
        'ef_electricity_used': ef_electricity_used,
    }

    return eb, e_efs, meta

from .costs import analyze_energy_costs, analyze_material_costs
from .transforms import (
    apply_fuel_substitutions,
    apply_dict_overrides,
    apply_recipe_overrides,
)

# Transforms/overrides (gas routing implemented locally)

# Reference helpers (local implementations)

DEFAULT_PRODUCER_PRIORITY = (
    # Core downstream/defaults
    "Continuous Casting (R)",
    "Hot Rolling",
    "Cold Rolling",
    "Basic Oxygen Furnace",
    "Electric Arc Furnace",
    # Additional priorities aligned with API legacy to ensure deterministic picks
    "Bypass Raw→IP3",
    "Bypass CR→IP3",
    "Nitrogen Production",
    "Oxygen Production",
    "Dolomite Production",
    "Burnt Lime Production",
    "Coke Production",
    "Natural gas from Market",
    "LPG from Market",
    "Biomethane from Market",
    "Hydrogen (Methane reforming) from Market",
    "Hydrogen (Electrolysis) from Market",
)


def _build_producers_index(recipes: List[Process]) -> Dict[str, List[Process]]:
    prod: Dict[str, List[Process]] = {}
    for r in recipes:
        for m in r.outputs:
            prod.setdefault(m, []).append(r)
    return prod


def _build_routes_from_picks(
    recipes: List[Process],
    demand_mat: str,
    picks_by_material: Dict[str, str],
    pre_mask: Optional[Dict[str, int]] = None,
    pre_select: Optional[Dict[str, int]] = None,
    fallback_materials: Optional[Set[str]] = None,
) -> Dict[str, int]:
    producers = _build_producers_index(recipes)
    pre_mask = dict(pre_mask or {})
    pre_select = dict(pre_select or {})
    chosen: Dict[str, int] = {}

    def score(proc: Process):
        try:
            idx = DEFAULT_PRODUCER_PRIORITY.index(proc.name)
            return (0, idx, proc.name)
        except ValueError:
            return (1, 0, proc.name)

    from collections import deque
    q = deque([demand_mat])
    visited_mats: Set[str] = {demand_mat}

    while q:
        mat = q.popleft(); visited_mats.discard(mat)
        cand_all = producers.get(mat, [])
        if not cand_all:
            continue
        allowed = [p for p in cand_all if pre_mask.get(p.name, 1) > 0]
        if not allowed:
            continue
        pick_name = picks_by_material.get(mat, "")
        pick = next((p for p in allowed if p.name == pick_name), None) if pick_name else None
        if pick is None:
            enabled = [p for p in allowed if pre_select.get(p.name, 1) > 0] or allowed
            enabled.sort(key=score)
            pick = enabled[0]
        chosen[pick.name] = 1
        for r in cand_all:
            if r.name != pick.name:
                chosen[r.name] = 0
        for im in pick.inputs.keys():
            if im not in visited_mats:
                q.append(im); visited_mats.add(im)
    return chosen


def _ensure_fallback_processes(
    recipes: List[Process],
    production_routes: Dict[str, int],
    fallback_materials: Optional[Set[str]],
) -> None:
    if not fallback_materials:
        return
    fallback_set = {str(m).strip() for m in fallback_materials if str(m).strip()}
    existing = {r.name for r in recipes}
    for mat in fallback_set:
        proc_name = f"External {mat} (auto)"
        if proc_name not in existing:
            recipes.append(Process(proc_name, inputs={}, outputs={mat: 1.0}))
            existing.add(proc_name)
        production_routes.setdefault(proc_name, 1)


def compute_inside_gas_reference_for_share(
    recipes: List[Process],
    energy_int: Dict[str, float],
    energy_shares: Dict[str, Dict[str, float]],
    energy_content: Dict[str, float],
    params,
    route_key: str,
    demand_qty: float,
    stage_ref: str = "IP3",
    stage_lookup: Optional[Dict[str, str]] = None,
    gas_carriers: Optional[List[str]] = None,
    fallback_materials: Optional[Set[str]] = None,
) -> float:
    pre_mask = build_route_mask(route_key, recipes)
    stage_map = stage_lookup or STAGE_MATS
    if stage_ref not in stage_map:
        raise KeyError(f"Stage '{stage_ref}' not found while computing gas reference.")
    demand_mat = stage_map[stage_ref]
    production_routes_full = _build_routes_from_picks(
        recipes, demand_mat, picks_by_material={}, pre_mask=pre_mask, fallback_materials=fallback_materials
    )
    final_demand_full = {demand_mat: float(demand_qty)}
    import copy as _copy
    recipes_full = _copy.deepcopy(recipes)
    _ensure_fallback_processes(recipes_full, production_routes_full, fallback_materials)
    balance_matrix_full, prod_levels_full = calculate_balance_matrix(
        recipes_full, final_demand_full, production_routes_full
    )
    if balance_matrix_full is None:
        return 0.0
    energy_balance_full = calculate_energy_balance(prod_levels_full, energy_int, energy_shares)
    total = 0.0
    carriers = gas_carriers or ["Gas"]
    rows = [r for r in energy_balance_full.index if r not in ("TOTAL",)]
    for carrier in carriers:
        if carrier in energy_balance_full.columns:
            total += float(energy_balance_full.loc[rows, carrier].clip(lower=0).sum())
    return total


def compute_inside_elec_reference_for_share(
    recipes: List[Process],
    energy_int: Dict[str, float],
    energy_shares: Dict[str, Dict[str, float]],
    energy_content: Dict[str, float],
    params,
    route_key: str,
    demand_qty: float,
    stage_ref: str = "IP3",
    stage_lookup: Optional[Dict[str, str]] = None,
    fallback_materials: Optional[Set[str]] = None,
) -> float:
    pre_mask = build_route_mask(route_key, recipes)
    stage_map = stage_lookup or STAGE_MATS
    if stage_ref not in stage_map:
        raise KeyError(f"Stage '{stage_ref}' not found while computing electricity reference.")
    demand_mat = stage_map[stage_ref]
    production_routes_full = _build_routes_from_picks(
        recipes, demand_mat, picks_by_material={}, pre_mask=pre_mask, fallback_materials=fallback_materials
    )
    final_demand_full = {demand_mat: float(demand_qty)}
    import copy as _copy
    recipes_full = _copy.deepcopy(recipes)
    _ensure_fallback_processes(recipes_full, production_routes_full, fallback_materials)
    balance_matrix_full, prod_levels_full = calculate_balance_matrix(
        recipes_full, final_demand_full, production_routes_full
    )
    if balance_matrix_full is None:
        return 0.0
    energy_balance_full = calculate_energy_balance(prod_levels_full, energy_int, energy_shares)
    total = 0.0
    if 'Electricity' in energy_balance_full.columns:
        rows = [r for r in energy_balance_full.index if r not in ("TOTAL",)]
        total = float(energy_balance_full.loc[rows, 'Electricity'].clip(lower=0).sum())
    return total

__all__ = [
    # calcs
    "expand_energy_tables_for_active",
    "calculate_internal_electricity",
    "adjust_energy_balance",
    "derive_energy_shares",
    "analyze_energy_costs",
    "analyze_material_costs",
    # transforms
    "apply_fuel_substitutions",
    "apply_dict_overrides",
    "apply_recipe_overrides",
    "apply_gas_routing_and_credits",
    # refs
    "compute_inside_elec_reference_for_share",
    "compute_inside_gas_reference_for_share",
]
