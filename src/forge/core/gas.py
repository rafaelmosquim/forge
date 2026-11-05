"""
Recovered process gas: one pool, allocated so that every MJ is charged once.

The blast furnace, coke ovens and BOF recover gas. Each source is credited, at
EF_process_gas, for every MJ it produces, and whoever burns the gas is charged
at the same factor. Carbon is therefore conserved exactly as long as the pool
is fully allocated, which this module guarantees. The pool is spent in order:

  1. losses         a fixed fraction is flared or lost (charged, no useful heat);
  2. fixed burners  processes whose recipes burn process gas natively (blast-
                    furnace stoves, coke-oven underfiring, BOF) take what they
                    need; any shortfall is met with purchased natural gas;
  3. flexible use   a share of the plant's natural-gas demand (reheating
                    furnaces, kilns) is met with recovered gas instead;
  4. power          the rest is burned in the utility plant, up to the plant's
                    electricity demand;
  5. overflow       anything still left is flared.

The allocation is computed ONCE, on the full-plant reference (``stage_ref``), and
turned into per-process factors (f_native, f_internal_gas, f_internal) plus a
flare fraction. Every process therefore carries the same inventory at any
reporting boundary: a narrower boundary leaves out the downstream processes
together with the gas and power they burn, while the source credits and the
flare row belong to processes that are inside every boundary.

Defaults for the loss fraction and the flexible share come from sector.yml
(gas.default_loss_fraction, gas.default_flexible_share); a scenario can override
them through gas_routing: {loss_fraction, flexible_share}. The legacy
direct_use_fraction / electricity_fraction inputs no longer drive anything: a
fixed 50/50 split ignored the gas that recipes already burn and could charge
more gas than the plant makes. They are reported back as realised shares.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from .models import Process
from .engine import calculate_balance_matrix, calculate_energy_balance
from .routing import STAGE_MATS, build_route_mask
from .compute import _build_routes_from_picks, _ensure_fallback_processes


_LEGACY_SPLIT_WARNED = False


def calculate_internal_electricity(prod_level: Dict[str, float], recipes_dict: Dict[str, Process], params) -> float:
    """Compute internal electricity from recovered gases (BF, BOF and Coke).

    Legacy diagnostic: it assumes every recovered MJ goes to the utility plant.
    The emission calculation uses the allocation in apply_gas_routing_and_credits.
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
    """Subtract internal electricity from TOTAL and add the Utility Plant export."""
    df = energy_df.copy()
    if 'Electricity' not in df.columns:
        df['Electricity'] = 0.0
    if 'TOTAL' not in df.index:
        try:
            df.loc['TOTAL'] = df.sum(numeric_only=True)
        except Exception:
            df.loc['TOTAL'] = 0.0
    df.loc['TOTAL', 'Electricity'] = float(df.loc['TOTAL'].get('Electricity', 0.0)) - float(internal_elec)
    df.loc['Utility Plant'] = 0.0
    df.loc['Utility Plant', 'Electricity'] = -float(internal_elec)
    return df


def _clip01(value, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float(default)
    return max(0.0, min(1.0, v))


def allocate_gas_pool(
    gas_MJ: float,
    native_demand_MJ: float,
    ng_demand_MJ: float,
    elec_demand_MJ: float,
    util_eff: float,
    loss_fraction: float,
    flexible_share: float,
) -> Dict[str, float]:
    """Spend a recovered-gas pool in the fixed order described in the module docstring.

    All arguments describe ONE plant (the reference). Returns the MJ in each
    destination plus the per-process factors derived from them. By construction
    losses + native + substituted + power + overflow == gas_MJ.
    """
    G = max(0.0, float(gas_MJ))
    native_demand = max(0.0, float(native_demand_MJ))
    ng_demand = max(0.0, float(ng_demand_MJ))
    elec_demand = max(0.0, float(elec_demand_MJ))

    losses = loss_fraction * G
    avail = G - losses

    native_used = min(native_demand, avail)
    avail -= native_used

    substituted = min(flexible_share * ng_demand, avail)
    avail -= substituted

    power_gas = avail
    if util_eff <= 1e-12:
        power_gas = 0.0
    elif elec_demand > 0 and power_gas * util_eff > elec_demand:
        power_gas = elec_demand / util_eff
    overflow = avail - power_gas
    internal_elec = power_gas * util_eff if util_eff > 1e-12 else 0.0

    flared = losses + overflow
    return {
        'gas_MJ': G,
        'losses_MJ': losses,
        'native_used_MJ': native_used,
        'substituted_MJ': substituted,
        'power_gas_MJ': power_gas,
        'overflow_MJ': overflow,
        'flared_MJ': flared,
        'internal_elec_MJ': internal_elec,
        # per-process factors
        'f_native': (native_used / native_demand) if native_demand > 1e-12 else 1.0,
        'f_internal_gas': (substituted / ng_demand) if ng_demand > 1e-12 else 0.0,
        'f_internal': min(1.0, internal_elec / elec_demand) if elec_demand > 1e-12 else 0.0,
        'flare_fraction': (flared / G) if G > 1e-12 else 0.0,
    }


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
    compute_inside_reference_fn=None,
):
    """Allocate recovered gas, set the process-gas EF and adjust the energy balance."""
    global _LEGACY_SPLIT_WARNED
    route_preset = (scenario.get("route_preset") or "").upper()
    recipes_dict = {r.name: r for r in recipes}

    gas_config = scenario.get('gas_config', {}) or {}
    process_gas_carrier = gas_config.get('process_gas_carrier') or 'Process Gas'
    natural_gas_carrier = gas_config.get('natural_gas_carrier') or 'Gas'
    utility_process_name = gas_config.get('utility_process') or 'Utility Plant'
    gas_carrier_names: Set[str] = {process_gas_carrier}
    for spec in gas_config.get('process_gas_sources') or []:
        if spec.get('carrier'):
            gas_carrier_names.add(str(spec['carrier']))
    gas_carrier_names.update((gas_config.get('process_gases') or {}).keys())

    def _blend_EF(shares: Dict[str, float], efs: Dict[str, float]) -> float:
        # Fuel-weighted EF of a process, excluding electricity and recovered gas.
        fuels = [
            (c, s)
            for c, s in (shares or {}).items()
            if c != 'Electricity' and c not in gas_carrier_names and s and s > 0
        ]
        if not fuels:
            return 0.0
        denom = sum(s for _, s in fuels) or 1e-12
        return sum(s * float(efs.get(c, 0.0)) for c, s in fuels) / denom

    def _feedstock_EF(proc_name: str) -> float:
        # EF of the fuels a process converts (e.g. coking coal in the coke ovens).
        proc = recipes_dict.get(proc_name)
        if not proc:
            return 0.0
        num = den = 0.0
        for carrier, qty in (proc.inputs or {}).items():
            if carrier in gas_carrier_names:
                continue
            try:
                lhv = float(energy_content.get(carrier, 0.0) or 0.0)
                ef = float(e_efs.get(carrier, 0.0) or 0.0)
                q = float(qty or 0.0)
            except (TypeError, ValueError):
                continue
            if lhv > 0 and ef > 0 and q > 0:
                num += q * lhv * ef
                den += q * lhv
        return num / den if den > 0 else 0.0

    def _source_EF(proc_name: str, carrier: Optional[str] = None) -> float:
        # Recovered gas carries the carbon intensity of what its source burns or,
        # for a conversion process fired on its own gas, of what it converts.
        ef = _blend_EF(energy_shares.get(proc_name, {}), e_efs)
        if ef <= 0:
            ef = _feedstock_EF(proc_name)
        if ef <= 0 and carrier:
            ef = float(e_efs.get(carrier, e_efs.get(process_gas_carrier, 0.0)) or 0.0)
        return ef

    def _collect_sources(specs: List[Dict[str, Any]], levels: Dict[str, float]) -> List[Dict[str, Any]]:
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
            runs = float(levels.get(proc_name, 0.0) or 0.0)
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

    ef_natural_gas_grid = float(e_efs.get(natural_gas_carrier, 0.0) or 0.0)
    ef_electricity_grid = float(e_efs.get('Electricity', 0.0) or 0.0)

    EF_coke_gas = _source_EF('Coke Production') if 'Coke Production' in recipes_dict else 0.0
    EF_bf_gas = _source_EF('Blast Furnace') if 'Blast Furnace' in recipes_dict else 0.0

    process_gas_specs = gas_config.get('process_gas_sources') or []
    if route_preset and route_preset not in {"BF-BOF", "BF_BOF"}:
        process_gas_specs = []

    # ---- gas produced in THIS run, and the EF it carries -------------------
    contributions = _collect_sources(process_gas_specs, prod_levels)
    gas_source_details: Dict[str, float] = {}
    for row in contributions:
        gas_source_details[row['process']] = gas_source_details.get(row['process'], 0.0) + row['mj']
    total_gas_MJ = float(sum(gas_source_details.values()))
    gas_coke_MJ = gas_source_details.get('Coke Production', 0.0)
    gas_bf_MJ = gas_source_details.get('Blast Furnace', 0.0)

    if total_gas_MJ > 1e-9:
        ef_weighted = sum(_source_EF(r['process'], r['carrier']) * r['mj'] for r in contributions)
        EF_process_gas = ef_weighted / total_gas_MJ
    else:
        # No recovered gas on this route: anything a recipe books as process gas
        # is bought as natural gas (moved below), so give it that factor.
        EF_process_gas = ef_natural_gas_grid
    if EF_process_gas <= 0:
        EF_process_gas = ef_natural_gas_grid

    try:
        util_eff = float(recipes_dict.get(utility_process_name, Process('', {}, {})).outputs.get('Electricity', 0.0))
    except Exception:
        util_eff = 0.0
    if util_eff <= 0 and utility_process_name != 'Utility Plant':
        try:
            util_eff = float(recipes_dict.get('Utility Plant', Process('', {}, {})).outputs.get('Electricity', 0.0))
        except Exception:
            util_eff = 0.0

    # ---- routing parameters -------------------------------------------------
    gas_routing = scenario.get('gas_routing', {}) or {}
    loss_fraction = _clip01(
        gas_routing.get('loss_fraction', gas_config.get('default_loss_fraction')), 0.0)
    flexible_share = _clip01(
        gas_routing.get('flexible_share', gas_config.get('default_flexible_share')), 1.0)
    legacy_keys = {'direct_use_fraction', 'electricity_fraction'} & set(gas_routing)
    if legacy_keys and not _LEGACY_SPLIT_WARNED:
        warnings.warn(
            "gas_routing.direct_use_fraction/electricity_fraction are ignored: recovered gas "
            "is now allocated by loss_fraction and flexible_share (see forge.core.gas).",
            DeprecationWarning,
            stacklevel=2,
        )
        _LEGACY_SPLIT_WARNED = True

    # Processes outside the mill (finishing and manufacturing lines downstream of
    # the plant) buy natural gas and grid power: recovered gas and internal
    # electricity serve the mill only.
    outside = set(scenario.get('outside_mill_procs') or [])

    # ---- the full-plant reference the allocation is computed on ------------
    def _demands(eb: pd.DataFrame) -> Tuple[float, float, float]:
        if eb is None or eb.empty:
            return 0.0, 0.0, 0.0
        rows = [r for r in eb.index if r not in ('TOTAL', utility_process_name) and r not in outside]

        def _sum(col: str) -> float:
            if col not in eb.columns or not rows:
                return 0.0
            return float(eb.loc[rows, col].clip(lower=0).sum())
        return _sum(process_gas_carrier), _sum(natural_gas_carrier), _sum('Electricity')

    native_run, ng_run, elec_run = _demands(energy_balance)
    ref_totals: Dict[str, Any] = {}
    if compute_inside_reference_fn:
        try:
            ref_totals = compute_inside_reference_fn(
                recipes=recipes,
                energy_int=energy_int,
                energy_shares=energy_shares,
                energy_content=energy_content,
                params=params,
                route_key=scenario.get('route_preset', None) or '',
                demand_qty=float(scenario.get('demand_qty', 1000.0)),
                stage_ref=scenario.get('stage_ref', 'IP3'),
                gas_carriers=[natural_gas_carrier, process_gas_carrier],
            ) or {}
        except Exception:
            ref_totals = {}

    by_carrier = ref_totals.get('gas_by_carrier') if isinstance(ref_totals, dict) else None
    if by_carrier:
        native_ref = float(by_carrier.get(process_gas_carrier, 0.0))
        ng_ref = float(by_carrier.get(natural_gas_carrier, 0.0))
        ref_levels = ref_totals.get('prod_levels') or prod_levels
        gas_ref = float(sum(r['mj'] for r in _collect_sources(process_gas_specs, ref_levels)))
    else:
        native_ref, ng_ref = native_run, ng_run
        gas_ref = total_gas_MJ

    elec_ref = float(scenario.get('inside_elec_ref', 0.0) or 0.0)
    if elec_ref <= 0.0:
        elec_ref = float(ref_totals.get('electricity_total', 0.0) or 0.0) if ref_totals else 0.0
    if elec_ref <= 0.0:
        elec_ref = elec_run

    pool = allocate_gas_pool(
        gas_MJ=gas_ref,
        native_demand_MJ=native_ref,
        ng_demand_MJ=ng_ref,
        elec_demand_MJ=elec_ref,
        util_eff=util_eff,
        loss_fraction=loss_fraction,
        flexible_share=flexible_share,
    )
    f_native = pool['f_native']
    f_internal_gas = pool['f_internal_gas']
    f_internal = pool['f_internal']
    flare_fraction = pool['flare_fraction']

    # ---- apply the factors to THIS run's energy balance ---------------------
    eb = energy_balance.copy()
    for col in (natural_gas_carrier, process_gas_carrier):
        if col not in eb.columns:
            eb[col] = 0.0
    proc_rows = [r for r in eb.index if r not in ('TOTAL', utility_process_name)]
    inside_rows = [r for r in proc_rows if r not in outside]
    outside_rows = [r for r in proc_rows if r in outside]
    if inside_rows:
        ng0 = eb.loc[inside_rows, natural_gas_carrier].astype(float).clip(lower=0)
        pg0 = eb.loc[inside_rows, process_gas_carrier].astype(float).clip(lower=0)
        # recovered gas replaces part of the natural gas; native process-gas
        # burners that the pool cannot serve buy natural gas instead
        eb.loc[inside_rows, natural_gas_carrier] = ng0 * (1.0 - f_internal_gas) + pg0 * (1.0 - f_native)
        eb.loc[inside_rows, process_gas_carrier] = pg0 * f_native + ng0 * f_internal_gas
    if outside_rows:
        ng0 = eb.loc[outside_rows, natural_gas_carrier].astype(float).clip(lower=0)
        pg0 = eb.loc[outside_rows, process_gas_carrier].astype(float).clip(lower=0)
        eb.loc[outside_rows, natural_gas_carrier] = ng0 + pg0
        eb.loc[outside_rows, process_gas_carrier] = 0.0
    if 'TOTAL' in eb.index and proc_rows:
        for col in (natural_gas_carrier, process_gas_carrier):
            eb.loc['TOTAL', col] = float(eb.loc[proc_rows, col].sum())

    # Internal power is shared over this run's consumers at the plant-wide
    # fraction, so a narrower boundary does not claim the whole plant's output.
    internal_elec_run = f_internal * elec_run
    eb = adjust_energy_balance(eb, internal_elec_run)

    e_efs = dict(e_efs)
    e_efs[natural_gas_carrier] = ef_natural_gas_grid
    e_efs[process_gas_carrier] = EF_process_gas
    ef_internal_electricity = (EF_process_gas / util_eff) if util_eff > 1e-12 else 0.0
    ef_electricity_used = f_internal * ef_internal_electricity + (1.0 - f_internal) * ef_electricity_grid
    ef_gas_blended = f_internal_gas * EF_process_gas + (1.0 - f_internal_gas) * ef_natural_gas_grid

    # Run-scaled quantities: the run makes total_gas_MJ; the reference split applies.
    scale = (total_gas_MJ / gas_ref) if gas_ref > 1e-12 else 0.0
    flared_gas_MJ = flare_fraction * total_gas_MJ
    direct_use_gas_MJ = (pool['native_used_MJ'] + pool['substituted_MJ']) * scale
    electricity_gas_MJ = pool['power_gas_MJ'] * scale

    gas_credit_details: Dict[str, Dict[str, float]] = {}
    for proc_name, mj in gas_source_details.items():
        share = mj / total_gas_MJ if total_gas_MJ > 0 else 0.0
        gas_credit_details[proc_name] = {
            'mj': mj,
            'share': share,
            'direct_use_MJ': share * direct_use_gas_MJ,
            'electricity_MJ': share * electricity_gas_MJ,
            'flared_MJ': share * flared_gas_MJ,
        }

    meta = {
        'total_process_gas_MJ': total_gas_MJ,
        'gas_coke_MJ': gas_coke_MJ,
        'gas_bf_MJ': gas_bf_MJ,
        'gas_sources_MJ': total_gas_MJ,
        'gas_source_details': gas_source_details,
        'gas_credit_details': gas_credit_details,
        # allocation (run-scaled MJ)
        'direct_use_gas_MJ': direct_use_gas_MJ,
        'electricity_gas_MJ': electricity_gas_MJ,
        'flared_gas_MJ': flared_gas_MJ,
        'loss_fraction': loss_fraction,
        'flexible_share': flexible_share,
        # realised shares of the recovered gas (reference plant)
        'direct_use_fraction': (pool['native_used_MJ'] + pool['substituted_MJ']) / gas_ref if gas_ref > 1e-12 else 0.0,
        'electricity_fraction': pool['power_gas_MJ'] / gas_ref if gas_ref > 1e-12 else 0.0,
        'flare_fraction': flare_fraction,
        # reference plant
        'gas_reference_MJ': gas_ref,
        'native_pg_demand_ref': native_ref,
        'ng_demand_ref': ng_ref,
        'total_gas_consumption_plant': native_ref + ng_ref,
        'pool_reference': pool,
        'f_native': f_native,
        'f_internal_gas': f_internal_gas,
        'f_internal': f_internal,
        'internal_elec_run_MJ': internal_elec_run,
        'inside_elec_ref': elec_ref,
        # emission factors (g CO2e/MJ)
        'ef_gas_blended': ef_gas_blended,
        'EF_coke_gas': EF_coke_gas,
        'EF_bf_gas': EF_bf_gas,
        'EF_process_gas': EF_process_gas,
        'util_eff': util_eff,
        'ef_internal_electricity': ef_internal_electricity,
        'process_gas_carrier': process_gas_carrier,
        'natural_gas_carrier': natural_gas_carrier,
        'utility_process': utility_process_name,
        'fallback_materials': list(scenario.get('fallback_materials', []) or []),
        'ef_nat_gas_grid': ef_natural_gas_grid,
        'ef_electricity_grid': ef_electricity_grid,
        'ef_electricity_used': ef_electricity_used,
    }

    return eb, e_efs, meta


def compute_inside_energy_reference_for_share(
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
    material_credit_map: Optional[Dict[str, Tuple[str, float]]] = None,
    exclude_processes: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Single full-plant reference run: gas and electricity demand, and its production levels.

    ``exclude_processes`` (the outside-mill lines) are left out of the demand:
    they buy natural gas and grid power, not the plant's recovered gas.
    """
    pre_mask = build_route_mask(route_key, recipes)
    stage_map = stage_lookup or STAGE_MATS
    if stage_ref not in stage_map:
        raise KeyError(f"Stage '{stage_ref}' not found while computing gas/electricity reference.")
    demand_mat = stage_map[stage_ref]
    production_routes_full = _build_routes_from_picks(
        recipes, demand_mat, picks_by_material={}, pre_mask=pre_mask, fallback_materials=fallback_materials
    )
    final_demand_full = {demand_mat: float(demand_qty)}
    import copy as _copy
    recipes_full = _copy.deepcopy(recipes)
    _ensure_fallback_processes(recipes_full, production_routes_full, fallback_materials)
    balance_matrix_full, prod_levels_full = calculate_balance_matrix(
        recipes_full, final_demand_full, production_routes_full, material_credit_map
    )
    if balance_matrix_full is None:
        return {"gas_total": 0.0, "electricity_total": 0.0, "gas_by_carrier": {}, "prod_levels": {}}
    energy_balance_full = calculate_energy_balance(prod_levels_full, energy_int, energy_shares)
    skip = set(exclude_processes or ())
    rows = [r for r in energy_balance_full.index if r not in ("TOTAL",) and r not in skip]
    carriers = gas_carriers or ["Gas"]
    by_carrier: Dict[str, float] = {}
    for carrier in carriers:
        if carrier in energy_balance_full.columns:
            by_carrier[carrier] = float(energy_balance_full.loc[rows, carrier].clip(lower=0).sum())
        else:
            by_carrier[carrier] = 0.0
    elec_total = 0.0
    if 'Electricity' in energy_balance_full.columns:
        elec_total = float(energy_balance_full.loc[rows, 'Electricity'].clip(lower=0).sum())
    return {
        "gas_total": float(sum(by_carrier.values())),
        "gas_by_carrier": by_carrier,
        "electricity_total": elec_total,
        "prod_levels": dict(prod_levels_full),
    }


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
    material_credit_map: Optional[Dict[str, Tuple[str, float]]] = None,
) -> float:
    refs = compute_inside_energy_reference_for_share(
        recipes=recipes,
        energy_int=energy_int,
        energy_shares=energy_shares,
        energy_content=energy_content,
        params=params,
        route_key=route_key,
        demand_qty=demand_qty,
        stage_ref=stage_ref,
        stage_lookup=stage_lookup,
        gas_carriers=gas_carriers,
        fallback_materials=fallback_materials,
        material_credit_map=material_credit_map,
    )
    try:
        return float(refs.get("gas_total", 0.0))
    except Exception:
        return 0.0


__all__ = [
    "calculate_internal_electricity",
    "adjust_energy_balance",
    "allocate_gas_pool",
    "apply_gas_routing_and_credits",
    "compute_inside_energy_reference_for_share",
    "compute_inside_gas_reference_for_share",
]
