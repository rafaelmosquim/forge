# -*- coding: utf-8 -*-
"""
steel_core_api_v2.py
Central API between the Streamlit app and the core model.

- RouteConfig / ScenarioInputs / RunOutputs dataclasses
- run_scenario(...) loads data, applies scenario overrides, locks route,
  builds production route from UI picks, runs balances & emissions
- Optional gating for BF process-gas → electricity credits driven by scenario
- Robust call to calculate_emissions(...) across versions
- write_run_log(...) helper for simple JSON logs
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple, Set
from collections import deque
from datetime import datetime
import inspect

import pandas as pd

# Core functions & models from your existing engine
from steel_model_core import (
    # IO
    load_data_from_yaml,
    load_parameters,
    load_recipes_from_yaml,
    load_market_config,
    load_electricity_intensity,
    # Overrides
    apply_fuel_substitutions,
    apply_dict_overrides,
    apply_recipe_overrides,
    # Route helpers
    STAGE_MATS,
    build_route_mask,
    enforce_eaf_feed,
    # Calculations
    calculate_balance_matrix,
    expand_energy_tables_for_active,
    calculate_internal_electricity,
    calculate_energy_balance,
    adjust_energy_balance,
    calculate_emissions,  # signature may vary; we guard below
    # Data classes/types
    Process,
    OUTSIDE_MILL_PROCS,
    compute_inside_elec_reference_for_share,
)

# ==============================
# Dataclasses
# ==============================
@dataclass
class RouteConfig:
    """Configuration for steel production route selection.
    
    Args:
        route_preset: Steel production route ('BF-BOF', 'DRI-EAF', etc.); avoids unreal routes by pre-selecting;
        stage_key: Production stage to stop at (key from STAGE_MATS); ensures no downstream process is used if not asked;
        demand_qty: Quantity demanded at the specified stage; locked at 1000 kg to ease comparisons; 
        picks_by_material: User selections for material producers; resolves ambiguity (i.e. N2 produced in house or bought);
        pre_select_soft: Processes to enable/disable by default
    """    
    route_preset: str               # 'BF-BOF' | 'DRI-EAF' | 'EAF-Scrap' | 'External' | 'auto'
    stage_key: str                  # key in STAGE_MATS
    demand_qty: float               # demand at stage
    picks_by_material: Dict[str, str] = field(default_factory=dict)
    pre_select_soft: Optional[Dict[str, int]] = None  # processes softly disabled (0/1)


@dataclass
class ScenarioInputs:

    """Execute the steel model for given scenario inputs. 
    Core model crashes if ambiguous producers are not resolved;
    App UI creates a single scenario, which API passes into the core for calculations.
    
    Args:
        data_dir: Path to directory containing YAML configuration files
        scn: Scenario configuration including route and parameters
        
    Returns:
        RunOutputs: Complete model results with balances and emissions
        
    Raises:
        FileNotFoundError: If required data files are missing
        ValueError: If scenario configuration is invalid
    """
    country_code: Optional[str]     # Electricity EF country (optional)
    scenario: Dict[str, Any]        # Full scenario dict (overrides, flags, etc.)
    route: RouteConfig              # Route config


@dataclass
class RunOutputs:

    """Complete results from a steel model scenario run.
    
    Attributes:
        production_routes: Dict mapping process names to enabled state (0/1)
        prod_levels: Dict mapping process names to production run counts
        energy_balance: DataFrame of energy flows by process and carrier (MJ)
        emissions: DataFrame of CO2e emissions by process (kg)
        total_co2e_kg: Total CO2e emissions for the scenario (kg)
        balance_matrix: DataFrame of material balances
        meta: Dictionary of metadata about the run
    """    
    production_routes: Dict[str, int]
    prod_levels: Dict[str, float]
    energy_balance: pd.DataFrame
    emissions: Optional[pd.DataFrame]
    total_co2e_kg: Optional[float]
    balance_matrix: Optional[pd.DataFrame] = None   # ← add this line
    meta: Dict[str, Any] = field(default_factory=dict)


# ==============================
# Helpers
# ==============================

def _credit_enabled(scn: dict | None) -> bool:
    """
    Returns True if recovered process-gas → electricity credit should be applied.
    Recognized flags in scenario:
      - process_gas_credit: true/false
      - bf_gas_credit: true/false
      - credits: { process_gas: true/false }
    Default True (enabled) unless explicitly disabled.
    """
    if not isinstance(scn, dict):
        return True
    for k in ("process_gas_credit", "bf_gas_credit"):
        if k in scn:
            v = scn[k]
            if isinstance(v, str):
                return v.strip().lower() not in {"false", "0", "no", "off"}
            return bool(v)
    credits = scn.get("credits")
    if isinstance(credits, dict) and "process_gas" in credits:
        v = credits["process_gas"]
        if isinstance(v, str):
            return v.strip().lower() not in {"false", "0", "no", "off"}
        return bool(v)
    return True


def _build_producers_index(recipes: List[Process]) -> Dict[str, List[Process]]:
    prod = {}
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
) -> Dict[str, int]:
    """
    Traverse upstream from demand_mat and build a 0/1 mask of chosen processes.
    If multiple producers exist for a material:
      - use picks_by_material[material] if present,
      - else pick the first enabled producer (deterministic).
    pre_mask / pre_select can disable producers (0/1).
    """
    pre_mask = pre_mask or {}
    pre_select = pre_select or {}
    producers = _build_producers_index(recipes)
    chosen: Dict[str, int] = {}
    visited_mats: Set[str] = set()
    q = deque([demand_mat])

    while q:
        mat = q.popleft()
        if mat in visited_mats:
            continue
        visited_mats.add(mat)

        cand = producers.get(mat, [])
        enabled = [r for r in cand if pre_mask.get(r.name, 1) > 0 and pre_select.get(r.name, 1) > 0]
        if not enabled:
            continue

        # Decide pick
        if len(enabled) == 1:
            pick = enabled[0]
        else:
            pick_name = picks_by_material.get(mat)
            if pick_name is None:
                pick = enabled[0]  # deterministic default
            else:
                pick = next((r for r in enabled if r.name == pick_name), enabled[0])

        # Record chosen vs others
        chosen[pick.name] = 1
        for r in cand:
            if r.name != pick.name:
                chosen[r.name] = 0

        # Recurse on inputs
        for im in pick.inputs.keys():
            if im not in visited_mats:
                q.append(im)

    return chosen


def _infer_eaf_mode(route_preset: str) -> Optional[str]:
    return {
        "EAF-Scrap": "scrap",
        "DRI-EAF": "dri",
        "BF-BOF": None,
        "External": None,
        "auto": None,
    }.get(route_preset, None)


def _robust_call_calculate_emissions(calc_fn, **kwargs):
    """
    Call calculate_emissions with only the parameters it actually accepts.
    This avoids crashes across slightly different local signatures.
    """
    sig = inspect.signature(calc_fn)
    usable = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return calc_fn(**usable)


def write_run_log(log_dir: str, payload: Dict[str, Any]) -> str:
    """
    Write a compact JSON log (config + total CO2e). Returns the file path.

        Args:
        log_dir: Directory path where log file should be written
        payload: Dictionary containing scenario config and results
        
    Returns:
        str: Path to the created log file
        
    Note:
        Log files are named with UTC timestamp: run_YYYYMMDDTHHMMSSZ.json
    """
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"run_{ts}.json"
    fpath = os.path.join(log_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return fpath

def compute_inside_gas_reference_for_share(
    recipes: List[Process],
    energy_int: Dict[str, float],
    energy_shares: Dict[str, Dict[str, float]],
    energy_content: Dict[str, float],
    params: Any,
    route_key: str,
    demand_qty: float,
    stage_ref: str = "IP3"
) -> float:
    """
    Compute total plant-level gas consumption for the entire production route.
    This provides a fixed reference regardless of user's stop-at-stage.
    """
    # Build a production route for the entire plant (to final product)
    from steel_model_core import build_route_mask, calculate_balance_matrix
    
    pre_mask = build_route_mask(route_key, recipes)
    demand_mat = STAGE_MATS[stage_ref]
    
    # Build production route deterministically (no picks)
    production_routes_full = _build_routes_from_picks(
        recipes,
        demand_mat,
        picks_by_material={},  # Use defaults
        pre_mask=pre_mask,
    )
    
    final_demand_full = {demand_mat: demand_qty}
    balance_matrix_full, prod_levels_full = calculate_balance_matrix(
        recipes, final_demand_full, production_routes_full
    )
    
    if balance_matrix_full is None:
        return 0.0
    
    # Calculate energy balance for full route
    energy_balance_full = calculate_energy_balance(prod_levels_full, energy_int, energy_shares)
    
    # Sum all gas consumption across the plant
    total_gas_consumption = 0.0
    if 'Gas' in energy_balance_full.columns:
        total_gas_consumption = energy_balance_full['Gas'].sum()
    
    return total_gas_consumption
# ==============================
# Main API
# ==============================
def run_scenario(data_dir: str, scn: ScenarioInputs) -> RunOutputs:
    """
    Execute the model for the given ScenarioInputs.
    Steps:
      1) Load base tables
      2) Apply scenario overrides (fuels, energy tables, EFs, params, recipes)
      3) Lock route preset; build route mask; enforce EAF feed
      4) Build production route from UI picks (or deterministically)
      5) Solve balances
      6) Apply/disable internal-electricity credit per scenario
      7) Compute emissions (robust to signature)
    """
    scenario: Dict[str, Any] = scn.scenario or {}
    route_preset: str = scn.route.route_preset or "auto"
    stage_key: str = scn.route.stage_key
    demand_qty: float = float(scn.route.demand_qty)
    picks_by_material: Dict[str, str] = scn.route.picks_by_material or {}
    pre_select_soft: Dict[str, int] = scn.route.pre_select_soft or {}
    country_code: Optional[str] = scn.country_code or None

    # Flag for BF process-gas → electricity credit
    credit_on: bool = _credit_enabled(scenario)

    # ---------- Load base data ----------
    base = os.path.join(data_dir, "")
    # allow scenario to select an alternate energy_int file
    _ei_file = (scenario.get('energy_int_file') or 'energy_int.yml').strip()
    # simple guard against weird inputs
    _allowed = {'energy_int.yml', 'energy_int_min.yml', 'energy_int_max.yml', 'energy_int_likely.yml'}
    if _ei_file not in _allowed:
        _ei_file = 'energy_int.yml'
    energy_int = load_data_from_yaml(os.path.join(base, _ei_file))
    # (optional) record in meta later
    energy_shares  = load_data_from_yaml(os.path.join(base, 'energy_matrix.yml'))
    energy_content = load_data_from_yaml(os.path.join(base, 'energy_content.yml'))
    e_efs          = load_data_from_yaml(os.path.join(base, 'emission_factors.yml'))
    params         = load_parameters      (os.path.join(base, 'parameters.yml'))
    mkt_cfg        = load_market_config   (os.path.join(base, 'mkt_config.yml'))
    elec_map       = load_electricity_intensity(os.path.join(base, 'electricity_intensity.yml')) or {}

    # Country-driven Electricity EF (if provided in UI)
    if country_code and country_code in elec_map:
        try:
            e_efs['Electricity'] = float(elec_map[country_code])
        except Exception:
            pass  # keep original if casting fails

    # Initial recipes
    recipes = load_recipes_from_yaml(
        os.path.join(base, 'recipes.yml'),
        params, energy_int, energy_shares, energy_content
    )
    # ---- Enforce route-locked energy_int override (defense-in-depth) ----
    allowed_proc_by_route = {
        "BF-BOF":    "Blast Furnace",
        "DRI-EAF":   "Direct Reduction Iron",
        "EAF-Scrap": "Electric Arc Furnace",
    }
    allowed_proc = allowed_proc_by_route.get(route_preset)
    if allowed_proc:
        ei = scenario.get('energy_int')
        if isinstance(ei, dict):
            # keep only the allowed key; drop all others
            scenario['energy_int'] = {k: v for k, v in ei.items() if k == allowed_proc and v is not None}
        else:
            scenario['energy_int'] = {}
    else:
        # No overrides allowed for 'External'/'auto'
        scenario['energy_int'] = {}

    # ---------- Scenario overrides ----------
    apply_fuel_substitutions(scenario.get('fuel_substitutions', {}), energy_shares, energy_int, energy_content, e_efs)
    apply_dict_overrides(energy_int,     scenario.get('energy_int', {}))
    apply_dict_overrides(energy_shares,  scenario.get('energy_matrix', {}))
    apply_dict_overrides(energy_content, scenario.get('energy_content', {}))
    apply_dict_overrides(e_efs,          scenario.get('emission_factors', {}))

    # Parameters (light/deep merge)
    from types import SimpleNamespace
    def _recursive_ns_update(ns, patch):
        for k, v in (patch or {}).items():
            if isinstance(v, dict):
                cur = getattr(ns, k, None)
                if not isinstance(cur, SimpleNamespace):
                    cur = SimpleNamespace()
                    setattr(ns, k, cur)
                _recursive_ns_update(cur, v)
            else:
                setattr(ns, k, v)

    _param_patch = scenario.get('param_overrides', None)
    if _param_patch is None:
        _param_patch = scenario.get('parameters', {})
    _recursive_ns_update(params, _param_patch)

    # Intensity adjustments
    from steel_model_core import adjust_blast_furnace_intensity, adjust_process_gas_intensity
    adjust_blast_furnace_intensity(energy_int, energy_shares, params)
    adjust_process_gas_intensity('Coke Production', 'process_gas_coke', energy_int, energy_shares, params)

    # Re-load recipes to re-evaluate expressions with updated params; then recipe overrides
    recipes = load_recipes_from_yaml(
        os.path.join(base, 'recipes.yml'),
        params, energy_int, energy_shares, energy_content
    )
    recipes = apply_recipe_overrides(recipes, scenario.get('recipe_overrides', {}), params, energy_int, energy_shares, energy_content)

    # ---------- Route mask & feed enforcement ----------
    pre_mask = build_route_mask(route_preset, recipes)
    eaf_mode = _infer_eaf_mode(route_preset)

    # Work on a copy for calculations (enforce feed)
    import copy
    recipes_calc = copy.deepcopy(recipes)
    enforce_eaf_feed(recipes_calc, eaf_mode)

    # ---------- Build production route from picks ----------
    demand_mat = STAGE_MATS[stage_key]
    production_routes: Dict[str, int] = _build_routes_from_picks(
        recipes_calc,
        demand_mat,
        picks_by_material,
        pre_mask=pre_mask,
        pre_select=pre_select_soft,
    )

    # ---------- Solve balances ----------
    final_demand = {demand_mat: demand_qty}

    balance_matrix, prod_levels = calculate_balance_matrix(recipes_calc, final_demand, production_routes)
    
    if balance_matrix is None:
        # Return empty-ish structures with a message rather than crashing
        return RunOutputs(
            production_routes=production_routes,
            prod_levels={},
            energy_balance=pd.DataFrame(),
            emissions=None,
            total_co2e_kg=None,
            balance_matrix=pd.DataFrame(),
            meta={"error": "Material balance failed"},
        )

    # Ensure energy tables have rows for all active variants
    active_procs = [p for p, r in prod_levels.items() if r > 1e-9]
    from steel_model_core import expand_energy_tables_for_active
    expand_energy_tables_for_active(active_procs, energy_shares, energy_int)

    # Internal electricity from recovered gases (before credit)
    recipes_dict = {r.name: r for r in recipes_calc}
    internal_elec = calculate_internal_electricity(prod_levels, recipes_dict, params)

    # Energy balance (base)
    energy_balance = calculate_energy_balance(prod_levels, energy_int, energy_shares)

    # Optional fix-ups to BF / Coke carriers (if your CLI does this)
    try:
        if 'Blast Furnace' in energy_balance.index and hasattr(params, 'bf_base_intensity'):
            bf_runs = float(prod_levels.get('Blast Furnace', 0.0))
            base_bf = float(params.bf_base_intensity)
            bf_sh   = energy_shares.get('Blast Furnace', {})
            for carrier in energy_balance.columns:
                if carrier != 'Electricity':
                    energy_balance.loc['Blast Furnace', carrier] = bf_runs * base_bf * float(bf_sh.get(carrier, 0.0))
        cp_runs = float(prod_levels.get('Coke Production', 0.0))
        base_cp = float(getattr(params, 'coke_production_base_intensity', energy_int.get('Coke Production', 0.0)))
        cp_sh   = energy_shares.get('Coke Production', {})
        if cp_runs and cp_sh:
            for carrier in energy_balance.columns:
                if carrier != 'Electricity':
                    energy_balance.loc['Coke Production', carrier] = cp_runs * base_cp * float(cp_sh.get(carrier, 0.0))
    except Exception:
        pass  # if fields missing, skip the fix-up

    # Compute recovered gas flows for dynamic EF (if needed by your emissions fn)
    try:
        gas_coke_MJ = prod_levels.get('Coke Production', 0.0) * recipes_dict.get('Coke Production', Process('',{},{})).outputs.get('Process Gas', 0.0)
    except Exception:
        gas_coke_MJ = 0.0
    try:
        # If params store bf_adj_intensity / bf_base_intensity
        bf_adj = float(getattr(params, 'bf_adj_intensity', 0.0))
        bf_base = float(getattr(params, 'bf_base_intensity', 0.0))
        gas_bf_MJ = (bf_adj - bf_base) * prod_levels.get('Blast Furnace', 0.0)
    except Exception:
        gas_bf_MJ = 0.0
    total_gas_MJ = float(gas_coke_MJ + gas_bf_MJ)

    # Dynamic EF for process gas based on energy shares & EF table
    def _blend_EF(shares: Dict[str, float], efs: Dict[str, float]) -> float:
        fuels = [(c, s) for c, s in shares.items() if c != 'Electricity' and s > 0]
        if not fuels:
            return 0.0
        denom = sum(s for _, s in fuels) or 1e-12
        return sum(s * float(efs.get(c, 0.0)) for c, s in fuels) / denom

    # brute force ef
    EF_coke_gas = 40 #_blend_EF(energy_shares.get('Coke Production', {}), e_efs)
    EF_bf_gas   = 260 #_blend_EF(energy_shares.get('Blast Furnace', {}), e_efs)
    EF_process_gas = EF_coke_gas if total_gas_MJ <= 1e-9 else (
        (EF_coke_gas * (gas_coke_MJ / max(1e-12, total_gas_MJ))) + (EF_bf_gas * (gas_bf_MJ / max(1e-12, total_gas_MJ)))
    )

    # Utility efficiency for converting process gas to electricity
    try:
        util_eff = recipes_dict.get('Utility Plant', Process('',{},{})).outputs.get('Electricity', 0.0)
    except Exception:
        util_eff = 0.0

    # NEW: Split gas 50% direct use, 50% electricity generation
    gas_routing = scenario.get('gas_routing', {})
    direct_use_fraction = gas_routing.get('direct_use_fraction', 0.5)
    electricity_fraction = gas_routing.get('electricity_fraction', 0.5)
    #direct_use_fraction = 0.5  # Hard-coded 50% for now
    #electricity_fraction = 0.5

    # Calculate split amounts
    direct_use_gas_MJ = total_gas_MJ * direct_use_fraction
    electricity_gas_MJ = total_gas_MJ * electricity_fraction

    # Internal electricity from only the electricity portion
    internal_elec = electricity_gas_MJ * util_eff

    # FIXED PLANT-LEVEL CALCULATIONS (not dependent on user's stop-at-stage)
    # Compute reference in-mill electricity (IP3 boundary)
    # FIX: Check if this function exists, provide fallback
    try:
        inside_elec_ref = compute_inside_elec_reference_for_share(
            recipes=recipes,
            energy_int=energy_int,
            energy_shares=energy_shares,
            energy_content=energy_content,
            params=params,
            route_key=route_preset,
            demand_qty=demand_qty,
            stage_ref="IP3",
        )
    except NameError:
        # Fallback if function doesn't exist
        inside_elec_ref = energy_balance.get('Electricity', pd.Series([0])).sum()

    # NEW: Compute plant-level total gas consumption (similar to electricity)
    total_gas_consumption_plant = compute_inside_gas_reference_for_share(
        recipes=recipes,
        energy_int=energy_int,
        energy_shares=energy_shares,
        energy_content=energy_content,
        params=params,
        route_key=route_preset,
        demand_qty=demand_qty,
        stage_ref="IP3",  # Fixed reference point
    )

    # FIX: Calculate f_internal_gas HERE, before using it
    if total_gas_consumption_plant > 1e-9:
        f_internal_gas = min(1.0, direct_use_gas_MJ / total_gas_consumption_plant)
    else:
        f_internal_gas = 0.0

    # Calculate blended emission factor for gas carrier
    ef_natural_gas = e_efs.get('Gas', 0.0)  # Default purchased gas EF
    ef_process_gas = EF_process_gas  # Our calculated process gas EF

    # Blend based on internal fraction
    ef_gas_blended = (f_internal_gas * ef_process_gas + 
                    (1 - f_internal_gas) * ef_natural_gas)

    # Update the emission factor for gas
    e_efs['Gas'] = ef_gas_blended

    # Also keep track of process gas separately if needed
    e_efs['Process Gas'] = ef_process_gas

    # Fixed plant-level internal electricity fraction
    if inside_elec_ref > 1e-9:
        f_internal = min(1.0, internal_elec / inside_elec_ref)
    else:
        f_internal = 0.0

    # Fixed internal electricity EF
    ef_internal_electricity = (EF_process_gas / util_eff) if util_eff > 1e-9 else 0.0

    # Apply/disable internal electricity AND gas credit
    if credit_on:
        energy_balance = adjust_energy_balance(energy_balance, internal_elec)
        
        # NEW: Apply direct gas use using plant-level fixed consumption
        if direct_use_gas_MJ > 0 and total_gas_consumption_plant > 1e-9:
            # Apply this fraction to reduce purchased gas across all processes
            for process_name in energy_balance.index:
                if 'Gas' in energy_balance.columns:
                    current_gas = energy_balance.loc[process_name, 'Gas']
                    if current_gas > 0:
                        # Reduce purchased gas by the internal fraction
                        reduction = current_gas * f_internal_gas
                        energy_balance.loc[process_name, 'Gas'] = current_gas - reduction
                        
                        # Track process gas usage (optional)
                        if 'Process Gas' not in energy_balance.columns:
                            energy_balance['Process Gas'] = 0.0
                        energy_balance.loc[process_name, 'Process Gas'] += reduction
    else:
        # FIX: Move this else block to only disable when credit is off
        internal_elec  = 0.0
        total_gas_MJ   = 0.0
        direct_use_gas_MJ = 0.0 # NEW: Also disable direct use when credit is off
        EF_process_gas = 0.0
        # do not adjust energy_balance when credit is off

    # Load process-emissions yaml for direct process emissions (if needed)
    process_emissions_table = load_data_from_yaml(os.path.join(base, 'process_emissions.yml'))

    # Emissions (robust to differing signatures)
    emissions = _robust_call_calculate_emissions(
        calculate_emissions,
        mkt_cfg=mkt_cfg,
        prod_lvl=prod_levels,            # some versions expect prod_lvl
        prod_level=prod_levels,          # others expect prod_level
        energy_balance=energy_balance,   # some versions expect energy_balance
        energy_df=energy_balance,        # others expect energy_df
        # emission factors
        e_efs=e_efs,                     # some versions expect e_efs
        energy_efs=e_efs,                # others expect energy_efs
        # process emissions table
        process_emissions_table=process_emissions_table,  # some versions expect this name
        process_efs=process_emissions_table,              # others expect process_efs
        internal_elec=internal_elec,
        final_demand=final_demand,
        total_gas_MJ=total_gas_MJ,
        EF_process_gas=EF_process_gas,
        # Fixed plant-level values
        internal_fraction_plant=f_internal,
        ef_internal_electricity=ef_internal_electricity,
    )

    # Ensure a TOTAL row if your function doesn't add it
    total_co2 = None
    try:
        if emissions is not None and not emissions.empty:
            if 'TOTAL' not in emissions.index and 'TOTAL CO2e' in emissions.columns:
                emissions.loc['TOTAL'] = emissions.sum()
            if 'TOTAL' in emissions.index and 'TOTAL CO2e' in emissions.columns:
                total_co2 = float(emissions.loc['TOTAL', 'TOTAL CO2e'])
            elif 'TOTAL CO2e' in emissions.columns:
                total_co2 = float(emissions['TOTAL CO2e'].sum())
    except Exception:
        pass
    fyield = float(getattr(params, "finished_yield", 0.85))    
    meta = {
        "route_preset": route_preset,
        "stage_key": stage_key,
        "demand_qty": demand_qty,
        "country_code": country_code,
        "process_gas_credit_enabled": bool(credit_on),
        "inside_elec_ref": inside_elec_ref,
        "f_internal": f_internal,
        "ef_internal_electricity": ef_internal_electricity,
        "finished_yield": fyield,
        # NEW: Gas routing information with plant-level logic
        "total_process_gas_MJ": total_gas_MJ,
        "direct_use_gas_MJ": direct_use_gas_MJ,
        "electricity_gas_MJ": electricity_gas_MJ,
        "total_gas_consumption_plant": total_gas_consumption_plant,
        "f_internal_gas": f_internal_gas,
        "ef_gas_blended": ef_gas_blended,
        "direct_use_fraction": direct_use_fraction,
        "electricity_fraction": electricity_fraction
    }

    # Debug prints for emission factors
    print("\n=== EMISSION FACTOR DEBUG ===")
    print(f"ELECTRICITY:")
    print(f"  Grid EF: {e_efs.get('Electricity', 0.0):.2f} kg CO2e/MJ")
    print(f"  Internal EF: {ef_internal_electricity:.2f} kg CO2e/MJ") 
    print(f"  Internal Fraction: {f_internal:.3f}")
    print(f"  Blended EF: {f_internal * ef_internal_electricity + (1 - f_internal) * e_efs.get('Electricity', 0.0):.2f} kg CO2e/MJ")

    print(f"\nGAS:")
    print(f"  Natural Gas EF: {ef_natural_gas:.2f} kg CO2e/MJ")
    print(f"  Process Gas EF: {ef_process_gas:.2f} kg CO2e/MJ")
    print(f"  Internal Fraction: {f_internal_gas:.3f}")
    print(f"  Blended EF: {ef_gas_blended:.2f} kg CO2e/MJ")

    print(f"\nPROCESS GAS BREAKDOWN:")
    print(f"  Coke Gas: {gas_coke_MJ:.1f} MJ (EF: {EF_coke_gas:.1f})")
    print(f"  BF Gas: {gas_bf_MJ:.1f} MJ (EF: {EF_bf_gas:.1f})")
    print(f"  Total: {total_gas_MJ:.1f} MJ (EF: {EF_process_gas:.1f})")
    print(f"  Direct Use: {direct_use_gas_MJ:.1f} MJ")
    print(f"  Electricity: {electricity_gas_MJ:.1f} MJ")
    print("============================\n")

    return RunOutputs(
        production_routes=production_routes,
        prod_levels=prod_levels,
        energy_balance=energy_balance,
        emissions=emissions,
        total_co2e_kg=total_co2,
        balance_matrix=balance_matrix,
        meta=meta,
    )