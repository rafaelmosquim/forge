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
from functools import partial

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
    calculate_lci,
    adjust_energy_balance,
    analyze_energy_costs,
    analyze_material_costs,
    calculate_emissions,  # signature may vary; we guard below
    # Data classes/types
    Process,
    OUTSIDE_MILL_PROCS,
    compute_inside_elec_reference_for_share,
    compute_inside_gas_reference_for_share,
    apply_gas_routing_and_credits,
    set_prefer_internal_processes,
    apply_inhouse_clamp,
)

from sector_descriptor import load_sector_descriptor
from scenario_resolver import (
    build_stage_material_map,
    build_route_mask_for_descriptor,
    reference_stage_for_gas,
    resolve_feed_mode,
    resolve_stage_material,
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
        stage_role: Descriptor menu key (e.g. 'validation', 'crude'); disambiguates shared stage_ids like 'Cast';
        demand_qty: Quantity demanded at the specified stage; locked at 1000 kg to ease comparisons; 
        picks_by_material: User selections for material producers; resolves ambiguity (i.e. N2 produced in house or bought);
        pre_select_soft: Processes to enable/disable by default
    """    
    route_preset: str               # 'BF-BOF' | 'DRI-EAF' | 'EAF-Scrap' | 'External' | 'auto'
    stage_key: str                  # key in STAGE_MATS
    demand_qty: float               # demand at stage
    stage_role: Optional[str] = None
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
        lci: DataFrame with material and energy inputs per process
        meta: Dictionary of metadata about the run
    """    
    production_routes: Dict[str, int]
    prod_levels: Dict[str, float]
    energy_balance: pd.DataFrame
    emissions: Optional[pd.DataFrame]
    total_co2e_kg: Optional[float]
    total_cost: Optional[float] = None
    material_cost: Optional[float] = None
    balance_matrix: Optional[pd.DataFrame] = None   # ← add this line
    lci: Optional[pd.DataFrame] = None
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


DEFAULT_PRODUCER_PRIORITY: Tuple[str, ...] = (
    "Continuous Casting (R)",
    "Hot Rolling",
    "Cold Rolling",
    "Basic Oxygen Furnace",
    "Electric Arc Furnace",
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
    fallback_materials: Optional[Set[str]] = None,
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
    fallback_set: Set[str] = set(fallback_materials or [])
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
        if fallback_set and mat in fallback_set:
            for r in cand:
                chosen[r.name] = 0
            continue
        enabled = [r for r in cand if pre_mask.get(r.name, 1) > 0 and pre_select.get(r.name, 1) > 0]
        if not enabled:
            continue

        # Decide pick
        if len(enabled) == 1:
            pick = enabled[0]
        else:
            pick_name = picks_by_material.get(mat)
            if pick_name is None:
                def _score(proc):
                    try:
                        idx = DEFAULT_PRODUCER_PRIORITY.index(proc.name)
                        return (0, idx, proc.name)
                    except ValueError:
                        return (1, proc.name)

                pick = min(enabled, key=_score)
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


def _resolve_stage_role(descriptor, stage_key: str, provided_role: Optional[str]) -> str:
    """Derive the stage role key ('validation', 'crude', etc.) when possible."""
    stage_role = (provided_role or "").strip().lower()
    if stage_role:
        return stage_role
    matches = [
        (item.key or "").strip().lower()
        for item in descriptor.stage_menu
        if str(item.stage_id).strip().lower() == str(stage_key).strip().lower()
    ]
    if len(matches) == 1:
        return matches[0]
    return stage_role


def _apply_route_overrides(
    pre_select: Dict[str, int],
    pre_mask: Dict[str, int],
    overrides: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Blend scenario route_overrides into the current pre-select/mask dictionaries."""
    if not overrides:
        return pre_select, pre_mask
    ps = dict(pre_select or {})
    pm = dict(pre_mask or {})
    for proc, raw in overrides.items():
        key = str(proc)
        try:
            val = float(raw)
        except Exception:
            val = 1.0 if bool(raw) else 0.0
        enabled = 1 if val > 0.0 else 0
        ps[key] = enabled
        if enabled:
            # remove hard bans when scenario explicitly forces enable
            if pm.get(key, 1) == 0:
                pm.pop(key, None)
        else:
            pm[key] = 0
    return ps, pm


def _ensure_fallback_processes(
    recipes: List[Process],
    production_routes: Dict[str, int],
    fallback_materials: Optional[Set[str]],
) -> None:
    if not fallback_materials:
        return
    fallback_set = {str(mat).strip() for mat in (fallback_materials or set()) if str(mat).strip()}
    if not fallback_set:
        return
    existing = {r.name for r in recipes}
    for mat in fallback_set:
        proc_name = f"External {mat} (auto)"
        if proc_name not in existing:
            recipes.append(Process(proc_name, inputs={}, outputs={mat: 1.0}))
            existing.add(proc_name)
        production_routes.setdefault(proc_name, 1)


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
    stage_ref: str = "IP3",
    stage_lookup: Optional[Dict[str, str]] = None,
    gas_carrier: str = "Gas",
    fallback_materials: Optional[Set[str]] = None,
) -> float:
    """
    Compute total plant-level gas consumption for the entire production route.
    This provides a fixed reference regardless of user's stop-at-stage.
    """
    # Build a production route for the entire plant (to final product)
    from steel_model_core import build_route_mask, calculate_balance_matrix
    
    pre_mask = build_route_mask(route_key, recipes)
    stage_map = stage_lookup or STAGE_MATS
    if stage_ref not in stage_map:
        raise KeyError(f"Stage '{stage_ref}' not found while computing gas reference.")
    demand_mat = stage_map[stage_ref]
    
    # Build production route deterministically (no picks)
    production_routes_full = _build_routes_from_picks(
        recipes,
        demand_mat,
        picks_by_material={},  # Use defaults
        pre_mask=pre_mask,
        fallback_materials=fallback_materials,
    )
    
    final_demand_full = {demand_mat: demand_qty}
    import copy as _copy
    recipes_full = _copy.deepcopy(recipes)
    _ensure_fallback_processes(recipes_full, production_routes_full, fallback_materials)
    balance_matrix_full, prod_levels_full = calculate_balance_matrix(
        recipes_full, final_demand_full, production_routes_full
    )
    
    if balance_matrix_full is None:
        return 0.0
    
    # Calculate energy balance for full route
    energy_balance_full = calculate_energy_balance(prod_levels_full, energy_int, energy_shares)
    
    # Sum all gas consumption across the plant
    total_gas_consumption = 0.0
    if gas_carrier in energy_balance_full.columns:
        total_gas_consumption = float(energy_balance_full[gas_carrier].sum())
    
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
        # ADD THIS - it will show you the exact structure
    print("🎯 SCENARIO STRUCTURE CAPTURED:")
    print("Scenario keys:", list(scn.scenario.keys()))
    print("Full scenario:")
    import json
    print(json.dumps(scn.scenario, indent=2, default=str))
    
    # Save to file for inspection
    with open('DEBUG_scenario_structure.json', 'w') as f:
        json.dump(scn.scenario, f, indent=2, default=str)
        
    scenario: Dict[str, Any] = scn.scenario or {}
    route_preset: str = scn.route.route_preset or "auto"
    stage_key: str = scn.route.stage_key
    stage_role_input: str = (scn.route.stage_role or "").strip().lower()
    demand_qty: float = float(scn.route.demand_qty)
    picks_by_material: Dict[str, str] = scn.route.picks_by_material or {}
    pre_select_soft: Dict[str, int] = scn.route.pre_select_soft or {}
    country_code: Optional[str] = scn.country_code or None

    scenario.setdefault('gas_routing', {})

    # Flag for BF process-gas → electricity credit
    credit_on: bool = _credit_enabled(scenario)

    # ---------- Load base data ----------
    base = os.path.join(data_dir, "")
    descriptor = load_sector_descriptor(base)
    stage_material_map = build_stage_material_map(descriptor)
    STAGE_MATS.update(stage_material_map)
    
    stage_role = _resolve_stage_role(descriptor, stage_key, stage_role_input)
    is_validation = (stage_role == 'validation')
    os.environ['STEEL_MODEL_STAGE'] = 'validation' if is_validation else ''
    print(
        f"🔍 Stage key: {stage_key}, Stage role: {stage_role or '(unspecified)'}, "
        f"Environment stage: {os.environ.get('STEEL_MODEL_STAGE', '')}"
    )

    # For validation stage, override any user picks for auxiliaries
    if is_validation:
        # Force market purchases for auxiliaries
        picks_by_material.update({
            'Nitrogen': 'Nitrogen from market',
            'Oxygen': 'Oxygen from market',
            'Dolomite': 'Dolomite from market',
            'Burnt Lime': 'Burnt Lime from market'
        })
        # Ensure these cannot be overridden
        pre_select_soft.update({
            'Nitrogen Production': 0,
            'Oxygen Production': 0,
            'Dolomite Production': 0,
            'Burnt Lime Production': 0,
            'Nitrogen from market': 1,
            'Oxygen from market': 1,
            'Dolomite from market': 1,
            'Burnt Lime from market': 1
        })

    fallback_materials = set(descriptor.balance_fallback_materials or set())
    scenario['fallback_materials'] = list(fallback_materials)
    # Determine process preferences based on stage
    if is_validation:
        # For validation stage, force market purchases
        prefer_internal_map = {
            "Nitrogen Production": ["Nitrogen from market"],
            "Oxygen Production": ["Oxygen from market"],
            "Dolomite Production": ["Dolomite from market"],
            "Burnt Lime Production": ["Burnt Lime from market"]
        }
    else:
        # For other stages, use descriptor preferences
        raw_prefer_internal = descriptor.prefer_internal_processes or {}
        prefer_internal_map = {}
        for market_proc, internal_proc in raw_prefer_internal.items():
            if not internal_proc:
                continue
            internal_key = str(internal_proc)
            prefer_internal_map.setdefault(internal_key, []).append(str(market_proc))
    
    # Set the preferences in the scenario
    scenario['prefer_internal_processes'] = prefer_internal_map
    external_purchase_rows = list(descriptor.costing.external_purchase_rows or [])
    scenario['external_purchase_rows'] = external_purchase_rows
    set_prefer_internal_processes(prefer_internal_map)
    gas_reference_stage = reference_stage_for_gas(descriptor)
    gas_config = {
        "process_gas_carrier": descriptor.gas.process_gas_carrier or "Process Gas",
        "natural_gas_carrier": descriptor.gas.natural_gas_carrier or "Gas",
        "utility_process": descriptor.gas.utility_process,
        "default_direct_use_fraction": descriptor.gas.default_direct_use_fraction,
    }
    gas_sources = [
        name for name, roles in descriptor.process_roles.items()
        if any(str(r).lower() == 'gas_source' for r in roles)
    ]
    if gas_sources:
        gas_config['gas_sources'] = gas_sources
    process_gas_carrier = gas_config["process_gas_carrier"]
    natural_gas_carrier = gas_config["natural_gas_carrier"]
    if (
        'direct_use_fraction' not in scenario['gas_routing']
        and gas_config["default_direct_use_fraction"] is not None
    ):
        scenario['gas_routing']['direct_use_fraction'] = gas_config["default_direct_use_fraction"]
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
            e_val = float(elec_map[country_code])
            e_efs['Electricity'] = e_val
            ef_overrides = scenario.get('emission_factors')
            if isinstance(ef_overrides, dict) and 'Electricity' in ef_overrides:
                ef_overrides = dict(ef_overrides)
                ef_overrides.pop('Electricity', None)
                scenario['emission_factors'] = ef_overrides
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

    # ---------- Build route constraints ----------
    # For validation stage, we want to:
    # 1. Only clamp auxiliary processes (nitrogen, oxygen, etc) to be market-purchased
    # 2. Keep other route choices (BF/EAF etc) flexible according to picks
    # 3. Not apply the fixed downstream path from build_pre_for_route
    if is_validation:
        print("🔍 Applying validation stage constraints")
        # For validation stage, explicitly force market purchases
        pre_mask = {
            "Nitrogen Production": 0,
            "Oxygen Production": 0,
            "Dolomite Production": 0,
            "Burnt Lime Production": 0
        }
        # Forcibly enable market purchases AND add to picks
        market_purchases = {
            "Nitrogen from market": 1,
            "Oxygen from market": 1,
            "Dolomite from market": 1,
            "Burnt Lime from market": 1
        }
        pre_select_soft.update(market_purchases)
        # Also force these in picks_by_material
        picks_by_material.update({
            'Nitrogen': 'Nitrogen from market',
            'Oxygen': 'Oxygen from market',
            'Dolomite': 'Dolomite from market',
            'Burnt Lime': 'Burnt Lime from market'
        })
        
        # Add EAF feed mode for validation if needed
        eaf_mode = resolve_feed_mode(descriptor, route_preset)
        if eaf_mode is None:
            eaf_mode = _infer_eaf_mode(route_preset)
            
        # Ensure validation stage is set in environment
        os.environ['STEEL_MODEL_STAGE'] = 'validation'
    else:
        # Non-validation: apply full route masks and in-house preferences
        pre_mask = (
            build_route_mask_for_descriptor(descriptor, route_preset, recipes)
            or build_route_mask(route_preset, recipes)
        )
        eaf_mode = resolve_feed_mode(descriptor, route_preset)
        if eaf_mode is None:
            eaf_mode = _infer_eaf_mode(route_preset)

        # Apply in-house preferences for auxiliaries based on stage
        if is_validation:
            # For validation, force market purchase of auxiliaries
            aux_mask = {
                "Nitrogen Production": 0,
                "Oxygen Production": 0,
                "Dolomite Production": 0,
                "Burnt Lime Production": 0,
            }
            aux_select = {
                "Nitrogen from market": 1,
                "Oxygen from market": 1,
                "Dolomite from market": 1,
                "Burnt Lime from market": 1,
            }
            pre_mask.update(aux_mask)
            pre_select_soft.update(aux_select)
        else:
            # Apply normal in-house clamp for non-validation stages
            pre_select_soft, pre_mask = apply_inhouse_clamp(pre_select_soft, pre_mask, prefer_internal_map)

    # Apply explicit route overrides from scenario (defensive copy)
    pre_select_soft, pre_mask = _apply_route_overrides(
        pre_select_soft,
        pre_mask,
        scenario.get('route_overrides'),
    )

    # Work on a copy for calculations (enforce feed)
    import copy
    recipes_calc = copy.deepcopy(recipes)
    enforce_eaf_feed(recipes_calc, eaf_mode)

    # ---------- Build production route from picks ----------
    demand_mat = resolve_stage_material(descriptor, stage_key)
    production_routes: Dict[str, int] = _build_routes_from_picks(
        recipes_calc,
        demand_mat,
        picks_by_material,
        pre_mask=pre_mask,
        pre_select=pre_select_soft,
        fallback_materials=fallback_materials,
    )

    _ensure_fallback_processes(recipes_calc, production_routes, fallback_materials)

    # ---------- Solve balances ----------
    final_demand = {demand_mat: demand_qty}

    balance_matrix, prod_levels = calculate_balance_matrix(recipes_calc, final_demand, production_routes)

    # ADD DEBUG CODE RIGHT HERE:
    print("🎯 CORE INPUTS CAPTURED:")
    print("1. final_demand:", final_demand)
    print("2. production_routes sample:", dict(list(production_routes.items())[:3]))
    print("3. recipes_calc count:", len(recipes_calc))
    print("4. scenario keys:", list(scenario.keys()))

    # Save what core actually receives
    core_inputs = {
        "final_demand": final_demand,
        "production_routes": production_routes,
        "scenario_dict": scenario,
        "recipes_sample": [r.name for r in recipes_calc[:5]],  # Just names to avoid large output
    }

    with open('DEBUG_core_balance_inputs.json', 'w') as f:
        json.dump(core_inputs, f, indent=2, default=str)

    print("✅ Saved balance inputs to DEBUG_core_balance_inputs.json")
    
    if balance_matrix is None:
        # Return empty-ish structures with a message rather than crashing
        return RunOutputs(
            production_routes=production_routes,
            prod_levels={},
            energy_balance=pd.DataFrame(),
            emissions=None,
            total_co2e_kg=None,
            total_cost=None,
            material_cost=None,
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
        if 'Blast Furnace' in energy_balance.index:
            bf_runs = float(prod_levels.get('Blast Furnace', 0.0))
            bf_intensity = float(
                getattr(
                    params,
                    'bf_adj_intensity',
                    getattr(params, 'bf_base_intensity', energy_int.get('Blast Furnace', 0.0)),
                )
            )
            bf_sh   = energy_shares.get('Blast Furnace', {})
            for carrier in energy_balance.columns:
                if carrier != 'Electricity':
                    energy_balance.loc['Blast Furnace', carrier] = bf_runs * bf_intensity * float(bf_sh.get(carrier, 0.0))
        cp_runs = float(prod_levels.get('Coke Production', 0.0))
        base_cp = float(getattr(params, 'coke_production_base_intensity', energy_int.get('Coke Production', 0.0)))
        cp_sh   = energy_shares.get('Coke Production', {})
        if cp_runs and cp_sh:
            for carrier in energy_balance.columns:
                if carrier != 'Electricity':
                    energy_balance.loc['Coke Production', carrier] = cp_runs * base_cp * float(cp_sh.get(carrier, 0.0))
    except Exception:
        pass  # if fields missing, skip the fix-up

    # Defer LCI build until after gas routing adjustments
    lci_df = None

    # Delegate gas routing, EF blending and credit application to core helper
    gas_reference_fn = partial(
        compute_inside_gas_reference_for_share,
        stage_lookup=stage_material_map,
        gas_carrier=natural_gas_carrier,
        fallback_materials=fallback_materials,
    )

    energy_balance, e_efs, gas_meta = apply_gas_routing_and_credits(
        energy_balance=energy_balance,
        recipes=recipes_calc,
        prod_levels=prod_levels,
        params=params,
        energy_shares=energy_shares,
        energy_int=energy_int,
        energy_content=energy_content,
        e_efs=e_efs,
        scenario={
            'gas_routing': scenario.get('gas_routing', {}),
            'route_preset': route_preset,
            'demand_qty': demand_qty,
            'stage_ref': gas_reference_stage,
            'gas_config': gas_config,
            'process_roles': descriptor.process_roles,
            'stage_lookup': stage_material_map,
            'fallback_materials': list(fallback_materials),
        },
        credit_on=credit_on,
        compute_inside_gas_reference_fn=gas_reference_fn,
    )

    # merge gas_meta into local variables for meta reporting
    total_gas_MJ = gas_meta.get('total_process_gas_MJ', 0.0)
    direct_use_gas_MJ = gas_meta.get('direct_use_gas_MJ', 0.0)
    electricity_gas_MJ = gas_meta.get('electricity_gas_MJ', 0.0)
    total_gas_consumption_plant = gas_meta.get('total_gas_consumption_plant', 0.0)
    f_internal_gas = gas_meta.get('f_internal_gas', 0.0)
    ef_gas_blended = gas_meta.get('ef_gas_blended', 0.0)
    # Load process-emissions yaml for direct process emissions (if needed)
    process_emissions_table = load_data_from_yaml(os.path.join(base, 'process_emissions.yml'))

    print("🎯 CORE ENERGY/EMISSIONS INPUTS:")
    print("1. prod_levels sample:", dict(list(prod_levels.items())[:3]))
    print("2. energy_balance shape:", energy_balance.shape)
    print("3. energy_int sample:", dict(list(energy_int.items())[:3]))
    print("4. e_efs sample:", dict(list(e_efs.items())[:3]))

    emissions_inputs = {
        "prod_levels_sample": dict(list(prod_levels.items())[:5]),
        "energy_balance_sample": energy_balance.head(3).to_dict(),
        "energy_int_sample": dict(list(energy_int.items())[:3]),
        "e_efs_sample": dict(list(e_efs.items())[:3]),
    }

    with open('DEBUG_core_emissions_inputs.json', 'w') as f:
        json.dump(emissions_inputs, f, indent=2, default=str)

    print("✅ Saved emissions inputs to DEBUG_core_emissions_inputs.json")

    # ensure we have plant-level inside_elec_ref and fixed internal ef values
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
    except Exception:
        inside_elec_ref = 0.0

    # Pull internal electricity diagnostics from gas_meta if available
    f_internal = gas_meta.get('f_internal', 0.0)
    ef_internal_electricity = gas_meta.get('ef_internal_electricity', 0.0)

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
        EF_process_gas=gas_meta.get('EF_process_gas', 0.0),
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

    # ----------- Energy Cost Calculation -----------
    try:
        # Load energy prices
        energy_prices_path = os.path.join(base, 'energy_prices.yml')
        energy_prices = load_data_from_yaml(energy_prices_path) or {}

        # Debug: Check what we are working with
        print(f"🔍 Material prices loaded: {bool(energy_prices)}")
        print(f"🔍 Material prices keys: {list(energy_prices.keys()) if energy_prices else 'None'}")

        # Calculate total cost using core function
        total_cost = analyze_energy_costs(energy_balance, energy_prices)
        print(f"🔍 Total cost calculated: {total_cost}")
        
    except Exception as e:
        print(f"❌ Error in cost calculation: {e}")
        import traceback
        traceback.print_exc()
        total_cost = 0.0  # Default to 0 instead of None

        # ----------- Material Cost Calculation -----------
    try:
        # Load energy prices
        material_prices_path = os.path.join(base, 'material_prices.yml')
        material_prices = load_data_from_yaml(material_prices_path) or {}

        # Debug: Check what we are working with
        print(f"🔍 Material prices loaded: {bool(material_prices)}")
        print(f"🔍 Material prices keys: {list(material_prices.keys()) if material_prices else 'None'}")

        # Calculate total cost using core function
        material_cost = analyze_material_costs(balance_matrix, material_prices, external_rows=external_purchase_rows)
        print(f"🔍 Material cost calculated: {material_cost}")
        
    except Exception as e:
        print(f"❌ Error in cost calculation: {e}")
        import traceback
        traceback.print_exc()
        material_cost = 0.0  # Default to 0 instead of None    

    meta = {
        "route_preset": route_preset,
        "stage_key": stage_key,
        "stage_role": stage_role,
        "sector_key": descriptor.key,
        "sector_name": descriptor.name,
        "demand_qty": demand_qty,
        "demand_material": demand_mat,
        "country_code": country_code,
        "process_gas_credit_enabled": bool(credit_on),
        "inside_elec_ref": inside_elec_ref,
        "f_internal": f_internal,
        "ef_internal_electricity": ef_internal_electricity,
        "finished_yield": fyield,
        "gas_reference_stage": gas_reference_stage,
        "fallback_materials": list(fallback_materials),
        "prefer_internal_processes": prefer_internal_map,
        "external_purchase_rows": external_purchase_rows,
        # NEW: Gas routing information with plant-level logic (from gas_meta)
        "total_process_gas_MJ": gas_meta.get('total_process_gas_MJ', 0.0),
        "gas_coke_MJ": gas_meta.get('gas_coke_MJ', 0.0),
        "gas_bf_MJ": gas_meta.get('gas_bf_MJ', 0.0),
        "gas_sources_MJ": gas_meta.get('gas_sources_MJ', 0.0),
        "direct_use_gas_MJ": gas_meta.get('direct_use_gas_MJ', 0.0),
        "electricity_gas_MJ": gas_meta.get('electricity_gas_MJ', 0.0),
        "total_gas_consumption_plant": gas_meta.get('total_gas_consumption_plant', 0.0),
        "f_internal_gas": gas_meta.get('f_internal_gas', 0.0),
        "ef_gas_blended": gas_meta.get('ef_gas_blended', 0.0),
        "EF_coke_gas": gas_meta.get('EF_coke_gas', 0.0),
        "EF_bf_gas": gas_meta.get('EF_bf_gas', 0.0),
        "EF_process_gas": gas_meta.get('EF_process_gas', 0.0),
        "util_eff": gas_meta.get('util_eff', 0.0),
        "direct_use_fraction": gas_meta.get('direct_use_fraction', 0.5),
        "electricity_fraction": gas_meta.get('electricity_fraction', 0.5),
        "process_gas_carrier": gas_meta.get('process_gas_carrier', None),
        "natural_gas_carrier": gas_meta.get('natural_gas_carrier', None),
        "utility_process": gas_meta.get('utility_process', None),
    }

    # Build final LCI with carrier splits
    lci_df = calculate_lci(
        prod_level=prod_levels,
        recipes=recipes_calc,
        energy_balance=energy_balance,
        electricity_internal_fraction=f_internal,
        gas_internal_fraction=f_internal_gas,
        natural_gas_carrier=gas_meta.get('natural_gas_carrier', natural_gas_carrier),
        process_gas_carrier=gas_meta.get('process_gas_carrier', process_gas_carrier),
    )

    # Debug prints for emission factors
    print("\n=== EMISSION FACTOR DEBUG ===")
    print(f"ELECTRICITY:")
    print(f"  Grid EF: {e_efs.get('Electricity', 0.0):.2f} kg CO2e/MJ")
    print(f"  Internal EF: {ef_internal_electricity:.2f} kg CO2e/MJ")
    print(f"  Internal Fraction: {f_internal:.3f}")
    print(f"  Blended EF: {f_internal * ef_internal_electricity + (1 - f_internal) * e_efs.get('Electricity', 0.0):.2f} kg CO2e/MJ")

    print(f"\nGAS:")
    print(f"  Natural Gas EF: {gas_meta.get('Gas', e_efs.get('Gas',0.0)):.2f} kg CO2e/MJ")
    print(f"  Process Gas EF: {gas_meta.get('EF_process_gas', 0.0):.2f} kg CO2e/MJ")
    print(f"  Internal Fraction: {gas_meta.get('f_internal_gas', 0.0):.3f}")
    print(f"  Blended EF: {gas_meta.get('ef_gas_blended', 0.0):.2f} kg CO2e/MJ")

    print(f"\nPROCESS GAS BREAKDOWN:")
    print(f"  Coke Gas: {gas_meta.get('gas_coke_MJ',0.0):.1f} MJ (EF: {gas_meta.get('EF_coke_gas',0.0):.1f})")
    print(f"  BF Gas: {gas_meta.get('gas_bf_MJ',0.0):.1f} MJ (EF: {gas_meta.get('EF_bf_gas',0.0):.1f})")
    print(f"  Total: {gas_meta.get('total_process_gas_MJ',0.0):.1f} MJ (EF: {gas_meta.get('EF_process_gas',0.0):.1f})")
    print(f"  Direct Use: {gas_meta.get('direct_use_gas_MJ',0.0):.1f} MJ")
    print(f"  Electricity: {gas_meta.get('electricity_gas_MJ',0.0):.1f} MJ")
    print("============================\n")

    return RunOutputs(
        production_routes=production_routes,
        prod_levels=prod_levels,
        energy_balance=energy_balance,
        emissions=emissions,
        total_co2e_kg=total_co2,
        total_cost=total_cost,
        material_cost=material_cost,
        balance_matrix=balance_matrix,
        lci=lci_df,
        meta=meta,
    )
