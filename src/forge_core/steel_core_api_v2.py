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
from forge_core.steel_model_core import (
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
    calculate_energy_balance,
    compute_fixed_plant_elec_model,
    calculate_emissions,  # signature may vary; we guard below
    analyze_energy_costs,
    # Data classes/types
    Process,
    OUTSIDE_MILL_PROCS,
    #compute_inside_elec_reference_for_share,
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
    total_cost: Optional[float] = None
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


# ==============================
# Main API (rewritten block)
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
      6) Dispatch process gas to steam/direct/power with internal-first policy
      7) Compute emissions (robust to signature)
    """
    # ----------- Unpack scenario -----------
    scenario: Dict[str, Any] = scn.scenario or {}
    route_preset: str = scn.route.route_preset or "auto"
    stage_key: str = scn.route.stage_key
    demand_qty: float = float(scn.route.demand_qty)
    picks_by_material: Dict[str, str] = scn.route.picks_by_material or {}
    pre_select_soft: Dict[str, int] = scn.route.pre_select_soft or {}
    country_code: Optional[str] = scn.country_code or None

    # (Optional) BF process-gas → elec credit gate
    credit_on: bool = _credit_enabled(scenario)

    # ----------- Load base data -----------
    base = os.path.join(data_dir, "")
    _ei_file = (scenario.get("energy_int_file") or "energy_int.yml").strip()
    _allowed = {
        "energy_int.yml",
        "energy_int_min.yml",
        "energy_int_max.yml",
        "energy_int_likely.yml",
    }
    if _ei_file not in _allowed:
        _ei_file = "energy_int.yml"

    energy_int   = load_data_from_yaml(os.path.join(base, _ei_file))
    energy_shares = load_data_from_yaml(os.path.join(base, "energy_matrix.yml"))
    energy_content = load_data_from_yaml(os.path.join(base, "energy_content.yml"))
    e_efs        = load_data_from_yaml(os.path.join(base, "emission_factors.yml"))
    params       = load_parameters(os.path.join(base, "parameters.yml"))
    mkt_cfg      = load_market_config(os.path.join(base, "mkt_config.yml"))
    elec_map     = load_electricity_intensity(os.path.join(base, "electricity_intensity.yml")) or {}

    # Country electricity EF override (UI)
    if country_code and country_code in elec_map:
        try:
            e_efs["Electricity"] = float(elec_map[country_code])
        except Exception:
            pass

    # Initial recipes (first pass)
    recipes = load_recipes_from_yaml(
        os.path.join(base, "recipes.yml"),
        params, energy_int, energy_shares, energy_content
    )

    # ---- Enforce route-locked energy_int override (defense-in-depth) ----
    allowed_proc_by_route = {
        "BF-BOF": "Blast Furnace",
        "DRI-EAF": "Direct Reduction Iron",
        "EAF-Scrap": "Electric Arc Furnace",
    }
    allowed_proc = allowed_proc_by_route.get(route_preset)
    if allowed_proc:
        ei = scenario.get("energy_int")
        if isinstance(ei, dict):
            scenario["energy_int"] = {k: v for k, v in ei.items() if k == allowed_proc and v is not None}
        else:
            scenario["energy_int"] = {}
    else:
        scenario["energy_int"] = {}

    # ----------- Apply scenario overrides -----------
    apply_fuel_substitutions(
        scenario.get("fuel_substitutions", {}),
        energy_shares, energy_int, energy_content, e_efs
    )
    apply_dict_overrides(energy_int,     scenario.get("energy_int", {}))
    apply_dict_overrides(energy_shares,  scenario.get("energy_matrix", {}))
    apply_dict_overrides(energy_content, scenario.get("energy_content", {}))
    apply_dict_overrides(e_efs,          scenario.get("emission_factors", {}))

    # Parameters deep-merge into SimpleNamespace
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

    _param_patch = scenario.get("param_overrides", None)
    if _param_patch is None:
        _param_patch = scenario.get("parameters", {})
    _recursive_ns_update(params, _param_patch)

    # Re-load recipes to re-evaluate expressions with updated params; then recipe overrides
    recipes = load_recipes_from_yaml(
        os.path.join(base, "recipes.yml"),
        params, energy_int, energy_shares, energy_content
    )
    recipes = apply_recipe_overrides(
        recipes,
        scenario.get("recipe_overrides", {}),
        params, energy_int, energy_shares, energy_content
    )

    # ----------- Route mask & feed enforcement -----------
    pre_mask = build_route_mask(route_preset, recipes)
    eaf_mode = _infer_eaf_mode(route_preset)

    import copy
    recipes_calc = copy.deepcopy(recipes)
    enforce_eaf_feed(recipes_calc, eaf_mode)

    # ----------- Build production route from picks -----------
    demand_mat = STAGE_MATS[stage_key]
    production_routes: Dict[str, int] = _build_routes_from_picks(
        recipes_calc,
        demand_mat,
        picks_by_material,
        pre_mask=pre_mask,
        pre_select=pre_select_soft,
    )

    # ----------- Solve balances -----------
    final_demand = {demand_mat: demand_qty}
    balance_matrix, prod_levels = calculate_balance_matrix(recipes_calc, final_demand, production_routes)
    if balance_matrix is None:
        return RunOutputs(
            production_routes=production_routes,
            prod_levels={},
            energy_balance=pd.DataFrame(),
            emissions=None,
            total_co2e_kg=None,
            total_cost=None,
            balance_matrix=pd.DataFrame(),
            meta={"error": "Material balance failed"},
        )

    # Ensure energy tables have rows for all active variants
    active_procs = [p for p, r in prod_levels.items() if r > 1e-9]
    from forge_core.steel_model_core import expand_energy_tables_for_active
    expand_energy_tables_for_active(active_procs, energy_shares, energy_int)

    # ----------- Energy balance (base) -----------
    energy_balance = calculate_energy_balance(prod_levels, energy_int, energy_shares)

    # ----------- Present-boundary electricity demand -----------
    def _inside_elec_present(eb: pd.DataFrame) -> float:
        if "Electricity" not in eb.columns:
            return 0.0
        idx = [p for p in eb.index if p not in ("TOTAL", "Utility Plant") and p not in OUTSIDE_MILL_PROCS]
        return float(eb.loc[idx, "Electricity"].clip(lower=0).sum())

    inside_elec_dyn = _inside_elec_present(energy_balance)

    # ----------- Recovered process gas (BF/COG/BOFG) -----------
    recipes_dict = {r.name: r for r in recipes_calc}
    recovery = getattr(params, "gas_recovery_rates", SimpleNamespace())
    rr = {
        "Blast Furnace": float(getattr(recovery, "Blast_Furnace", getattr(recovery, "Blast Furnace", 0.0)) or 0.0),
        "Coke Production": float(getattr(recovery, "Coke_Production", getattr(recovery, "Coke Production", 0.0)) or 0.0),
        "Basic Oxygen Furnace": float(getattr(recovery, "Basic_Oxygen_Furnace", getattr(recovery, "Basic Oxygen Furnace", 0.0)) or 0.0),
    }

    def _proc_gas_MJ(proc_name: str) -> float:
        runs = float(prod_levels.get(proc_name, 0.0))
        if runs <= 0.0:
            return 0.0
        total_MJ = runs * float(energy_int.get(proc_name, 0.0))
        return total_MJ * float(rr.get(proc_name, 0.0))

    gas_bf_MJ  = _proc_gas_MJ("Blast Furnace")
    gas_cp_MJ  = _proc_gas_MJ("Coke Production")
    gas_bof_MJ = _proc_gas_MJ("Basic Oxygen Furnace")
    total_gas_MJ = float(gas_bf_MJ + gas_cp_MJ + gas_bof_MJ)

    def _proc_gas_EF(proc_name: str) -> float:
        shares = dict(energy_shares.get(proc_name, {}))
        fuels = [(c, s) for c, s in shares.items() if c != "Electricity" and s > 0]
        if not fuels:
            return 0.0
        denom = sum(s for _, s in fuels) or 1e-12
        return sum(s * float(e_efs.get(c, 0.0)) for c, s in fuels) / denom

    EF_bf  = _proc_gas_EF("Blast Furnace")
    EF_cp  = _proc_gas_EF("Coke Production")
    EF_bof = _proc_gas_EF("Basic Oxygen Furnace")
    EF_process_gas = (EF_bf * gas_bf_MJ + EF_cp * gas_cp_MJ + EF_bof * gas_bof_MJ) / total_gas_MJ if total_gas_MJ > 1e-9 else 0.0

    # Dicts used by dispatch
    gas_MJ_by_source = {"BF": gas_bf_MJ, "COG": gas_cp_MJ, "BOFG": gas_bof_MJ}
    gas_ef_by_source = {"BF": EF_bf,      "COG": EF_cp,     "BOFG": EF_bof}
    gas_pooled_ef_g_per_MJ = EF_process_gas

    # Utility Plant (optional) electric efficiency if you keep it in recipes (for quick prints only)
    try:
        util_eff = recipes_dict.get("Utility Plant", Process("", {}, {})).outputs.get("Electricity", 0.0)
    except Exception:
        util_eff = 0.0

    # ----------- Utility dispatch (internal-first) -----------
    from forge_core.utility_dispatch import dispatch, DispatchParams, Demands

    # Steam demand (inside) if modeled
    steam_col = "Steam"
    steam_demand_MJ = 0.0
    if steam_col in energy_balance.columns:
        inside_mask = energy_balance.index.map(lambda p: p not in ("TOTAL", "Utility Plant") and p not in OUTSIDE_MILL_PROCS)
        steam_demand_MJ = float(energy_balance.loc[inside_mask, steam_col].clip(lower=0.0).sum())

    # Read utility config safely from params (SimpleNamespace)
    u = getattr(params, "utility", SimpleNamespace())
    utility_cfg = {
        "dispatch_priority": getattr(u, "dispatch_priority", ["steam", "direct", "power"]),
        "boiler_eff": getattr(u, "boiler_eff", 0.85),
        "power_eff": getattr(u, "power_eff", 0.35),
        "allow_export": getattr(u, "allow_export", False),
        "allow_flare": getattr(u, "allow_flare", True),
        "purchased_steam_carrier": getattr(u, "purchased_steam_carrier", "Steam"),
        "market_fuel_for_direct": getattr(u, "market_fuel_for_direct", "Natural Gas"),
        "direct_heat_eligible_fraction": getattr(u, "direct_heat_eligible_fraction", 0.0),
    }

    # Direct-heat eligible demand only when no explicit Steam is modeled
    non_electric_cols = [c for c in energy_balance.columns if c not in ("Electricity", steam_col)]
    direct_heat_eligible_MJ = 0.0
    if steam_demand_MJ <= 0.0 and utility_cfg["direct_heat_eligible_fraction"] > 0.0 and non_electric_cols:
        inside_mask = energy_balance.index.map(lambda p: p not in ("TOTAL", "Utility Plant") and p not in OUTSIDE_MILL_PROCS)
        non_elec_sum = float(energy_balance.loc[inside_mask, non_electric_cols].clip(lower=0.0).sum().sum())
        direct_heat_eligible_MJ = float(utility_cfg["direct_heat_eligible_fraction"]) * non_elec_sum

    # Electricity demand (present boundary)
    elec_demand_MJ = float(max(0.0, inside_elec_dyn))

    # Gas sources + EFs for dispatch
    pooled_gas_ef = float(max(0.0, gas_pooled_ef_g_per_MJ))
    gas_sources = {
        src: {"MJ": float(max(0.0, MJ)), "ef_g_per_MJ": float(max(0.0, gas_ef_by_source.get(src, pooled_gas_ef)))}
        for src, MJ in gas_MJ_by_source.items()
    }

    dp = DispatchParams(
        dispatch_priority=list(utility_cfg["dispatch_priority"]),
        boiler_eff=float(utility_cfg["boiler_eff"]),
        power_eff=float(utility_cfg["power_eff"]),
        allow_export=bool(utility_cfg["allow_export"]),
        allow_flare=bool(utility_cfg["allow_flare"]),


    )

    res = dispatch(
        gas_sources=gas_sources,
        demands=Demands(
            steam_MJ=steam_demand_MJ,
            direct_heat_MJ=direct_heat_eligible_MJ,
            electricity_MJ=elec_demand_MJ,
        ),
        params=dp,
    )

    # ----------- Apply market top-ups (Utility Plant row) -----------
    utility_row_name = "Utility Plant"
    if utility_row_name not in energy_balance.index:
        energy_balance.loc[utility_row_name] = {c: 0.0 for c in energy_balance.columns}

    # steam shortfall → purchased Steam (if modeled) else equivalent market fuel
    purchased_steam_carrier = str(utility_cfg["purchased_steam_carrier"])
    if res.shortfall_steam_MJ > 0.0:
        if purchased_steam_carrier in energy_balance.columns:
            energy_balance.loc[utility_row_name, purchased_steam_carrier] = \
                float(energy_balance.loc[utility_row_name, purchased_steam_carrier]) + res.shortfall_steam_MJ
        else:
            market_fuel = str(utility_cfg["market_fuel_for_direct"])
            gas_equiv = res.shortfall_steam_MJ / max(1e-12, dp.boiler_eff)
            if market_fuel not in energy_balance.columns:
                energy_balance[market_fuel] = 0.0
            energy_balance.loc[utility_row_name, market_fuel] = \
                float(energy_balance.loc[utility_row_name, market_fuel]) + gas_equiv

    # direct-heat shortfall → market fuel
    if res.shortfall_direct_MJ > 0.0:
        market_fuel = str(utility_cfg["market_fuel_for_direct"])
        if market_fuel not in energy_balance.columns:
            energy_balance[market_fuel] = 0.0
        energy_balance.loc[utility_row_name, market_fuel] = \
            float(energy_balance.loc[utility_row_name, market_fuel]) + res.shortfall_direct_MJ

    # ----------- Reference (fixed) plant electricity model for meta only -----------
    inside_elec_ref_fixed, f_internal_fixed, ef_internal_fixed = compute_fixed_plant_elec_model(
        recipes=recipes,  # base recipes
        energy_int=energy_int,
        energy_shares=energy_shares,
        energy_content=energy_content,
        params=params,
        route_key=route_preset,
        demand_qty=demand_qty,
        stage_ref="IP3",
    )

    # ----------- Emissions: use DISPATCH results -----------
    # internal electricity (MJ), internal fraction, and EF from dispatch
    internal_elec_MJ = float(res.produced_electricity_MJ)
    internal_fraction_plant = float(res.internal_electricity_fraction)
    ef_internal_electricity = float(res.ef_internal_electricity_g_per_MJ)

    # If gas credit is disabled, zero out internal elec & process-gas meta to mimic "no credit"
    if not credit_on:
        internal_elec_MJ = 0.0
        total_gas_MJ = 0.0
        EF_process_gas = 0.0
        internal_fraction_plant = 0.0
        ef_internal_electricity = 0.0

    process_emissions_table = load_data_from_yaml(os.path.join(base, "process_emissions.yml"))

    emissions = _robust_call_calculate_emissions(
        calculate_emissions,
        mkt_cfg=mkt_cfg,
        prod_lvl=prod_levels,
        prod_level=prod_levels,
        energy_balance=energy_balance,
        energy_df=energy_balance,
        e_efs=e_efs,
        energy_efs=e_efs,
        process_emissions_table=process_emissions_table,
        process_efs=process_emissions_table,
        internal_elec=internal_elec_MJ,                   # <- DISPATCH
        final_demand=final_demand,
        total_gas_MJ=total_gas_MJ,                        # for reference/meta if your fn uses it
        EF_process_gas=EF_process_gas,                    # pooled EF of the gas (not double-counted)
        internal_fraction_plant=internal_fraction_plant,  # <- DISPATCH
        ef_internal_electricity=ef_internal_electricity,  # <- DISPATCH
    )

    # ----------- TOTAL CO2e (robust) -----------
    total_co2 = None
    try:
        if emissions is not None and not emissions.empty:
            if "TOTAL" not in emissions.index and "TOTAL CO2e" in emissions.columns:
                emissions.loc["TOTAL"] = emissions.sum()
            if "TOTAL" in emissions.index and "TOTAL CO2e" in emissions.columns:
                total_co2 = float(emissions.loc["TOTAL", "TOTAL CO2e"])
            elif "TOTAL CO2e" in emissions.columns:
                total_co2 = float(emissions["TOTAL CO2e"].sum())
    except Exception:
        pass

    fyield = float(getattr(params, "finished_yield", 0.85))

    # ----------- Energy Cost Calculation -----------
    try:
        # Load energy prices
        energy_prices_path = os.path.join(base, 'energy_prices.yml')
        energy_prices = load_data_from_yaml(energy_prices_path) or {}
        
        # DEBUG: Check what we're working with
        print(f"🔍 Energy prices loaded: {bool(energy_prices)}")
        print(f"🔍 Energy prices keys: {list(energy_prices.keys()) if energy_prices else 'None'}")
        
        # Calculate total cost using core function
        total_cost = analyze_energy_costs(energy_balance, energy_prices)
        print(f"🔍 Total cost calculated: {total_cost}")
        
    except Exception as e:
        print(f"❌ Error in cost calculation: {e}")
        import traceback
        traceback.print_exc()
        total_cost = 0.0  # Default to 0 instead of None

    # ----------- Meta (merge; keep dispatch block) -----------
    meta: Dict[str, Any] = {
        "route_preset": route_preset,
        "stage_key": stage_key,
        "demand_qty": demand_qty,
        "country_code": country_code,
        "process_gas_credit_enabled": bool(credit_on),
        "inside_elec_ref_fixed": inside_elec_ref_fixed,
        "f_internal_fixed": f_internal_fixed,
        "ef_internal_electricity_fixed": ef_internal_fixed,
        "finished_yield": fyield,
        "dispatch": {
            "gas_available_MJ": {
                **{k: float(v) for k, v in gas_MJ_by_source.items()},
                "TOTAL": float(sum(gas_MJ_by_source.values())),
            },
            "allocations_MJ": {
                "to_steam": float(res.gas_to_steam_MJ),
                "to_direct": float(res.gas_to_direct_MJ),
                "to_power": float(res.gas_to_power_MJ),
                "export": float(res.gas_export_MJ),
                "flare": float(res.gas_flare_MJ),
            },
            "produced": {
                "steam_MJ": float(res.produced_steam_MJ),
                "electricity_MJ": float(res.produced_electricity_MJ),
            },
            "shortfalls_market": {
                "steam_MJ": float(res.shortfall_steam_MJ),
                "direct_MJ": float(res.shortfall_direct_MJ),
                "electricity_MJ": float(res.shortfall_electricity_MJ),
            },
            "ef_internal_electricity": float(ef_internal_electricity),
            "internal_electricity_fraction": float(internal_fraction_plant),
        },
    }

    return RunOutputs(
        production_routes=production_routes,
        prod_levels=prod_levels,
        energy_balance=energy_balance,
        emissions=emissions,
        total_co2e_kg=total_co2,
        total_cost=total_cost,
        balance_matrix=balance_matrix,
        meta=meta,
    )
