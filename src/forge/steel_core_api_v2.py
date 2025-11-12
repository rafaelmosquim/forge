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
from functools import partial

import pandas as pd

# Core functions & models from your existing engine
from forge.core.io import (
    load_data_from_yaml,
    load_parameters,
    load_recipes_from_yaml,
    load_market_config,
    load_electricity_intensity,
)
from forge.core.models import Process, OUTSIDE_MILL_PROCS
from forge.core.routing import STAGE_MATS
from forge.core.compute import (
    calculate_lci,
    compute_inside_elec_reference_for_share,
)
from forge.core.runner import run_core_scenario
from forge.scenarios.builder import build_core_scenario
from forge.scenarios.transforms import apply_dri_mix, apply_charcoal_expansion
from forge.reporting.lci_aug import augment_lci_and_debug
from forge.core.transforms import (
    apply_fuel_substitutions,
    apply_dict_overrides,
    apply_recipe_overrides,
    apply_energy_int_efficiency_scaling,
    apply_energy_int_floor,
)
# Engine compute is orchestrated via core.runner now

from forge.descriptor import load_sector_descriptor
from forge.descriptor import (
    build_stage_material_map,
    reference_stage_for_gas,
    reference_stage_for_electricity,
    resolve_stage_material,
)


def _env_flag_truthy(var_name: str) -> bool:
    """
    Return True when the environment variable is set to a truthy value.
    Accepted truthy values: '1', 'true', 'yes', 'on' (case insensitive).
    """
    try:
        raw = os.environ.get(var_name, "")
    except Exception:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_lci_enabled() -> bool:
    """
    Global feature flag for LCI calculations.
    LCI outputs are experimental and disabled by default; enable by setting
    FORGE_ENABLE_LCI=1 (or any truthy value accepted by `_env_flag_truthy`).
    """
    return _env_flag_truthy("FORGE_ENABLE_LCI")


def _is_debug_io_enabled() -> bool:
    return _env_flag_truthy("FORGE_DEBUG_IO")


def _debug_print(*args, **kwargs) -> None:
    if _is_debug_io_enabled():
        print(*args, **kwargs)


def _coerce_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_process_from_carrier(carrier: str) -> Optional[str]:
    if not carrier:
        return None
    name = carrier.lower()
    if "coke" in name:
        return "Coke Production"
    if "bof" in name or "basic oxygen" in name:
        return "Basic Oxygen Furnace"
    if "bf" in name or "blast furnace" in name:
        return "Blast Furnace"
    return None


def _load_process_gas_table(base_path: str) -> Dict[str, Any]:
    path = os.path.join(base_path, "process_gases.yml")
    try:
        data = load_data_from_yaml(path, unwrap_single_key=False)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    if "process_gases" in data and isinstance(data["process_gases"], dict):
        return data["process_gases"]
    return data


def _normalize_process_gas_entries(raw: Dict[str, Any], energy_content: Dict[str, float]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    specs: List[Dict[str, Any]] = []
    if not isinstance(raw, dict):
        return normalized, specs

    for carrier, meta in raw.items():
        if not isinstance(meta, dict):
            continue
        entry: Dict[str, Any] = {}
        energy_mj_per_nm3 = _coerce_float(meta.get("energy_mj_per_nm3"))
        energy_gj_per_ndam3 = _coerce_float(
            meta.get("energy_gj_per_ndam3") or meta.get("energy_GJ_per_Ndam3") or meta.get("Energy Content")
        )
        if energy_mj_per_nm3 is None and energy_gj_per_ndam3 is not None:
            ndam3_to_nm3 = _coerce_float(meta.get("ndam3_to_nm3")) or 1000.0
            energy_mj_per_nm3 = (energy_gj_per_ndam3 * 1000.0) / max(1.0, ndam3_to_nm3)
        density = _coerce_float(meta.get("density_kg_per_nm3") or meta.get("density_kg_per_Nm3") or meta.get("Density"))
        entry["energy_mj_per_nm3"] = energy_mj_per_nm3
        entry["density_kg_per_nm3"] = density

        energy_mj_per_unit = None
        if energy_mj_per_nm3 is not None and density and density > 0:
            energy_mj_per_unit = energy_mj_per_nm3 / density
        elif isinstance(energy_content.get(carrier), (int, float)):
            try:
                energy_mj_per_unit = float(energy_content[carrier])
            except Exception:
                energy_mj_per_unit = None

        if energy_mj_per_unit is not None:
            energy_content[carrier] = energy_mj_per_unit
        entry["energy_mj_per_unit"] = energy_mj_per_unit

        recovery = meta.get("recovery_fraction")
        if recovery is None:
            for key, val in meta.items():
                if "recovery" in str(key).lower():
                    recovery = val
                    break
        entry["recovery_fraction"] = _coerce_float(recovery)

        source_process = meta.get("source_process") or meta.get("process") or meta.get("producer") or _infer_process_from_carrier(carrier)
        entry["source_process"] = source_process
        entry["outputs_in_MJ"] = bool(meta.get("outputs_in_MJ", False))

        normalized[carrier] = entry
        if source_process:
            specs.append(
                {
                    "process": source_process,
                    "carrier": carrier,
                    "outputs_in_MJ": entry["outputs_in_MJ"],
                    "energy_per_unit": entry["energy_mj_per_unit"],
                }
            )
    return normalized, specs


def _apply_process_gas_metadata(base_path: str, energy_content: Dict[str, float], params, gas_config: Dict[str, Any]) -> Dict[str, Any]:
    raw = _load_process_gas_table(base_path)
    normalized, specs = _normalize_process_gas_entries(raw, energy_content)
    if not normalized:
        return gas_config

    if isinstance(params, object):
        setattr(params, "process_gases", normalized)

    updated = dict(gas_config or {})
    updated["process_gases"] = normalized
    if specs:
        updated["process_gas_sources"] = specs
    return updated


def _build_energy_flow_summary(
    process_names: List[str],
    energy_balance: Optional[pd.DataFrame],
    recipes: List[Any],
    prod_levels: Dict[str, float],
    energy_content: Dict[str, float],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    recipe_map = {r.name: r for r in (recipes or [])}
    def _energy_for_carrier(carrier: str, qty: float) -> Optional[float]:
        if carrier in energy_content:
            return qty * float(energy_content[carrier])
        if carrier.lower() == "electricity":
            return qty  # already MJ
        return None

    for name in process_names:
        inputs: Dict[str, float] = {}
        outputs: Dict[str, float] = {}
        if energy_balance is not None and name in energy_balance.index:
            row = energy_balance.loc[name]
            inputs = {
                carrier: float(val)
                for carrier, val in row.items()
                if isinstance(val, (int, float)) and float(val) > 0
            }
        proc = recipe_map.get(name)
        runs = float(prod_levels.get(name, 0.0) or 0.0)
        if proc and runs > 0:
            for in_name, qty in proc.inputs.items():
                try:
                    qty_val = float(qty)
                except (TypeError, ValueError):
                    continue
                energy_val = _energy_for_carrier(in_name, qty_val * runs)
                if energy_val is not None and (not inputs or in_name not in inputs):
                    inputs[in_name] = energy_val

            for out_name, qty in proc.outputs.items():
                try:
                    qty_val = float(qty)
                except (TypeError, ValueError):
                    continue
                energy_val = _energy_for_carrier(out_name, qty_val * runs)
                if energy_val is not None:
                    outputs[out_name] = energy_val
        summary[name] = {"inputs_MJ": inputs, "outputs_MJ": outputs}
    return summary


# ==============================
# Scenario transforms
# ==============================
# moved to forge.core.transforms: apply_energy_int_efficiency_scaling / apply_energy_int_floor


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


## Emissions call robustness handled by core.runner


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

## NOTE: Canonical gas reference is provided by forge.core.compute.compute_inside_gas_reference_for_share
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
    # Optional debug dump of scenario payload
    if _is_debug_io_enabled():
        _debug_print("🎯 SCENARIO STRUCTURE CAPTURED:")
        _debug_print("Scenario keys:", list(scn.scenario.keys()))
        _debug_print("Full scenario:")
        _debug_print(json.dumps(scn.scenario, indent=2, default=str))
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
    _debug_print(
        f"🔍 Stage key: {stage_key}, Stage role: {stage_role or '(unspecified)'}, "
        f"Environment stage: {os.environ.get('STEEL_MODEL_STAGE', '')}"
    )

    fallback_materials = set(descriptor.balance_fallback_materials or set())
    external_purchase_rows = list(descriptor.costing.external_purchase_rows or [])
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
    gas_config     = _apply_process_gas_metadata(base, energy_content, params, gas_config)
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

    # Apply uniform/scheduled efficiency improvements to intensities BEFORE adjustments
    apply_energy_int_efficiency_scaling(energy_int, scenario)
    # Optional per-process minimum floors (applied after schedule)
    apply_energy_int_floor(energy_int, scenario)

    apply_dri_mix(energy_shares, scenario)
    apply_charcoal_expansion(energy_shares, scenario)

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

    # Intensity adjustments (from refactored core)
    from forge.core.compute import (
        adjust_blast_furnace_intensity, adjust_process_gas_intensity
    )
    adjust_blast_furnace_intensity(energy_int, energy_shares, params)
    adjust_process_gas_intensity('Coke Production', 'process_gas_coke', energy_int, energy_shares, params)

    # Re-load recipes to re-evaluate expressions with updated params; then recipe overrides
    recipes = load_recipes_from_yaml(
        os.path.join(base, 'recipes.yml'),
        params, energy_int, energy_shares, energy_content
    )
    recipes = apply_recipe_overrides(recipes, scenario.get('recipe_overrides', {}), params, energy_int, energy_shares, energy_content)

    # ---------- Build demand material (builder constructs route) ----------
    demand_mat = resolve_stage_material(descriptor, stage_key)

    # ---------- Compute via core runner ----------
    # Load process-emissions yaml for direct process emissions mapping
    process_emissions_table = load_data_from_yaml(os.path.join(base, 'process_emissions.yml')) or {}
    # Load optional price tables (API does I/O, core does math)
    energy_prices = load_data_from_yaml(os.path.join(base, 'energy_prices.yml')) or {}
    material_prices = load_data_from_yaml(os.path.join(base, 'material_prices.yml')) or {}

    build_result = build_core_scenario(
        descriptor=descriptor,
        stage_role=stage_role,
        route_preset=route_preset,
        demand_qty=demand_qty,
        demand_material=demand_mat,
        recipes=recipes,
        energy_int=energy_int,
        energy_shares=energy_shares,
        energy_content=energy_content,
        e_efs=e_efs,
        params=params,
        picks_by_material=picks_by_material,
        pre_select_soft=pre_select_soft,
        route_overrides=scenario.get('route_overrides'),
        fallback_materials=fallback_materials,
        gas_config=gas_config,
        gas_routing=scenario.get('gas_routing', {}),
        process_efs=process_emissions_table,
        external_purchase_rows=external_purchase_rows,
        energy_prices=energy_prices,
        material_prices=material_prices,
        outside_mill_procs=set(OUTSIDE_MILL_PROCS or []),
        enable_lci=is_lci_enabled(),
    )
    core_inputs = build_result.core
    prefer_internal_map = build_result.prefer_internal_map
    fallback_materials = build_result.fallback_materials

    core_res = run_core_scenario(core_inputs)

    balance_matrix = core_res.balance_matrix
    prod_levels = core_res.prod_levels
    energy_balance = core_res.energy_balance
    emissions = core_res.emissions
    total_co2 = core_res.total_co2e_kg
    lci_df = core_res.lci
    gas_meta = core_res.meta
    e_efs = core_res.energy_efs_out

    # Diagnostics comparable to prior debug
    try:
        emissions_inputs = {
            "prod_levels_sample": dict(list(prod_levels.items())[:5]),
            "energy_balance_sample": energy_balance.head(3).to_dict() if isinstance(energy_balance, pd.DataFrame) else {},
            "energy_int_sample": dict(list(energy_int.items())[:3]),
            "e_efs_sample": dict(list(e_efs.items())[:3]),
        }
        with open('DEBUG_core_emissions_inputs.json', 'w') as f:
            json.dump(emissions_inputs, f, indent=2, default=str)
    except Exception:
        pass

    # ensure we have plant-level inside_elec_ref and fixed internal ef values
    try:
        elec_ref_stage = reference_stage_for_electricity(descriptor)
        inside_elec_ref = compute_inside_elec_reference_for_share(
            recipes=recipes,
            energy_int=energy_int,
            energy_shares=energy_shares,
            energy_content=energy_content,
            params=params,
            route_key=route_preset,
            demand_qty=demand_qty,
            stage_ref=elec_ref_stage,
        )
    except Exception:
        inside_elec_ref = 0.0

    # Pull internal electricity diagnostics from gas_meta if available
    f_internal = float(gas_meta.get('f_internal', 0.0))
    f_internal_gas = float(gas_meta.get('f_internal_gas', 0.0))
    ef_internal_electricity = float(gas_meta.get('ef_internal_electricity', 0.0))
    fyield = float(getattr(params, "finished_yield", 0.85))    

    # Costs were computed inside core runner when price tables are present
    total_cost = core_res.total_cost if core_res.total_cost is not None else 0.0
    material_cost = core_res.material_cost if core_res.material_cost is not None else 0.0

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
        "gas_source_details": gas_meta.get('gas_source_details', {}),
        "gas_credit_details": gas_meta.get('gas_credit_details', {}),
        "direct_use_gas_MJ": gas_meta.get('direct_use_gas_MJ', 0.0),
        "electricity_gas_MJ": gas_meta.get('electricity_gas_MJ', 0.0),
        "total_gas_consumption_plant": gas_meta.get('total_gas_consumption_plant', 0.0),
        "f_internal_gas": gas_meta.get('f_internal_gas', 0.0),
        "ef_gas_blended": gas_meta.get('ef_gas_blended', 0.0),
        "ef_nat_gas_grid": gas_meta.get('ef_nat_gas_grid', e_efs.get('Gas', 0.0)),
        "EF_coke_gas": gas_meta.get('EF_coke_gas', 0.0),
        "EF_bf_gas": gas_meta.get('EF_bf_gas', 0.0),
        "EF_process_gas": gas_meta.get('EF_process_gas', 0.0),
        "util_eff": gas_meta.get('util_eff', 0.0),
        "direct_use_fraction": gas_meta.get('direct_use_fraction', 0.5),
        "electricity_fraction": gas_meta.get('electricity_fraction', 0.5),
        "process_gas_carrier": gas_meta.get('process_gas_carrier', None),
        "natural_gas_carrier": gas_meta.get('natural_gas_carrier', None),
        "utility_process": gas_meta.get('utility_process', None),
        # Electricity EF diagnostics (units: g CO2e/MJ)
        "ef_electricity_grid": gas_meta.get('ef_electricity_grid', e_efs.get('Electricity', 0.0)),
        "ef_electricity_used": gas_meta.get('ef_electricity_used', f_internal * gas_meta.get('ef_internal_electricity', 0.0) + (1 - f_internal) * e_efs.get('Electricity', 0.0)),
    }

    energy_flow_processes = [
        "Coke Production",
        "Blast Furnace",
        "Basic Oxygen Furnace",
        gas_config.get("utility_process") or "Utility Plant",
    ]
    energy_flow_summary = _build_energy_flow_summary(
        energy_flow_processes,
        energy_balance,
        core_inputs.recipes,
        prod_levels,
        energy_content,
    )
    utility_name = gas_config.get("utility_process") or "Utility Plant"
    if utility_name in energy_flow_summary:
        util_entry = energy_flow_summary[utility_name]
        elec_gas = float(gas_meta.get('electricity_gas_MJ', 0.0) or 0.0)
        if elec_gas > 0:
            carrier_name = process_gas_carrier or "Process Gas"
            util_entry['inputs_MJ'].setdefault(carrier_name, elec_gas)
            util_eff = float(gas_meta.get('util_eff', 0.0) or 0.0)
            util_entry['outputs_MJ'].setdefault('Electricity', elec_gas * util_eff)
    meta["energy_flow_summary"] = energy_flow_summary

    enable_lci = is_lci_enabled()
    if enable_lci and lci_df is None:
        # Build final LCI with carrier splits
        lci_df = calculate_lci(
            prod_level=prod_levels,
            recipes=core_inputs.recipes,
            energy_balance=energy_balance,
            electricity_internal_fraction=f_internal,
            gas_internal_fraction=f_internal_gas,
            natural_gas_carrier=gas_meta.get('natural_gas_carrier', natural_gas_carrier),
            process_gas_carrier=gas_meta.get('process_gas_carrier', process_gas_carrier),
        )
        # Augment LCI and print EF debug
        lci_df = augment_lci_and_debug(
            lci_df=lci_df,
            prod_levels=prod_levels,
            recipes=core_inputs.recipes,
            energy_balance=energy_balance,
            energy_shares=energy_shares,
            energy_content=energy_content,
            params=params,
            gas_meta=gas_meta,
            e_efs=e_efs,
            meta=meta,
            base_path=base,
            natural_gas_carrier=gas_meta.get('natural_gas_carrier', natural_gas_carrier),
            process_gas_carrier=gas_meta.get('process_gas_carrier', process_gas_carrier),
            f_internal=f_internal,
        )
    else:
        lci_df = None

    return RunOutputs(
        production_routes=core_res.production_routes,
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
