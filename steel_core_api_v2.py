# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:39:43 2025

@author: rafae
"""

# steel_core_api_v2.py
# -----------------------------------------------------------------------------
# Thin, stable API layer that wraps your existing core logic (steel_model_core.py)
# so the Streamlit app can focus ONLY on route building + logging.
#
# Exposes:
#   - RouteConfig, ScenarioInputs, RunOutputs (dataclasses)
#   - run_scenario(data_dir: str, scn: ScenarioInputs) -> RunOutputs
#   - build_picks_index(...) helper to surface ambiguous materials (for UI)
#   - write_run_log(log_dir: str, payload: dict) -> str
#
# This module deliberately imports your existing functions from steel_model_core
# and does no reimplementation of balances or emissions.
# -----------------------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import os, json, pathlib, copy
from datetime import datetime

import pandas as pd

# Import the implementation you already have
from steel_model_core import (
    # data models & loaders
    Process,
    load_data_from_yaml,
    load_parameters,
    load_recipes_from_yaml,
    load_market_config,
    load_electricity_intensity,
    apply_fuel_substitutions,
    apply_dict_overrides,
    apply_recipe_overrides,
    # calcs
    adjust_blast_furnace_intensity,
    adjust_process_gas_intensity,
    calculate_balance_matrix,
    calculate_energy_balance,
    calculate_internal_electricity,
    adjust_energy_balance,
    calculate_emissions,
    expand_energy_tables_for_active,
    # route helpers & constants
    STAGE_MATS,
    OUTSIDE_MILL_PROCS,
    build_route_mask,
    enforce_eaf_feed,
)

# ==============================
# Dataclasses (stable contract)
# ==============================
@dataclass(frozen=True)
class RouteConfig:
    route_preset: str                           # "BF-BOF" | "DRI-EAF" | "EAF-Scrap" | "External" | "auto"
    stage_key: str                              # key from STAGE_MATS (e.g., "Finished")
    demand_qty: float                           # demand quantity at stage
    picks_by_material: Dict[str, str]           # material -> chosen producer name
    pre_select_soft: Optional[Dict[str, int]] = None  # optional soft disables (0/1)

@dataclass(frozen=True)
class ScenarioInputs:
    country_code: Optional[str]                 # ISO3 for electricity EF; may be None
    scenario: Dict[str, Any]                    # raw scenario dict loaded from YAML
    route: RouteConfig

@dataclass
class RunOutputs:
    production_routes: Dict[str, int]           # process -> {0/1}
    prod_levels: Dict[str, float]               # process -> runs
    energy_balance: pd.DataFrame
    emissions: Optional[pd.DataFrame]
    total_co2e_kg: Optional[float]
    meta: Dict[str, Any]

# ==============================
# Helpers
# ==============================

def _ns_to_dict(ns):
    try:
        return {k: _ns_to_dict(getattr(ns, k)) for k in vars(ns)}
    except Exception:
        if isinstance(ns, dict):
            return {k: _ns_to_dict(v) for k, v in ns.items()}
        if isinstance(ns, (list, tuple)):
            return [_ns_to_dict(v) for v in ns]
        return ns


def _build_producers_index(recipes: List[Process]) -> Dict[str, List[Process]]:
    prod = {}
    for r in recipes:
        for m in r.outputs:
            prod.setdefault(m, []).append(r)
    return prod


def build_picks_index(recipes: List[Process], demand_mat: str,
                       pre_mask: Optional[Dict[str, int]] = None,
                       pre_select: Optional[Dict[str, int]] = None
                      ) -> List[Tuple[str, List[str]]]:
    """Return [(material, [enabled_producer_names,...]), ...] encountered upstream.
       Use this to populate the UI with radios/selects. Non-ambiguous nodes
       are omitted.
    """
    pre_mask = pre_mask or {}
    pre_select = pre_select or {}
    producers = _build_producers_index(recipes)

    out: List[Tuple[str, List[str]]] = []
    seen_mats: set[str] = set()
    q = deque([demand_mat])
    while q:
        mat = q.popleft()
        if mat in seen_mats:
            continue
        seen_mats.add(mat)
        cand = producers.get(mat, [])
        enabled = [r for r in cand if pre_mask.get(r.name, 1) > 0 and pre_select.get(r.name, 1) > 0]
        if not cand:
            continue
        if len(enabled) <= 1:
            pick = enabled[0] if len(enabled) == 1 else None
            if pick:
                for im in pick.inputs:
                    if im not in seen_mats:
                        q.append(im)
            continue
        out.append((mat, [r.name for r in enabled]))
        for r in enabled:
            for im in r.inputs:
                if im not in seen_mats:
                    q.append(im)
    return out


def build_routes_from_picks(recipes: List[Process], demand_mat: str,
                             picks_by_material: Dict[str, str],
                             pre_mask: Optional[Dict[str, int]] = None,
                             pre_select: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Deterministic on/off map from material→chosen producer, walking upstream."""
    pre_mask = pre_mask or {}
    pre_select = pre_select or {}
    producers = _build_producers_index(recipes)
    chosen: Dict[str, int] = {}
    visited: set[str] = set()
    q = deque([demand_mat])
    while q:
        mat = q.popleft()
        if mat in visited:
            continue
        visited.add(mat)
        cand = producers.get(mat, [])
        enabled = [r for r in cand if pre_mask.get(r.name, 1) > 0 and pre_select.get(r.name, 1) > 0]
        if not enabled:
            continue
        # pick
        if len(enabled) == 1:
            pick = enabled[0]
        else:
            pick_name = picks_by_material.get(mat)
            pick = next((r for r in enabled if r.name == pick_name), enabled[0])
        chosen[pick.name] = 1
        for r in cand:
            if r.name != pick.name:
                chosen[r.name] = 0
        for im in pick.inputs.keys():
            if im not in visited:
                q.append(im)
    return chosen


# ==============================
# Core API: run_scenario
# ==============================

def run_scenario(data_dir: str, scn: ScenarioInputs) -> RunOutputs:
    """
    Load all base tables, apply scenario overrides, compile route mask, and run
    the full pipeline: material balance → energy → internal electricity credit →
    emissions. Returns RunOutputs with total CO₂e in kg.
    """
    base = os.path.join(data_dir, "")

    # --- Load base inputs
    energy_int     = load_data_from_yaml(os.path.join(base, 'energy_int.yml'))
    energy_shares  = load_data_from_yaml(os.path.join(base, 'energy_matrix.yml'))
    energy_content = load_data_from_yaml(os.path.join(base, 'energy_content.yml'))
    e_efs          = load_data_from_yaml(os.path.join(base, 'emission_factors.yml'))
    params         = load_parameters      (os.path.join(base, 'parameters.yml'))
    mkt_cfg        = load_market_config   (os.path.join(base, 'mkt_config.yml'))
    elec_map       = load_electricity_intensity(os.path.join(base, 'electricity_intensity.yml'))

    # --- Recipes (first pass)
    recipes = load_recipes_from_yaml(os.path.join(base, 'recipes.yml'), params, energy_int, energy_shares, energy_content)

    # --- Scenario-level overrides
    scenario = scn.scenario or {}
    apply_fuel_substitutions(scenario.get('fuel_substitutions', {}), energy_shares, energy_int, energy_content, e_efs)
    apply_dict_overrides(energy_int,     scenario.get('energy_int', {}))
    apply_dict_overrides(energy_shares,  scenario.get('energy_matrix', {}))
    apply_dict_overrides(energy_content, scenario.get('energy_content', {}))
    apply_dict_overrides(e_efs,          scenario.get('emission_factors', {}))

    # --- Country-specific electricity EF
    if scn.country_code and scn.country_code in elec_map:
        e_efs['Electricity'] = float(elec_map[scn.country_code])

    # --- Params merge & renormalize blend if needed
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
    def _renorm_blend(ns):
        try:
            b = ns.blend
            s = float(getattr(b, 'sinter', 0.0))
            p_ = float(getattr(b, 'pellet', 0.0))
            l = float(getattr(b, 'lump',   0.0))
            tot = s + p_ + l
            if tot and abs(tot - 1.0) > 1e-9:
                b.sinter = s / tot; b.pellet = p_ / tot; b.lump = l / tot
        except AttributeError:
            pass
    _param_patch = scenario.get('param_overrides') or scenario.get('parameters', {})
    _recursive_ns_update(params, _param_patch)
    _renorm_blend(params)

    # --- Intensity adjustments (process-gas recovery, etc.)
    adjust_blast_furnace_intensity(energy_int, energy_shares, params)
    adjust_process_gas_intensity('Coke Production', 'process_gas_coke', energy_int, energy_shares, params)

    # --- Recipes again to refresh formulas after param updates
    recipes = load_recipes_from_yaml(os.path.join(base, 'recipes.yml'), params, energy_int, energy_shares, energy_content)
    recipes = apply_recipe_overrides(recipes, scenario.get('recipe_overrides', {}), params, energy_int, energy_shares, energy_content)

    # --- Route preset → pre-mask & EAF feed enforcement copy for UI logic
    demand_mat = STAGE_MATS[scn.route.stage_key]
    pre_mask = build_route_mask(scn.route.route_preset, recipes)

    recipes_for_route = copy.deepcopy(recipes)
    eaf_feed_mode = {
        "EAF-Scrap": "scrap",
        "DRI-EAF":   "dri",
        "BF-BOF":    None,
        "External":  None,
        "auto":      None,
    }.get(scn.route.route_preset)
    enforce_eaf_feed(recipes_for_route, eaf_feed_mode)

    # --- Build production on/off map from picks
    prod_routes = build_routes_from_picks(
        recipes_for_route,
        demand_mat,
        scn.route.picks_by_material,
        pre_mask=pre_mask,
        pre_select=scn.route.pre_select_soft or {},
    )

    # --- Solve material balance
    balance_matrix, prod_lvl = calculate_balance_matrix(recipes_for_route, {demand_mat: scn.route.demand_qty}, prod_routes)
    if balance_matrix is None:
        raise RuntimeError("Material balance failed.")

    # Make sure energy tables include all active variants
    active_procs = [p for p, r in prod_lvl.items() if r > 1e-9]
    expand_energy_tables_for_active(active_procs, energy_shares, energy_int)

    # --- Internal electricity and energy balance
    recipes_dict = {r.name: r for r in recipes_for_route}
    internal_elec = calculate_internal_electricity(prod_lvl, recipes_dict, params)

    energy_balance = calculate_energy_balance(prod_lvl, energy_int, energy_shares)
    energy_balance = adjust_energy_balance(energy_balance, internal_elec)

    # --- Process-gas EF mix and totals (Coke + BF top-gas)
    gas_coke_MJ = prod_lvl.get('Coke Production', 0.0) * recipes_dict.get('Coke Production', Process('',{},{})).outputs.get('Process Gas', 0.0)
    gas_bf_MJ   = (getattr(params, 'bf_adj_intensity', 0.0) - getattr(params, 'bf_base_intensity', 0.0)) * prod_lvl.get('Blast Furnace', 0.0)
    total_gas_MJ = float(gas_coke_MJ + gas_bf_MJ)

    cp_shares = (load_data_from_yaml(os.path.join(base, 'energy_matrix.yml')) or {}).get('Coke Production', {})
    bf_shares = (load_data_from_yaml(os.path.join(base, 'energy_matrix.yml')) or {}).get('Blast Furnace', {})
    fuels_cp = [c for c in cp_shares if c != 'Electricity' and cp_shares[c] > 0]
    fuels_bf = [c for c in bf_shares if c != 'Electricity' and bf_shares[c] > 0]
    EF_coke_gas = (sum(cp_shares[c] * (load_data_from_yaml(os.path.join(base, 'emission_factors.yml')) or {}).get(c, 0.0) for c in fuels_cp) / max(1e-12, sum(cp_shares[c] for c in fuels_cp))) if fuels_cp else 0.0
    EF_bf_gas   = (sum(bf_shares[c] * (load_data_from_yaml(os.path.join(base, 'emission_factors.yml')) or {}).get(c, 0.0) for c in fuels_bf) / max(1e-12, sum(bf_shares[c] for c in fuels_bf))) if fuels_bf else 0.0
    EF_process_gas = EF_coke_gas if total_gas_MJ <= 1e-9 else (
        (EF_coke_gas * (gas_coke_MJ / max(1e-12, total_gas_MJ))) + (EF_bf_gas * (gas_bf_MJ / max(1e-12, total_gas_MJ)))
    )

    # --- Emissions
    emissions = calculate_emissions(
        mkt_cfg,
        prod_lvl,
        energy_balance,
        e_efs,
        load_data_from_yaml(os.path.join(base, 'process_emissions.yml')),
        internal_elec,
        {demand_mat: scn.route.demand_qty},
        total_gas_MJ,
        EF_process_gas,
    )

    total = None
    if emissions is not None and not emissions.empty:
        if ("TOTAL" in emissions.index) and ("TOTAL CO2e" in emissions.columns):
            total = float(emissions.loc["TOTAL", "TOTAL CO2e"])  # already in kg in your core
        elif "TOTAL CO2e" in emissions.columns:
            total = float(emissions["TOTAL CO2e"].sum())

    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "country_code": scn.country_code,
        "route_preset": scn.route.route_preset,
        "stage_key": scn.route.stage_key,
        "demand_qty": scn.route.demand_qty,
    }

    return RunOutputs(
        production_routes=prod_routes,
        prod_levels=dict(prod_lvl),
        energy_balance=energy_balance,
        emissions=emissions,
        total_co2e_kg=total,
        meta=meta,
    )


# ==============================
# Logging
# ==============================

def write_run_log(log_dir: str, payload: Dict[str, Any]) -> str:
    """Write a single JSON log with config + CO₂e. Returns file path."""
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"run_{ts}.json"
    fpath = str(pathlib.Path(log_dir) / fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return fpath
