# streamlit_app.py
# Streamlit front-end for your steel model (with branded header + logos)
# ---------------------------------------------------------
# Expected layout (relative paths):
#   .
#   ├── steel_model_core.py   # your existing script (renamed)
#   ├── streamlit_app.py      # this file
#   ├── assets/
#   │   ├── university_logo.png
#   │   └── faculty_logo.png
#   └── data/
#       ├── energy_int.yml
#       ├── energy_matrix.yml
#       ├── energy_content.yml
#       ├── emission_factors.yml
#       ├── electricity_intensity.yml
#       ├── parameters.yml
#       ├── process_emissions.yml
#       ├── recipes.yml
#       ├── mkt_config.yml
#       └── scenarios/
#           └── *.yml
# ---------------------------------------------------------

from __future__ import annotations
import os
import io
import re
import zipfile
import pathlib
from typing import Dict, List, Tuple, Set
import json, uuid, hashlib, sys, platform
from datetime import datetime

import streamlit as st
import pandas as pd
import yaml

# Import everything we need from your existing codebase
from steel_model_core import (
    # Data models & loaders
    Process,
    load_data_from_yaml,
    load_parameters,
    load_recipes_from_yaml,
    load_market_config,
    load_electricity_intensity,
    apply_fuel_substitutions,
    apply_dict_overrides,
    apply_recipe_overrides,
    # Calculations
    adjust_blast_furnace_intensity,
    adjust_process_gas_intensity,
    calculate_balance_matrix,
    calculate_energy_balance,
    calculate_internal_electricity,
    adjust_energy_balance,
    calculate_emissions,
    derive_energy_shares,
    # Plot builders
    make_mass_sankey,
    make_energy_sankey,
    make_energy_to_process_sankey,
    make_hybrid_sankey,
    # Route helpers & constants
    STAGE_MATS,
    OUTSIDE_MILL_PROCS,
    build_route_mask,
    enforce_eaf_feed,
    expand_energy_tables_for_active,
)

from steel_core_api_v2 import RouteConfig, ScenarioInputs, run_scenario


DATA_ROOT = "data"  # fixed data folder


def _route_from_scenario(scenario: dict | None, scenario_name: str) -> str:
    if scenario:
        for k in ("route_preset", "route"):
            v = scenario.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    name = (scenario_name or "").lower()
    if "dri" in name and "eaf" in name: return "DRI-EAF"
    if "eaf" in name and "scrap" in name: return "EAF-Scrap"
    if "bf" in name or "bof" in name:    return "BF-BOF"
    if "external" in name:               return "External"
    return "auto"


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(
    page_title="Steel Model – Routes & Treatments",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --primary-color: #2563eb; }
div[data-baseweb="radio"] [aria-checked="true"] > div:first-child {
  background-color: #2563eb !important; border-color: #2563eb !important;
}
div[data-baseweb="radio"] [aria-checked="false"] > div:first-child {
  border-color: #2563eb !important;
}
[role="radiogroup"] [role="radio"][aria-checked="true"] > div:first-child {
  background-color: #2563eb !important; border-color: #2563eb !important;
}
div[role="radiogroup"] > label:has(input:checked) {
  border-color: #2563eb; background: rgba(37, 99, 235, 0.08);
}
button[kind="primary"], [data-testid="baseButton-primary"] {
  background-color: #16a34a !important; border-color: #16a34a !important; color: #fff !important;
}
button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover {
  background-color: #15803d !important; border-color: #15803d !important;
}
</style>
""", unsafe_allow_html=True)


# --- Branding header (replaces old st.title/st.caption) ----------------------
def app_header(title: str = "Steel Carbon Intensity Model"):
    st.markdown("""
    <style>
      .app-hero { padding:.25rem 0 0 0; text-align:center; }
      .app-hero h1 { margin:0; font-size:1.8rem; line-height:1.2; }
      .hr { height:1px; background:linear-gradient(to right,#e5e7eb,#cbd5e1,#e5e7eb); margin:.25rem 0 .75rem 0; border:0; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="app-hero"><h1>{title}</h1></div>', unsafe_allow_html=True)

app_header(title="Steel Carbon Intensity Model")

# Sidebar – data location and scenario selection
with st.sidebar:
    # logos (optional)
    logo_paths = [ "assets/university_logo.png", "assets/faculty_logo.png" ]
    logo_paths = [p for p in logo_paths if os.path.exists(p)]
    if logo_paths:
        cols = st.columns(len(logo_paths))
        for col, path in zip(cols, logo_paths):
            with col:
                st.image(path, width=100)

    # Scenario picker (folder-only; no upload)
    def list_scenarios(data_dir: str) -> List[str]:
        sc_dir = pathlib.Path(data_dir) / "scenarios"
        if not sc_dir.exists():
            return []
        return sorted([p.name for p in sc_dir.glob("*.yml")])

    available = list_scenarios(DATA_ROOT)
    if available:
        scenario_choice = st.selectbox(
            "Scenario file (data/scenarios)",
            options=available,
            index=0
        )
        scenario_path = pathlib.Path(DATA_ROOT) / "scenarios" / scenario_choice
        scenario = load_data_from_yaml(
            str(scenario_path),
            default_value=None,
            unwrap_single_key=False
        )
        scenario_name = scenario_choice
    else:
        scenario_choice = None
        scenario = {}
        scenario_name = "(no scenario)"
        st.warning("No scenario .yml files found in data/scenarios")

    # --- Run options
    st.header("Run options")
    route = _route_from_scenario(scenario, scenario_name)
    st.caption(f"Route preset: **{route}** (locked by scenario)")
    if route == "auto":
        st.warning("Scenario does not specify a route; using AUTO (no preset mask).")

    stage_key = st.selectbox("Stop at stage", options=list(STAGE_MATS.keys()), index=0)
    demand_qty = st.number_input("Demand quantity at selected stage", value=1000.0, min_value=0.0, step=100.0)

    # --- Logging controls (fixed)
    st.header("Logging")
    log_runs = st.checkbox("Create run bundle (ZIP) & save", value=False)
    log_folder = st.text_input("Save folder", value="run_logs")
    uploaded = None  # (no file-upload UI at the moment)

data_dir = DATA_ROOT

# Load base tables once per change in data_dir
@st.cache_data(show_spinner=False)
def _load_base(data_dir: str):
    base = os.path.join(data_dir, "")
    energy_int     = load_data_from_yaml(os.path.join(base, 'energy_int.yml'))
    energy_shares  = load_data_from_yaml(os.path.join(base, 'energy_matrix.yml'))
    energy_content = load_data_from_yaml(os.path.join(base, 'energy_content.yml'))
    e_efs          = load_data_from_yaml(os.path.join(base, 'emission_factors.yml'))
    params         = load_parameters      (os.path.join(base, 'parameters.yml'))
    mkt_cfg        = load_market_config   (os.path.join(base, 'mkt_config.yml'))
    elec_map       = load_electricity_intensity(os.path.join(base, 'electricity_intensity.yml'))
    recipes        = load_recipes_from_yaml(os.path.join(base, 'recipes.yml'), params, energy_int, energy_shares, energy_content)
    return energy_int, energy_shares, energy_content, e_efs, params, mkt_cfg, elec_map, recipes

energy_int, energy_shares, energy_content, e_efs, params, mkt_cfg, elec_map, recipes = _load_base(data_dir)

# Country selection for electricity EF (scenario hint → selectbox override)
pre_code = (scenario.get('grid_country') or scenario.get('country') or '').upper()
country_code = None
if elec_map:
    options = sorted(elec_map.keys())
    idx = options.index(pre_code) if pre_code in options else 0
    country_code = st.selectbox("Grid electricity country (sets Electricity EF)", options=options, index=idx)
    if country_code:
        e_efs['Electricity'] = float(elec_map[country_code])
        params.grid_country = country_code
        st.caption(f"Electricity EF: {e_efs['Electricity']:.2f} gCO₂/MJ (country {country_code})")
else:
    st.warning("No electricity_intensity.yml found. Using default Electricity EF from emission_factors.yml.")

# Scenario-level overrides
apply_fuel_substitutions(scenario.get('fuel_substitutions', {}), energy_shares, energy_int, energy_content, e_efs)
apply_dict_overrides(energy_int,     scenario.get('energy_int', {}))
apply_dict_overrides(energy_shares,  scenario.get('energy_matrix', {}))
apply_dict_overrides(energy_content, scenario.get('energy_content', {}))
apply_dict_overrides(e_efs,          scenario.get('emission_factors', {}))

# Params (deep-ish merge)
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
            b.sinter = s / tot
            b.pellet = p_ / tot
            b.lump   = l / tot
    except AttributeError:
        pass

_param_patch = scenario.get('param_overrides', None)
if _param_patch is None:
    _param_patch = scenario.get('parameters', {})
_recursive_ns_update(params, _param_patch)
_renorm_blend(params)

# Intensity adjustments (after overrides)
adjust_blast_furnace_intensity(energy_int, energy_shares, params)
adjust_process_gas_intensity('Coke Production', 'process_gas_coke', energy_int, energy_shares, params)

# Recipes again (to re-evaluate expressions with updated params) + recipe overrides
recipes = load_recipes_from_yaml(os.path.join(data_dir, 'recipes.yml'), params, energy_int, energy_shares, energy_content)
recipes = apply_recipe_overrides(recipes, scenario.get('recipe_overrides', {}), params, energy_int, energy_shares, energy_content)

# Pre-mask for route & pre-select disabling of conflicting upstream cores
pre_mask = build_route_mask(route, recipes)
feed_mode = {
    "EAF-Scrap": "scrap",
    "DRI-EAF":   "dri",
    "BF-BOF":    None,
    "External":  None,
    "auto":      None,
}.get(route)

# Enforce EAF feed on a copy of recipes so user can flip route without side-effects
import copy
recipes_for_ui = copy.deepcopy(recipes)
enforce_eaf_feed(recipes_for_ui, feed_mode)

# Soft pre-select disabling for upstream cores (clearer UI defaults)
UPSTREAM_CORE = {
    "Blast Furnace", "Basic Oxygen Furnace", "Direct Reduction Iron", "Electric Arc Furnace", "Scrap Purchase"
}
route_disable = {
    "EAF-Scrap": {"Blast Furnace", "Basic Oxygen Furnace", "Direct Reduction Iron"},
    "DRI-EAF":   {"Blast Furnace", "Basic Oxygen Furnace"},
    "BF-BOF":    {"Direct Reduction Iron", "Electric Arc Furnace"},
    "External":  set(),
    "auto":      set(),
}.get(route, set())
pre_select_soft = {p: 0 for p in route_disable if p in UPSTREAM_CORE}

# ----------------------------- helpers ---------------------------------------
def _ns_to_dict(ns):
    try:
        return {k: _ns_to_dict(getattr(ns, k)) for k in vars(ns)}
    except Exception:
        if isinstance(ns, dict):
            return {k: _ns_to_dict(v) for k, v in ns.items()}
        if isinstance(ns, (list, tuple)):
            return [_ns_to_dict(v) for v in ns]
        return ns

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def build_producers_index(recipes: List[Process]) -> Dict[str, List[Process]]:
    prod = {}
    for r in recipes:
        for m in r.outputs:
            prod.setdefault(m, []).append(r)
    return prod

from collections import deque

def gather_ambiguous_chain_materials(
    recipes: List[Process],
    demand_mat: str,
    pre_mask: Dict[str, int] | None = None,
    pre_select: Dict[str, int] | None = None,
) -> List[Tuple[str, List[str]]]:
    pre_mask = pre_mask or {}
    pre_select = pre_select or {}
    producers = build_producers_index(recipes)
    out: List[Tuple[str, List[str]]] = []
    seen_mats: Set[str] = set()
    q = deque([demand_mat])
    while q:
        mat = q.popleft()
        if mat in seen_mats: continue
        seen_mats.add(mat)
        cand = producers.get(mat, [])
        enabled = [r for r in cand if pre_mask.get(r.name, 1) > 0 and pre_select.get(r.name, 1) > 0]
        if not cand: continue
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


def build_routes_from_ui(
    recipes: List[Process],
    demand_mat: str,
    picks_by_material: Dict[str, str],
    pre_mask: Dict[str, int] | None = None,
    pre_select: Dict[str, int] | None = None,
) -> Dict[str, int]:
    pre_mask = pre_mask or {}
    pre_select = pre_select or {}
    producers = build_producers_index(recipes)
    chosen: Dict[str, int] = {}
    visited_mats: Set[str] = set()
    q = deque([demand_mat])
    while q:
        mat = q.popleft()
        if mat in visited_mats: continue
        visited_mats.add(mat)
        cand = producers.get(mat, [])
        enabled = [r for r in cand if pre_mask.get(r.name, 1) > 0 and pre_select.get(r.name, 1) > 0]
        if not enabled: continue
        pick: Process | None = None
        if len(enabled) == 1:
            pick = enabled[0]
        else:
            pick_name = picks_by_material.get(mat)
            if pick_name is None:
                pick = enabled[0]
            else:
                pick = next((r for r in enabled if r.name == pick_name), enabled[0])
        chosen[pick.name] = 1
        for r in cand:
            if r.name != pick.name:
                chosen[r.name] = 0
        for im in pick.inputs.keys():
            if im not in visited_mats:
                q.append(im)
    return chosen

# ----------------------------- UI --------------------------------------------
demand_mat = STAGE_MATS[stage_key]
ambiguous = gather_ambiguous_chain_materials(recipes_for_ui, demand_mat, pre_mask=pre_mask, pre_select=pre_select_soft)

st.subheader("Route & treatment choices")
if not ambiguous:
    st.info("No ambiguous producers along the chain with this setup — the path is unique.")

# Persist user picks across reruns
if "picks_by_material" not in st.session_state:
    st.session_state.picks_by_material = {}

cols = st.columns(2)
with cols[0]:
    st.write("Select one producer per material (only where multiple options exist):")

def _stage_label_for(mat_name: str) -> str:
    if "Finished" in mat_name:
        return "Finish"
    m = re.search(r"\(IP(\d)\)", mat_name)
    if m:
        return f"IP{m.group(1)}"
    if "Liquid" in mat_name:
        return "Liquid"
    return "Other"

OPTION_LABELS = {
    "Finished Products": {
        "No Coating": "NONE",
        "Hot Dip Metal Coating FP": "HDG",
        "Electrolytic Metal Coating FP": "EG",
        "Organic or Sintetic Coating (painting)": "PAINT",
    },
}

from collections import defaultdict
groups = defaultdict(list)
for mat, options in ambiguous:
    groups[_stage_label_for(mat)].append((mat, options))

_stage_order = ["Finished", "IP4", "IP3", "IP2", "IP1", "Liquid", "Other"]

def _use_selectbox(options: list[str]) -> bool:
    return (len(options) > 5) or (max(len(o) for o in options) > 28)

for stage in _stage_order:
    items = groups.get(stage, [])
    if not items:
        continue
    st.markdown(f"**{stage}**")
    cols = st.columns(2) if len(items) > 1 else [st.container()]
    for i, (mat, options) in enumerate(items):
        with cols[i % len(cols)]:
            default_proc = st.session_state.picks_by_material.get(mat, options[0])
            default_idx = options.index(default_proc) if default_proc in options else 0
            show_label = len(items) > 1
            if _use_selectbox(options):
                st.session_state.picks_by_material[mat] = st.selectbox(
                    label = mat if show_label else "",
                    options = options,
                    index = default_idx,
                    key = f"pick_{mat}",
                    label_visibility = "visible" if show_label else "collapsed",
                )
            else:
                choice_idx = st.radio(
                    label = mat if show_label else "",
                    options = list(range(len(options))),
                    index = default_idx,
                    format_func = lambda i, opts=options: opts[i],
                    horizontal = True,
                    key = f"radio_{mat}",
                    label_visibility = "visible" if show_label else "collapsed",
                )
                st.session_state.picks_by_material[mat] = options[choice_idx]

c1, c2, c3 = st.columns([1,1,3])
with c1:
    if st.button("Reset picks"):
        st.session_state.picks_by_material = {}
        for k in list(st.session_state.keys()):
            if k.startswith("radio_"):
                del st.session_state[k]
        st.experimental_rerun()
with c2:
    run_now = st.button("Run model", type="primary")

# -----------------------------
# Execute model when requested
# -----------------------------
if run_now:
    with st.spinner("Running model (core)…"):
        final_demand = {demand_mat: float(demand_qty)}
        route_cfg = RouteConfig(
            route_preset=route,            # ← forced by scenario (your code above)
            stage_key=stage_key,
            demand_qty=float(demand_qty),
            picks_by_material=dict(st.session_state.picks_by_material),
            pre_select_soft=pre_select_soft,
        )
        scn = ScenarioInputs(
            country_code=(country_code or None),
            scenario=scenario,
            route=route_cfg,
        )
        out = run_scenario(data_dir, scn)
    
        production_routes = out.production_routes
        prod_lvl = out.prod_levels
        energy_balance = out.energy_balance
        emissions = out.emissions
        total = out.total_co2e_kg
    
        # No local recomputation of these (kept for your bundle keys)
        internal_elec = None
        total_gas_MJ = None
        EF_process_gas = None

# Footer note
st.caption("© 2025 UNICAMP – Faculdade de Engenharia Mecânica. App v0.9 (beta)")
