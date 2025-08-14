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
import yaml  # make sure this is available at top-level (you already use it)
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
/* === Make radios blue (works across Streamlit/BaseWeb DOMs) =============== */

/* 1) Nudge theme primary color (helps many components) */
:root { --primary-color: #2563eb; }

/* 2) BaseWeb radios: selected dot & ring */
div[data-baseweb="radio"] [aria-checked="true"] > div:first-child {
  background-color: #2563eb !important;
  border-color: #2563eb !important;
}
div[data-baseweb="radio"] [aria-checked="false"] > div:first-child {
  border-color: #2563eb !important;
}

/* 3) Older/fallback structure */
[role="radiogroup"] [role="radio"][aria-checked="true"] > div:first-child {
  background-color: #2563eb !important;
  border-color: #2563eb !important;
}

/* 4) Your chip highlight for the whole label when selected */
div[role="radiogroup"] > label:has(input:checked) {
  border-color: #2563eb;
  background: rgba(37, 99, 235, 0.08);
}

/* === Primary buttons (Run model) → green =================================== */
button[kind="primary"], [data-testid="baseButton-primary"] {
  background-color: #16a34a !important;
  border-color: #16a34a !important;
  color: #fff !important;
}
button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover {
  background-color: #15803d !important;
  border-color: #15803d !important;
}
</style>
""", unsafe_allow_html=True)


# --- Branding header (replaces old st.title/st.caption) ----------------------

def app_header(
    title: str = "Steel Carbon Intensity Model",
):
    st.markdown("""
    <style>
      .app-hero { padding:.25rem 0 0 0; text-align:center; }
      .app-hero h1 { margin:0; font-size:1.8rem; line-height:1.2; }
      .hr { height:1px; background:linear-gradient(to right,#e5e7eb,#cbd5e1,#e5e7eb); margin:.25rem 0 .75rem 0; border:0; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="app-hero"><h1>{title}</h1></div>', unsafe_allow_html=True)
    #st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    
app_header(title="Steel Carbon Intensity Model")

# Sidebar – data location and scenario selection
with st.sidebar:
    # --- Top logos side-by-side (80 px each)
    logo_paths = [ "assets/university_logo.png", "assets/faculty_logo.png" ]
    logo_paths = [p for p in logo_paths if os.path.exists(p)]  # only those that exist
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

    # --- Run options (moved up, right under Setup) ---------------------------
    #st.divider()
    st.header("Run options")

    # lock route to scenario
    route = _route_from_scenario(scenario, scenario_name)
    st.caption(f"Route preset: **{route}** (locked by scenario)")
    if route == "auto":
        st.warning("Scenario does not specify a route; using AUTO (no preset mask).")

    stage_key = st.selectbox("Stop at stage", options=list(STAGE_MATS.keys()), index=0)
    demand_qty = st.number_input("Demand quantity at selected stage", value=1000.0, min_value=0.0, step=100.0)

    #st.divider()
    #st.checkbox("Advanced: custom overrides (energy/process)", key="show_adv")

       
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

# Params (deep-ish merge): reuse the helper from core script
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

# Pre-mask for route & pre-select disabling of conflicting upstream cores (mirrors CLI)
pre_mask = build_route_mask(route, recipes)
feed_mode = {
    "EAF-Scrap": "scrap",
    "DRI-EAF":   "dri",
    "BF-BOF":    None,
    "External":  None,
    "auto":      None,
}.get(route)

# Enforce EAF feed on a *copy* of recipes so user can flip route without side-effects
import copy
recipes_for_ui = copy.deepcopy(recipes)
enforce_eaf_feed(recipes_for_ui, feed_mode)

# Soft pre-select disabling for upstream cores (for clearer UI defaults)
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

# Ambiguous materials along the chain (after route constraints)
from typing import Set

def _ns_to_dict(ns):
    # Convert SimpleNamespace → plain dict (recursive)
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
        if mat in visited_mats:
            continue
        visited_mats.add(mat)
        cand = producers.get(mat, [])
        enabled = [r for r in cand if pre_mask.get(r.name, 1) > 0 and pre_select.get(r.name, 1) > 0]
        if not enabled:
            continue
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

# --- staged radio helpers (UI-only; no model logic changed) -------------------
def _stage_label_for(mat_name: str) -> str:
    if "Finished" in mat_name:
        return "Finish"
    m = re.search(r"\(IP(\d)\)", mat_name)
    if m:
        return f"IP{m.group(1)}"
    if "Liquid" in mat_name:
        return "Liquid"
    return "Other"

# Short labels for specific materials (fallback: show original process name)
OPTION_LABELS = {
    "Finished Products": {
        "No Coating": "NONE",
        "Hot Dip Metal Coating FP": "HDG",
        "Electrolytic Metal Coating FP": "EG",
        "Organic or Sintetic Coating (painting)": "PAINT",
    },
    # You can add more material-specific label maps here if you want:
    # "Manufactured Feed (IP4)": { ... }
}

from collections import defaultdict
groups = defaultdict(list)
for mat, options in ambiguous:
    groups[_stage_label_for(mat)].append((mat, options))

_stage_order = ["Finished", "IP4", "IP3", "IP2", "IP1", "Liquid", "Other"]

def _use_selectbox(options: list[str]) -> bool:
    # Heuristic: too many or too long → dropdown saves space
    return (len(options) > 5) or (max(len(o) for o in options) > 28)

for stage in _stage_order:
    items = groups.get(stage, [])
    if not items:
        continue

    st.markdown(f"**{stage}**")

    # 2 columns to reduce vertical scrolling
    cols = st.columns(2) if len(items) > 1 else [st.container()]

    for i, (mat, options) in enumerate(items):
        with cols[i % len(cols)]:
            # Current choice (full process name)
            default_proc = st.session_state.picks_by_material.get(mat, options[0])
            default_idx = options.index(default_proc) if default_proc in options else 0

            show_label = len(items) > 1  # hide redundant label if single item in stage

            if _use_selectbox(options):
                # Compact dropdown when many/long options
                st.session_state.picks_by_material[mat] = st.selectbox(
                    label = mat if show_label else "",
                    options = options,                     # full names
                    index = default_idx,
                    key = f"pick_{mat}",
                    label_visibility = "visible" if show_label else "collapsed",
                )
            else:
                # Chip-style radios (full names displayed; CSS makes them compact)
                choice_idx = st.radio(
                    label = mat if show_label else "",
                    options = list(range(len(options))),
                    index = default_idx,
                    format_func = lambda i, opts=options: opts[i],  # show full names
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
    with st.spinner("Running model…"):
        # Build production_routes from UI picks
        production_routes = build_routes_from_ui(
            recipes_for_ui,
            demand_mat,
            st.session_state.picks_by_material,
            pre_mask=pre_mask,
            pre_select=pre_select_soft,
        )

        final_demand = {demand_mat: float(demand_qty)}

        # Solve material balance
        balance_matrix, prod_lvl = calculate_balance_matrix(recipes_for_ui, final_demand, production_routes)
        if balance_matrix is None:
            st.error("Material balance failed.")
            st.stop()

        # Ensure energy tables have rows for all active variants
        active_procs = [p for p, r in prod_lvl.items() if r > 1e-9]
        expand_energy_tables_for_active(active_procs, energy_shares, energy_int)

        # Internal electricity from recovered gases (before credit)
        recipes_dict = {r.name: r for r in recipes_for_ui}
        internal_elec = calculate_internal_electricity(prod_lvl, recipes_dict, params)

        # Energy balance (and repair BF/CP to base carriers, mirroring CLI)
        energy_balance = calculate_energy_balance(prod_lvl, energy_int, energy_shares)
        if 'Blast Furnace' in energy_balance.index and hasattr(params, 'bf_base_intensity'):
            bf_runs = float(prod_lvl.get('Blast Furnace', 0.0))
            base_bf = float(params.bf_base_intensity)
            bf_sh   = energy_shares.get('Blast Furnace', {})
            for carrier in energy_balance.columns:
                if carrier != 'Electricity':
                    energy_balance.loc['Blast Furnace', carrier] = bf_runs * base_bf * float(bf_sh.get(carrier, 0.0))
        cp_runs = float(prod_lvl.get('Coke Production', 0.0))
        base_cp = float(getattr(params, 'coke_production_base_intensity', energy_int.get('Coke Production', 0.0)))
        cp_sh   = energy_shares.get('Coke Production', {})
        if cp_runs and cp_sh:
            for carrier in energy_balance.columns:
                if carrier != 'Electricity':
                    energy_balance.loc['Coke Production', carrier] = cp_runs * base_cp * float(cp_sh.get(carrier, 0.0))

        # Apply internal electricity credit
        energy_balance = adjust_energy_balance(energy_balance, internal_elec)

        # Compute dynamic EF for recovered gas (matches CLI)
        gas_coke_MJ = prod_lvl.get('Coke Production', 0.0) * recipes_dict.get('Coke Production', Process('',{},{})).outputs.get('Process Gas', 0.0)
        gas_bf_MJ   = (getattr(params, 'bf_adj_intensity', 0.0) - getattr(params, 'bf_base_intensity', 0.0)) * prod_lvl.get('Blast Furnace', 0.0)
        total_gas_MJ = float(gas_coke_MJ + gas_bf_MJ)

        cp_shares = energy_shares.get('Coke Production', {})
        fuels_cp  = [c for c in cp_shares if c != 'Electricity' and cp_shares[c] > 0]
        EF_coke_gas = (sum(cp_shares[c] * e_efs.get(c, 0.0) for c in fuels_cp) / max(1e-12, sum(cp_shares[c] for c in fuels_cp))) if fuels_cp else 0.0
        bf_shares = energy_shares.get('Blast Furnace', {})
        fuels_bf  = [c for c in bf_shares if c != 'Electricity' and bf_shares[c] > 0]
        EF_bf_gas = (sum(bf_shares[c] * e_efs.get(c, 0.0) for c in fuels_bf) / max(1e-12, sum(bf_shares[c] for c in fuels_bf))) if fuels_bf else 0.0
        EF_process_gas = EF_coke_gas if total_gas_MJ <= 1e-9 else (
            (EF_coke_gas * (gas_coke_MJ / max(1e-12, total_gas_MJ))) + (EF_bf_gas * (gas_bf_MJ / max(1e-12, total_gas_MJ)))
        )

        util_eff = recipes_dict.get('Utility Plant', Process('',{},{})).outputs.get('Electricity', 0.0)
        internal_elec = total_gas_MJ * util_eff  # recompute defensively

        # Emissions
        emissions = calculate_emissions(
            mkt_cfg,
            prod_lvl,
            energy_balance,
            e_efs,
            load_data_from_yaml(os.path.join(data_dir, 'process_emissions.yml')),
            internal_elec,
            final_demand,
            total_gas_MJ,
            EF_process_gas,
        )
        
        
        if emissions is not None and 'TOTAL' not in emissions.index:
            emissions.loc['TOTAL'] = emissions.sum()


    # -----------------------------
    # Display results
    # -----------------------------
    st.success("Model run complete.")
    
    # Keep df_runs so the logging code can write production_runs.csv
    df_runs = pd.DataFrame(sorted(prod_lvl.items()), columns=["Process", "Runs"]).set_index("Process")
    
    # Compute total safely; don't render any tables
    total = None
    if (emissions is not None) and (not emissions.empty):
        if ("TOTAL" in emissions.index) and ("TOTAL CO2e" in emissions.columns):
            total = float(emissions.loc["TOTAL", "TOTAL CO2e"])
        elif "TOTAL CO2e" in emissions.columns:
            total = float(emissions["TOTAL CO2e"].sum())
    
    if total is not None:
        st.metric("Total CO₂e", f"{total:,.2f} kg")
    else:
        st.info("No emissions available for this run.")

    # Sankey figures
    st.subheader("Sankey diagrams")
    recipes_dict_live = {r.name: r for r in recipes_for_ui}

    fig_mass = make_mass_sankey(
        prod_lvl=prod_lvl,
        recipes_dict=recipes_dict_live,
        min_flow=0.5,
        title=f"Mass Flow Sankey — {demand_qty:.0f} units {STAGE_MATS[stage_key]} ({scenario_name})",
    )
    st.plotly_chart(fig_mass, use_container_width=True)

    fig_energy = make_energy_sankey(
        energy_balance_df=energy_balance,
        min_MJ=25.0,
        title="Energy Flow Sankey — Process Carriers",
    )
    st.plotly_chart(fig_energy, use_container_width=True)

    if emissions is not None and not emissions.empty:
        fig_hybrid = make_hybrid_sankey(
            energy_balance_df=energy_balance,
            emissions_df=emissions,
            min_MJ=25.0,
            min_kg=1.0,
            co2_scale=None,
            include_direct_and_energy_sinks=True,
        )
        st.plotly_chart(fig_hybrid, use_container_width=True)

        fig_energy_ranked = make_energy_to_process_sankey(
            energy_balance_df=energy_balance,
            emissions_df=emissions,
            title="Energy → Processes (ranked by CO₂e)",
            min_MJ=25.0,
            sort_by="emissions",
        )
        st.plotly_chart(fig_energy_ranked, use_container_width=True)

    # Downloads (CSV + HTMLs zipped)
    st.subheader("Downloads")

    # CSVs
    csv_col1, csv_col2, csv_col3 = st.columns(3)
    csv_col1.download_button(
        label="Download production runs (CSV)",
        data=df_runs.to_csv().encode("utf-8"),
        file_name="production_runs.csv",
        mime="text/csv",
    )
    csv_col2.download_button(
        label="Download energy balance (CSV)",
        data=energy_balance.to_csv().encode("utf-8"),
        file_name="energy_balance.csv",
        mime="text/csv",
    )
    if emissions is not None and not emissions.empty:
        csv_col3.download_button(
            label="Download emissions (CSV)",
            data=emissions.to_csv().encode("utf-8"),
            file_name="emissions.csv",
            mime="text/csv",
        )

    # HTMLs (Plotly figures) packed in a zip for convenience
    html_files = {
        "mass_sankey.html": fig_mass.to_html(include_plotlyjs="cdn"),
        "energy_sankey.html": fig_energy.to_html(include_plotlyjs="cdn"),
    }
    if emissions is not None and not emissions.empty:
        html_files["hybrid_sankey.html"] = fig_hybrid.to_html(include_plotlyjs="cdn")
        html_files["energy_to_process_sankey.html"] = fig_energy_ranked.to_html(include_plotlyjs="cdn")
        
    # -----------------------------
    # Create "run bundle" ZIP (config + results + figures) + save to disk
    # -----------------------------
    if log_runs:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        bundle_name = f"run_{run_id}.zip"
    
        # Scenario bytes + SHA (works for selected file, uploaded file, or inline)
        scen_bytes = None
        scen_sha = None
        scen_filename = "inline_scenario.yml"
        try:
            if uploaded is not None:
                scen_bytes = uploaded.getvalue()
                scen_filename = getattr(uploaded, "name", "uploaded.yml")
            elif scenario_choice:
                p = pathlib.Path(data_dir) / "scenarios" / scenario_choice
                scen_bytes = p.read_bytes()
                scen_filename = p.name
            else:
                # No scenario file on disk—dump the in-memory dict so we have a record
                scen_bytes = yaml.safe_dump(scenario or {}, sort_keys=True).encode("utf-8")
                scen_filename = "scenario_from_session.yml"
            scen_sha = _sha256(scen_bytes) if scen_bytes is not None else None
        except Exception:
            pass
    
        # Build manifest with config + results metadata
        # NOTE: keep only JSON-safe primitives
        params_snapshot = _ns_to_dict(params)
        manifest = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "app": {"title": "Brazilian Steel Decarbonization Model", "version": "1.0"},
            "data_dir": data_dir,
            "scenario_file": scen_filename,
            "scenario_sha256": scen_sha,
            "route_preset": route,
            "stage_key": stage_key,
            "demand_qty": float(demand_qty),
            "grid_country": country_code,
            "electricity_EF_gCO2_per_MJ": float(e_efs.get("Electricity", 0.0)),
            "picks_by_material": dict(st.session_state.get("picks_by_material", {})),
            "production_routes": {k: int(v) for k, v in (production_routes or {}).items()},
            "final_demand": {k: float(v) for k, v in (final_demand or {}).items()},
            "internal_electricity_MJ": float(internal_elec),
            "total_process_gas_MJ": float(total_gas_MJ),
            "EF_process_gas_gCO2_per_MJ": float(EF_process_gas),
            "emissions_total_kgCO2e": float(total) if (emissions is not None and not emissions.empty) else None,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "streamlit": getattr(st, "__version__", "unknown"),
                "pandas": pd.__version__,
            },
            "parameters_snapshot": params_snapshot,  # proof of knobs used
        }
    
        # Build the ZIP (and a file manifest with hashes for integrity)
        file_hashes = {}
        def _add_bytes(zf, arcname: str, b: bytes):
            zf.writestr(arcname, b)
            file_hashes[arcname] = {"sha256": _sha256(b), "bytes": len(b)}
    
        bundle_buf = io.BytesIO()
        with zipfile.ZipFile(bundle_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Config
            if scen_bytes is not None:
                _add_bytes(zf, f"scenario/{scen_filename}", scen_bytes)
            _add_bytes(zf, "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    
            # Results (CSVs)
            _add_bytes(zf, "results/production_runs.csv", df_runs.to_csv().encode("utf-8"))
            _add_bytes(zf, "results/energy_balance.csv", energy_balance.to_csv().encode("utf-8"))
            if emissions is not None and not emissions.empty:
                _add_bytes(zf, "results/emissions.csv", emissions.to_csv().encode("utf-8"))
    
            # Figures (HTML)
            for fname, html in html_files.items():
                _add_bytes(zf, f"figures/{fname}", html.encode("utf-8"))
    
            # File integrity index
            _add_bytes(zf, "manifest_files.json", json.dumps(file_hashes, indent=2, sort_keys=True).encode("utf-8"))
    
        # Offer download and also persist locally
        st.download_button(
            label="Download run log bundle (ZIP)",
            data=bundle_buf.getvalue(),
            file_name=bundle_name,
            mime="application/zip",
        )
    
        try:
            os.makedirs(log_folder, exist_ok=True)
            out_path = os.path.join(log_folder, bundle_name)
            with open(out_path, "wb") as f:
                f.write(bundle_buf.getvalue())
            st.caption(f"Saved: `{out_path}`")
        except Exception as e:
            st.warning(f"Could not save to disk: {e}")
        
    
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, html in html_files.items():
                zf.writestr(fname, html)
        st.download_button(
            label="Download Sankey HTMLs (ZIP)",
            data=buf.getvalue(),
            file_name="sankey_charts.zip",
            mime="application/zip",
        )
    
    

# Footer note
st.caption("© 2025 UNICAMP – Faculdade de Engenharia Mecânica. App v0.9 (beta)")
