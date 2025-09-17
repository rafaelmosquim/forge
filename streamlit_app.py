# -*- coding: utf-8 -*-
"""
Steel Model – Routes & Treatments (scenario-locked route; core-calculated)
Clean "Stop at stage" (Pig iron / Liquid steel / Finished).
No Finished-tailoring UI.

- Force the route from scenario (content or filename). No route radio.
- Keep the original UI (per-material "treatments" and options).
- Delegate balances/emissions to steel_core_api_v2.run_scenario (no duplicate math).
- Show results and write a simple JSON log (config + total CO₂e).
"""

from __future__ import annotations
import os
import re
import pathlib
import base64
import math
import copy as _copy
import json
from typing import Dict, List, Tuple, Set
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd
import yaml
import numpy as np
from copy import deepcopy
from types import SimpleNamespace
import textwrap, json, yaml
from steel_core_api_v2 import ScenarioInputs, run_scenario


# --- Core wrappers (no duplicate math here) ---
from steel_core_api_v2 import (
    RouteConfig,
    ScenarioInputs,
    run_scenario,
    write_run_log,  # for simple JSON logging
)

# --- Original core imports for UI-building only (recipes graph & helpers) ---
from steel_model_core import (
    Process,
    load_data_from_yaml,
    load_parameters,
    load_recipes_from_yaml,
    load_market_config,
    load_electricity_intensity,
    apply_fuel_substitutions,
    apply_dict_overrides,
    apply_recipe_overrides,
    adjust_blast_furnace_intensity,
    adjust_process_gas_intensity,
    # Sankey plot builders (work from prod levels / energy balance)
    make_mass_sankey,
    make_energy_sankey,
    make_energy_to_process_sankey,
    make_hybrid_sankey,
    # Route helpers & constants
    STAGE_MATS,
    build_route_mask,
    enforce_eaf_feed,
)

st.set_page_config(
    page_title="Steel Model – Routes & Treatments",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_PROFILE = os.getenv("APP_PROFILE", "dev").lower()
IS_PAPER = (APP_PROFILE == "paper")
DATA_ROOT = st.session_state.get("DATA_ROOT", "data")



# -----------------------------
# Helpers
# -----------------------------



from pathlib import Path

APP_DIR = Path(__file__).parent
ASSETS  = APP_DIR / "assets"
MAP_PNG = ASSETS / "process_map.png"

def _render_svg(path: Path, width_px: int = 100):
    """Inline an SVG in Streamlit via data URI."""
    if not path.exists():
        return
    try:
        svg_text = path.read_text(encoding="utf-8")
        b64 = base64.b64encode(svg_text.encode("utf-8")).decode("utf-8")
        st.markdown(
            f"<div style='display:flex;justify-content:center;'>"
            f"<img src='data:image/svg+xml;base64,{b64}' width='{width_px}' />"
            f"</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

def _pick_first(*names: str):
    for n in names:
        p = ASSETS / n
        if p.exists():
            return p
    return None

def _coalesce_df(*candidates):
    for x in candidates:
        if isinstance(x, pd.DataFrame):
            return x
    return None

def _pick(o, name, default=None):
    if o is None: return default
    if isinstance(o, dict): return o.get(name, default)
    return getattr(o, name, default)

def _find_balance_matrix(o):
    """
    Try common attribute names; validate by presence of 'External Inputs'/'Final Demand'.
    """
    if o is None: return None
    candidates = [
        "balance_matrix", "material_balance", "mass_balance",
        "balance_df", "materials_df", "balance", "matrix"
    ]
    for nm in candidates:
        df = _pick(o, nm)
        if isinstance(df, pd.DataFrame):
            idx = df.index.astype(str)
            if any("External Inputs" == i for i in idx) and any("Final Demand" == i for i in idx):
                return df
    # As a last resort, scan all DataFrame attrs on the object
    for nm in dir(o):
        try:
            df = getattr(o, nm)
        except Exception:
            continue
        if isinstance(df, pd.DataFrame):
            idx = df.index.astype(str)
            if "External Inputs" in idx and "Final Demand" in idx:
                return df
    return None

def _find_params(last):
    # 1) session_state
    ps = st.session_state.get("params")
    if ps is not None:
        return ps
    # 2) attributes on last
    for nm in ["params", "parameters", "model_params", "run_params"]:
        ps = _pick(last, nm)
        if ps is not None:
            return ps
    # 3) fallback minimal namespace (enough for UI defaults)
    return SimpleNamespace(
        fC_coal_coking=0.82, fC_pci=0.80, fC_coke_product=0.90,
        limestone_purity=1.0, fC_hot_metal=0.045, fC_liquid_steel=0.0015, fC_dri=0.01
    )

def render_sidebar_logos(
    left="faculty_logo.png",
    right="university_logo.png",
    middle_svg="fgv-logo.png",
    width_px=100,
    middle_width_px=None,
):
    left_path   = ASSETS / left
    right_path  = ASSETS / right
    middle_path = ASSETS / middle_svg

    with st.sidebar:
        col1, colm, col2 = st.columns([1, 1, 1])
        with col1:
            if left_path.exists():
                st.image(str(left_path), width=width_px)
        with colm:
            if middle_path.exists():
                if middle_path.suffix.lower() == ".svg":
                    _render_svg(middle_path, width_px=(middle_width_px or width_px))
                else:
                    st.image(str(middle_path), width=(middle_width_px or width_px))
        with col2:
            if right_path.exists():
                st.image(str(right_path), width=width_px)
        st.markdown("")  # small spacer

# -----------------------------
# Sweep helpers (self‑contained)
# -----------------------------

def _snapshot_current_setup_for_sweep(
    route_preset: str,
    stage_key: str,
    demand_qty: float,
    country_code: str | None,
    scenario: dict,
    picks_by_material: dict,
    pre_select_soft: dict,
):
    """Capture an immutable snapshot so Sweep runs are apples‑to‑apples.
    We deep‑copy the scenario and picks to avoid cross‑pollution with the main tab.
    """
    snap = {
        "route_preset": route_preset,
        "stage_key": stage_key,
        "demand_qty": float(demand_qty),
        "country_code": (country_code or None),
        "scenario": _copy.deepcopy(scenario),
        "picks_by_material": _copy.deepcopy(picks_by_material or {}),
        "pre_select_soft": _copy.deepcopy(pre_select_soft or {}),
    }
    return snap


def _linspace(a: float, b: float, steps: int) -> list[float]:
    if steps <= 1:
        return [float(a)]
    d = (b - a) / (steps - 1)
    return [a + i * d for i in range(steps)]


def _ensure_energy_int(scn: dict) -> dict:
    scn.setdefault("energy_int", {})
    return scn


def _total_energy_mj_boundary(energy_balance_df: pd.DataFrame) -> float:
    """Compute boundary energy (no double count of onsite secondaries like Coke/Electricity).
    Works with your wide table shape (Process × carriers)."""
    if energy_balance_df is None or energy_balance_df.empty:
        return 0.0
    EB = energy_balance_df.copy()
    cols = list(EB.columns)
    if cols and cols[0] != "Process":
        EB = EB.rename(columns={cols[0]: "Process"})
    carriers = [c for c in EB.columns if c != "Process" and pd.api.types.is_numeric_dtype(EB[c])]
    if not carriers:
        return 0.0
    total = float(EB[carriers].sum().sum())
    # Coke: subtract primaries in Coke Production (keep Coke at point of use)
    if "Coke" in carriers:
        mask = EB["Process"].astype(str).str.contains("Coke Production", case=False, na=False)
        prim_in_coke = float(EB.loc[mask, [c for c in carriers if c != "Coke"]].sum().sum())
        total -= prim_in_coke
    # Charcoal (if present)
    if "Charcoal" in carriers:
        mask = EB["Process"].astype(str).str.contains("Charcoal Production", case=False, na=False)
        prim_in_char = float(EB.loc[mask, [c for c in carriers if c != "Charcoal"]].sum().sum())
        total -= prim_in_char
    # Electricity production / utility plant (if present)
    if "Electricity" in carriers:
        mask = EB["Process"].astype(str).str.contains("Electricity Production|Utility Plant", case=False, na=False)
        prim_in_elec = float(EB.loc[mask, [c for c in carriers if c != "Electricity"]].sum().sum())
        total -= prim_in_elec
    return max(total, 0.0)

def _route_from_scenario(scenario: dict | None, scenario_name: str) -> str:
    """Infer the route preset from scenario dict or its filename."""
    if scenario:
        for k in ("route_preset", "route", "preset"):
            v = scenario.get(k)
            if isinstance(v, str) and v.strip():
                v2 = v.strip().lower().replace("_", "-")
                if "dri-eaf" in v2 or v2 in {"drieaf", "dri"}:
                    return "DRI-EAF"
                if "eaf-scrap" in v2 or v2 in {"eafscrap"}:
                    return "EAF-Scrap"
                if "bf-bof" in v2 or "bf" in v2 or "bof" in v2:
                    return "BF-BOF"
                if "external" in v2:
                    return "External"
    name = (scenario_name or "").lower()
    if "dri" in name and "eaf" in name: return "DRI-EAF"
    if "eaf" in name and "scrap" in name: return "EAF-Scrap"
    if "bf" in name or "bof" in name:    return "BF-BOF"
    if "external" in name:               return "External"
    return "auto"


def _detect_stage_keys() -> Dict[str, str]:
    """
    Map stage roles to the keys in STAGE_MATS:
      pig_iron          : BF hot metal
      liquid_steel      : after BOF/EAF, before CC
      as_cast           : crude steel as-cast (post-CC)
      after_cr          : after cold rolling (IP2)
      finished          : finished products
    """
    found = {}
    for k, mat in STAGE_MATS.items():
        m = str(mat).lower()

        # pig iron / hot metal
        if ("pig iron" in m) or ("hot metal" in m) or ("hot-metal" in m) or ("hotmetal" in m):
            found.setdefault("pig_iron", k)

        # liquid steel (tap) – before CC
        if (("liquid" in m and "steel" in m) or ("ingot" in m)) and ("finished" not in m):
            found.setdefault("liquid_steel", k)

        # crude steel (as-cast) – right after CC (IP1), not Raw Products
        if ("(ip1)" in m) or ("cast steel" in m) or ("as-cast" in m) or ("continuous casting" in m and "cast" in m):
            found.setdefault("as_cast", k)

        # after cold rolling (often IP2)
        if ("cold raw steel" in m) or ("after cold rolling" in m) or ("cold rolling" in m) or ("(ip2)" in m):
            found.setdefault("after_cr", k)

        # finished
        if "finished" in m:
            found.setdefault("finished", k)

    # Fallbacks (conservative, keep previous behavior if missing)
    if "pig_iron" not in found:
        found["pig_iron"] = sorted(STAGE_MATS.keys())[0]
        
    if "liquid_steel" not in found:
        lk = next((k for k, v in STAGE_MATS.items()
                   if isinstance(v, str) and (("liquid" in v.lower() and "steel" in v.lower()) or ("ingot" in v.lower()))), None)
        found["liquid_steel"] = lk or found["pig_iron"]
        
    if "as_cast" not in found:
        ak = next((k for k, v in STAGE_MATS.items()
                   if isinstance(v, str) and "(ip1)" in v.lower()), None)
        found["as_cast"] = ak or found.get("liquid_steel", sorted(STAGE_MATS.keys())[0])
        
    if "after_cr" not in found:
        ik = next((k for k in STAGE_MATS if "ip2" in str(k).lower()), None)
        found["after_cr"] = ik or found["as_cast"]
        
    if "finished" not in found:
        fk = next((k for k, v in STAGE_MATS.items() if isinstance(v, str) and ("finished" in v.lower())), None)
        found["finished"] = fk or list(STAGE_MATS.keys())[-1]

    return found


def _load_for_picks(data_dir: str, route_preset: str, stage_key: str, scenario: dict):
    """
    Build a UI-ready recipe graph consistent with the scenario and route,
    without re-doing model math. This mirrors the light preprocessing the core
    uses so ambiguous choices match the actual runnable graph.
    """
    base = os.path.join(data_dir, "")
    energy_int     = load_data_from_yaml(os.path.join(base, 'energy_int.yml'))
    energy_shares  = load_data_from_yaml(os.path.join(base, 'energy_matrix.yml'))
    energy_content = load_data_from_yaml(os.path.join(base, 'energy_content.yml'))
    e_efs          = load_data_from_yaml(os.path.join(base, 'emission_factors.yml'))
    params         = load_parameters      (os.path.join(base, 'parameters.yml'))

    # Initial recipes
    recipes = load_recipes_from_yaml(
        os.path.join(base, 'recipes.yml'),
        params, energy_int, energy_shares, energy_content
    )

    # Scenario overrides that change which producers are available/connected
    apply_fuel_substitutions(scenario.get('fuel_substitutions', {}), energy_shares, energy_int, energy_content, e_efs)
    apply_dict_overrides(energy_int,     scenario.get('energy_int', {}))
    apply_dict_overrides(energy_shares,  scenario.get('energy_matrix', {}))
    apply_dict_overrides(energy_content, scenario.get('energy_content', {}))
    apply_dict_overrides(e_efs,          scenario.get('emission_factors', {}))

    # Parameters overrides (light merge)
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

    # Intensity adjustments that affect connectivity/modes
    adjust_blast_furnace_intensity(energy_int, energy_shares, params)
    adjust_process_gas_intensity('Coke Production', 'process_gas_coke', energy_int, energy_shares, params)

    # Re-load recipes to re-evaluate expressions with new params
    recipes = load_recipes_from_yaml(
        os.path.join(base, 'recipes.yml'),
        params, energy_int, energy_shares, energy_content
    )
    recipes = apply_recipe_overrides(recipes, scenario.get('recipe_overrides', {}), params, energy_int, energy_shares, energy_content)

    # Route mask + EAF feed mode (for UI)
    pre_mask = build_route_mask(route_preset, recipes)
    import copy
    recipes_for_ui = copy.deepcopy(recipes)
    eaf_mode = {"EAF-Scrap":"scrap","DRI-EAF":"dri","BF-BOF":None,"External":None,"auto":None}.get(route_preset)
    enforce_eaf_feed(recipes_for_ui, eaf_mode)

    demand_mat = STAGE_MATS[stage_key]
    return recipes_for_ui, pre_mask, demand_mat


def build_producers_index(recipes: List[Process]) -> Dict[str, List[Process]]:
    prod = {}
    for r in recipes:
        for m in r.outputs:
            prod.setdefault(m, []).append(r)
    return prod

def _pick(o, name, default=None):
    # works for dicts and objects (dataclass/SimpleNamespace/attrs)
    if o is None:
        return default
    if isinstance(o, dict):
        return o.get(name, default)
    return getattr(o, name, default)


def gather_ambiguous_chain_materials(
    recipes: List[Process],
    demand_mat: str,
    pre_mask: Dict[str, int] | None = None,
    pre_select: Dict[str, int] | None = None,
) -> List[Tuple[str, List[str]]]:
    """Traverse upstream from demand_mat and list materials with >1 enabled producers."""
    pre_mask   = pre_mask or {}
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


# -----------------------------
# Page setup
# -----------------------------

st.markdown("""
<style>
:root { --primary-color: #2563eb; }
div[data-baseweb="radio"] [aria-checked="true"] > div:first-child { background-color: #2563eb !important; border-color: #2563eb !important; }
div[data-baseweb="radio"] [aria-checked="false"] > div:first-child { border-color: #2563eb !important; }
[role="radiogroup"] [role="radio"][aria-checked="true"] > div:first-child { background-color: #2563eb !important; border-color: #2563eb !important; }
button[kind="primary"], [data-testid="baseButton-primary"] { background-color: #16a34a !important; border-color: #16a34a !important; color: #fff !important; }
button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover { background-color: #15803d !important; border-color: #15803d !important; }
.hr { height:1px; background:linear-gradient(to right,#e5e7eb,#cbd5e1,#e5e7eb); margin:.25rem 0 .75rem 0; border:0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div style="text-align:center;"><h3>FORGE  - Flexible Optimization of Routes for GHG & Energy</h3></div>', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;"><h6>A steel plant energy & emissions model</h6></div>',
    unsafe_allow_html=True
)

if IS_PAPER:
    (tab_main,) = st.tabs(["Main Model"])
else:
    tab_main, tab_sweep, tab_static = st.tabs(["Main Model", "Sensitivity", "Static Mods"])

# Defaults for yield UI even when the Static Mods tab is hidden (paper mode)
# --- Safe defaults for variables created in hidden tabs (paper profile) ---
use_yield  = bool(st.session_state.get("static_yield_apply", False))
try:
    yield_frac = float(st.session_state.get("static_yield_value", 1.0))
except Exception:
    yield_frac = 1.0

# Default for the run button in case the widget hasn't rendered yet in this pass
run_now = False

# -----------------------------
# Sidebar – data & scenario
# -----------------------------

render_sidebar_logos(width_px=80, middle_svg="fgv-logo.png", middle_width_px=120)
with st.sidebar:
    st.header("Main options")
    # Scenario picker (from data/scenarios)
    def list_scenarios(data_dir: str) -> List[str]:
        sc_dir = pathlib.Path(data_dir) / "scenarios"
        if not sc_dir.exists():
            return []
        return sorted([p.name for p in sc_dir.glob("*.yml")])
    
    # Choose which data folder to use
    data_choice = st.selectbox(
        "Data set",
        ["Likely (data/)", "Low (data_min/)", "High (data_max/)"],
        index=0,
        help="Switches all YAML loads to the chosen folder."
    )
    _map = {
        "Likely (data/)": "data",
        "Low (data_min/)": "data_min",
        "High (data_max/)": "data_max",
    }
    DATA_ROOT = _map[data_choice]
    st.session_state["DATA_ROOT"] = DATA_ROOT
    st.caption(f"Using data folder: {DATA_ROOT}/")


    available = list_scenarios(DATA_ROOT)
    PREFERRED = "BF_BOF_coal.yml"
    default_idx = available.index(PREFERRED) if PREFERRED in available else 0

    if available:
        scenario_choice = st.selectbox(
            "Route",
            options=available,
            index=default_idx
        )
        scenario_path = pathlib.Path(DATA_ROOT) / "scenarios" / scenario_choice
        scenario = load_data_from_yaml(str(scenario_path), default_value=None, unwrap_single_key=False)
        scenario_name = scenario_choice
    else:
        scenario_choice = None
        scenario = {}
        scenario_name = "(no scenario)"
        st.warning("No scenario .yml files found in data/scenarios")


    route = _route_from_scenario(scenario, scenario_name)
    #st.caption(f"Route preset: **{route}** (locked by scenario)")
    if route == "auto":
        st.warning("Scenario does not specify a route; using AUTO (no preset mask).")
        
        
        
     # Cleaned "Stop at stage" (Pig iron / Liquid steel / Finished)
    stage_keys = _detect_stage_keys()
    stage_menu = {
        "Pig iron": stage_keys["pig_iron"],
        #"Liquid steel": stage_keys["liquid_steel"],
        "Crude steel": stage_keys["as_cast"],
        #"Steel-mill steel": stage_keys["after_cr"],
        "Finished": stage_keys["finished"],
    }
    default_stage_label = "Crude steel"
    
    stage_label = st.radio(
        "Product",
        options=list(stage_menu.keys()),
        index=list(stage_menu.keys()).index(default_stage_label),
        help=(
            "Pig iron: hot metal after Blast Furnace/DRI."
            #"Liquid steel: after BOF/EAF (before continuous casting)."
            "Crude steel: after continuous casting."
            #"Steel-mill steel: plant boundary.  "
            "Finished: include off-site processing."
        ),
    )
    stage_key = stage_menu[stage_label]
    is_as_cast  = (stage_label == "Crude steel")
    is_after_cr = (stage_label == "Steel-mill steel")
    is_finished = (stage_label == "Finished")
    
    demand_qty = 1000
    
    # Grid EF country (passed to core; EF not altered in-app)
    elec_map = load_electricity_intensity(os.path.join(DATA_ROOT, "electricity_intensity.yml")) or {}
    country_opts = sorted(elec_map.keys()) if elec_map else []
    country_code = st.selectbox("Country grid electricity", options=["BRA"] + country_opts, index=0, 
                                help = "Sets emission factor for electricity. Values for 2024, " \
                                "from https://ourworldindata.org/electricity-mix")
                
# -----------------------------
# Build UI graph (for picks)
# -----------------------------
with tab_main:
    top_metrics = st.empty()
    main_after_run = st.container() 
    try:
        recipes_for_ui, pre_mask, demand_mat = _load_for_picks(DATA_ROOT, route, stage_key, scenario)
    except Exception as e:
        st.error("Setup failed while preparing route/treatments UI.")
        st.exception(e)
        st.stop()
    
    #pre_mask.pop("Coke", None)  # allow both "Coke Production" and "Coke from market"

    # ---------------------------------
    # Read Post-CC values (no UI here)
    # ---------------------------------
    enable_post_cc = (is_after_cr or is_finished)
    
    
    if enable_post_cc:
        cc_choice_val = st.session_state.get("cc_choice_radio", "Hot Rolling")
        cr_toggle_val = st.session_state.get("cr_toggle", False)
    else:
        cc_choice_val = None
        cr_toggle_val = False
        # Clean up stray state so radios/checkboxes don't leak into crude steel
        for k in ("cc_choice_radio", "cr_toggle"):
            if k in st.session_state:
                del st.session_state[k]
    
    # Stage-specific forced picks
    forced_pre_select = {}
    
    # 1) Lock CRUDE STEEL to Regular (R) and remove any lingering post-CC picks
    if is_as_cast:
        forced_pre_select["Cast Steel (IP1)"] = "Continuous Casting (R)"
        pbm = st.session_state.setdefault("picks_by_material", {})
        for k in ("Raw Products (types)", "Intermediate Process 3"):
            pbm.pop(k, None)
    
    # 2) For Steel-mill steel / Finished, propagate Post-CC choice AND force the IP3 bridge
    if enable_post_cc:
        # HR vs Rod/bar path after CC
        if cc_choice_val:
            forced_pre_select["Raw Products (types)"] = cc_choice_val
    
        # Also pin the IP3 bypass so the flow continues beyond CC
        if is_after_cr or is_finished:
            forced_pre_select["Intermediate Process 3"] = (
                "Bypass CR→IP3" if (cc_choice_val == "Hot Rolling" and cr_toggle_val)
                else "Bypass Raw→IP3"
            )
            
    
    # Disable conflicting upstream cores for defaulting (soft)
    UPSTREAM_CORE = {"Blast Furnace", "Basic Oxygen Furnace", "Direct Reduction Iron", "Electric Arc Furnace", "Scrap Purchase"}
    route_disable = {
        "EAF-Scrap": {"Blast Furnace", "Basic Oxygen Furnace", "Direct Reduction Iron"},
        "DRI-EAF":   {"Blast Furnace", "Basic Oxygen Furnace"},
        "BF-BOF":    {"Direct Reduction Iron", "Electric Arc Furnace"},
        "External":  set(),
        "auto":      set(),
    }.get(route, set())
    pre_select_soft = {p: 0 for p in route_disable if p in UPSTREAM_CORE}
    
    # # Force picks derived from Post-CC controls (use session values)
    # forced_pre_select = {}
    # if enable_post_cc and cc_choice_val:
    #     forced_pre_select["Raw Products (types)"] = cc_choice_val  # HR or Rod/bar/section
    # if stage_key == "Finished":
    #     # CR only matters if Hot Rolling is selected
    #     forced_pre_select["Intermediate Process 3"] = (
    #         "Bypass CR→IP3" if (cc_choice_val == "Hot Rolling" and cr_toggle_val) else "Bypass Raw→IP3"
    #     )
    
    # Persist these forced decisions
    if "picks_by_material" not in st.session_state:
        st.session_state.picks_by_material = {}
    st.session_state.picks_by_material.update(forced_pre_select)
    
    # Merge and compute ambiguity
    pre_select = {**pre_select_soft, **forced_pre_select}
    ambiguous = gather_ambiguous_chain_materials(
        recipes_for_ui, demand_mat, pre_mask=pre_mask, pre_select=pre_select
    )
    
    # Don't render UI for materials we already forced above
    if forced_pre_select:
        ambiguous = [(m, opts) for (m, opts) in ambiguous if m not in forced_pre_select]
    
    # -----------------------------
    # Route & treatment choices (display + layout)
    # -----------------------------
    
    # UI-only renames (do NOT alter engine tokens)
    PROC_RENAMES = {
        "Continuous Casting (R)": "Regular (R)",
        "Continuous Casting (L)": "Low-alloy (L)",
        "Continuous Casting (H)": "High-alloy (H)",
    }
    LABEL_RENAMES = {
        "Cast Steel (IP1)": "Alloying (choose class)",
        "Manufactured Feed (IP4)": "Shaping (IP4)",
    }
    STAGE_DISPLAY = {
        "IP1": "Alloying",
        "Raw": "Post-CC",
        "IP4": "Shaping (off-site)",
        "Finished": "Finishing (off-site)"
    }
    
    def _fmt_proc(name: str) -> str:
        return PROC_RENAMES.get(name, name)
    
    # Materials that should appear in the Upstream section (add more as needed)
    UPSTREAM_MATS = {
        "Nitrogen",
        "Oxygen",
        #"Coal",
        "Coke",
        "Dolomite",
        "Burnt Lime"
        # "Natural Gas", "Coal", "Coke", "Dolomite", "Burnt Lime",  # uncomment/extend if useful
    }
    
    # UI-only renames for producers (optional niceties)
    PROC_RENAMES.update({
        "Nitrogen Production": "Onsite",
        "Nitrogen from market": "Purchase",
        "Oxygen Production": "Onsite",
        "Oxygen from market": "Purchase",
        "Coke Production": "Onsite",
        "Coke Petroleum from Market": "Purchase (pet)",
        "Coke Mineral from Market": "Purchase (raw)",
        "Dolomite Production": "Onsite",
        "Dolomite from market": "Purchase",
        "Burnt Lime Production": "Onsite",
        "Burnt Lime from market": "Purchase",
        "Coal from Market": "Coking Coal",
        "Anthracite Coal from Market": "Anthracite",
        "PCI Coal from Market": "Pulverized"
    })
    
    def _stage_label_for(mat_name: str) -> str:
        if mat_name in UPSTREAM_MATS:
            return "Upstream"
        if "Finished" in mat_name: return "Finished"
        if "Manufactured Feed (IP4)" in mat_name: return "IP4"
        if "Intermediate Process 3" in mat_name: return "IP3"
        if "Cold Raw Steel (IP2)" in mat_name: return "IP2"
        if "Raw Products" in mat_name: return "Raw"
        m = re.search(r"\(IP(\d)\)", mat_name)
        if m: return f"IP{m.group(1)}"
        if "Liquid" in mat_name: return "Liquid"
        return "Other"
    
    # Group ambiguous by stage
    groups = defaultdict(list)
    if is_as_cast:
        groups.pop("IP1", None)   # no Alloying column for crude steel
    for mat, options in ambiguous:
        groups[_stage_label_for(mat)].append((mat, options))
    
    # Desired left→right column order
    primary_order = ["IP1", "Raw", "IP4", "Finished"]
    
           
    # --- Downstream header + "Process Map" button, same font as Upstream
    left, right = st.columns([1, 3])
    with left:
        st.subheader("Downstream choices")        # same font as Upstream

    with right:
        if MAP_PNG.exists():
            b64 = base64.b64encode(MAP_PNG.read_bytes()).decode()
            st.components.v1.html(
                f"""
                <button
                    onclick="(function(){{
                        const w = window.open('about:blank','_blank'); if(!w) return;
                        w.document.title='Process Map';
                        w.document.body.style.margin='0';
                        w.document.body.style.background='#0b0b0b';
                        const img=new Image();
                        img.style.display='block';
                        img.style.maxWidth='50%';
                        img.style.height='auto';
                        img.style.margin='0 auto';
                        img.src='data:image/png;base64,{b64}';
                        img.onload=()=>w.document.body.appendChild(img);
                    }})()"
                    style="padding:.35rem .6rem;border:1px solid #cbd5e1;border-radius:.5rem;
                        cursor:pointer;background:none;">
                    Process Map
                </button>
                """,
                height=40,
            )
        else:
            st.caption(f"Map not found at: {MAP_PNG}")

        
    cols = st.columns(len(primary_order))
    
    def _use_selectbox(options: list[str]) -> bool:
        return (len(options) > 5) or (max(len(o) for o in options) > 28)
    
    def _render_group(stage: str, container, show_title: bool = True, n_cols: int | None = None):
        items = groups.get(stage, [])
        if not items:
            return
        with container:
            if show_title:
                st.markdown(f"**{STAGE_DISPLAY.get(stage, stage)}**")
            # If n_cols is provided, use it; else keep the old behavior (2 cols when >1 item)
            n = (n_cols if (isinstance(n_cols, int) and n_cols > 0)
                 else (2 if len(items) > 1 else 1))
            inner_cols = st.columns(n)
            for i, (mat, options) in enumerate(items):
                with inner_cols[i % len(inner_cols)]:
                    label_text = LABEL_RENAMES.get(mat, mat)
                    default_proc = st.session_state.picks_by_material.get(mat, options[0])
                    default_idx = options.index(default_proc) if default_proc in options else 0
                    show_label = len(items) > 1
                    if _use_selectbox(options):
                        st.session_state.picks_by_material[mat] = st.selectbox(
                            label = label_text if show_label else "",
                            options = options,
                            index = default_idx,
                            key = f"pick_{mat}",
                            label_visibility = "visible" if show_label else "collapsed",
                            format_func = _fmt_proc,
                        )
                    else:
                        choice_idx = st.radio(
                            label = label_text if show_label else "",
                            options = list(range(len(options))),
                            index = default_idx,
                            format_func = lambda i, opts=options: _fmt_proc(opts[i]),
                            horizontal = True,
                            key = f"radio_{mat}",
                            label_visibility = "visible" if show_label else "collapsed",
                        )
                        st.session_state.picks_by_material[mat] = options[choice_idx]
    
    # 1) Alloying (IP1)
    #'_render_group("IP1", cols[0])' if False else None  # placeholder to avoid accidental edits
    if not is_as_cast:
        _render_group("IP1", cols[0])
    
    with cols[1]:
        if enable_post_cc:
            st.markdown("**After Continuous Casting**")
            cc_choice_widget = st.radio(
                "",
                ["Hot Rolling", "Rod/bar/section Mill"],
                index = 0 if st.session_state.get("cc_choice_radio", "Hot Rolling") == "Hot Rolling" else 1,
                key = "cc_choice_radio",
                horizontal = True,
                label_visibility="collapsed",
            )
            if cc_choice_widget == "Hot Rolling":
                st.checkbox(
                    "Apply Cold Rolling",
                    value = st.session_state.get("cr_toggle", False),
                    key = "cr_toggle",
                )
            else:
                st.session_state["cr_toggle"] = False

    
    # 3) Shaping (IP4)
    _render_group("IP4", cols[2])
    
    # 4) Finished
    _render_group("Finished", cols[3])

    # --- Upstream picks, no title line ---
    st.subheader("Upstream choices", help="Model considers Scopes 1+2 only; upstream purchases excludes emissions from this process.")
    if groups.get("Upstream"):
        _render_group("Upstream", st.container(), show_title=False, n_cols = 6)
        
    
    # Reset & Run row
    # c1, c2 = st.columns([1, 1])
    # with c2:
    #     if st.button("Reset picks"):
    #         st.session_state.picks_by_material = {}
    #         for k in list(st.session_state.keys()):
    #             if k.startswith("radio_") or k.startswith("pick_") or k in ("cc_choice_radio", "cr_toggle"):
    #                 del st.session_state[k]
    #         st.experimental_rerun()
    # with c1:
    run_now = st.button("Run model", type="primary")
    
    st.markdown("<hr class='hr'>", unsafe_allow_html=True)

# -----------------------------
# Sweep Tab UI
# -----------------------------
if not IS_PAPER:
    with tab_sweep:
        st.subheader("Parameter sweep (1-D)")
        st.caption("Snapshot current setup (route, stage, picks) → vary one parameter over a range → record key metrics.")

        # 1) Snapshot row
        c_snap1, c_snap2 = st.columns([1,2])
        with c_snap1:
            if st.button("Use current setup", key="sweep_take_snapshot"):
                st.session_state["sweep_snapshot"] = _snapshot_current_setup_for_sweep(
                    route_preset=route,
                    stage_key=stage_key,
                    demand_qty=float(demand_qty),
                    country_code=(country_code or None),
                    scenario=scenario,
                    picks_by_material=dict(st.session_state.get("picks_by_material", {})),
                    pre_select_soft=pre_select_soft,
                )
        with c_snap2:
            snap = st.session_state.get("sweep_snapshot")
            if snap:
                st.success(
                    f"Snapshot ready • Route: {snap['route_preset']} • Stage key: {snap['stage_key']} • Demand: {snap['demand_qty']:.0f}"
                )
            else:
                st.info("Click **Use current setup** to capture a sweep snapshot.")

        # 2) Parameter selection
        st.markdown("**Parameter**")
        param_label = st.radio(
            "Choose a parameter to sweep",
            options=[
                "Blast Furnace base energy intensity (MJ/kg hot metal)",
                "BOF scrap share (fraction of metallic charge)",
                "Grid electricity intensity (min/avg/max)",
            ],
            index=0,
            help=(
                "BF intensity: writes scenario['energy_int']['Blast Furnace'].\n"
                "BOF scrap share: overrides Basic Oxygen Furnace recipe inputs → Scrap=s, Pig Iron=(1-s).\n"
                "Grid electricity: sets scenario['emission_factors']['Electricity'] to min/avg/max from electricity_intensity.yml."
            ),
        )

        # 2a) Range (and save ranges for MC)
        if param_label.startswith("Grid electricity"):
            try:
                elec_map = load_electricity_intensity(os.path.join(DATA_ROOT, "electricity_intensity.yml")) or {}
            except Exception:
                elec_map = {}
            vals = [(k, float(v)) for k, v in elec_map.items() if isinstance(v, (int, float))]
            if vals:
                min_code, min_val = min(vals, key=lambda kv: kv[1])
                max_code, max_val = max(vals, key=lambda kv: kv[1])
                avg_val = sum(v for _, v in vals) / len(vals)
            else:
                min_code = max_code = "—"
                min_val = max_val = avg_val = 0.0

            st.info(
                f"Grid EF points (gCO₂/MJ): "
                f"min **{min_val:.2f}** ({min_code}), "
                f"avg **{avg_val:.2f}**, "
                f"max **{max_val:.2f}** ({max_code})"
            )
            # Save for sweep/MC
            st.session_state["_grid_points"] = [float(min_val), float(avg_val), float(max_val)]
            st.session_state["_grid_labels"] = [f"min({min_code})", "avg", f"max({max_code})"]

        else:
            # Numeric range UI (one set of widgets only)
            if param_label.startswith("BOF scrap"):
                vmin_default, vmax_default, steps_default = 0.00, 0.25, 6
                step_size = 0.01
            else:
                vmin_default, vmax_default, steps_default = 11.0, 15.0, 9
                step_size = 0.1

            c_rng1, c_rng2, c_rng3 = st.columns(3)
            with c_rng1:
                vmin = st.number_input("Min", value=float(vmin_default), step=step_size)
            with c_rng2:
                vmax = st.number_input("Max", value=float(vmax_default), step=step_size)
            with c_rng3:
                steps = st.number_input("Steps", value=int(steps_default), min_value=2, max_value=200, step=1)

            # Save ranges for MC
            if param_label.startswith("BOF scrap"):
                st.session_state["mc_bof_range"] = (float(vmin), float(vmax))
            else:
                st.session_state["mc_bf_range"] = (float(vmin), float(vmax))

        # Metric to summarize
        metric_choice = st.selectbox(
            "Chart metric",
            options=["Total CO2e (kg)", "Gross EF (kg/unit)", "Final EF (kg/unit)", "Total energy (MJ, boundary)"],
            index=0,
        )

        # 3) Run 1-D sweep
        run_sweep = st.button("Run sweep", type="primary")
        if run_sweep:
            snap = st.session_state.get("sweep_snapshot")
            if not snap:
                st.warning("No snapshot yet. Click **Use current setup** first.")
                st.stop()

            # Build Xs
            if param_label.startswith("Grid electricity"):
                grid_points = st.session_state.get("_grid_points")
                if grid_points:
                    xs = list(grid_points)
                else:
                    try:
                        elec_map = load_electricity_intensity(os.path.join(DATA_ROOT, "electricity_intensity.yml")) or {}
                    except Exception:
                        elec_map = {}
                    vals = [float(v) for v in elec_map.values() if isinstance(v, (int, float))]
                    xs = [min(vals), sum(vals)/len(vals), max(vals)] if vals else [0.0, 0.0, 0.0]
            else:
                xs = _linspace(float(vmin), float(vmax), int(steps))

            sweep_rows = []
            prog = st.progress(0.0)

            for i, x in enumerate(xs, start=1):
                scn_dict = _copy.deepcopy(snap["scenario"])
                _ensure_energy_int(scn_dict)

                country_override = (snap["country_code"] or None)

                if param_label.startswith("Blast Furnace"):
                    scn_dict["energy_int"]["Blast Furnace"] = float(x)

                elif param_label.startswith("BOF scrap"):
                    s = max(0.0, min(0.25, float(x)))
                    pig = max(0.0, 1.0 - s)
                    ro = scn_dict.setdefault("recipe_overrides", {})
                    bof = ro.setdefault("Basic Oxygen Furnace", {})
                    bof_inputs = bof.setdefault("inputs", {})
                    bof_inputs["Scrap"] = float(s)
                    bof_inputs["Pig Iron"] = float(pig)

                elif param_label.startswith("Grid electricity"):
                    scn_dict.setdefault("emission_factors", {})["Electricity"] = float(x)
                    country_override = None  # force use of override

                route_cfg = RouteConfig(
                    route_preset=snap["route_preset"],
                    stage_key=snap["stage_key"],
                    demand_qty=float(snap["demand_qty"]),
                    picks_by_material=_copy.deepcopy(snap["picks_by_material"]),
                    pre_select_soft=_copy.deepcopy(snap["pre_select_soft"]),
                )
                scn_inputs = ScenarioInputs(
                    country_code=country_override,
                    scenario=scn_dict,
                    route=route_cfg,
                )

                try:
                    out_i = run_scenario(DATA_ROOT, scn_inputs)
                except Exception as e:
                    sweep_rows.append({
                        "x": x,
                        "Total CO2e (kg)": math.nan,
                        "Gross EF (kg/unit)": math.nan,
                        "Final EF (kg/unit)": math.nan,
                        "Total energy (MJ, boundary)": math.nan,
                        "_error": str(e),
                    })
                    prog.progress(i / len(xs))
                    continue

                total_i  = getattr(out_i, "total_co2e_kg", None)
                energy_i = getattr(out_i, "energy_balance", None)
                demand_i = float(snap["demand_qty"]) if float(snap["demand_qty"]) > 0 else math.nan
                gross_ef = (float(total_i) / demand_i) if (total_i is not None and math.isfinite(demand_i) and demand_i > 0) else math.nan

                # Use Static-tab yield (same as MC)
                use_yield = bool(st.session_state.get("static_yield_apply", False))
                yfrac     = float(st.session_state.get("static_yield_value", 1.0))
                final_ef  = (gross_ef / max(1e-9, yfrac)) if (use_yield and math.isfinite(gross_ef)) else gross_ef

                tot_mj   = _total_energy_mj_boundary(energy_i) if energy_i is not None else math.nan

                sweep_rows.append({
                    "x": float(x),
                    "Total CO2e (kg)": float(total_i) if total_i is not None else math.nan,
                    "Gross EF (kg/unit)": float(gross_ef) if gross_ef is not None else math.nan,
                    "Final EF (kg/unit)": float(final_ef) if final_ef is not None else math.nan,
                    "Total energy (MJ, boundary)": float(tot_mj) if tot_mj is not None else math.nan,
                })
                prog.progress(i / len(xs))

            res_df = pd.DataFrame(sweep_rows)
            ymap = {
                "Total CO2e (kg)": "Total CO2e (kg)",
                "Gross EF (kg/unit)": "Gross EF (kg/unit)",
                "Final EF (kg/unit)": "Final EF (kg/unit)",
                "Total energy (MJ, boundary)": "Total energy (MJ, boundary)",
            }
            ycol = ymap[metric_choice]

            cmin, cmed, cmax = st.columns(3)
            series = res_df[ycol].astype(float)
            if series.notna().any():
                with cmin: st.metric(f"Min {ycol}",    f"{series.min():,.2f}")
                with cmed: st.metric(f"Median {ycol}", f"{series.median():,.2f}")
                with cmax: st.metric(f"Max {ycol}",    f"{series.max():,.2f}")
            else:
                st.info("No valid points computed.")

        # ------------------------------------------
        # Monte Carlo (multi-parameter) – independent
        # ------------------------------------------
        st.markdown("---")
        st.subheader("Monte Carlo (multi-parameter)")

        mc_n    = st.number_input("Samples", min_value=50, max_value=10000, value=500, step=50, help="Number of random draws.")
        mc_seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        run_mc  = st.button("Run Monte Carlo", type="primary", key="btn_run_mc")

        if run_mc:
            snap = st.session_state.get("sweep_snapshot")
            if not snap:
                st.warning("No snapshot yet. Click **Use current setup** first.")
                st.stop()

            bf_min, bf_max     = st.session_state.get("mc_bf_range",  (12.5, 15.5))
            bof_min, bof_max   = st.session_state.get("mc_bof_range", (0.0, 0.20))

            grid_points = st.session_state.get("_grid_points")
            if not grid_points:
                try:
                    elec_map = load_electricity_intensity(os.path.join(DATA_ROOT, "electricity_intensity.yml")) or {}
                    vals = [float(v) for v in elec_map.values() if isinstance(v, (int, float))]
                except Exception:
                    vals = []
                grid_points = [min(vals), sum(vals)/len(vals), max(vals)] if vals else [0.0, 0.0, 0.0]

            st.caption(
                f"BF intensity ~ U[{bf_min:.3g}, {bf_max:.3g}] • "
                f"BOF scrap ~ U[{bof_min:.3g}, {bof_max:.3g}] • "
                f"Grid EF ∈ {{ {', '.join(f'{v:.2f}' for v in grid_points)} }} gCO₂/MJ"
            )

            rng = np.random.default_rng(int(mc_seed))
            mc_rows = []
            prog_mc = st.progress(0.0)
            N = int(mc_n)

            for i in range(N):
                bf_eint   = float(rng.uniform(bf_min, bf_max))     # MJ/kg hot metal
                bof_scrap = float(rng.uniform(bof_min, bof_max))   # fraction
                grid_ef   = float(rng.choice(grid_points))         # gCO2/MJ

                scn_dict = deepcopy(snap["scenario"])
                scn_dict.setdefault("energy_int", {})["Blast Furnace"] = bf_eint

                ro = scn_dict.setdefault("recipe_overrides", {}).setdefault("Basic Oxygen Furnace", {})
                inputs = ro.setdefault("inputs", {})
                s = max(0.0, min(1.0, bof_scrap))
                inputs["Scrap"] = s
                inputs["Pig Iron"] = 1.0 - s

                scn_dict.setdefault("emission_factors", {})["Electricity"] = grid_ef
                country_override = None

                route_cfg = RouteConfig(
                    route_preset=snap["route_preset"],
                    stage_key=snap["stage_key"],
                    demand_qty=float(snap["demand_qty"]),
                    picks_by_material=deepcopy(snap["picks_by_material"]),
                    pre_select_soft=deepcopy(snap["pre_select_soft"]),
                )
                scn_inputs = ScenarioInputs(
                    country_code=country_override,
                    scenario=scn_dict,
                    route=route_cfg,
                )

                try:
                    out_i = run_scenario(DATA_ROOT, scn_inputs)
                    total_i  = getattr(out_i, "total_co2e_kg", None)
                    energy_i = getattr(out_i, "energy_balance", None)

                    demand_i = float(snap["demand_qty"]) if float(snap["demand_qty"]) > 0 else float("nan")
                    gross_ef = (float(total_i) / demand_i) if (total_i is not None and math.isfinite(demand_i) and demand_i > 0) else float("nan")

                    use_yield = bool(st.session_state.get("static_yield_apply", False))
                    yfrac     = float(st.session_state.get("static_yield_value", 1.0))
                    final_ef  = (gross_ef / max(1e-9, yfrac)) if (use_yield and math.isfinite(gross_ef)) else gross_ef

                    tot_mj   = _total_energy_mj_boundary(energy_i) if energy_i is not None else float("nan")

                    mc_rows.append({
                        "BF MJ/kg": bf_eint,
                        "BOF scrap": s,
                        "Grid gCO2/MJ": grid_ef,
                        "Total CO2e (kg)": float(total_i) if total_i is not None else float("nan"),
                        "Gross EF (kg/unit)": float(gross_ef) if math.isfinite(gross_ef) else float("nan"),
                        "Final EF (kg/unit)": float(final_ef) if math.isfinite(final_ef) else float("nan"),
                        "Total energy (MJ, boundary)": float(tot_mj) if math.isfinite(tot_mj) else float("nan"),
                    })
                except Exception as e:
                    mc_rows.append({
                        "BF MJ/kg": bf_eint, "BOF scrap": s, "Grid gCO2/MJ": grid_ef,
                        "Total CO2e (kg)": math.nan, "Gross EF (kg/unit)": math.nan,
                        "Final EF (kg/unit)": math.nan, "Total energy (MJ, boundary)": math.nan,
                        "_error": str(e),
                    })

                prog_mc.progress((i + 1) / N)

            mc_df = pd.DataFrame(mc_rows)

            # Which metric from MC to plot
            metric_choice_mc = st.selectbox(
                "Plot metric",
                options=["Total CO2e (kg)", "Gross EF (kg/unit)", "Final EF (kg/unit)", "Total energy (MJ, boundary)"],
                index=1,
                key="mc_metric_choice"
            )

            series = pd.to_numeric(mc_df[metric_choice_mc], errors="coerce").dropna().to_numpy()
            if series.size == 0:
                st.info("No valid Monte Carlo results.")
            else:
                import numpy as np
                import matplotlib.pyplot as plt

                # Axis label (no auto unit switching tied to external refs)
                x_label = metric_choice_mc

                # Gaussian KDE (Silverman's bandwidth)
                n = series.size
                s = np.std(series, ddof=1) if n > 1 else 0.0
                h = 1.06 * s * (n ** (-1/5)) if s > 0 else max(1e-3, 0.01 * (np.mean(series) if series.size else 1.0))
                x_min, x_max = series.min(), series.max()
                pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
                xs = np.linspace(x_min - pad, x_max + pad, 400)

                # KDE evaluation
                u = (xs[:, None] - series[None, :]) / h
                kde = (np.exp(-0.5 * u * u).sum(axis=1) / (n * h * np.sqrt(2 * np.pi)))

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(xs, kde, linewidth=2, label=f"KDE (N={n})")
                ax.fill_between(xs, 0, kde, alpha=0.18)

                # Tiny rug so you still "see every run"
                ax.plot(series, np.zeros_like(series), "|", markersize=10, alpha=0.45)

                ax.set_xlabel(x_label)
                ax.set_ylabel("Density")
                ax.legend(loc="best")

                st.pyplot(fig)

        pass
# -----------------------------
# Static Tab UI
# -----------------------------
if not IS_PAPER:
    with tab_static:
        # ---- Route-locked process energy intensity override ----
        allowed_proc_by_route = {
            "BF-BOF":   ("Blast Furnace",        "BF base energy intensity (MJ/kg hot metal)"),
            "DRI-EAF":  ("Direct Reduction Iron","DRI base energy intensity (MJ/kg DRI)"),
            "EAF-Scrap":("Electric Arc Furnace","EAF base energy intensity (MJ/kg liquid steel)"),
        }
        allowed = allowed_proc_by_route.get(route)

        if allowed:
            proc_name, friendly_label = allowed
            existing_ei = (scenario.get('energy_int') or {})
            scenario['energy_int'] = {proc_name: existing_ei.get(proc_name)} if proc_name in existing_ei else {}

            with st.expander("Process energy intensity override", expanded=False):
                enable_override = st.checkbox(f"{proc_name}", value=proc_name in (scenario.get('energy_int') or {}), key="static_ei_enable")

                if enable_override:
                    default_val = None
                    try:
                        default_val = float((scenario.get('energy_int') or {}).get(proc_name))
                    except Exception:
                        default_val = None
                    if default_val is None:
                        try:
                            base_energy_int = load_data_from_yaml(os.path.join(DATA_ROOT, 'energy_int.yml')) or {}
                            default_val = float(base_energy_int.get(proc_name, 0.0))
                        except Exception:
                            default_val = 0.0

                    val = st.number_input(
                        friendly_label,
                        value=float(default_val),
                        min_value=0.0,
                        step=0.1,
                        key="static_ei_value",
                    )
                    scenario['energy_int'][proc_name] = float(val)
                else:
                    scenario['energy_int'] = {}
        else:
            scenario['energy_int'] = {}

        # ---- BF fuel split (only for BF-BOF) ----
        if route == "BF-BOF":
            with st.expander("Blast Furnace — fuel split", expanded=False):

                def _bf_base_shares_with_extras():
                    """Return (elec0, coke0, coal0, char0, others_dict) from scenario or base file."""
                    # scenario override (if any)
                    sc_bf = (scenario.get('energy_matrix') or {}).get('Blast Furnace', {}) or {}
                    # base file fallback
                    try:
                        base = load_data_from_yaml(os.path.join(DATA_ROOT, 'energy_matrix.yml')) or {}
                    except Exception:
                        base = {}
                    base_bf = base.get('Blast Furnace', {}) or {}

                    def _get(d, k, default=0.0):
                        v = d.get(k, None)
                        try:
                            return float(v) if v is not None else float(base_bf.get(k, default))
                        except Exception:
                            return float(base_bf.get(k, default))

                    # read core carriers
                    elec0 = _get(sc_bf, 'Electricity', 0.0)
                    coke0 = _get(sc_bf, 'Coke',       0.0)
                    coal0 = _get(sc_bf, 'Coal',       0.0)
                    char0 = _get(sc_bf, 'Charcoal',   0.0)

                    # preserve any other carriers (Gas, Wood, etc.)
                    others_keys = set(sc_bf.keys()) | set(base_bf.keys())
                    others_keys.difference_update({'Electricity', 'Coke', 'Coal', 'Charcoal'})
                    others = {}
                    for k in sorted(others_keys):
                        others[k] = _get(sc_bf, k, 0.0)

                    # tiny normalization guard (don’t change relative sizes)
                    total = elec0 + coke0 + coal0 + char0 + sum(others.values())
                    if total > 0:
                        scale = 1.0 / total
                        elec0 *= scale; coke0 *= scale; coal0 *= scale; char0 *= scale
                        for k in others: others[k] *= scale

                    return elec0, coke0, coal0, char0, others

                elec0, coke0, coal0, char0, others = _bf_base_shares_with_extras()

                # Thermal slice available to C/C/C = 1 - (electricity + others)
                others_sum = sum(others.values())
                thermal_total = max(1e-9, 1.0 - elec0 - others_sum)

                # Defaults for sliders expressed INSIDE the thermal slice
                coke_frac_th = (coke0 / thermal_total) if thermal_total > 0 else 0.0
                pci_th = max(0.0, 1.0 - coke_frac_th)  # remaining thermal is PCI
                pci_char_frac = (char0 / max(1e-9, (coal0 + char0))) if (coal0 + char0) > 0 else 0.0

                # UI
                coke_share_th = st.slider(
                    "Coke share (of BF thermal input)",
                    0.0, 1.0, float(coke_frac_th), 0.01, key="static_bf_coke"
                )
                pci_charcoal_frac = st.slider(
                    "Charcoal fraction inside PCI (coal↔charcoal)",
                    0.0, 1.0, float(pci_char_frac), 0.01, key="static_bf_charfrac",
                    help="0 → PCI is 100% Coal; 1 → PCI is 100% Charcoal. Electricity and other carriers stay fixed."
                )

                # Recompose absolute shares (keep electricity & others fixed)
                coke_abs = thermal_total * coke_share_th
                pci_abs  = thermal_total - coke_abs
                coal_abs = pci_abs * (1.0 - pci_charcoal_frac)
                char_abs = pci_abs * pci_charcoal_frac

                # Write back without touching Electricity or the 'others'
                scenario.setdefault('energy_matrix', {})
                bm = {'Electricity': elec0, 'Coke': coke_abs, 'Coal': coal_abs, 'Charcoal': char_abs, **others}

                # final tiny renorm to sum exactly 1.0, **locking** Electricity proportion
                total = sum(bm.values())
                if total > 0:
                    scale = 1.0 / total
                    # scale all, but rescale thermal/others uniformly (electricity included so sums to 1)
                    for k in bm:
                        bm[k] *= scale

                scenario['energy_matrix']['Blast Furnace'] = bm

                st.caption(
                    "Slider adjusts only the thermal slice (Coke vs PCI). Electricity and any other carriers stay fixed; "
                    "PCI is split between Coal and Charcoal."
                )


        # ---- Gas blend (NG / biomethane / Green H₂ / Blue H₂) ----
        with st.expander("Gas blend (sets EF for 'gas')", expanded=False):
            st.caption("Final EF for **Gas** = Σ shareᵢ × EFᵢ (gCO₂/MJ).")
            c1, c2 = st.columns(2)
            with c1:
                sh_ng  = st.number_input("Natural gas share", min_value=0.0, max_value=1.0, value=1.0, step=0.01, key="static_gas_ng")
                sh_bio = st.number_input("Biomethane share",  min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="static_gas_bio")
            with c2:
                sh_h2g = st.number_input("Green H₂ share",    min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="static_gas_h2g")
                sh_h2b = st.number_input("Blue H₂ share",     min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="static_gas_h2b")

            S = max(1e-9, sh_ng + sh_bio + sh_h2g + sh_h2b)
            w_ng, w_bio, w_h2g, w_h2b = (sh_ng/S, sh_bio/S, sh_h2g/S, sh_h2b/S)

            try:
                base_efs = load_data_from_yaml(os.path.join(DATA_ROOT, 'emission_factors.yml')) or {}
            except Exception:
                base_efs = {}

            def find_ef(d: dict, keys: list[str]) -> float | None:
                for k in keys:
                    v = d.get(k)
                    if v is not None:
                        try: return float(v)
                        except Exception: continue
                return None

            ef_ng  = find_ef(base_efs, ["Natural gas", "Natural Gas", "Gas"])
            ef_bio = find_ef(base_efs, ["Biomethane", "Bio-methane", "Upgraded biogas"])
            ef_h2g = find_ef(base_efs, ["Hydrogen (Electrolysis)", "Green hydrogen", "H2 - Electrolysis"])
            ef_h2b = find_ef(base_efs, ["Hydrogen (Methane reforming + CCS)", "Hydrogen (Methane reforming)", "Blue hydrogen", "H2 - SMR+CCS", "H2 - ATR+CCS"])

            all_missing = all(x is None for x in [ef_ng, ef_bio, ef_h2g, ef_h2b])
            ef_gas = float(base_efs.get("Gas", 0.0)) if all_missing else (
                w_ng  * (ef_ng  or 0.0) +
                w_bio * (ef_bio or 0.0) +
                w_h2g * (ef_h2g or 0.0) +
                w_h2b * (ef_h2b or 0.0)
            )

            st.metric("Effective EF for Gas", f"{ef_gas:.2f} gCO₂/MJ")
            scenario.setdefault('emission_factors', {})
            scenario['emission_factors']['Gas'] = float(ef_gas)
            scenario['gas_blend'] = {
                "shares": {"NG": w_ng, "Biomethane": w_bio, "H2_green": w_h2g, "H2_blue": w_h2b},
                "effective_gas_ef_gco2_per_MJ": ef_gas,
            }

        # --- Material yield (reporting) ---
        with st.expander("Material yield (reporting)", expanded=False):
            use_yield = st.checkbox(
                "Apply yield to emission factor",
                value=False,
                help="Final EF = (total emission / demand) ÷ yield ",
                key="static_yield_apply",
            )
            yield_frac = st.number_input(
                "Yield (fraction 0-1)",
                value=1.00, min_value=0.0, max_value=1.0, step=0.01,
                key="static_yield_value",
            )
        pass

# -----------------------------
# Execute model when requested
# -----------------------------

if run_now:
    try:
        with st.spinner("Running model (core)…"):
            route_cfg = RouteConfig(
                route_preset=route,
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
            out = run_scenario(DATA_ROOT, scn)
            
        # ---- Map core outputs (after the spinner) ----
        production_routes = getattr(out, "production_routes", None)
        prod_lvl         = getattr(out, "prod_levels", None)
        energy_balance   = getattr(out, "energy_balance", None)
        emissions        = getattr(out, "emissions", None)
        total            = getattr(out, "total_co2e_kg", None)

        # Try to grab the material balance from out under common names
        balance_matrix = next(
            (x for x in [
                getattr(out, "balance_matrix", None),
                getattr(out, "material_balance", None),
                getattr(out, "mass_balance", None),
                _find_balance_matrix(out),
            ] if isinstance(x, pd.DataFrame)),
            None
        )

        # Params (for the Validation tab sliders). Prefer session, else from out/meta, else a small fallback.
        params = (
            st.session_state.get("params")
            or getattr(out, "params", None)
            or getattr(out, "parameters", None)
            or SimpleNamespace(
                fC_coal_coking=0.82, fC_pci=0.80, fC_coke_product=0.90,
                limestone_purity=1.0, fC_hot_metal=0.045, fC_liquid_steel=0.0015, fC_dri=0.01
            )
        )

        # ✅ Persist exactly what the Validation tab needs
        st.session_state["last_run_outputs"] = SimpleNamespace(
            emissions=emissions,
            balance_matrix=balance_matrix,
            params=params,
        )
        # Optional but handy elsewhere
        st.session_state["last_run_recipes"] = getattr(out, "recipes", None)

        with main_after_run:
            st.success("Model run complete (core).")

            is_finished = stage_key in ("Finished", "Finished steel")
            raw_total   = float(total or 0.0)
            reported    = (raw_total/0.85) if is_finished else raw_total

            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    "Total CO₂e — raw",
                    f"{raw_total:,.0f} kg CO₂e per ton",
                    help="Direct model total (no yield adjustment)."
                )
            with c2:
                st.metric(
                    "Total CO₂e — reported (÷0.85)" if is_finished else "Total CO₂e — reported",
                    f"{reported:,.0f} kg CO₂e per ton",
                    help="This accounts for yield (0.85%) downstream if product is Finished."
                )

            # Downloads
            df_runs = pd.DataFrame(sorted(prod_lvl.items()), columns=["Process", "Runs"]).set_index("Process")
            d1, d2, d3 = st.columns(3)
            d1.download_button("Production runs (CSV)", data=df_runs.to_csv().encode("utf-8"),
                               file_name="production_runs.csv", mime="text/csv")
            if isinstance(energy_balance, pd.DataFrame) and not energy_balance.empty:
                d2.download_button("Energy balance (CSV)", data=energy_balance.to_csv().encode("utf-8"),
                                   file_name="energy_balance.csv", mime="text/csv")
            if isinstance(emissions, pd.DataFrame):
                d3.download_button("Emissions (CSV)", data=emissions.to_csv().encode("utf-8"),
                                   file_name="emissions.csv", mime="text/csv")

            # Tables
            if isinstance(emissions, pd.DataFrame) and not emissions.empty:
                st.dataframe(emissions)
            else:
                st.info("No per-process emissions table for this run.")

            if isinstance(energy_balance, pd.DataFrame) and not energy_balance.empty:
                st.dataframe(energy_balance)
            else:
                st.info("No energy balance table for this run.")
                
    except Exception as e:
        st.error("Run crashed. Traceback below:")
        st.exception(e)
        st.stop()

st.caption("© 2025 UNICAMP – Faculdade de Engenharia Mecânica. App v1.2 (scenario-locked route, core-calculated, clean stage selector)")

