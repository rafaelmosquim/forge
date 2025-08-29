# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 19:40:11 2025

@author: rafae
"""

# -*- coding: utf-8 -*-
"""
Steel Model – FULL Wizard/Stepper UI (scenario-locked route; core-calculated)

Step flow (left panel) + Live Preview (right panel):
  1) Scenario
  2) Stage & Demand
  3) Fuels & Overrides
  4) Route Picks
  5) Review & Run

- No sidebar. Important controls surfaced in steps.
- Preserves your core API calls and variable names where possible.
- Keeps picks in st.session_state.picks_by_material across steps.
- Maintains your BF fuel split, gas EF blending, and logging payload.

Drop-in replacement for your current app file.
"""

from __future__ import annotations
import os
import re
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, deque
from datetime import datetime

import streamlit as st
import pandas as pd

# --- Core wrappers (no duplicate math here) ---
from steel_core_api_v2 import (
    RouteConfig,
    ScenarioInputs,
    run_scenario,
    write_run_log,
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
    # Sankey plot builders
    make_mass_sankey,
    make_energy_sankey,
    make_energy_to_process_sankey,
    make_hybrid_sankey,
    # Route helpers & constants
    STAGE_MATS,
    build_route_mask,
    enforce_eaf_feed,
)

DATA_ROOT = "data"  # fixed data folder
APP_DIR = Path(__file__).parent
ASSETS  = APP_DIR / "assets"

# -----------------------------
# General page setup & styles
# -----------------------------
st.set_page_config(
    page_title="Steel Model — Wizard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
/***** Hide Streamlit sidebar chrome *****/
[data-testid="stSidebar"], [data-testid="collapsedControl"]{display:none;}

/***** Step bar styles *****/
.stepbar{position:sticky; top:0; z-index:100; background:#fff; padding:.25rem .25rem .5rem .25rem;}
.step-chip{display:inline-block; padding:.25rem .6rem; border-radius:999px; border:1px solid #d1d5db; margin-right:.25rem; font-size:.85rem;}
.step-chip.active{border-color:#2563eb; background:rgba(37,99,235,.08);}
.step-chip.done{border-color:#16a34a; background:rgba(22,163,74,.08);}
.hr { height:1px; background:linear-gradient(to right,#e5e7eb,#cbd5e1,#e5e7eb); margin:.25rem 0 .75rem 0; border:0; }

/***** Primary buttons *****/
button[kind="primary"], [data-testid="baseButton-primary"]{
  background-color:#16a34a !important; border-color:#16a34a !important; color:#fff !important;
}
button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover{
  background-color:#15803d !important; border-color:#15803d !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------

def _pick_first(*names: str):
    for n in names:
        p = ASSETS / n
        if p.exists():
            return p
    return None


def render_header_logos(left="faculty_logo.png", right="university_logo.png"):
    c1, cmid, c2 = st.columns([1, 6, 1])
    with c1:
        lp = _pick_first(left)
        if lp: st.image(str(lp), use_container_width=True)
    with cmid:
        st.markdown("<h1 style='text-align:center;margin:.25rem 0 0;'>Steel Carbon Intensity Model — Wizard</h1>", unsafe_allow_html=True)
        st.caption("Scenario-locked route; core-calculated; clean stage selector")
    with c2:
        rp = _pick_first(right)
        if rp: st.image(str(rp), use_container_width=True)


def _route_from_scenario(scenario: dict | None, scenario_name: str) -> str:
    if scenario:
        for k in ("route_preset", "route", "preset"):
            v = scenario.get(k)
            if isinstance(v, str) and v.strip():
                v2 = v.strip().lower().replace("_", "-")
                if "dri-eaf" in v2 or v2 in {"drieaf", "dri"}: return "DRI-EAF"
                if "eaf-scrap" in v2 or v2 in {"eafscrap"}:     return "EAF-Scrap"
                if "bf-bof" in v2 or "bf" in v2 or "bof" in v2: return "BF-BOF"
                if "external" in v2:                             return "External"
    name = (scenario_name or "").lower()
    if "dri" in name and "eaf" in name: return "DRI-EAF"
    if "eaf" in name and "scrap" in name: return "EAF-Scrap"
    if "bf" in name or "bof" in name:    return "BF-BOF"
    if "external" in name:               return "External"
    return "auto"


def _detect_stage_keys() -> Dict[str, str]:
    found = {}
    for k, mat in STAGE_MATS.items():
        m = str(mat).lower()
        if ("pig iron" in m) or ("hot metal" in m) or ("hot-metal" in m) or ("hotmetal" in m):
            found.setdefault("pig_iron", k)
        if (("liquid" in m and "steel" in m) or ("ingot" in m)) and ("finished" not in m):
            found.setdefault("liquid_steel", k)
        if "finished" in m:
            found.setdefault("finished", k)
    # Fallbacks
    if "pig_iron" not in found:
        found["pig_iron"] = sorted(STAGE_MATS.keys())[0]
    if "liquid_steel" not in found:
        lk = next((k for k, v in STAGE_MATS.items()
                   if isinstance(v, str) and (("liquid" in v.lower() and "steel" in v.lower()) or ("ingot" in v.lower()))), None)
        found["liquid_steel"] = lk or found["pig_iron"]
    if "finished" not in found:
        fk = next((k for k, v in STAGE_MATS.items() if isinstance(v, str) and ("finished" in v.lower())), None)
        found["finished"] = fk or list(STAGE_MATS.keys())[-1]
    return found


@st.cache_data(show_spinner=False)
def _load_for_picks(data_dir: str, route_preset: str, stage_key: str, scenario: dict):
    base = os.path.join(data_dir, "")
    energy_int     = load_data_from_yaml(os.path.join(base, 'energy_int.yml'))
    energy_shares  = load_data_from_yaml(os.path.join(base, 'energy_matrix.yml'))
    energy_content = load_data_from_yaml(os.path.join(base, 'energy_content.yml'))
    e_efs          = load_data_from_yaml(os.path.join(base, 'emission_factors.yml'))
    params         = load_parameters      (os.path.join(base, 'parameters.yml'))

    recipes = load_recipes_from_yaml(
        os.path.join(base, 'recipes.yml'),
        params, energy_int, energy_shares, energy_content
    )

    apply_fuel_substitutions(scenario.get('fuel_substitutions', {}), energy_shares, energy_int, energy_content, e_efs)
    apply_dict_overrides(energy_int,     scenario.get('energy_int', {}))
    apply_dict_overrides(energy_shares,  scenario.get('energy_matrix', {}))
    apply_dict_overrides(energy_content, scenario.get('energy_content', {}))
    apply_dict_overrides(e_efs,          scenario.get('emission_factors', {}))

    from types import SimpleNamespace
    def _recursive_ns_update(ns, patch):
        for k, v in (patch or {}).items():
            if isinstance(v, dict):
                cur = getattr(ns, k, None)
                if not isinstance(cur, SimpleNamespace):
                    cur = SimpleNamespace(); setattr(ns, k, cur)
                _recursive_ns_update(cur, v)
            else:
                setattr(ns, k, v)

    _param_patch = scenario.get('param_overrides', None)
    if _param_patch is None:
        _param_patch = scenario.get('parameters', {})
    _recursive_ns_update(params, _param_patch)

    adjust_blast_furnace_intensity(energy_int, energy_shares, params)
    adjust_process_gas_intensity('Coke Production', 'process_gas_coke', energy_int, energy_shares, params)

    recipes = load_recipes_from_yaml(
        os.path.join(base, 'recipes.yml'),
        params, energy_int, energy_shares, energy_content
    )
    recipes = apply_recipe_overrides(recipes, scenario.get('recipe_overrides', {}), params, energy_int, energy_shares, energy_content)

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


def gather_ambiguous_chain_materials(
    recipes: List[Process],
    demand_mat: str,
    pre_mask: Dict[str, int] | None = None,
    pre_select: Dict[str, int] | None = None,
) -> List[Tuple[str, List[str]]]:
    pre_mask   = pre_mask or {}
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
                    if im not in seen_mats: q.append(im)
            continue
        out.append((mat, [r.name for r in enabled]))
        for r in enabled:
            for im in r.inputs:
                if im not in seen_mats: q.append(im)
    return out

# -----------------------------
# Stepper state helpers
# -----------------------------
if "step" not in st.session_state: st.session_state.step = 1
if "scenario" not in st.session_state: st.session_state.scenario = {}
if "scenario_choice" not in st.session_state: st.session_state.scenario_choice = None
if "scenario_name" not in st.session_state: st.session_state.scenario_name = "(no scenario)"
if "route" not in st.session_state: st.session_state.route = "auto"
if "stage_key" not in st.session_state:
    # default to Finished if available
    skeys = list(STAGE_MATS.keys())
    st.session_state.stage_key = skeys[-1] if skeys else "Finished"
if "stage_label" not in st.session_state: st.session_state.stage_label = "Finished"
if "demand_qty" not in st.session_state: st.session_state.demand_qty = 1000.0
if "use_yield" not in st.session_state: st.session_state.use_yield = False
if "yield_frac" not in st.session_state: st.session_state.yield_frac = 1.0
if "country_code" not in st.session_state: st.session_state.country_code = ""
if "picks_by_material" not in st.session_state: st.session_state.picks_by_material = {}
if "do_log" not in st.session_state: st.session_state.do_log = True
if "log_dir" not in st.session_state: st.session_state.log_dir = "run_logs"

# gas EF ephemeral preview (stored in scenario['emission_factors']['Gas'] when set)
if "ef_gas_preview" not in st.session_state: st.session_state.ef_gas_preview = None


def step_bar():
    labels = ["1. Scenario", "2. Stage & Demand", "3. Fuels & Overrides", "4. Route Picks", "5. Review & Run"]
    cur = st.session_state.step
    html = ["<div class='stepbar'>"]
    for i, lab in enumerate(labels, start=1):
        cls = "step-chip"
        if i < cur: cls += " done"
        elif i == cur: cls += " active"
        html.append(f"<span class='{cls}'>{lab}</span>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def next_step():
    st.session_state.step = min(5, st.session_state.step + 1)

def prev_step():
    st.session_state.step = max(1, st.session_state.step - 1)

# -----------------------------
# Header & Step bar
# -----------------------------
render_header_logos()
step_bar()

left, right = st.columns([1.5, 2.5], gap="large")

# -----------------------------
# LEFT: STEP CONTENT
# -----------------------------
with left:
    step = st.session_state.step

    # ------------------ STEP 1: Scenario ------------------
    if step == 1:
        st.header("1. Scenario")

        def list_scenarios(data_dir: str) -> List[str]:
            sc_dir = pathlib.Path(data_dir) / "scenarios"
            if not sc_dir.exists(): return []
            return sorted([p.name for p in sc_dir.glob("*.yml")])

        available = list_scenarios(DATA_ROOT)
        if not available:
            st.warning("No scenario .yml files found in data/scenarios (using empty scenario).")
        scenario_choice = st.selectbox(
            "Scenario file (data/scenarios)",
            options=available or ["(none)"]
        )
        if available:
            scenario_path = pathlib.Path(DATA_ROOT) / "scenarios" / scenario_choice
            scenario = load_data_from_yaml(str(scenario_path), default_value=None, unwrap_single_key=False)
            scenario_name = scenario_choice
        else:
            scenario = {}
            scenario_name = "(no scenario)"

        st.session_state.scenario = scenario or {}
        st.session_state.scenario_choice = scenario_choice if available else None
        st.session_state.scenario_name = scenario_name

        # Route lock info
        route = _route_from_scenario(st.session_state.scenario, st.session_state.scenario_name)
        st.session_state.route = route
        st.info(f"Route preset: **{route}** (locked by scenario)")
        if route == "auto":
            st.warning("Scenario does not specify a route; using AUTO (no preset mask).")

        b = st.columns(2)
        if b[0].button("Next ▶", type="primary"): next_step()

    # --------------- STEP 2: Stage & Demand ---------------
    elif step == 2:
        st.header("2. Stage & Demand")
        stage_keys = _detect_stage_keys()
        stage_menu = {
            "Pig iron (BF hot metal)": stage_keys["pig_iron"],
            "Liquid steel (pre-CC ingot)": stage_keys["liquid_steel"],
            "Finished": stage_keys["finished"],
        }
        # Pre-select prior value if still valid
        reverse_menu = {v:k for k,v in stage_menu.items()}
        default_idx = list(stage_menu.keys()).index(
            reverse_menu.get(st.session_state.stage_key, "Finished")
        )
        stage_label = st.radio(
            "Stop at stage",
            options=list(stage_menu.keys()),
            index=default_idx,
            help="Pig iron: just after BF; Liquid steel: after BOF/EAF and before Continuous Casting; Finished: final products.",
        )
        st.session_state.stage_label = stage_label
        st.session_state.stage_key = stage_menu[stage_label]

        st.session_state.demand_qty = st.number_input(
            "Demand quantity at selected stage",
            value=float(st.session_state.demand_qty), min_value=0.0, step=100.0
        )

        with st.expander("Material yield (reporting)", expanded=False):
            st.session_state.use_yield = st.checkbox(
                "Apply yield to emission factor",
                value=bool(st.session_state.use_yield),
                help="Final EF = (total emission / demand) ÷ yield"
            )
            st.session_state.yield_frac = st.number_input(
                "Yield (fraction 0-1)",
                value=float(st.session_state.yield_frac), min_value=0.0, max_value=1.0, step=0.01
            )

        b = st.columns(2)
        if b[0].button("◀ Back"): prev_step()
        if b[1].button("Next ▶", type="primary"): next_step()

    # ----------- STEP 3: Fuels & Overrides -----------------
    elif step == 3:
        st.header("3. Fuels & Overrides")
        # Grid EF country
        elec_map = load_electricity_intensity(os.path.join(DATA_ROOT, "electricity_intensity.yml")) or {}
        country_opts = sorted(elec_map.keys()) if elec_map else []
        st.session_state.country_code = st.selectbox(
            "Grid electricity country (for Electricity EF)",
            options=[""] + country_opts,
            index=([""] + country_opts).index(st.session_state.country_code) if st.session_state.country_code in ([""] + country_opts) else 0,
        )

        st.markdown("**Gas blend (sets EF for 'Gas')**")
        c1, c2, c3, c4 = st.columns(4)
        sh_ng  = c1.slider("Natural gas share", 0.0, 1.0, 1.0, .01, key="sh_ng")
        sh_bio = c2.slider("Biomethane share",  0.0, 1.0, 0.0, .01, key="sh_bio")
        sh_h2g = c3.slider("Green H₂ share",    0.0, 1.0, 0.0, .01, key="sh_h2g")
        sh_h2b = c4.slider("Blue H₂ share",     0.0, 1.0, 0.0, .01, key="sh_h2b")
        S = max(1e-9, sh_ng + sh_bio + sh_h2g + sh_h2b)
        w_ng, w_bio, w_h2g, w_h2b = (sh_ng/S, sh_bio/S, sh_h2g/S, sh_h2b/S)

        # Default values for EF blending
        try:
            base_efs = load_data_from_yaml(os.path.join(DATA_ROOT, 'emission_factors.yml')) or {}
        except Exception:
            base_efs = {}

        def find_ef(d: dict, keys: list[str]) -> float | None:
            for k in keys:
                v = d.get(k)
                if v is None: continue
                try: return float(v)
                except Exception: pass
            return None

        ef_ng  = find_ef(base_efs, ["Natural gas", "Natural Gas", "Gas"]) \
                 if base_efs else None
        ef_bio = find_ef(base_efs, ["Biomethane", "Bio-methane", "Upgraded biogas"]) \
                 if base_efs else None
        ef_h2g = find_ef(base_efs, ["Hydrogen (Electrolysis)", "Green hydrogen", "H2 - Electrolysis"]) \
                 if base_efs else None
        ef_h2b = find_ef(base_efs, ["Hydrogen (Methane reforming + CCS)", "Hydrogen (Methane reforming)", "Blue hydrogen", "H2 - SMR+CCS", "H2 - ATR+CCS"]) \
                 if base_efs else None

        all_missing = all(x is None for x in [ef_ng, ef_bio, ef_h2g, ef_h2b])
        if all_missing:
            ef_gas = float(base_efs.get("Gas", 0.0))
        else:
            ef_gas = (
                (w_ng  * (ef_ng  or 0.0)) +
                (w_bio * (ef_bio or 0.0)) +
                (w_h2g * (ef_h2g or 0.0)) +
                (w_h2b * (ef_h2b or 0.0))
            )
        st.metric("Effective EF for Gas", f"{ef_gas:.2f} gCO₂/MJ")
        st.session_state.ef_gas_preview = ef_gas
        # Push into scenario core
        st.session_state.scenario.setdefault('emission_factors', {})
        st.session_state.scenario['emission_factors']['Gas'] = float(ef_gas)
        st.session_state.scenario['gas_blend'] = {
            "shares": {"NG": w_ng, "Biomethane": w_bio, "H2_green": w_h2g, "H2_blue": w_h2b},
            "effective_gas_ef_gco2_per_MJ": ef_gas,
        }

        # Route-locked process energy intensity override
        route = st.session_state.route
        allowed_proc_by_route = {
            "BF-BOF":   ("Blast Furnace",        "BF base energy intensity (MJ/kg hot metal)"),
            "DRI-EAF":  ("Direct Reduction Iron","DRI base energy intensity (MJ/kg DRI)"),
            "EAF-Scrap":("Electric Arc Furnace","EAF base energy intensity (MJ/kg liquid steel)"),
        }
        allowed = allowed_proc_by_route.get(route)

        if allowed:
            proc_name, friendly_label = allowed
            existing_ei = (st.session_state.scenario.get('energy_int') or {})
            st.session_state.scenario['energy_int'] = {proc_name: existing_ei.get(proc_name)} if proc_name in existing_ei else {}

            with st.expander("Process energy intensity override", expanded=False):
                enable_override = st.checkbox(f"{proc_name}", value=(proc_name in (st.session_state.scenario.get('energy_int') or {})))
                if enable_override:
                    default_val = None
                    try:
                        default_val = float((st.session_state.scenario.get('energy_int') or {}).get(proc_name))
                    except Exception:
                        default_val = None
                    if default_val is None:
                        try:
                            base_energy_int = load_data_from_yaml(os.path.join(DATA_ROOT, 'energy_int.yml')) or {}
                            default_val = float(base_energy_int.get(proc_name, 0.0))
                        except Exception:
                            default_val = 0.0
                    val = st.number_input(friendly_label, value=float(default_val), min_value=0.0, step=0.1)
                    st.session_state.scenario.setdefault('energy_int', {})
                    st.session_state.scenario['energy_int'][proc_name] = float(val)
                else:
                    st.session_state.scenario['energy_int'] = {}
        else:
            st.session_state.scenario['energy_int'] = {}

        # BF fuel split (only BF-BOF)
        if route == "BF-BOF":
            with st.expander("Blast Furnace — fuel split", expanded=False):
                def _get_default_shares():
                    bf_mx = (st.session_state.scenario.get('energy_matrix') or {}).get('Blast Furnace', {})
                    coke0 = float(bf_mx.get('Coke', None)) if bf_mx else None
                    coal0 = float(bf_mx.get('Coal', None)) if bf_mx else None
                    char0 = float(bf_mx.get('Charcoal', None)) if bf_mx else None
                    if coke0 is not None and (coal0 is not None or char0 is not None):
                        return coke0, (coal0 or 0.0), (char0 or 0.0)
                    try:
                        base_sh = load_data_from_yaml(os.path.join(DATA_ROOT, 'energy_shares.yml')) or {}
                        bf = base_sh.get('Blast Furnace', {})
                        return float(bf.get('Coke', 0.6)), float(bf.get('Coal', 0.4)), float(bf.get('Charcoal', 0.0))
                    except Exception:
                        return 0.6, 0.4, 0.0

                coke0, coal0, char0 = _get_default_shares()
                pci0 = max(0.0, min(1.0, coal0 + char0))
                coke0 = max(0.0, min(1.0, coke0))

                coke_share = st.slider("Coke share of BF thermal input (non-electric)", 0.0, 1.0, float(coke0), 0.01)
                pci_charcoal_frac = st.slider(
                    "Charcoal fraction inside PCI", 0.0, 1.0,
                    float(0.0 if pci0 <= 1e-9 else char0 / max(pci0, 1e-9)), 0.01,
                    help="0 → PCI is 100% Coal; 1 → PCI is 100% Charcoal. Coke is unchanged."
                )
                pci_share = 1.0 - coke_share
                coal_share = pci_share * (1.0 - pci_charcoal_frac)
                char_share = pci_share * pci_charcoal_frac

                st.session_state.scenario.setdefault('energy_matrix', {})
                Ssum = max(1e-9, coke_share + coal_share + char_share)
                st.session_state.scenario['energy_matrix']['Blast Furnace'] = {
                    'Coke':     float(coke_share / Ssum),
                    'Coal':     float(coal_share / Ssum),
                    'Charcoal': float(char_share / Ssum),
                }

        with st.expander("Logging", expanded=False):
            st.session_state.do_log = st.checkbox("Write JSON log (config + CO₂)", value=st.session_state.do_log)
            st.session_state.log_dir = st.text_input("Log folder", value=st.session_state.log_dir)

        b = st.columns(2)
        if b[0].button("◀ Back"): prev_step()
        if b[1].button("Next ▶", type="primary"): next_step()

    # ---------------- STEP 4: Route Picks ------------------
    elif step == 4:
        st.header("4. Route Picks")
        route       = st.session_state.route
        stage_key   = st.session_state.stage_key
        scenario    = st.session_state.scenario

        recipes_for_ui, pre_mask, demand_mat = _load_for_picks(DATA_ROOT, route, stage_key, scenario)
        pre_mask.pop("Coke", None)  # allow both Coke Production and Coke from market

        # Post-CC controls
        enable_post_cc = stage_key in {"Raw", "Finished"}
        cc_choice_val  = st.session_state.get("cc_choice_radio", "Hot Rolling")
        cr_toggle_val  = st.session_state.get("cr_toggle", False)

        forced_pre_select = {}
        if enable_post_cc and cc_choice_val:
            forced_pre_select["Raw Products (types)"] = cc_choice_val
        if stage_key == "Finished":
            forced_pre_select["Intermediate Process 3"] = (
                "Bypass CR→IP3" if (cc_choice_val == "Hot Rolling" and cr_toggle_val) else "Bypass Raw→IP3"
            )

        UPSTREAM_CORE = {"Blast Furnace", "Basic Oxygen Furnace", "Direct Reduction Iron", "Electric Arc Furnace", "Scrap Purchase"}
        route_disable = {
            "EAF-Scrap": {"Blast Furnace", "Basic Oxygen Furnace", "Direct Reduction Iron"},
            "DRI-EAF":   {"Blast Furnace", "Basic Oxygen Furnace"},
            "BF-BOF":    {"Direct Reduction Iron", "Electric Arc Furnace"},
            "External":  set(),
            "auto":      set(),
        }.get(route, set())
        pre_select_soft = {p: 0 for p in route_disable if p in UPSTREAM_CORE}

        st.session_state.picks_by_material.update(forced_pre_select)
        pre_select = {**pre_select_soft, **forced_pre_select}
        ambiguous = gather_ambiguous_chain_materials(recipes_for_ui, demand_mat, pre_mask=pre_mask, pre_select=pre_select)
        if forced_pre_select:
            ambiguous = [(m, opts) for (m, opts) in ambiguous if m not in forced_pre_select]

        # Renames and grouping (same as your app)
        PROC_RENAMES = {
            "Continuous Casting (R)": "Regular (R)",
            "Continuous Casting (L)": "Low-alloy (L)",
            "Continuous Casting (H)": "High-alloy (H)",
            "Nitrogen Production": "Onsite production",
            "Nitrogen from market": "Market purchase",
            "Oxygen Production": "Onsite production",
            "Oxygen from market": "Market purchase",
            "Coke Production": "Onsite production",
            "Coke from market": "Market purchase",
            "Dolomite from market": "Market purchase",
            "Burnt Lime from market": "Market purchase",
        }
        LABEL_RENAMES = {"Cast Steel (IP1)": "Alloying (choose class)", "Manufactured Feed (IP4)": "Shaping (IP4)"}
        STAGE_DISPLAY = {"IP1": "Alloying", "Raw": "Post-CC", "IP4": "Shaping", "Finished": "Finished"}

        UPSTREAM_MATS = {"Nitrogen", "Oxygen", "Coal", "Coke", "Dolomite", "Burnt Lime"}

        def _fmt_proc(name: str) -> str: return PROC_RENAMES.get(name, name)

        def _stage_label_for(mat_name: str) -> str:
            if mat_name in UPSTREAM_MATS: return "Upstream"
            if "Finished" in mat_name: return "Finished"
            if "Manufactured Feed (IP4)" in mat_name: return "IP4"
            if "Intermediate Process 3" in mat_name: return "IP3"
            if "Cold Raw Steel (IP2)" in mat_name: return "IP2"
            if "Raw Products" in mat_name: return "Raw"
            m = re.search(r"\(IP(\d)\)", mat_name)
            if m: return f"IP{m.group(1)}"
            if "Liquid" in mat_name: return "Liquid"
            return "Other"

        groups = defaultdict(list)
        for mat, options in ambiguous:
            groups[_stage_label_for(mat)].append((mat, options))

        def _use_selectbox(options: list[str]) -> bool:
            return (len(options) > 5) or (max(len(o) for o in options) > 28)

        def _render_group(stage: str, container, show_title: bool = True):
            items = groups.get(stage, [])
            if not items: return
            with container:
                st.markdown(f"**{STAGE_DISPLAY.get(stage, stage)}**")
                inner_cols = st.columns(2) if len(items) > 1 else [st.container()]
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

        # Layout for picks
        primary_order = ["IP1", "Raw", "IP4", "Finished"]
        cols = st.columns(len(primary_order))
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
                    st.checkbox("Apply Cold Rolling", value=st.session_state.get("cr_toggle", False), key="cr_toggle")
                else:
                    st.session_state["cr_toggle"] = False

        _render_group("IP4", cols[2])
        _render_group("Finished", cols[3])

        if groups.get("Upstream"):
            _render_group("Upstream", st.container(), show_title=False)

        b = st.columns(3)
        if b[0].button("Reset picks"):
            st.session_state.picks_by_material = {}
            for k in list(st.session_state.keys()):
                if k.startswith("radio_") or k.startswith("pick_") or k in ("cc_choice_radio", "cr_toggle"):
                    del st.session_state[k]
            st.experimental_rerun()
        if b[1].button("◀ Back"): prev_step()
        if b[2].button("Next ▶", type="primary"): next_step()

    # ---------------- STEP 5: Review & Run -----------------
    else:
        st.header("5. Review & Run")
        route       = st.session_state.route
        stage_key   = st.session_state.stage_key
        stage_label = st.session_state.stage_label
        demand_qty  = float(st.session_state.demand_qty)
        country_code= st.session_state.country_code or None
        scenario    = st.session_state.scenario

        # Summary
        st.markdown("**Summary**")
        sA, sB = st.columns(2)
        with sA:
            st.write(f"Route preset: **{route}**")
            st.write(f"Stage: **{stage_label}** → `{STAGE_MATS.get(stage_key,'?')}`")
            st.write(f"Demand: **{demand_qty:,.0f}** units")
            st.write(f"Grid country: **{country_code or '—'}**")
            st.write(f"Gas EF (blend): **{(st.session_state.ef_gas_preview or 0):.2f} gCO₂/MJ**")
        with sB:
            st.write("**Selected picks (subset)**")
            if st.session_state.picks_by_material:
                dfp = pd.DataFrame(sorted(st.session_state.picks_by_material.items()), columns=["Material","Producer"]) \
                        .set_index("Material")
                st.dataframe(dfp, use_container_width=True)
            else:
                st.caption("No ambiguous picks required.")

        # Action row
        b = st.columns(3)
        back = b[0].button("◀ Back")
        run_now = b[2].button("Run model", type="primary", use_container_width=True)
        if back: prev_step()

        st.markdown("<hr class='hr'>", unsafe_allow_html=True)

        if run_now:
            with st.spinner("Running model (core)…"):
                route_cfg = RouteConfig(
                    route_preset=route,
                    stage_key=stage_key,
                    demand_qty=float(demand_qty),
                    picks_by_material=dict(st.session_state.picks_by_material),
                    pre_select_soft={},  # already handled via graph mask above; optional to recompute
                )
                scn = ScenarioInputs(
                    country_code=country_code,
                    scenario=scenario,
                    route=route_cfg,
                )
                out = run_scenario(DATA_ROOT, scn)

            production_routes = out.production_routes
            prod_lvl         = out.prod_levels
            energy_balance   = out.energy_balance
            emissions        = out.emissions
            total            = out.total_co2e_kg

            st.subheader("Emission Factor")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total emissions", f"{(total or 0):,.0f} kg CO₂e")
            ef_gross = (total / demand_qty) if (total is not None and demand_qty and demand_qty > 0) else None
            with c2:
                st.metric(f"Gross EF (kg CO₂ per unit {stage_label})", f"{ef_gross:.3f}" if ef_gross is not None else "—")
            ef_final = (ef_gross / max(1e-9, st.session_state.yield_frac)) if (ef_gross is not None and st.session_state.use_yield) else None
            with c3:
                st.metric("Final EF (inc. yield)", f"{ef_final:.3f}" if ef_final is not None else "—", help="Computed as (total/demand) ÷ yield")

            st.success("Model run complete (core).")
            st.write(f"**Stage material**: {STAGE_MATS.get(stage_key,'?')}")
            if total is not None:
                st.metric("Total CO₂e", f"{total:,.2f} kg")
            else:
                st.info("No emissions available for this run.")

            st.subheader("Energy balance (MJ)")
            st.dataframe(energy_balance, use_container_width=True)

            if emissions is not None:
                st.subheader("Emissions (kg CO₂e)")
                st.dataframe(emissions, use_container_width=True)

            # Sankeys
            recipes_for_ui, _, _ = _load_for_picks(DATA_ROOT, route, stage_key, scenario)
            recipes_dict_live = {r.name: r for r in recipes_for_ui}

            fig_mass = make_mass_sankey(
                prod_lvl=prod_lvl,
                recipes_dict=recipes_dict_live,
                min_flow=0.5,
                title=f"Mass Flow Sankey — {demand_qty:.0f} units {STAGE_MATS.get(stage_key,'?')} ({st.session_state.scenario_name})",
            )
            st.plotly_chart(fig_mass, use_container_width=True)

            fig_energy = make_energy_sankey(
                energy_balance_df=energy_balance,
                min_MJ=25.0,
                title="Energy Flow Sankey — Process Carriers",
            )
            st.plotly_chart(fig_energy, use_container_width=True)

            if emissions is not None:
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

            # Downloads
            df_runs = pd.DataFrame(sorted(prod_lvl.items()), columns=["Process", "Runs"]).set_index("Process")
            d1, d2, d3 = st.columns(3)
            d1.download_button(
                "Production runs (CSV)", data=df_runs.to_csv().encode("utf-8"), file_name="production_runs.csv", mime="text/csv",
            )
            d2.download_button(
                "Energy balance (CSV)", data=energy_balance.to_csv().encode("utf-8"), file_name="energy_balance.csv", mime="text/csv",
            )
            if emissions is not None:
                d3.download_button(
                    "Emissions (CSV)", data=emissions.to_csv().encode("utf-8"), file_name="emissions.csv", mime="text/csv",
                )

            # JSON log
            if st.session_state.do_log:
                try:
                    payload = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "data_root": DATA_ROOT,
                        "route": {
                            "route_preset": route,
                            "stage_key": stage_key,
                            "stage_material": STAGE_MATS.get(stage_key),
                            "demand_qty": float(demand_qty),
                            "picks_by_material": dict(st.session_state.get("picks_by_material", {})),
                            "pre_select_soft": {},
                            "yield_applied": bool(st.session_state.use_yield),
                            "yield_fraction": float(st.session_state.yield_frac) if st.session_state.use_yield else None,
                            "ef_gross_kg_per_unit": float(ef_gross) if ef_gross is not None else None,
                            "ef_final_kg_per_unit": float(ef_final) if ef_final is not None else None,
                        },
                        "country_code": country_code,
                        "scenario_file": st.session_state.scenario_choice,
                        "total_co2e_kg": float(total) if total is not None else None,
                        "stage_label": stage_label,
                    }
                    log_path = write_run_log(st.session_state.log_dir or "run_logs", payload)
                    st.caption(f"Log written: `{log_path}`")
                except Exception as e:
                    st.warning(f"Could not write JSON log: {e}")

# -----------------------------
# RIGHT: LIVE PREVIEW PANEL
# -----------------------------
with right:
    st.subheader("Live Preview")
    st.caption("Key selections and quick metrics update as you proceed.")

    # Quick chips
    p1, p2, p3 = st.columns(3)
    with p1: st.metric("Route", st.session_state.route)
    with p2: st.metric("Stage", st.session_state.stage_label)
    with p3: st.metric("Demand", f"{st.session_state.demand_qty:,.0f}")

    st.metric("Gas EF (preview)", f"{(st.session_state.ef_gas_preview or 0):.2f} gCO₂/MJ")

    # Show a snapshot of picks if any
    if st.session_state.picks_by_material:
        dfp = pd.DataFrame(sorted(st.session_state.picks_by_material.items()), columns=["Material","Producer"]).set_index("Material")
        st.dataframe(dfp, use_container_width=True, height=280)
    else:
        st.caption("No ambiguous picks yet.")

    st.markdown("<hr class='hr'>", unsafe_allow_html=True)
    st.info("Use the buttons below each step to navigate. Results appear in Step 5.")
