# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:40:32 2025

@author: rafae
"""

# streamlit_route_builder_v2.py
# -----------------------------------------------------------------------------
# Streamlit UI dedicated to ROUTE BUILDING + RUN + LOGGING.
# All heavy lifting (balances, energy, emissions) is delegated to the
# steel_core_api_v2.run_scenario API which itself calls your core engine.
# -----------------------------------------------------------------------------
from __future__ import annotations
import os, pathlib, json, hashlib
from typing import Dict, List
from datetime import datetime

import streamlit as st
import pandas as pd
import yaml

from steel_core_api_v2 import (
    RouteConfig, ScenarioInputs, run_scenario,
    build_picks_index, write_run_log,
)
from steel_model_core import (
    STAGE_MATS,
    load_data_from_yaml,
    load_electricity_intensity,
    load_parameters,
)

# ==============================
# Page setup
# ==============================
st.set_page_config(page_title="Steel Route Builder v2", layout="wide")

st.markdown(
    """
    <style>
    :root { --c1: #2563eb; }
    div[data-baseweb="radio"] [aria-checked="true"] > div:first-child { background-color: var(--c1) !important; border-color: var(--c1) !important; }
    div[data-baseweb="radio"] [aria-checked="false"] > div:first-child { border-color: var(--c1) !important; }
    .hr { height:1px; background:linear-gradient(to right,#e5e7eb,#cbd5e1,#e5e7eb); margin:.25rem 0 .75rem 0; border:0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Steel Route Builder v2")
st.caption("Build route → run core → log config + CO₂. This app does not re-compute the model.")

# ==============================
# Sidebar controls
# ==============================
with st.sidebar:
    st.header("Run settings")
    data_root = st.text_input("Data folder", value="data")

    # Scenario file (optional overrides)
    sc_files: List[str] = []
    sc_dir = pathlib.Path(data_root) / "scenarios"
    if sc_dir.exists():
        sc_files = sorted([p.name for p in sc_dir.glob("*.yml")])
    scenario_choice = st.selectbox("Scenario (optional)", options=["(none)"] + sc_files, index=0)

    if scenario_choice != "(none)":
        sc_path = sc_dir / scenario_choice
        scenario = load_data_from_yaml(str(sc_path), default_value=None, unwrap_single_key=False)
        st.caption(f"Scenario loaded: {scenario_choice}")
    else:
        scenario = {}

    # Route preset
    route_preset = st.radio("Route preset", ["auto","BF-BOF","DRI-EAF","EAF-Scrap","External"], horizontal=True)

    # Stage + demand
    stage_key = st.selectbox("Stop at stage", options=list(STAGE_MATS.keys()), index=0)
    demand_qty = st.number_input("Demand quantity at stage", min_value=0.0, value=1000.0, step=100.0)

    # Grid EF country
    elec_map = load_electricity_intensity(os.path.join(data_root, "electricity_intensity.yml")) or {}
    country_opts = sorted(elec_map.keys()) if elec_map else []
    country_code = st.selectbox("Grid electricity country (for Electricity EF)", options=[""] + country_opts, index=0)

    # Logging
    st.header("Logging")
    do_log = st.checkbox("Write JSON log (config + CO₂)", value=True)
    log_dir = st.text_input("Log folder", value="run_logs_v2")

st.markdown("<hr class='hr'>", unsafe_allow_html=True)

# ==============================
# Route picks UI
# ==============================
@st.cache_data(show_spinner=False)
def _load_for_picks(data_dir: str, route_preset: str, stage_key: str, scenario: dict):
    from steel_model_core import (
        load_data_from_yaml, load_parameters, load_recipes_from_yaml, apply_fuel_substitutions,
        apply_dict_overrides, apply_recipe_overrides, adjust_blast_furnace_intensity,
        adjust_process_gas_intensity, build_route_mask, enforce_eaf_feed,
    )
    base = os.path.join(data_dir, "")
    energy_int     = load_data_from_yaml(os.path.join(base, 'energy_int.yml'))
    energy_shares  = load_data_from_yaml(os.path.join(base, 'energy_matrix.yml'))
    energy_content = load_data_from_yaml(os.path.join(base, 'energy_content.yml'))
    e_efs          = load_data_from_yaml(os.path.join(base, 'emission_factors.yml'))
    params         = load_parameters      (os.path.join(base, 'parameters.yml'))

    recipes = load_recipes_from_yaml(os.path.join(base, 'recipes.yml'), params, energy_int, energy_shares, energy_content)

    # scenario overrides that affect picks
    apply_fuel_substitutions(scenario.get('fuel_substitutions', {}), energy_shares, energy_int, energy_content, e_efs)
    apply_dict_overrides(energy_int,     scenario.get('energy_int', {}))
    apply_dict_overrides(energy_shares,  scenario.get('energy_matrix', {}))
    apply_dict_overrides(energy_content, scenario.get('energy_content', {}))

    adjust_blast_furnace_intensity(energy_int, energy_shares, params)
    adjust_process_gas_intensity('Coke Production', 'process_gas_coke', energy_int, energy_shares, params)

    # Route mask + EAF feed
    pre_mask = build_route_mask(route_preset, recipes)

    import copy
    recipes_for_ui = copy.deepcopy(recipes)
    eaf_mode = {"EAF-Scrap":"scrap","DRI-EAF":"dri","BF-BOF":None,"External":None,"auto":None}.get(route_preset)
    enforce_eaf_feed(recipes_for_ui, eaf_mode)

    demand_mat = STAGE_MATS[stage_key]
    return recipes_for_ui, pre_mask, demand_mat

recipes_for_ui, pre_mask, demand_mat = _load_for_picks(data_root, route_preset, stage_key, scenario)

st.subheader("Pick one producer where multiple options exist")
from steel_core_api_v2 import build_picks_index as _build
ambiguous = _build(recipes_for_ui, demand_mat, pre_mask=pre_mask, pre_select=None)

if not ambiguous:
    st.info("No ambiguous producers along the chain with this setup — unique path.")

# Persist picks across reruns
if "picks_by_material" not in st.session_state:
    st.session_state.picks_by_material = {}

colL, colR = st.columns(2)
with colL:
    for mat, options in ambiguous:
        default = st.session_state.picks_by_material.get(mat, options[0])
        idx = options.index(default) if default in options else 0
        choice = st.radio(mat, options=options, index=idx, key=f"rb_{mat}")
        st.session_state.picks_by_material[mat] = choice

with colR:
    if st.button("Reset picks"):
        st.session_state.picks_by_material = {}
        st.experimental_rerun()

st.markdown("<hr class='hr'>", unsafe_allow_html=True)

# ==============================
# Run core + Log
# ==============================
run_clicked = st.button("Run core model", type="primary")
if run_clicked:
    route = RouteConfig(
        route_preset=route_preset,
        stage_key=stage_key,
        demand_qty=float(demand_qty),
        picks_by_material=dict(st.session_state.picks_by_material),
        pre_select_soft=None,
    )
    scn = ScenarioInputs(
        country_code=(country_code or None),
        scenario=scenario,
        route=route,
    )

    with st.spinner("Running scenario…"):
        out = run_scenario(data_root, scn)

    st.success("Run complete")

    # Show key outputs
    if out.total_co2e_kg is not None:
        st.metric("Total CO₂e", f"{out.total_co2e_kg:,.2f} kg")
    st.subheader("Energy balance (MJ)")
    st.dataframe(out.energy_balance)
    if out.emissions is not None:
        st.subheader("Emissions (kg CO₂e)")
        st.dataframe(out.emissions)

    # --------------------
    # JSON run log
    # --------------------
    if do_log:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data_root": data_root,
            "route": {
                "route_preset": route.route_preset,
                "stage_key": route.stage_key,
                "demand_qty": route.demand_qty,
                "picks_by_material": route.picks_by_material,
            },
            "country_code": country_code or None,
            "scenario_file": (scenario_choice if scenario_choice != "(none)" else None),
            "total_co2e_kg": out.total_co2e_kg,
        }
        fpath = write_run_log(log_dir, payload)
        st.caption(f"Log written: {fpath}")

    # Optional downloads
    c1, c2, c3 = st.columns(3)
    c1.download_button("Production routes (JSON)", data=json.dumps(out.production_routes, indent=2).encode("utf-8"), file_name="production_routes.json")
    c2.download_button("Energy balance (CSV)", data=out.energy_balance.to_csv().encode("utf-8"), file_name="energy_balance.csv")
    if out.emissions is not None:
        c3.download_button("Emissions (CSV)", data=out.emissions.to_csv().encode("utf-8"), file_name="emissions.csv")

st.markdown("<hr class='hr'>", unsafe_allow_html=True)

st.caption("Steel Route Builder v2 — keeps the app thin and logs just the inputs + CO₂.")
