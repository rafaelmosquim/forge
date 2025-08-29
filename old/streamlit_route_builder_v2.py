# -*- coding: utf-8 -*-
"""
Streamlit Route Builder v2 (scenario-driven)

- No route preset radio
- No per-material producer pick UI
- Scenario (YAML content or filename) locks the route
- Optional picks are read from scenario (picks_by_material or picks)
- Core does all calculations; app only builds inputs + logs

Run:
    streamlit run streamlit_route_builder_v2.py
"""

from __future__ import annotations
import os
import pathlib
import json
from typing import List, Dict, Any
from datetime import datetime

import streamlit as st
import pandas as pd  # used for display; keep import
import yaml

from steel_core_api_v2 import (
    RouteConfig,
    ScenarioInputs,
    run_scenario,
    write_run_log,
)
from steel_model_core import (
    STAGE_MATS,
    load_data_from_yaml,
    load_electricity_intensity,
)

# ==============================
# Page setup
# ==============================
st.set_page_config(page_title="Steel Route Builder v2", layout="wide")

st.markdown(
    """
    <style>
    :root { --c1: #2563eb; }
    .hr { height:1px; background:linear-gradient(to right,#e5e7eb,#cbd5e1,#e5e7eb); margin:.25rem 0 .75rem 0; border:0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Steel Route Builder v2")
st.caption("Scenario controls route and (optionally) picks. App does not re-compute the model.")

# ------------------------------
# Helper: infer route from scenario (content or filename)
# ------------------------------
def _infer_route_preset_from_scenario(scn_dict: Dict[str, Any] | None, scn_filename: str | None) -> str | None:
    """
    Returns one of: 'BF-BOF', 'DRI-EAF', 'EAF-Scrap', 'External', or None.
    Priority:
      1) scenario['route_preset'] (or 'route'/'preset') if present
      2) filename tokens: 'bf-bof', 'dri-eaf', 'eaf-scrap', 'external'
    """
    def _norm(val: str) -> str | None:
        if not isinstance(val, str):
            return None
        v = val.strip().lower().replace("_", "-")
        if "bf-bof" in v or v in {"bf", "bfbof"}:
            return "BF-BOF"
        if "dri-eaf" in v or v in {"drieaf", "dri"}:
            return "DRI-EAF"
        if "eaf-scrap" in v or v in {"eafscrap"}:
            return "EAF-Scrap"
        if "external" in v:
            return "External"
        return None

    if scn_dict and isinstance(scn_dict, dict):
        for k in ("route_preset", "route", "preset"):
            got = _norm(scn_dict.get(k))
            if got:
                return got
    if scn_filename:
        got = _norm(scn_filename)
        if got:
            return got
    return None


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

    # Route preset is driven by scenario (no radio)
    implied_route = _infer_route_preset_from_scenario(scenario, scenario_choice if scenario_choice != "(none)" else None)
    route_preset_eff = implied_route or "auto"
    if implied_route:
        st.caption(f"Route preset forced by scenario → **{route_preset_eff}**")
    else:
        st.caption("No route in scenario — using **auto** (mask + core defaults).")

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
# (No producer-pick UI) — scenario drives route and optional picks.


# ==============================
# Run core + Log
# ==============================
run_clicked = st.button("Run core model", type="primary")
if run_clicked:
    # Use picks from scenario if provided; otherwise none (core auto-picks).
    scenario_picks: Dict[str, str] = {}
    if isinstance(scenario, dict):
        scenario_picks = scenario.get("picks_by_material") or scenario.get("picks") or {}
        if not isinstance(scenario_picks, dict):
            scenario_picks = {}

    route = RouteConfig(
        route_preset=route_preset_eff,
        stage_key=stage_key,
        demand_qty=float(demand_qty),
        picks_by_material=scenario_picks,
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
                "route_preset": route_preset_eff,
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
    c1.download_button(
        "Production routes (JSON)",
        data=json.dumps(out.production_routes, indent=2).encode("utf-8"),
        file_name="production_routes.json",
    )
    c2.download_button(
        "Energy balance (CSV)",
        data=out.energy_balance.to_csv().encode("utf-8"),
        file_name="energy_balance.csv",
    )
    if out.emissions is not None:
        c3.download_button(
            "Emissions (CSV)",
            data=out.emissions.to_csv().encode("utf-8"),
            file_name="emissions.csv",
        )

st.markdown("<hr class='hr'>", unsafe_allow_html=True)
st.caption("Steel Route Builder v2 — scenario-driven route; logs inputs + CO₂.")
