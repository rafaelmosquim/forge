"""Portal UI to choose material and dispatch to dedicated apps."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="FORGE Portal", layout="centered")

st.markdown("## FORGE Viewer\nSelect a material to open the dedicated UI.")

choice = st.radio("Material", ["Steel", "Aluminum"], index=0, horizontal=True)

if st.button("Open", type="primary"):
    if choice == "Steel":
        # Force steel-only UI and switch to the steel page
        st.session_state["portal_target"] = "steel"
        st.query_params.clear()
        st.query_params.update({"material": "steel"})
        st.switch_page("pages/steel_app.py")
    else:
        st.session_state["portal_target"] = "aluminum"
        st.query_params.clear()
        st.query_params.update({"material": "aluminum"})
        st.switch_page("pages/aluminum_app.py")

st.info("Steel uses lci-demo stack; Aluminum uses features/p-gas stack.", icon="ℹ️")
