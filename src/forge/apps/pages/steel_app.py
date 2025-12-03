import os
import streamlit as st

# Force the steel UI to stay on Steel only
os.environ["FORGE_FORCE_SECTOR"] = "Steel"

# Importing runs the steel app
import forge.apps.streamlit_app  # noqa: F401

st.stop()
