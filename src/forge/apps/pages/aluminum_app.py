import os
import streamlit as st

# Force the aluminum UI and canonical stack
os.environ["FORGE_FORCE_SECTOR"] = "Aluminum"

# Importing runs the canonical aluminum app
import forge.canonical.apps.streamlit_app  # noqa: F401

st.stop()
