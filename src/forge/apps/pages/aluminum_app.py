import os
import sys
import runpy
from pathlib import Path
import streamlit as st

# Ensure repo root on sys.path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Force the aluminum UI and canonical stack
os.environ["FORGE_FORCE_SECTOR"] = "Aluminum"

# Execute the canonical aluminum app script on each rerun so UI updates correctly
APP_PATH = ROOT_DIR / "src" / "forge" / "canonical" / "apps" / "streamlit_app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")

st.stop()
