import os
import sys
import runpy
from pathlib import Path
import streamlit as st

# Ensure repo root on sys.path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Force the steel UI to stay on Steel only
os.environ["FORGE_FORCE_SECTOR"] = "Steel"

# Execute the steel app script on each rerun so UI updates correctly
APP_PATH = ROOT_DIR / "src" / "forge" / "apps" / "streamlit_app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")

st.stop()
