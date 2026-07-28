"""
Regression coverage for the Streamlit front-page sector ("material") gate:
selecting Steel vs Aluminum must load the correct dataset afterward, both on
first load and when switching sectors mid-session via "Change sector".

Uses Streamlit's official AppTest harness (in-process script execution,
no browser/server required). Each st.stop() in the gate/reset logic ends a
run, so proceeding past one takes an extra at.run() with no new interaction.
"""
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest


APP_PATH = "src/forge/apps/streamlit_app.py"


def _select_sector(at: AppTest, sector: str) -> AppTest:
    """Drive the front-page gate: pick `sector`, click Continue, run past st.stop()."""
    at.radio(key="sector_gate_selection").set_value(sector)
    at.button(key="FormSubmitter:sector_gate_form-Continue").click()
    at.run(timeout=30)  # processes the click, hits st.stop() right after
    at.run(timeout=30)  # re-renders past the gate into the main body
    assert not at.exception, f"unexpected exception after selecting {sector}: {list(at.exception)}"
    return at


def _data_folder_caption(at: AppTest) -> str | None:
    for c in at.caption:
        if "Using data folder:" in str(c.value):
            return str(c.value)
    return None


@pytest.mark.integration
def test_front_page_gate_offers_steel_and_aluminum():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception

    radios = {r.key: r for r in at.radio}
    assert "sector_gate_selection" in radios
    assert radios["sector_gate_selection"].options == ["Steel", "Aluminum"]
    # No dataset should be selected yet -- still at the gate.
    assert _data_folder_caption(at) is None


@pytest.mark.integration
def test_selecting_steel_loads_steel_dataset():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _select_sector(at, "Steel")

    assert _data_folder_caption(at) == "Using data folder: datasets/steel/likely"

    datasets = {sb.key: sb for sb in at.selectbox}
    assert datasets["dataset_select_steel"].options == [
        "Likely", "Optimistic (Low)", "Pessimistic (High)", "Usiminas",
    ]

    route_boxes = [sb for sb in at.selectbox if (sb.label or "").lower() == "route"]
    assert route_boxes, "Route selectbox not found for Steel sector"
    assert set(route_boxes[0].options) >= {"BF-BOF", "DRI-EAF", "EAF-Scrap"}


@pytest.mark.integration
def test_selecting_aluminum_loads_aluminum_dataset():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _select_sector(at, "Aluminum")

    assert _data_folder_caption(at) == "Using data folder: datasets/aluminum/baseline"

    datasets = {sb.key: sb for sb in at.selectbox}
    assert datasets["dataset_select_aluminum"].options == ["Baseline"]

    route_boxes = [sb for sb in at.selectbox if (sb.label or "").lower() == "route"]
    assert route_boxes, "Route selectbox not found for Aluminum sector"
    assert set(route_boxes[0].options) == {"Primary Aluminum", "Secondary Aluminum"}


@pytest.mark.integration
def test_switching_sector_reloads_correct_dataset():
    """Steel -> Change sector -> Aluminum must not leave stale steel state behind."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    _select_sector(at, "Steel")
    assert _data_folder_caption(at) == "Using data folder: datasets/steel/likely"

    at.button(key="btn_change_sector").click()
    at.run(timeout=30)  # processes the click, hits st.stop() after resetting state
    at.run(timeout=30)  # re-renders the gate
    assert not at.exception

    radios = {r.key: r for r in at.radio}
    assert "sector_gate_selection" in radios, "Change sector did not return to the gate"
    assert _data_folder_caption(at) is None, "stale dataset caption survived the sector reset"

    _select_sector(at, "Aluminum")
    assert _data_folder_caption(at) == "Using data folder: datasets/aluminum/baseline"

    datasets = {sb.key: sb for sb in at.selectbox}
    assert "dataset_select_steel" not in datasets, "steel dataset selectbox leaked after switching to Aluminum"
    assert "dataset_select_aluminum" in datasets
