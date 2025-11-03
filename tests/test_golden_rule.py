import sys
import pathlib

import pytest

# Ensure src directory is on path for direct test execution
SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from forge.steel_core_api_v2 import RouteConfig, ScenarioInputs, run_scenario


@pytest.mark.integration
def test_bf_bof_validation_brazil_golden_rule(data_dir: pathlib.Path):
    """
    Golden-rule regression: Brazil, BF-BOF route, validation product boundary,
    gas routing split 50/50. Result pinned to the value observed after the
    descriptor / gas-routing refactor so future changes surface explicitly.
    """
    route = RouteConfig(
        route_preset="BF-BOF",
        stage_key="Cast",
        demand_qty=1000.0,
        stage_role="validation",
    )

    scenario_payload = {
        "gas_routing": {
            "direct_use_fraction": 0.5,
            "electricity_fraction": 0.5,
        },
    }

    scenario = ScenarioInputs(
        country_code="BRA",
        scenario=scenario_payload,
        route=route,
    )

    outputs = run_scenario(str(data_dir), scenario)

    # Expectation anchored on the latest validated pipeline
    assert outputs.total_co2e_kg == pytest.approx(2088.0584658163034, rel=1e-6)
