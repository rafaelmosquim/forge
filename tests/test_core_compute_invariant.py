import os
import pytest

from forge.steel_core_api_v2 import run_scenario, ScenarioInputs, RouteConfig


def test_compute_produces_nonnegative_emissions_and_rows(data_dir):
    # Minimal invariant via public API
    scn = ScenarioInputs(
        country_code=None,
        scenario={},
        route=RouteConfig(route_preset='BF-BOF', stage_key='Finished', demand_qty=1000.0),
    )
    out = run_scenario(str(data_dir), scn)
    assert out is not None
    # Emissions table exists and totals nonnegative
    assert out.emissions is not None and not out.emissions.empty
    total = float(out.emissions.get('TOTAL CO2e', out.emissions.sum(axis=1)).sum())
    assert total >= 0.0
