import os
import math
import pytest

from forge.steel_core_api_v2 import run_scenario, ScenarioInputs, RouteConfig


@pytest.mark.integration
@pytest.mark.parametrize(
    "direct,use_expected", [
        (0.0, 2998.0),
        (0.5, 2827.0),
        (1.0, 2651.0),
    ],
)
def test_bf_bof_validation_as_cast_bra_gas_routing_snapshots(data_dir, direct, use_expected):
    """Golden regression: Likely dataset, Brazil grid, Validation (as cast).

    Verifies total CO2e (kg) for BF-BOF under gas routing fractions
    0.0 (all natural gas), 0.5 split, and 1.0 (all internal process gas).
    """
    scn = ScenarioInputs(
        country_code="BRA",
        scenario={
            # Gas routing selection for the run
            "gas_routing": {"direct_use_fraction": direct},
        },
        route=RouteConfig(
            route_preset="BF-BOF",
            stage_key="Cast",            # "as cast" boundary
            stage_role="validation",     # validation mode
            demand_qty=1000.0,            # kg demand at boundary
        ),
    )

    out = run_scenario(str(data_dir), scn)
    assert out is not None and out.emissions is not None and not out.emissions.empty

    # total_co2e_kg should be reported in kilograms
    total_kg = float(out.total_co2e_kg or 0.0)

    # Allow small tolerance while converging (<= 10 kg)
    assert math.isclose(total_kg, use_expected, rel_tol=0.0, abs_tol=10.0), (
        f"Got {total_kg} kg for direct_use_fraction={direct}; expected {use_expected} kg"
    )


@pytest.mark.integration
def test_cli_style_run_matches_app_scenario(data_dir, yload):
    """Ensure CLI-style BF-BOF validation run matches the app's scenario-based run."""
    # CLI-style: what forge.cli.engine_cli uses under make reproduce-validation
    scn_cli = ScenarioInputs(
        country_code="BRA",
        scenario={},
        route=RouteConfig(
            route_preset="BF-BOF",
            stage_key="Cast",
            demand_qty=1000.0,
        ),
    )
    out_cli = run_scenario(str(data_dir), scn_cli)
    total_cli = float(out_cli.total_co2e_kg or 0.0)

    # App-style: uses the BF_BOF_coal.yml scenario selected in the UI
    scenario_path = data_dir / "scenarios" / "BF_BOF_coal.yml"
    scenario_app = yload(scenario_path)
    scn_app = ScenarioInputs(
        country_code="BRA",
        scenario=scenario_app,
        route=RouteConfig(
            route_preset="BF-BOF",
            stage_key="Cast",
            stage_role="validation",
            demand_qty=1000.0,
        ),
    )
    out_app = run_scenario(str(data_dir), scn_app)
    total_app = float(out_app.total_co2e_kg or 0.0)

    # CLI and app runs should agree numerically
    assert math.isclose(total_cli, total_app, rel_tol=0.0, abs_tol=1e-6), (
        f"CLI-style total {total_cli} kg != app-style total {total_app} kg"
    )


@pytest.mark.integration
@pytest.mark.parametrize("route_preset", ["DRI-EAF", "EAF-Scrap"])
def test_non_bf_routes_insensitive_to_direct_use_fraction(data_dir, route_preset):
    """For DRI-EAF and EAF-Scrap, gas routing should not affect results.

    Process-gas recovery logic is defined only for BF-BOF routes; changing
    direct_use_fraction must not change total emissions for DRI/EAF routes.
    """

    def _run_with_direct(direct):
        scn = ScenarioInputs(
            country_code="BRA",
            scenario={"gas_routing": {"direct_use_fraction": direct}},
            route=RouteConfig(
                route_preset=route_preset,
                stage_key="Finished",
                demand_qty=1000.0,
            ),
        )
        out = run_scenario(str(data_dir), scn)
        assert out is not None
        return float(out.total_co2e_kg or 0.0)

    total_0 = _run_with_direct(0.0)
    total_1 = _run_with_direct(1.0)

    # Results should be identical (within numerical noise) when toggling routing
    assert math.isclose(total_0, total_1, rel_tol=0.0, abs_tol=1e-6), (
        f"{route_preset} total CO2e changed with direct_use_fraction: "
        f"{total_0} vs {total_1}"
    )
