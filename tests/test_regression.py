import os
import math
import pytest

from forge.steel_core_api_v2 import run_scenario, ScenarioInputs, RouteConfig


@pytest.mark.integration
@pytest.mark.parametrize(
    "flexible,use_expected", [
        (0.0, 2447.8),
        (0.61, 2366.1),
        (0.95, 2320.5),
    ],
)
def test_bf_bof_validation_as_cast_bra_gas_routing_snapshots(data_dir, flexible, use_expected):
    """Golden regression: Likely dataset, Brazil grid, Validation (as cast).

    Total CO2e (kg) for BF-BOF as the share of the plant's natural-gas demand
    met with recovered gas varies: 0.0 (CSN-like, reheating on natural gas),
    0.61 (sector default, 13 plant-years) and 0.95 (Tubarao-like).

    Snapshot updated 2026-11-05 (was 3617.5/3352.8/3082.8 keyed on
    direct_use_fraction, already stale at 2440/2210/1972). The recovered gas
    is now allocated from one pool so every MJ is charged once (forge.core.gas),
    the coke ovens are fired on gas and charged for the coal they convert, and
    10% of the gas is flared. The old 50/50 split charged more gas than the
    plant made at one end and lost gas at the other.
    """
    scn = ScenarioInputs(
        country_code="BRA",
        scenario={
            # Gas routing selection for the run
            "gas_routing": {"flexible_share": flexible},
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
        f"Got {total_kg} kg for flexible_share={flexible}; expected {use_expected} kg"
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
def test_non_bf_routes_insensitive_to_gas_routing(data_dir, route_preset):
    """For DRI-EAF and EAF-Scrap, gas routing should not affect results.

    Process-gas recovery logic is defined only for BF-BOF routes; changing
    the gas routing must not change total emissions for DRI/EAF routes.
    """

    def _run_with_direct(share):
        scn = ScenarioInputs(
            country_code="BRA",
            scenario={"gas_routing": {"flexible_share": share, "loss_fraction": share}},
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
        f"{route_preset} total CO2e changed with gas routing: "
        f"{total_0} vs {total_1}"
    )
