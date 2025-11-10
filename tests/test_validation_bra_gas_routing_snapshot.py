import os
import math
import pytest

from forge.steel_core_api_v2 import run_scenario, ScenarioInputs, RouteConfig


@pytest.mark.integration
@pytest.mark.parametrize(
    "direct,use_expected", [
        (0.0, 2216.0),
        (0.5, 2087.0),
        (1.0, 1951.0),
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

    # Allow tiny numerical tolerance (<= 1 kg)
    assert math.isclose(total_kg, use_expected, rel_tol=0.0, abs_tol=1.0), (
        f"Got {total_kg} kg for direct_use_fraction={direct}; expected {use_expected} kg"
    )

