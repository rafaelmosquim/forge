import sys
import pathlib
from types import SimpleNamespace


# Ensure src directory on path
SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from forge.models import Process
from forge.gas_routing import apply_gas_routing_and_credits
from forge.steel_model_core import calculate_energy_balance


def _make_stub_recipes():
    util = Process("Utility Plant", inputs={"Process Gas": 1.0}, outputs={"Electricity": 0.25})
    coke = Process("Coke Production", inputs={"Coal": 1.0}, outputs={"Process Gas": 4.0})
    return [util, coke]


def test_reference_callbacks_keep_totals_constant():
    recipes = _make_stub_recipes()
    prod_levels = {"Coke Production": 1.0, "Utility Plant": 0.0}
    energy_shares = {
        "Coke Production": {"Coal": 1.0},
        "Utility Plant": {"Process Gas": 1.0},
    }
    energy_int = {"Coke Production": 12.0, "Utility Plant": 0.0}
    energy_balance = calculate_energy_balance(prod_levels, energy_int, energy_shares)

    base_params = SimpleNamespace()
    energy_content = {}
    emission_factors = {"Coal": 15.0, "Gas": 45.0, "Process Gas": 0.0, "Electricity": 90.0}

    scenario_template = {
        "gas_config": {},
        "process_roles": {},
        "fallback_materials": [],
    }

    callback_gas = lambda **_: 40.0
    callback_elec = lambda **_: 18.0

    result_direct, efs_direct, meta_direct = apply_gas_routing_and_credits(
        energy_balance=energy_balance,
        recipes=recipes,
        prod_levels=prod_levels,
        params=base_params,
        energy_shares=energy_shares,
        energy_int=energy_int,
        energy_content=energy_content,
        e_efs=emission_factors,
        scenario={**scenario_template, "gas_routing": {"direct_use_fraction": 1.0}},
        credit_on=True,
        compute_inside_gas_reference_fn=callback_gas,
        compute_inside_elec_reference_fn=callback_elec,
    )

    result_split, efs_split, meta_split = apply_gas_routing_and_credits(
        energy_balance=energy_balance,
        recipes=recipes,
        prod_levels=prod_levels,
        params=base_params,
        energy_shares=energy_shares,
        energy_int=energy_int,
        energy_content=energy_content,
        e_efs=emission_factors,
        scenario={**scenario_template, "gas_routing": {"direct_use_fraction": 0.3, "electricity_fraction": 0.4}},
        credit_on=True,
        compute_inside_gas_reference_fn=callback_gas,
        compute_inside_elec_reference_fn=callback_elec,
    )

    assert meta_direct["total_gas_consumption_plant"] == 40.0
    assert meta_split["total_gas_consumption_plant"] == 40.0
    assert meta_direct["f_internal_gas"] != meta_split["f_internal_gas"]

    # Process gas EF should remain anchored to coal regardless of routing fractions
    assert meta_direct["EF_process_gas"] == meta_split["EF_process_gas"] == 15.0
