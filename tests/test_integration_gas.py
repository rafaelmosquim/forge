import sys
import pathlib
import pandas as pd
import types
from types import SimpleNamespace

# Ensure src directory is on sys.path so package imports resolve during tests
SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from forge.core.models import Process
from forge.core.gas import apply_gas_routing_and_credits
from forge.core.engine import calculate_energy_balance


def make_minimal_recipes():
    # Utility Plant that converts Process Gas to Electricity at 0.2 MJ/MJ
    util = Process('Utility Plant', inputs={'Process Gas': 1.0}, outputs={'Electricity': 0.2})

    # Coke Production emits Process Gas
    coke = Process('Coke Production', inputs={'Coal': 1.0}, outputs={'Process Gas': 5.0})

    return [util, coke]


def test_apply_gas_routing_and_credits_basic():
    recipes = make_minimal_recipes()
    recipes_dict = {r.name: r for r in recipes}

    # production levels: 1 run of Coke Production
    prod_levels = {'Coke Production': 1.0}

    # simple energy shares: Coke uses Coal (fuel) so its "process gas" EF is derived from Coal
    energy_shares = {'Coke Production': {'Coal': 1.0}, 'Utility Plant': {'Process Gas': 1.0}}
    energy_int = {'Coke Production': 10.0, 'Utility Plant': 0.0}
    energy_content = {}

    # Starting emission factors: Coal (fuel) vs grid Gas
    e_efs = {'Coal': 10.0, 'Gas': 50.0, 'Electricity': 100.0}

    # Minimal energy balance derived from production levels
    energy_balance = calculate_energy_balance(prod_levels, energy_int, energy_shares)

    params = SimpleNamespace()
    # Route all process gas to direct use so blended Gas EF should equal process-gas EF
    scenario = {
        'gas_config': {
            'process_gas_carrier': 'Process Gas',
            'natural_gas_carrier': 'Gas',
            'utility_process': 'Utility Plant',
            'process_gas_sources': [
                {'process': 'Coke Production', 'carrier': 'Process Gas', 'outputs_in_MJ': True},
            ],
        },
        'gas_routing': {'direct_use_fraction': 1.0, 'electricity_fraction': 0.0},
        'inside_elec_ref': 0.0,
    }

    eb_new, e_efs_new, meta = apply_gas_routing_and_credits(
        energy_balance=energy_balance,
        recipes=recipes,
        prod_levels=prod_levels,
        params=params,
        energy_shares=energy_shares,
        energy_int=energy_int,
        energy_content=energy_content,
        e_efs=e_efs,
        scenario=scenario,
        compute_inside_gas_reference_fn=lambda *a, **k: 5.0,
    )

    # With all process gas routed to electricity and util eff 0.2, internal elec should be > 0
    assert 'Utility Plant' in eb_new.index
    # Process gas EF should be derived from Coal (10.0) and applied to both 'Process Gas' and blended 'Gas'
    assert e_efs_new.get('Process Gas', None) == 10.0
    assert e_efs_new.get('Gas', None) == 10.0
    assert 'total_process_gas_MJ' in meta
    assert meta['total_process_gas_MJ'] == 5.0
