import types
import pandas as pd

from forge.core.compute import apply_gas_routing_and_credits
from forge.core.models import Process


def test_apply_gas_routing_updates_efs_and_meta():
    # Minimal synthetic setup
    recipes = [
        Process('Coke Production', inputs={'Coal': 1.0}, outputs={'Process Gas': 5.0}),
        Process('Blast Furnace', inputs={'Coal': 1.0}, outputs={'Pig Iron': 1.0}),
        Process('Utility Plant', inputs={'Process Gas': 1.0}, outputs={'Electricity': 0.2}),
    ]
    prod_levels = {'Coke Production': 10.0, 'Blast Furnace': 2.0}
    energy_shares = {
        'Coke Production': {'Coal': 1.0},
        'Blast Furnace': {'Coal': 1.0},
    }
    energy_int = {'Blast Furnace': 10.0}
    energy_content = {}
    e_efs = {'Coal': 10.0, 'Gas': 100.0}
    params = types.SimpleNamespace()
    eb = pd.DataFrame({'TOTAL': [0.0]}, index=['TOTAL'])
    scenario = {
        'gas_config': {
            'process_gas_carrier': 'Process Gas',
            'natural_gas_carrier': 'Gas',
            'utility_process': 'Utility Plant',
            'process_gas_sources': [
                {'process': 'Coke Production', 'carrier': 'Process Gas', 'outputs_in_MJ': True},
            ],
        },
        'gas_routing': {'direct_use_fraction': 0.5},
    }

    eb2, efs2, meta = apply_gas_routing_and_credits(
        energy_balance=eb,
        recipes=recipes,
        prod_levels=prod_levels,
        params=params,
        energy_shares=energy_shares,
        energy_int=energy_int,
        energy_content=energy_content,
        e_efs=e_efs,
        scenario=scenario,
        compute_inside_gas_reference_fn=None,
    )

    # EFs should contain updated Gas and Process Gas
    assert 'Gas' in efs2 and 'Process Gas' in efs2
    # Meta should include these keys
    for k in (
        'total_process_gas_MJ', 'direct_use_gas_MJ', 'electricity_gas_MJ',
        'ef_gas_blended', 'EF_process_gas', 'util_eff',
    ):
        assert k in meta
