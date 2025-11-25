import types
import math
import pandas as pd

from forge.core.gas import apply_gas_routing_and_credits
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
        compute_inside_reference_fn=None,
    )

    # EFs should contain updated Gas and Process Gas
    assert 'Gas' in efs2 and 'Process Gas' in efs2
    # Meta should include these keys
    for k in (
        'total_process_gas_MJ', 'direct_use_gas_MJ', 'electricity_gas_MJ',
        'ef_gas_blended', 'EF_process_gas', 'util_eff',
    ):
        assert k in meta


def test_process_gas_efs_in_expected_range_and_bof_uses_carrier_ef():
    """Ensure Coke/BF gas EFs come from combustion mix and BOF gas from carrier EF.

    All resulting process-gas EFs (per source and blended) should be in the
    'hundreds' range (between 100 and 200 gCO2/MJ), and Coke/BF should not
    fall back to their carrier EFs.
    """
    recipes = [
        Process('Coke Production', inputs={'Coal': 1.0}, outputs={'Coke Gas': 1.0}),
        Process('Blast Furnace', inputs={'Coal': 1.0}, outputs={'BF Gas': 1.0}),
        Process('Basic Oxygen Furnace', inputs={'Electricity': 1.0}, outputs={'BOF Gas': 1.0}),
        Process('Utility Plant', inputs={'Process Gas': 1.0}, outputs={'Electricity': 0.5}),
    ]
    prod_levels = {
        'Coke Production': 1.0,
        'Blast Furnace': 1.0,
        'Basic Oxygen Furnace': 1.0,
    }
    energy_shares = {
        # Coke/BF use Coal → combustion-based EFs
        'Coke Production': {'Coal': 1.0},
        'Blast Furnace': {'Coal': 1.0},
        # BOF uses only electricity + process gas → forces fallback path
        'Basic Oxygen Furnace': {'Electricity': 1.0, 'Process Gas': 0.0},
    }
    energy_int = {}
    energy_content = {}
    e_efs = {
        # Combustion fuels
        'Coal': 120.0,
        'Gas': 70.0,
        'Electricity': 40.0,
        # Gas carriers: exaggerated values for Coke/BF so we can detect misuse
        'Coke Gas': 999.0,
        'BF Gas': 999.0,
        # BOF gas uses a realistic "hundreds" EF
        'BOF Gas': 150.0,
        # Base process-gas carrier EF stays zero
        'Process Gas': 0.0,
    }
    params = types.SimpleNamespace()
    eb = pd.DataFrame({'TOTAL': [0.0]}, index=['TOTAL'])
    gas_config = {
        'process_gas_carrier': 'Process Gas',
        'natural_gas_carrier': 'Gas',
        'utility_process': 'Utility Plant',
        'process_gas_sources': [
            {'process': 'Coke Production', 'carrier': 'Coke Gas', 'outputs_in_MJ': True},
            {'process': 'Blast Furnace', 'carrier': 'BF Gas', 'outputs_in_MJ': True},
            {'process': 'Basic Oxygen Furnace', 'carrier': 'BOF Gas', 'outputs_in_MJ': True},
        ],
    }
    scenario = {
        'gas_config': gas_config,
        'gas_routing': {'direct_use_fraction': 1.0},
    }

    _, efs2, meta = apply_gas_routing_and_credits(
        energy_balance=eb,
        recipes=recipes,
        prod_levels=prod_levels,
        params=params,
        energy_shares=energy_shares,
        energy_int=energy_int,
        energy_content=energy_content,
        e_efs=e_efs,
        scenario=scenario,
        compute_inside_reference_fn=None,
    )

    ef_coke = meta.get('EF_coke_gas')
    ef_bf = meta.get('EF_bf_gas')
    ef_process = meta.get('EF_process_gas')

    # All EFs we care about should live in the 100–200 range
    for val in (ef_coke, ef_bf, ef_process):
        assert val is not None
        assert 100.0 <= float(val) <= 200.0

    # Coke/BF gas EFs must come from combustion (Coal), not their carrier EF (999)
    assert not math.isclose(ef_coke, e_efs['Coke Gas'])
    assert not math.isclose(ef_bf, e_efs['BF Gas'])

    # With equal MJ contributions from each source, blended EF should be the
    # simple average of EF_coke, EF_bf, and BOF gas EF (150).
    expected_blend = (ef_coke + ef_bf + e_efs['BOF Gas']) / 3.0
    assert math.isclose(ef_process, expected_blend, rel_tol=1e-6)
