import pandas as pd

from forge.core.viz import make_energy_sankey


def test_make_energy_sankey_returns_figure():
    df = pd.DataFrame(
        {
            'Electricity': [50.0, 0.0],
            'Gas': [0.0, 20.0],
            'TOTAL': [50.0, 20.0],
        },
        index=['ProcA', 'ProcB']
    )
    fig = make_energy_sankey(df, min_MJ=1.0)
    # Basic sanity: figure-like object has data
    assert hasattr(fig, 'to_dict')
    d = fig.to_dict()
    assert 'data' in d and len(d['data']) >= 1

