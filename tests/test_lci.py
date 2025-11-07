import pandas as pd

from forge.core.lci import calculate_lci
from forge.core.models import Process


def test_calculate_lci_simple_energy_split():
    # One process: 2 runs, outputs 1 kg per run → denom = 2 kg
    proc = Process('P', inputs={'Electricity': 10.0}, outputs={'Out': 1.0})
    prod = {'P': 2.0}
    eb = pd.DataFrame({'Electricity': [20.0]}, index=['P'])

    lci = calculate_lci(prod, [proc], energy_balance=eb)
    assert not lci.empty
    # Energy input per kg should be 10 MJ/kg
    row = lci[(lci['Process'] == 'P') & (lci['Category'] == 'Energy') & (lci['Input'] == 'Electricity')]
    assert abs(float(row['Amount'].iloc[0]) - 10.0) < 1e-9

