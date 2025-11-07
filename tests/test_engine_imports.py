def test_core_engine_trio_imports():
    from forge.core.engine import (
        calculate_balance_matrix,
        calculate_energy_balance,
        calculate_emissions,
    )
    assert callable(calculate_balance_matrix)
    assert callable(calculate_energy_balance)
    assert callable(calculate_emissions)

