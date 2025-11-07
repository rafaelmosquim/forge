"""Core engine trio kept together here.

Exports the main compute functions while we gradually refactor internals.
Currently delegates to the legacy monolith to preserve behavior.
"""
from __future__ import annotations

from forge import steel_model_core as _core

calculate_balance_matrix = _core.calculate_balance_matrix
calculate_energy_balance = _core.calculate_energy_balance
calculate_emissions = _core.calculate_emissions

__all__ = [
    "calculate_balance_matrix",
    "calculate_energy_balance",
    "calculate_emissions",
]

