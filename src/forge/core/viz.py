"""Plotting helpers (wrappers for Sankey builders)."""
from __future__ import annotations

from forge import steel_model_core as _core

make_mass_sankey = getattr(_core, "make_mass_sankey", None)
make_energy_sankey = getattr(_core, "make_energy_sankey", None)
make_energy_to_process_sankey = getattr(_core, "make_energy_to_process_sankey", None)
make_hybrid_sankey = getattr(_core, "make_hybrid_sankey", None)

__all__ = [
    "make_mass_sankey",
    "make_energy_sankey",
    "make_energy_to_process_sankey",
    "make_hybrid_sankey",
]

