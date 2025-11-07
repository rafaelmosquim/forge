"""Models and core data structures (duplicated from monolith)."""
from __future__ import annotations

from typing import Any, Dict


class Process:
    """Represents a single recipe with its inputs and outputs."""
    __slots__ = ("name", "inputs", "outputs")

    def __init__(self, name: str, inputs: Dict[str, float], outputs: Dict[str, float]):
        self.name = name
        self.inputs = dict(inputs or {})
        self.outputs = dict(outputs or {})


# Keep the same constant as the monolith so route utilities can reference it.
OUTSIDE_MILL_PROCS = {
    "Ship Pig Iron (Exit)",
    "Ingot Casting (R) – Exit",
    "Ingot Casting (L) – Exit",
    "Ingot Casting (H) – Exit",
    "Direct use of Basic Steel Products (Exit)",
    "Direct use after Cold Rolling (Exit)",
    "Direct use of Basic Steel Products (IP4)",
    "Casting/Extrusion/Conformation",
    "Stamping/calendering/lamination",
    "Machining",
    "No Coating",
    "Hot Dip Metal Coating FP",
    "Electrolytic Metal Coating FP",
    "Organic or Sintetic Coating (painting)",
}


__all__ = ["Process", "OUTSIDE_MILL_PROCS"]

