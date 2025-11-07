"""Models and core data structures.

For now, these are thin aliases to the legacy monolith
`forge.steel_model_core` to avoid duplication during the transition.
"""
from __future__ import annotations

from typing import Any

# Import from the monolith and re-export
from forge import steel_model_core as _core

# Classes
Process = _core.Process

# Common constants (if present)
OUTSIDE_MILL_PROCS: Any = getattr(_core, "OUTSIDE_MILL_PROCS", None)

__all__ = [
    "Process",
    "OUTSIDE_MILL_PROCS",
]

