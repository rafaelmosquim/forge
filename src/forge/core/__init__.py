"""
Core package façade.

This package provides a structured import surface for the steel core while
keeping the legacy monolith `forge.steel_model_core` as the single source of
truth for now. Each submodule wraps and re-exports functions/classes from the
monolith. This allows a gradual migration without breaking existing imports.

Submodules:
  - models: data structures (e.g., Process)
  - io: YAML/config loaders
  - routing: route helpers and constants (STAGE_MATS, build_route_mask, ...)
  - compute: core computations, transforms, emissions/LCI
  - viz: plotting helpers (Sankey builders)

Downstream code can start importing from `forge.core.*` immediately.
"""

from . import models, io, routing, compute, viz  # baseline
from . import costs, lci, transforms  # additional subpackages

__all__ = [
    "models",
    "io",
    "routing",
    "compute",
    "viz",
    "costs",
    "lci",
    "transforms",
]
