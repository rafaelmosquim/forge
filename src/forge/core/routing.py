"""Routing helpers and constants (wrappers)."""
from __future__ import annotations

from typing import Any, Dict

from forge import steel_model_core as _core

# Public constants
STAGE_MATS = getattr(_core, "STAGE_MATS", {})

# Route helpers
build_route_mask = getattr(_core, "build_route_mask")
enforce_eaf_feed = getattr(_core, "enforce_eaf_feed")
set_prefer_internal_processes = getattr(_core, "set_prefer_internal_processes", lambda *a, **k: None)
apply_inhouse_clamp = getattr(_core, "apply_inhouse_clamp", lambda *a, **k: None)

__all__ = [
    "STAGE_MATS",
    "build_route_mask",
    "enforce_eaf_feed",
    "set_prefer_internal_processes",
    "apply_inhouse_clamp",
]

