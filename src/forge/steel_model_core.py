"""Compatibility shim.

This module preserves the public name `forge.steel_model_core` while the legacy
implementation has been renamed (e.g., to `steel_model_core_legacy`). Importing
from here allows the refactored modules that still delegate to the monolith to
run unmodified.

If you renamed the file to `steel_model_core_legacy.py`, this shim re-exports
everything from that module. If not, it falls back to importing the original
module.
"""
from __future__ import annotations

try:
    # Preferred: load the renamed legacy module if present
    from .steel_model_core_legacy import *  # type: ignore  # noqa: F401,F403
except Exception as e:
    # If not present, instruct to restore the original filename
    raise ImportError(
        "forge.steel_model_core shim: could not import steel_model_core_legacy. "
        "Either restore the original filename 'steel_model_core.py' or keep a "
        "copy named 'steel_model_core_legacy.py' alongside this shim."
    ) from e
