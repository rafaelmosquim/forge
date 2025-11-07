"""I/O utilities: YAML and config loaders (wrappers).

Thin wrappers to the legacy `forge.steel_model_core` functions to provide a
cleaner import path during migration.
"""
from __future__ import annotations

from typing import Any, Dict

from forge import steel_model_core as _core

safe_yaml_load = _core.safe_yaml_load
load_parameters = _core.load_parameters
load_data_from_yaml = _core.load_data_from_yaml
load_recipes_from_yaml = _core.load_recipes_from_yaml
load_market_config = _core.load_market_config
load_electricity_intensity = _core.load_electricity_intensity

__all__ = [
    "safe_yaml_load",
    "load_parameters",
    "load_data_from_yaml",
    "load_recipes_from_yaml",
    "load_market_config",
    "load_electricity_intensity",
]

