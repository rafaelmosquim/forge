"""I/O utilities: YAML and config loaders.

This module duplicates the core I/O logic from the monolith so callers can
depend on `forge.core.io` without dragging the entire monolith. The legacy
`forge.steel_model_core` keeps its own copies for now; once migration is
complete we can make it re-export these.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def safe_yaml_load(filepath: str | Path, default=None):
    """Safe YAML loader: returns default when file missing or invalid."""
    try:
        p = Path(filepath)
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or default
    except FileNotFoundError:
        logger.warning("YAML file not found: %s — returning default", filepath)
        return default
    except Exception as e:
        logger.error("Error reading YAML %s: %s", filepath, e)
        return default


def load_parameters(filepath: str):
    """Load parameters YAML into a SimpleNamespace (nested)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            params_dict = yaml.safe_load(f) or {}
            params = json.loads(json.dumps(params_dict), object_hook=lambda d: SimpleNamespace(**d))
            logger.debug("Loaded parameters YAML from %s", filepath)
            logger.debug("process_gas: %s", getattr(params, "process_gas", "NOT FOUND"))
            return params
    except FileNotFoundError:
        logger.error("Parameters file not found: %s", filepath)
        return SimpleNamespace()


def load_data_from_yaml(filepath: str | Path, default_value=0, unwrap_single_key=True):
    """Load generic YAML into a normalized dict of scalars and mappings.

    - Unwraps single-key dicts when `unwrap_single_key` is True
    - Converts scalar strings to float/int when possible
    - Preserves None by mapping to `default_value`
    """
    data = safe_yaml_load(filepath, default={}) or {}

    if unwrap_single_key and isinstance(data, dict) and len(data) == 1:
        data = next(iter(data.values())) or {}

    if not isinstance(data, dict):
        return data  # allow non-dict YAML (e.g., plain list)

    cleaned: Dict[str, Any] = {}
    for k, v in data.items():
        key = str(k).strip()
        if v is None:
            cleaned[key] = default_value
        elif isinstance(v, str):
            try:
                cleaned[key] = float(v) if "." in v else int(v)
            except ValueError:
                cleaned[key] = v
        else:
            cleaned[key] = v
    return cleaned


def load_recipes_from_yaml(filepath: str | Path, params, energy_int, energy_shares, energy_content):
    """Load recipes with expressions evaluated in a restricted context.

    Returns a list of Process-like dicts with `name`, `inputs`, `outputs` if
    the consuming code constructs the actual Process class elsewhere. In the
    monolith this function returns `Process` instances; for compatibility with
    existing call sites that expect objects with `.name/.inputs/.outputs`, we
    mirror that shape during the transition.
    """
    recipe_data = safe_yaml_load(filepath, default=[]) or []
    if not recipe_data:
        logger.warning("No recipes loaded from %s", filepath)

    # Context for evaluating expressions
    context = {
        **(vars(params) if isinstance(params, SimpleNamespace) else getattr(params, "__dict__", {}) ),
        "energy_int": energy_int,
        "energy_shares": energy_shares,
        "energy_content": energy_content,
    }

    recipes = []

    def _restricted_eval(expr: str, ctx: dict):
        g = {"__builtins__": {}}
        if ctx:
            for k, v in ctx.items():
                g[k] = v
        return eval(expr, g)

    for item in (recipe_data or []):
        name = (item.get("process") or "").strip()
        if not name:
            continue
        inputs: Dict[str, float] = {}
        outputs: Dict[str, float] = {}

        for mat, formula in (item.get("inputs") or {}).items():
            if isinstance(formula, str):
                try:
                    inputs[mat] = float(_restricted_eval(formula, context))
                except Exception as e:
                    logger.warning("Error evaluating input %s for %s: %s", mat, name, e)
                    inputs[mat] = 0.0
            else:
                inputs[mat] = float(formula)

        for mat, formula in (item.get("outputs") or {}).items():
            if isinstance(formula, str):
                out_ctx = {**context, "inputs": inputs}
                try:
                    outputs[mat] = float(_restricted_eval(formula, out_ctx))
                except Exception as e:
                    logger.warning("Error evaluating output %s for %s: %s", mat, name, e)
                    outputs[mat] = 0.0
            else:
                outputs[mat] = float(formula)

        # Return simple objects with the expected attributes
        recipes.append(type("_Recipe", (), {"name": name, "inputs": inputs, "outputs": outputs})())

    return recipes


def load_market_config(filepath: str | Path) -> Dict[str, Any]:
    cfg = safe_yaml_load(filepath, default=[]) or []
    try:
        return {i["name"].strip(): i["value"] for i in cfg}
    except Exception:
        logger.error("Invalid market config at %s", filepath)
        return {}


def load_electricity_intensity(filepath: str | Path) -> Dict[str, float]:
    """Return dict {ISO3: gCO2_per_MJ_electricity} from electricity_intensity.yml."""
    raw = safe_yaml_load(filepath, default={}) or {}
    items = raw.get("electricity_intensity", [])
    out: Dict[str, float] = {}
    for it in items:
        try:
            code = str(it["code"]).upper()
            val = float(it["intensity"])
            out[code] = val
        except Exception:
            pass
    return out


__all__ = [
    "safe_yaml_load",
    "load_parameters",
    "load_data_from_yaml",
    "load_recipes_from_yaml",
    "load_market_config",
    "load_electricity_intensity",
]
