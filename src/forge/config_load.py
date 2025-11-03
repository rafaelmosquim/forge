"""
Configuration loading utilities for Forge application.
Handles loading parameters, recipes, market configurations,
and electricity intensity data from YAML files.
"""

import json
import logging
from types import SimpleNamespace
import yaml
from .models import Process

logger = logging.getLogger(__name__)
# ===================================================================
#                         Configuration Loaders
# ===================================================================
def load_parameters(filepath):
    logger = logging.getLogger(__name__)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            params_dict = yaml.safe_load(f) or {}
            # Convert to SimpleNamespace via JSON round-trip for nested dicts
            params = json.loads(json.dumps(params_dict),
                        object_hook=lambda d: SimpleNamespace(**d))
            logger.debug("Loaded parameters YAML from %s", filepath)
            logger.debug("process_gas: %s", getattr(params, 'process_gas', 'NOT FOUND'))
            return params
    except FileNotFoundError:
        logger.error("Parameters file not found: %s", filepath)
        return SimpleNamespace()


def safe_yaml_load(filepath, default=None):
    """Safe YAML loader: returns default when file missing or invalid."""
    logger = logging.getLogger(__name__)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or default
    except FileNotFoundError:
        logger.warning("YAML file not found: %s — returning default", filepath)
        return default
    except Exception as e:
        logger.error("Error reading YAML %s: %s", filepath, e)
        return default

def load_data_from_yaml(filepath, default_value=0, unwrap_single_key=True):
    data = safe_yaml_load(filepath, default={}) or {}

    # only unwrap when requested (for legacy files like {"processes": {...}})
    if unwrap_single_key and isinstance(data, dict) and len(data) == 1:
        data = next(iter(data.values())) or {}

    cleaned = {}
    if not isinstance(data, dict):
        return data  # allow non-dict YAML (e.g., plain list/string)

    for k, v in data.items():
        key = str(k).strip()
        if v is None:
            cleaned[key] = default_value
        elif isinstance(v, str):
            try:
                cleaned[key] = float(v) if '.' in v else int(v)
            except ValueError:
                cleaned[key] = v
        else:
            cleaned[key] = v
    return cleaned

def load_recipes_from_yaml(filepath, params, energy_int, energy_shares, energy_content):
    recipe_data = safe_yaml_load(filepath, default=[]) or []
    if not recipe_data:
        logging.getLogger(__name__).warning("No recipes loaded from %s", filepath)

    # Context for evaluating expressions
    context = {**vars(params),
               'energy_int': energy_int,
               'energy_shares': energy_shares,
               'energy_content': energy_content}

    recipes = []
    for item in recipe_data:
        name = (item.get('process') or '').strip()
        if not name:
            continue

        # Restricted eval: use eval but with no builtins to reduce risk while preserving expressions
        def _restricted_eval(expr: str, ctx: dict):
            # Prepare globals mapping with empty builtins, then inject ctx as names
            g = {"__builtins__": {}}
            # Copy ctx entries into globals so names resolve as before
            if ctx:
                for k, v in ctx.items():
                    g[k] = v
            # Evaluate expression using restricted globals
            return eval(expr, g)

        # Evaluate inputs
        inputs, outputs = {}, {}
        for mat, formula in (item.get('inputs') or {}).items():
            if isinstance(formula, str):
                try:
                    inputs[mat] = float(_restricted_eval(formula, context))
                except Exception as e:
                    logger.warning("Error evaluating input %s for %s: %s", mat, name, e)
                    inputs[mat] = 0.0
            else:
                inputs[mat] = float(formula)

        # Evaluate outputs (can reference 'inputs')
        for mat, formula in (item.get('outputs') or {}).items():
            if isinstance(formula, str):
                out_ctx = {**context, 'inputs': inputs}
                try:
                    outputs[mat] = float(_restricted_eval(formula, out_ctx))
                except Exception as e:
                    logger.warning("Error evaluating output %s for %s: %s", mat, name, e)
                    outputs[mat] = 0.0
            else:
                outputs[mat] = float(formula)

        recipes.append(Process(name, inputs, outputs))

    return recipes

def load_market_config(filepath):
    cfg = safe_yaml_load(filepath, default=[]) or []
    try:
        return {i['name'].strip(): i['value'] for i in cfg}
    except Exception:
        logging.getLogger(__name__).error("Invalid market config at %s", filepath)
        return {}

def load_electricity_intensity(filepath):
    """Return dict {ISO3: gCO2_per_MJ_electricity} from electricity_intensity.yml."""
    raw = safe_yaml_load(filepath, default={}) or {}
    items = raw.get('electricity_intensity', [])
    out = {}
    for it in items:
        try:
            code = str(it['code']).upper()
            val  = float(it['intensity'])
            out[code] = val
        except Exception:
            pass
    return out
