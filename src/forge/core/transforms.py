"""Scenario transforms and override helpers.

Duplicated from the monolith so they can be imported without pulling the
entire legacy module.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)


def apply_fuel_substitutions(
    sub_map: Dict[str, str],
    energy_shares: Dict[str, Dict[str, float]],
    energy_int: Dict[str, float],
    energy_content: Dict[str, float],
    emission_factors: Dict[str, float],
    recipes: Optional[Iterable] = None,
):
    """Re-map energy shares for carriers and optionally rename recipe IO.

    Only shares (and optionally recipe IO names) are affected; intensities and
    EFs are left to other transforms.
    """
    for old, new in (sub_map or {}).items():
        if old == new:
            continue
        for _proc, shares in energy_shares.items():
            if old in shares:
                old_val = shares.get(old) or 0.0
                shares[new] = (shares.get(new) or 0.0) + old_val
                shares[old] = 0.0
        if recipes is not None:
            for r in recipes:
                if old in r.inputs:
                    r.inputs[new] = r.inputs.pop(old)
                # Keep Coke Production output as process gas
                if r.name != 'Coke Production' and old in r.outputs:
                    r.outputs[new] = r.outputs.pop(old)


def apply_dict_overrides(target_dict: Dict, overrides: Optional[Dict]) -> None:
    """Shallow update helper for energy_int, content, efs, etc."""
    target_dict.update(overrides or {})


def apply_recipe_overrides(
    recipes: Iterable,
    overrides: Optional[Dict[str, Dict]],
    params: SimpleNamespace,
    energy_int: Dict[str, float],
    energy_shares: Dict[str, Dict[str, float]],
    energy_content: Dict[str, float],
):
    if not overrides:
        return list(recipes)
    by_name = {r.name: r for r in recipes}
    base_ctx = {
        **(vars(params) if isinstance(params, SimpleNamespace) else {}),
        'energy_int': energy_int,
        'energy_shares': energy_shares,
        'energy_content': energy_content,
    }
    for name, spec in overrides.items():
        r = by_name.get(name)
        if not r:
            continue
        if 'inputs' in spec:
            new_in = {}
            for k, v in (spec.get('inputs') or {}).items():
                new_in[k] = eval(v, {"__builtins__": {}}, base_ctx) if isinstance(v, str) else v
            r.inputs = new_in
        if 'outputs' in spec:
            out_ctx = {**base_ctx, 'inputs': r.inputs}
            new_out = {}
            for k, v in (spec.get('outputs') or {}).items():
                new_out[k] = eval(v, {"__builtins__": {}}, out_ctx) if isinstance(v, str) else v
            r.outputs = new_out
    return list(by_name.values())


def adjust_blast_furnace_intensity(energy_int, energy_shares, params):
    """Scale BF intensity and store base/adjusted in params.

    Top-gas available = adjusted_intensity – base_intensity.
    """
    pg = getattr(params, 'process_gas', 0.0)
    if 'Blast Furnace' not in energy_int:
        return

    base = float(energy_int['Blast Furnace'])
    params.bf_base_intensity = base

    shares = energy_shares.get('Blast Furnace', {}) or {}
    carriers = ['Gas', 'Coal', 'Coke', 'Charcoal']
    S = sum(float(shares.get(c, 0.0) or 0.0) for c in carriers)
    denom = max(1e-9, 1 - float(pg) * S)

    adj = base / denom
    energy_int['Blast Furnace'] = adj
    params.bf_adj_intensity = adj
    logger.info("Adjusted BF intensity: %0.2f -> %0.2f MJ/t steel (recovering %0.1f%% of carriers)", base, adj, float(pg)*100)


def adjust_process_gas_intensity(proc_name, param_key, energy_int, energy_shares, params):
    pg = getattr(params, param_key, 0.0)
    if proc_name not in energy_int or float(pg) <= 0:
        return
    base = float(energy_int[proc_name])
    safe = proc_name.replace(' ', '_').lower()
    setattr(params, f"{safe}_base_intensity", base)

    shares = energy_shares.get(proc_name, {}) or {}
    S = sum(float(shares.get(c, 0.0) or 0.0) for c in ['Gas', 'Coal', 'Coke', 'Charcoal'])
    denom = max(1e-9, 1 - float(pg) * S)
    adj = base / denom
    energy_int[proc_name] = adj
    setattr(params, f"{safe}_adj_intensity", adj)
    logger.info("Adjusted %s: %0.2f -> %0.2f MJ/run", proc_name, base, adj)


__all__ = [
    'apply_fuel_substitutions',
    'apply_dict_overrides',
    'apply_recipe_overrides',
    'adjust_blast_furnace_intensity',
    'adjust_process_gas_intensity',
]

