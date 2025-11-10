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
    'apply_energy_int_efficiency_scaling',
    'apply_energy_int_floor',
]


def apply_energy_int_efficiency_scaling(energy_int: Dict[str, float], scenario: Dict[str, object]) -> None:
    """Apply uniform and scheduled efficiency scaling to energy intensities.

    Recognized keys in `scenario` (all optional):
      - energy_int_factor: float multiplier applied to all numeric intensities
      - energy_int_schedule: mapping with one or more of:
            rate_pct_per_year | annual_pct | rate_pct | rate
            baseline_year (default 2023)
            target_year   (default 2050)
            max_year      (default 2050)
            years         (directly specify years)

    Non-numeric entries and zeros are left untouched.
    """
    if not isinstance(energy_int, dict) or not isinstance(scenario, dict):
        return

    try:
        factor = float(scenario.get('energy_int_factor', 1.0))
    except Exception:
        factor = 1.0

    sched = scenario.get('energy_int_schedule') or scenario.get('efficiency_schedule')
    if isinstance(sched, dict):
        try:
            rate = sched.get('rate_pct_per_year', sched.get('annual_pct', sched.get('rate_pct', sched.get('rate'))))
            rate = float(rate) if rate is not None else None
        except Exception:
            rate = None
        if rate is not None:
            years = None
            try:
                y_raw = sched.get('years')
                years = int(y_raw) if y_raw is not None else None
            except Exception:
                years = None
            if years is None:
                try:
                    baseline = int(sched.get('baseline_year', 2023))
                except Exception:
                    baseline = 2023
                try:
                    tgt = int(sched.get('target_year', 2050))
                except Exception:
                    tgt = 2050
                try:
                    cap = int(sched.get('max_year', 2050))
                except Exception:
                    cap = 2050
                tgt = min(tgt, cap)
                years = max(0, tgt - baseline)
            annual = max(0.0, 1.0 - float(rate) / 100.0)
            factor *= (annual ** int(years))

    if abs(factor - 1.0) < 1e-12:
        return

    for k, v in list(energy_int.items()):
        try:
            val = float(v)
        except Exception:
            continue
        if val == 0.0:
            continue
        energy_int[k] = val * factor


def apply_energy_int_floor(energy_int: Dict[str, float], scenario: Dict[str, object]) -> None:
    """Apply per-process minimum intensity floors after schedule scaling.

    Scenario may include::
        energy_int_floor: { "Blast Furnace": 11.0, ... }
    which enforces energy_int[proc] = max(floor, current) for numeric entries.
    """
    if not isinstance(energy_int, dict) or not isinstance(scenario, dict):
        return
    floors = scenario.get('energy_int_floor')
    if not isinstance(floors, dict):
        return
    for k, v in floors.items():
        try:
            floor_val = float(v)
        except Exception:
            continue
        try:
            cur = float(energy_int.get(k, 0.0) or 0.0)
        except Exception:
            cur = 0.0
        energy_int[k] = max(cur, floor_val)
