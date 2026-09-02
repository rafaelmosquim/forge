"""Scenario-specific transforms shared across entry points."""
from __future__ import annotations

from typing import Dict, Any


def apply_dri_mix(energy_shares: Dict[str, Dict[str, float]], scenario: Dict[str, Any]) -> None:
    mix_name = (scenario.get('dri_mix') or '').strip()
    if not mix_name:
        return
    try:
        year = int(scenario.get('snapshot_year', 2030))
    except Exception:
        year = 2030

    default_defs = {
        'Blue': {
            2030: {'Gas': 1.0},
            2040: {'Gas': 0.70, 'Biomethane': 0.10, 'Green hydrogen': 0.20},
        },
        'Green': {
            2030: {'Gas': 0.70, 'Biomethane': 0.10, 'Green hydrogen': 0.20},
            2040: {'Gas': 0.40, 'Biomethane': 0.20, 'Green hydrogen': 0.40},
        },
    }
    defs = scenario.get('dri_mix_definitions') or default_defs

    plan = None
    options = defs.get(mix_name) if isinstance(defs, dict) else None
    if isinstance(options, dict):
        try:
            keys = sorted({int(k) for k in options.keys()})
        except Exception:
            keys = []
        chosen = None
        for k in keys:
            if k <= year:
                chosen = k
        if chosen is None and keys:
            chosen = min(keys)
        if chosen is not None:
            plan = options.get(str(chosen), options.get(chosen))
    if not isinstance(plan, dict):
        return

    es = energy_shares.get('Direct Reduction Iron')
    if not isinstance(es, dict):
        return
    try:
        base_gas = float(es.get('Gas', 0.0) or 0.0)
    except Exception:
        base_gas = 0.0
    if base_gas <= 0:
        return

    def _as_frac(x):
        try:
            v = float(x)
            return v / 100.0 if v > 1.0 else v
        except Exception:
            return 0.0

    carriers = ('Gas', 'Biomethane', 'Green hydrogen')
    fractions = {str(k): _as_frac(v) for k, v in plan.items()}

    # The mix is a SUBSTITUTION of the DRI unit's gas demand, not an addition to
    # it. Only the share the plan leaves unassigned stays as gas; everything the
    # plan assigns to another carrier must leave the gas line, or the unit ends
    # up burning its full gas demand PLUS the substitute and cleaner fuels come
    # out dirtier than the one they replace.
    allocated = sum(float(fractions.get(c, 0.0) or 0.0) for c in carriers)
    for carrier in carriers:
        frac = float(fractions.get(carrier, 0.0) or 0.0)
        if carrier == 'Gas':
            es['Gas'] = base_gas * frac
        else:
            add_val = base_gas * frac
            if add_val > 0:
                es[carrier] = es.get(carrier, 0.0) + add_val
    unallocated = base_gas * max(0.0, 1.0 - allocated)
    if unallocated > 0:
        es['Gas'] = es.get('Gas', 0.0) + unallocated


def apply_charcoal_expansion(energy_shares: Dict[str, Dict[str, float]], scenario: Dict[str, Any]) -> None:
    mode = (scenario.get('charcoal_expansion') or '').strip().lower()
    if mode not in {'expansion'}:
        return
    try:
        year = int(scenario.get('snapshot_year', 2030))
    except Exception:
        year = 2030
    schedule = scenario.get('charcoal_share_schedule') or {}
    frac = None
    if isinstance(schedule, dict):
        try:
            years = sorted({int(k) for k in schedule.keys()})
        except Exception:
            years = []
        chosen = None
        for y in years:
            if y <= year:
                chosen = y
        if chosen is None and years:
            chosen = min(years)
        if chosen is not None:
            raw = schedule.get(str(chosen), schedule.get(chosen))
            try:
                val = float(raw)
                frac = val / 100.0 if val > 1.0 else val
            except Exception:
                frac = None
    if frac is None:
        return
    frac = max(0.0, min(1.0, float(frac)))
    es_bf = energy_shares.get('Blast Furnace')
    if not isinstance(es_bf, dict):
        return
    coal = float(es_bf.get('Coal', 0.0) or 0.0)
    coke = float(es_bf.get('Coke', 0.0) or 0.0)
    if (coal + coke) <= 0:
        return
    new_coal = coal * (1.0 - frac)
    new_coke = coke * (1.0 - frac)
    moved = (coal - new_coal) + (coke - new_coke)
    es_bf['Coal'] = new_coal
    es_bf['Coke'] = new_coke
    es_bf['Charcoal'] = es_bf.get('Charcoal', 0.0) + moved

