"""Routing constants and helpers (duplicated from monolith)."""
from __future__ import annotations

import logging
from typing import Dict, Iterable

logger = logging.getLogger(__name__)


STAGE_MATS = {
    "Finished": "Finished Products",
    "IP4": "Manufactured Feed (IP4)",
    "IP3": "Intermediate Process 3",
    "Raw": "Raw Products (types)",
    "Cast": "Cast Steel (IP1)",
    "Liquid": "Liquid Steel",
    "PigIron": "Pig Iron",
    "GradeR": "Liquid Steel R",
    "GradeL": "Liquid Steel L",
    "GradeH": "Liquid Steel H",
    "PigIronExit": "Pig Iron (Exit)",
    "IngotExit": "Ingot (Exit)",
    "DirectExit": "Basic Steel (Exit)",
}


try:
    # Delegate to monolith if available to preserve behavior
    from forge.steel_model_core import set_prefer_internal_processes as _mon_set_prefer
except Exception:  # pragma: no cover
    _mon_set_prefer = None


def set_prefer_internal_processes(mapping: dict | None) -> None:
    if _mon_set_prefer is not None:
        return _mon_set_prefer(mapping)
    return None


try:
    from forge.steel_model_core import apply_inhouse_clamp as _mon_inhouse_clamp
except Exception:  # pragma: no cover
    _mon_inhouse_clamp = None


def apply_inhouse_clamp(pre_select: dict | None, pre_mask: dict | None, prefer_map: dict | None = None):
    if _mon_inhouse_clamp is not None:
        return _mon_inhouse_clamp(pre_select, pre_mask, prefer_map)
    return dict(pre_select or {}), dict(pre_mask or {})


def build_route_mask(route_name, recipes):
    """Return a pre-mask {process: 0/1} to restrict upstream based on route.

    Keeps downstream shaping/coating available; only clamps upstream.
    """
    ban = set()
    if route_name == 'EAF-Scrap':
        ban = {'Blast Furnace', 'Basic Oxygen Furnace', 'Direct Reduction Iron'}
    elif route_name == 'DRI-EAF':
        ban = {'Blast Furnace'}
    elif route_name == 'BF-BOF':
        ban = {'Direct Reduction Iron', 'Electric Arc Furnace'}
    elif route_name == 'External':
        ban = {
            'Blast Furnace', 'Basic Oxygen Furnace', 'Direct Reduction Iron', 'Electric Arc Furnace',
            'Coke Production', 'Charcoal Production', 'Sintering', 'Pelletizing'
        }
    ban |= {"Ingot Casting", "Direct use of Basic Steel Products (IP4)"}
    return {r.name: (0 if r.name in ban else 1) for r in recipes}


def _first_existing(candidates: Iterable[str], pool: Iterable[str]):
    for k in candidates:
        if k in pool:
            return k
    return None


def enforce_eaf_feed(recipes, mode: str | None):
    """Force EAF to one feed: 'scrap' | 'dri' | 'pigiron'. Safe no-op otherwise."""
    if not mode:
        return
    eaf = next((r for r in recipes if r.name == "Electric Arc Furnace"), None)
    if not eaf:
        logger.warning("enforce_eaf_feed: EAF recipe not found — skipping.")
        return
    all_mats = set()
    for r in recipes:
        all_mats.update(r.outputs.keys())
    if mode == "scrap":
        want = _first_existing(("Scrap", "Scrap Steel"), all_mats)
    elif mode == "dri":
        want = _first_existing(("Direct Reduced Iron", "DRI", "HBI"), all_mats)
    elif mode == "pigiron":
        want = _first_existing(("Pig Iron", "Hot Metal"), all_mats)
    else:
        logger.warning("enforce_eaf_feed: unknown mode '%s' — skipping.", mode)
        return
    if not want:
        logger.warning("enforce_eaf_feed: no matching material for mode '%s' — skipping.", mode)
        return
    feed_keys = {"Pig Iron", "Hot Metal", "Direct Reduced Iron", "DRI", "HBI", "Scrap", "Scrap Steel"}
    non_feed = {k: v for k, v in eaf.inputs.items() if k not in feed_keys}
    eaf.inputs = {**non_feed, want: 1.0}
    logger.info("EAF feed forced to '%s' (%s)", want, mode)


__all__ = [
    "STAGE_MATS",
    "build_route_mask",
    "enforce_eaf_feed",
    "set_prefer_internal_processes",
    "apply_inhouse_clamp",
]
