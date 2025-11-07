"""Routing constants and helpers (duplicated from monolith)."""
from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, Optional

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


INHOUSE_FORCE = {
    # Note: Nitrogen intentionally left market by default; others prefer in-house
    "Oxygen Production": "Oxygen from market",
    "Dolomite Production": "Dolomite from market",
    "Burnt Lime Production": "Burnt Lime from market",
    "Coke Production": [
        "Coke from market",
        "Coke Mineral from Market",
        "Coke Petroleum from Market",
    ],
}

PREFER_INTERNAL_OVERRIDE: Dict[str, object] = {}


def set_prefer_internal_processes(mapping: dict | None) -> None:
    """Override default in-house preference mapping (descriptor-driven)."""
    global PREFER_INTERNAL_OVERRIDE
    if not mapping:
        PREFER_INTERNAL_OVERRIDE = {}
        return
    converted: Dict[str, object] = {}
    for k, v in mapping.items():
        key = str(k)
        if isinstance(v, (list, tuple, set)):
            converted[key] = [str(item) for item in v]
        else:
            converted[key] = str(v)
    PREFER_INTERNAL_OVERRIDE = converted


def apply_inhouse_clamp(
    pre_select: Optional[dict], pre_mask: Optional[dict], prefer_map: Optional[dict] = None
):
    """Prefer in-house production and optionally force market in validation stage."""
    ps = dict(pre_select or {})
    pm = dict(pre_mask or {})

    stage = os.environ.get('STEEL_MODEL_STAGE', '')
    if stage == 'validation':
        # Force auxiliaries to be market-purchased
        aux_rules = {
            "Nitrogen Production": ("Nitrogen from market", 0),
            "Oxygen Production": ("Oxygen from market", 0),
            "Dolomite Production": ("Dolomite from market", 0),
            "Burnt Lime Production": ("Burnt Lime from market", 0),
        }
        for prod_proc, (market_proc, _) in aux_rules.items():
            pm[prod_proc] = 0
            ps[market_proc] = 1
        return ps, pm

    mapping = prefer_map or PREFER_INTERNAL_OVERRIDE or INHOUSE_FORCE
    for prod_proc, market_proc in mapping.items():
        ps[prod_proc] = 1
        targets = list(market_proc) if isinstance(market_proc, (list, tuple, set)) else [market_proc]
        for target in targets:
            if not target:
                continue
            if pm.get(target) == 0:
                pm.pop(target, None)
            if target not in ps:
                ps[target] = 1
    return ps, pm


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
