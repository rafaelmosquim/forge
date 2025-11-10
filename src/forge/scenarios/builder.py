"""Scenario builder: shape inputs into a CoreScenario for execution.

No file I/O here. Accepts already-loaded tables and a descriptor, applies
scenario semantics (stage role, validation rules, route masks, in-house
preferences, EAF feed, route overrides, fallback injection), and returns a
CoreScenario ready for `forge.core.runner.run_core_scenario`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import copy as _copy

from forge.core.models import Process
from forge.core.routing import (
    build_route_mask,
    enforce_eaf_feed,
    apply_inhouse_clamp,
)
from forge.core.compute import (
    _build_routes_from_picks,
    _ensure_fallback_processes,
)
from forge.core.runner import CoreScenario
from forge.descriptor.scenario_resolver import (
    build_route_mask_for_descriptor,
    resolve_feed_mode,
)


def _apply_validation_overrides(
    *, picks_by_material: Dict[str, str], pre_select_soft: Dict[str, int]
) -> Tuple[Dict[str, str], Dict[str, int]]:
    picks = dict(picks_by_material or {})
    ps = dict(pre_select_soft or {})
    picks.update({
        'Nitrogen': 'Nitrogen from market',
        'Oxygen': 'Oxygen from market',
        'Dolomite': 'Dolomite from market',
        'Burnt Lime': 'Burnt Lime from market',
    })
    ps.update({
        'Nitrogen Production': 0,
        'Oxygen Production': 0,
        'Dolomite Production': 0,
        'Burnt Lime Production': 0,
        'Nitrogen from market': 1,
        'Oxygen from market': 1,
        'Dolomite from market': 1,
        'Burnt Lime from market': 1,
    })
    return picks, ps


def build_core_scenario(
    *,
    descriptor,
    stage_key: str,
    stage_role: Optional[str],
    route_preset: str,
    demand_qty: float,
    demand_material: str,
    recipes: List[Process],
    energy_int: Dict[str, float],
    energy_shares: Dict[str, Dict[str, float]],
    energy_content: Dict[str, float],
    e_efs: Dict[str, float],
    params: Any,
    picks_by_material: Optional[Dict[str, str]] = None,
    pre_select_soft: Optional[Dict[str, int]] = None,
    route_overrides: Optional[Dict[str, Any]] = None,
    fallback_materials: Optional[Set[str]] = None,
    stage_lookup: Optional[Dict[str, str]] = None,
    gas_config: Optional[Dict[str, Any]] = None,
    process_efs: Optional[Dict[str, float]] = None,
    external_purchase_rows: Optional[Iterable[str]] = None,
    energy_prices: Optional[Dict[str, float]] = None,
    material_prices: Optional[Dict[str, float]] = None,
    outside_mill_procs: Optional[Set[str]] = None,
    enable_lci: bool = False,
) -> CoreScenario:
    is_validation = (str(stage_role or '').strip().lower() == 'validation')

    picks = dict(picks_by_material or {})
    ps_soft = dict(pre_select_soft or {})

    if is_validation:
        picks, ps_soft = _apply_validation_overrides(picks_by_material=picks, pre_select_soft=ps_soft)

    # Build route pre-mask
    pre_mask = (
        build_route_mask_for_descriptor(descriptor, route_preset, recipes)
        or build_route_mask(route_preset, recipes)
    )

    # Apply in-house clamp for non-validation stages
    if not is_validation:
        prefer_internal = {}
        raw = descriptor.prefer_internal_processes or {}
        for market_proc, internal_proc in raw.items():
            if internal_proc:
                prefer_internal.setdefault(str(internal_proc), []).append(str(market_proc))
        ps_soft, pre_mask = apply_inhouse_clamp(ps_soft, pre_mask, prefer_internal)

    # EAF feed mode
    eaf_mode = resolve_feed_mode(descriptor, route_preset)

    # Prepare recipes list for calculation and enforce EAF feed if requested
    recipes_calc = _copy.deepcopy(recipes)
    enforce_eaf_feed(recipes_calc, eaf_mode)

    # Route overrides (soft enable/disable)
    if route_overrides:
        for proc, raw in route_overrides.items():
            key = str(proc)
            try:
                val = float(raw)
            except Exception:
                val = 1.0 if bool(raw) else 0.0
            enabled = 1 if val > 0.0 else 0
            ps_soft[key] = enabled
            if not enabled:
                pre_mask[key] = 0
            elif pre_mask.get(key, 1) == 0:
                pre_mask.pop(key, None)

    # Build production routes
    production_routes = _build_routes_from_picks(
        recipes_calc,
        demand_material,
        picks,
        pre_mask=pre_mask,
        pre_select=ps_soft,
        fallback_materials=fallback_materials,
    )

    _ensure_fallback_processes(recipes_calc, production_routes, fallback_materials)

    # Assemble CoreScenario
    core_scn = CoreScenario(
        recipes=recipes_calc,
        energy_int=energy_int,
        energy_shares=energy_shares,
        energy_content=energy_content,
        params=params,
        production_routes=production_routes,
        demand_material=demand_material,
        demand_qty=float(demand_qty),
        mkt_cfg={},  # prefer explicit outside_mill_procs over inferring from mkt_cfg
        energy_efs=e_efs,
        process_efs=process_efs or {},
        route_preset=route_preset,
        stage_ref=(descriptor.gas.reference_stage_id or 'IP3') if descriptor else 'IP3',
        gas_config=gas_config or {},
        gas_routing=(gas_config and gas_config.get('gas_routing', {})) or {},
        fallback_materials=fallback_materials or set(),
        allow_direct_onsite=None,
        outside_mill_procs=outside_mill_procs or set(),
        enable_lci=enable_lci,
        energy_prices=energy_prices,
        material_prices=material_prices,
        external_purchase_rows=list(external_purchase_rows or []),
    )
    return core_scn


__all__ = [
    'build_core_scenario',
]
