"""Scenario builder: shape resolved inputs into a CoreScenario."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import copy as _copy

from forge.canonical.core.models import Process
from forge.canonical.core.runner import CoreScenario
from forge.canonical.core.routing import (
    build_route_mask,
    enforce_eaf_feed,
    apply_inhouse_clamp,
)
from forge.canonical.core.compute import (
    _build_routes_from_picks,
    _ensure_fallback_processes,
)
from forge.canonical.descriptor.scenario_resolver import (
    build_route_mask_for_descriptor,
    resolve_feed_mode,
)


def _apply_validation_overrides(
    picks_by_material: Dict[str, str], pre_select_soft: Dict[str, int]
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


def _apply_route_overrides(
    pre_select: Dict[str, int],
    pre_mask: Dict[str, int],
    overrides: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    if not overrides:
        return pre_select, pre_mask
    ps = dict(pre_select or {})
    pm = dict(pre_mask or {})
    for proc, raw in overrides.items():
        key = str(proc)
        try:
            val = float(raw)
        except Exception:
            val = 1.0 if bool(raw) else 0.0
        enabled = 1 if val > 0.0 else 0
        ps[key] = enabled
        if enabled:
            if pm.get(key, 1) == 0:
                pm.pop(key, None)
        else:
            pm[key] = 0
    return ps, pm


def _infer_eaf_mode(route_preset: str) -> Optional[str]:
    return {
        'EAF-Scrap': 'scrap',
        'DRI-EAF': 'dri',
        'BF-BOF': None,
        'External': None,
        'auto': None,
    }.get(route_preset, None)


@dataclass
class ScenarioBuildResult:
    core: CoreScenario
    prefer_internal_map: Dict[str, List[str]]
    fallback_materials: Set[str]


def build_core_scenario(
    *,
    descriptor,
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
    gas_config: Optional[Dict[str, Any]] = None,
    gas_routing: Optional[Dict[str, Any]] = None,
    process_efs: Optional[Dict[str, float]] = None,
    external_purchase_rows: Optional[Iterable[str]] = None,
    energy_prices: Optional[Dict[str, float]] = None,
    material_prices: Optional[Dict[str, float]] = None,
    outside_mill_procs: Optional[Set[str]] = None,
    allow_direct_onsite: Optional[Iterable[str]] = None,
    material_credit_map: Optional[Dict[str, Any]] = None,
) -> ScenarioBuildResult:
    fallback_set = set(fallback_materials or set())
    is_validation = (str(stage_role or '').strip().lower() == 'validation')

    picks = dict(picks_by_material or {})
    ps_soft = dict(pre_select_soft or {})

    if is_validation:
        picks, ps_soft = _apply_validation_overrides(picks, ps_soft)
        pre_mask = {
            'Nitrogen Production': 0,
            'Oxygen Production': 0,
            'Dolomite Production': 0,
            'Burnt Lime Production': 0,
        }
        prefer_internal_map = {
            'Nitrogen Production': ['Nitrogen from market'],
            'Oxygen Production': ['Oxygen from market'],
            'Dolomite Production': ['Dolomite from market'],
            'Burnt Lime Production': ['Burnt Lime from market'],
        }
    else:
        pre_mask = (
            build_route_mask_for_descriptor(descriptor, route_preset, recipes)
            or build_route_mask(route_preset, recipes)
        )
        prefer_internal_map: Dict[str, List[str]] = {}
        raw_prefer_internal = descriptor.prefer_internal_processes or {}
        for market_proc, internal_proc in raw_prefer_internal.items():
            if internal_proc:
                prefer_internal_map.setdefault(str(internal_proc), []).append(str(market_proc))
        ps_soft, pre_mask = apply_inhouse_clamp(ps_soft, pre_mask, prefer_internal_map)

    ps_soft, pre_mask = _apply_route_overrides(ps_soft, pre_mask, route_overrides)

    recipes_calc = _copy.deepcopy(recipes)
    eaf_mode = resolve_feed_mode(descriptor, route_preset)
    if eaf_mode is None:
        eaf_mode = _infer_eaf_mode(route_preset)
    enforce_eaf_feed(recipes_calc, eaf_mode)

    production_routes = _build_routes_from_picks(
        recipes_calc,
        demand_material,
        picks,
        pre_mask=pre_mask,
        pre_select=ps_soft,
        fallback_materials=fallback_set,
    )
    _ensure_fallback_processes(recipes_calc, production_routes, fallback_set)

    allow_direct_set = None
    if allow_direct_onsite:
        allow_direct_set = {
            str(proc).strip()
            for proc in allow_direct_onsite
            if isinstance(proc, str) and str(proc).strip()
        } or None

    credit_map_clean: Dict[str, Tuple[str, float]] = {}
    for source, raw_spec in (material_credit_map or {}).items():
        try:
            src_name = str(source).strip()
        except Exception:
            src_name = ""
        if not src_name:
            continue
        target_name = None
        ratio_val = 1.0
        if isinstance(raw_spec, str):
            target_name = raw_spec.strip()
        elif isinstance(raw_spec, dict):
            target_name = str(
                raw_spec.get("target")
                or raw_spec.get("to")
                or raw_spec.get("material")
                or raw_spec.get("dest")
                or raw_spec.get("destination")
                or ""
            ).strip()
            ratio_raw = raw_spec.get("ratio", raw_spec.get("factor", raw_spec.get("multiplier", 1.0)))
            try:
                ratio_val = float(ratio_raw)
            except Exception:
                ratio_val = 1.0
        elif isinstance(raw_spec, (tuple, list)) and raw_spec:
            target_name = str(raw_spec[0]).strip()
            if len(raw_spec) > 1:
                try:
                    ratio_val = float(raw_spec[1])
                except Exception:
                    ratio_val = 1.0
        if not target_name:
            continue
        credit_map_clean[src_name] = (target_name, ratio_val)

    core_scn = CoreScenario(
        recipes=recipes_calc,
        energy_int=energy_int,
        energy_shares=energy_shares,
        energy_content=energy_content,
        params=params,
        production_routes=production_routes,
        demand_material=demand_material,
        demand_qty=float(demand_qty),
        mkt_cfg={},
        energy_efs=e_efs,
        process_efs=process_efs or {},
        route_preset=route_preset,
        stage_ref=(descriptor.gas.reference_stage_id or 'IP3') if descriptor else 'IP3',
        gas_config=gas_config or {},
        gas_routing=gas_routing or {},
        fallback_materials=fallback_set,
        allow_direct_onsite=allow_direct_set,
        outside_mill_procs=outside_mill_procs or set(),
        material_credit_map=credit_map_clean or None,
        energy_prices=energy_prices,
        material_prices=material_prices,
        external_purchase_rows=list(external_purchase_rows or []),
    )

    return ScenarioBuildResult(
        core=core_scn,
        prefer_internal_map=prefer_internal_map,
        fallback_materials=fallback_set,
    )


__all__ = [
    'build_core_scenario',
    'ScenarioBuildResult',
]
