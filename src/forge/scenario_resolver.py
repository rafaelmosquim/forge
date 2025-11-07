"""
Shared helpers to translate sector descriptors into runtime-friendly
structures for scenario execution.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

try:
    from forge.core.routing import STAGE_MATS  # preferred path
except Exception:  # pragma: no cover
    try:
        from forge.steel_model_core import STAGE_MATS  # fallback to monolith
    except Exception:
        STAGE_MATS: Dict[str, str] = {}

from forge.sector_descriptor import SectorDescriptor, RoutePreset


def build_stage_material_map(
    descriptor: Optional[SectorDescriptor],
) -> Dict[str, str]:
    """
    Combine descriptor-defined stages with legacy ``STAGE_MATS`` fallback.
    """
    stage_map: Dict[str, str] = {}
    if descriptor is not None:
        stage_map.update({
            stage_id: stage.material
            for stage_id, stage in descriptor.stages.items()
        })
    stage_map.update(STAGE_MATS)
    return stage_map


def resolve_stage_material(
    descriptor: Optional[SectorDescriptor],
    stage_key: str,
) -> str:
    stage_map = build_stage_material_map(descriptor)
    return stage_map.get(stage_key, stage_key)


def reference_stage_for_gas(descriptor: Optional[SectorDescriptor]) -> str:
    if descriptor and descriptor.gas.reference_stage_id:
        return descriptor.gas.reference_stage_id
    return "IP3"


def _route_aliases(preset: RoutePreset) -> Iterable[str]:
    aliases = {preset.id.lower(), *preset.aliases}
    for alias in aliases:
        if not alias:
            continue
        yield alias.lower()
        yield alias.replace(" ", "-")
        yield alias.replace(" ", "_")
        yield alias.replace("-", "_")


def match_route(
    descriptor: Optional[SectorDescriptor],
    token: Optional[str],
) -> Optional[str]:
    if descriptor is None or not token:
        return None
    token_norm = token.strip().lower()
    for preset in descriptor.routes.values():
        if token_norm == preset.id.lower():
            return preset.id
        if token_norm in preset.aliases:
            return preset.id
    return None


def match_route_in_name(
    descriptor: Optional[SectorDescriptor],
    name: Optional[str],
) -> Optional[str]:
    if descriptor is None or not name:
        return None
    name_lower = name.lower()
    for preset in descriptor.routes.values():
        if any(alias in name_lower for alias in _route_aliases(preset)):
            return preset.id
    return None


def build_route_mask_for_descriptor(
    descriptor: Optional[SectorDescriptor],
    route_id: str,
    recipes,
) -> Optional[Dict[str, int]]:
    if descriptor is None:
        return None
    route = descriptor.routes.get(route_id)
    if route is None:
        return None
    mask = {getattr(r, "name", str(r)): 1 for r in recipes}
    for proc in route.disable:
        mask[proc] = 0
    for proc in route.enable:
        mask[proc] = 1
    return mask


_DEFAULT_FEED_MODES = {
    "EAF-Scrap": "scrap",
    "DRI-EAF": "dri",
    "BF-BOF": None,
    "External": None,
    "auto": None,
}


def resolve_feed_mode(
    descriptor: Optional[SectorDescriptor],
    route_id: str,
) -> Optional[str]:
    if descriptor:
        route = descriptor.routes.get(route_id)
        if route:
            feed = route.feed_mode
            if isinstance(feed, str):
                feed = feed.strip().lower()
                if feed in {"", "none", "null"}:
                    return None
                return feed
            if feed is not None:
                return feed
    return _DEFAULT_FEED_MODES.get(route_id)
