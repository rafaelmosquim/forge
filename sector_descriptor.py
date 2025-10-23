"""
Sector descriptor loader for industry-agnostic configuration.

Each dataset folder (e.g. ``data/`` for steel or ``aluminum/``) provides a
``sector.yml`` file describing stage mappings, route presets, process roles,
and other knobs that were previously hard-coded for steel.

The structures defined here keep the core model generic by moving
sector-specific assumptions into data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import yaml


@dataclass(frozen=True)
class StageDefinition:
    """Single stage entry mapping an identifier to a material name."""

    id: str
    material: str
    label: Optional[str] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class StageMenuItem:
    """Stage menu entry used by the UI."""

    key: str
    stage_id: str
    label: str
    description: Optional[str] = None


@dataclass(frozen=True)
class RoutePreset:
    """Route preset describing enable/disable rules."""

    id: str
    label: str
    aliases: Set[str] = field(default_factory=set)
    disable: Set[str] = field(default_factory=set)
    enable: Set[str] = field(default_factory=set)
    feed_mode: Optional[str] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class CostingConfig:
    external_purchase_rows: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class GasConfig:
    process_gas_carrier: Optional[str] = None
    natural_gas_carrier: Optional[str] = None
    utility_process: Optional[str] = None
    reference_stage_id: Optional[str] = None
    default_direct_use_fraction: Optional[float] = None


@dataclass(frozen=True)
class SectorDescriptor:
    key: str
    name: str
    stages: Dict[str, StageDefinition] = field(default_factory=dict)
    stage_menu: List[StageMenuItem] = field(default_factory=list)
    routes: Dict[str, RoutePreset] = field(default_factory=dict)
    process_roles: Dict[str, Set[str]] = field(default_factory=dict)
    balance_fallback_materials: Set[str] = field(default_factory=set)
    prefer_internal_processes: Dict[str, str] = field(default_factory=dict)
    costing: CostingConfig = field(default_factory=CostingConfig)
    gas: GasConfig = field(default_factory=GasConfig)

    def get_stage(self, stage_id: str) -> StageDefinition:
        if stage_id not in self.stages:
            raise KeyError(f"Stage '{stage_id}' not defined for sector '{self.key}'.")
        return self.stages[stage_id]

    def find_route_by_alias(self, alias: str) -> Optional[RoutePreset]:
        alias_norm = (alias or "").strip().lower()
        for preset in self.routes.values():
            if alias_norm == preset.id.lower():
                return preset
            if alias_norm in preset.aliases:
                return preset
        return None

    def stage_menu_items(self) -> Iterable[StageMenuItem]:
        return list(self.stage_menu)


def _as_set(value) -> Set[str]:
    if not value:
        return set()
    if isinstance(value, set):
        return value
    if isinstance(value, (list, tuple)):
        return {str(v) for v in value}
    return {str(value)}


def _parse_stage(entry_id: str, payload: dict) -> StageDefinition:
    material = str(payload.get("material") or "").strip()
    if not material:
        raise ValueError(f"Stage '{entry_id}' is missing required 'material'.")
    return StageDefinition(
        id=entry_id,
        material=material,
        label=payload.get("label"),
        description=payload.get("description"),
    )


def _parse_stage_menu(items: List[dict]) -> List[StageMenuItem]:
    result: List[StageMenuItem] = []
    for item in items or []:
        stage_id = str(item.get("stage_id") or "").strip()
        key = str(item.get("key") or stage_id).strip()
        label = str(item.get("label") or stage_id)
        if not stage_id:
            raise ValueError("Stage menu item missing 'stage_id'.")
        result.append(
            StageMenuItem(
                key=key,
                stage_id=stage_id,
                label=label,
                description=item.get("description"),
            )
        )
    return result


def _parse_route(entry: dict) -> RoutePreset:
    route_id = str(entry.get("id") or "").strip()
    label = str(entry.get("label") or route_id).strip()
    if not route_id:
        raise ValueError("Route preset missing 'id'.")
    aliases = {a.strip().lower() for a in entry.get("aliases", []) if str(a).strip()}
    disable = _as_set(entry.get("disable"))
    enable = _as_set(entry.get("enable"))
    feed_mode = entry.get("feed_mode")
    return RoutePreset(
        id=route_id,
        label=label or route_id,
        aliases=aliases,
        disable=disable,
        enable=enable,
        feed_mode=feed_mode,
        description=entry.get("description"),
    )


def _parse_costing(data: dict) -> CostingConfig:
    return CostingConfig(
        external_purchase_rows=_as_set(data.get("external_purchase_rows")),
    )


def _parse_gas(data: dict) -> GasConfig:
    return GasConfig(
        process_gas_carrier=data.get("process_gas_carrier"),
        natural_gas_carrier=data.get("natural_gas_carrier"),
        utility_process=data.get("utility_process"),
        reference_stage_id=data.get("reference_stage_id"),
        default_direct_use_fraction=data.get("default_direct_use_fraction"),
    )


def load_sector_descriptor(data_dir: str | Path) -> SectorDescriptor:
    """
    Load ``sector.yml`` from a dataset directory.

    Args:
        data_dir: Path to dataset folder that contains ``sector.yml``.
    """
    data_path = Path(data_dir)
    descriptor_path = data_path / "sector.yml"
    if not descriptor_path.exists():
        raise FileNotFoundError(f"Sector descriptor not found: {descriptor_path}")

    with descriptor_path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}

    key = str(payload.get("sector_key") or "").strip() or data_path.name
    name = str(payload.get("sector_name") or key).strip()

    stages_raw = payload.get("stages") or {}
    stages = {sid: _parse_stage(sid, entry or {}) for sid, entry in stages_raw.items()}

    stage_menu = _parse_stage_menu(payload.get("stage_menu") or [])

    routes: Dict[str, RoutePreset] = {}
    for entry in payload.get("routes") or []:
        preset = _parse_route(entry or {})
        routes[preset.id] = preset

    process_roles = {
        proc: _as_set(roles)
        for proc, roles in (payload.get("process_roles") or {}).items()
    }

    costing = _parse_costing(payload.get("costing") or {})
    gas = _parse_gas(payload.get("gas") or {})
    fallbacks = _as_set(payload.get("balance_fallback_materials"))
    prefer_internal = {
        str(k): str(v)
        for k, v in (payload.get("prefer_internal_processes") or {}).items()
        if str(k).strip() and str(v).strip()
    }

    return SectorDescriptor(
        key=key,
        name=name,
        stages=stages,
        stage_menu=stage_menu,
        routes=routes,
        process_roles=process_roles,
        balance_fallback_materials=fallbacks,
        prefer_internal_processes=prefer_internal,
        costing=costing,
        gas=gas,
    )
