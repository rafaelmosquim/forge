"""Core computations and transforms.

This module starts as wrappers around the legacy monolith and incrementally
duplicates selected functions so callers can depend on `forge.core.compute`
without the monolith. Where we provide local implementations, they take
precedence; otherwise we delegate to `forge.steel_model_core`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import pandas as pd
from .models import Process
from .routing import STAGE_MATS, build_route_mask
from .engine import calculate_balance_matrix, calculate_energy_balance

# -----------------------
# Local implementations
# -----------------------


def expand_energy_tables_for_active(active_names, energy_shares, energy_int):
    """Copy base energy rows to variant names like 'Continuous Casting (R)'."""
    def base(n: str) -> str:
        return n.split(" (")[0]
    # removes any suffix like " (R)" to let model continue working

    for n in list(active_names or []):
        b = base(n)
        if n not in energy_shares and b in energy_shares:
            energy_shares[n] = dict(energy_shares[b])
        if n not in energy_int and b in energy_int:
            energy_int[n] = energy_int[b]


def derive_energy_shares(recipes: List[Process], energy_content: Dict[str, float]):
    """Derive energy carrier shares per process from recipe inputs.

    Electricity is taken directly (MJ units). Fuels are converted using
    lower heating values provided in `energy_content`.
    Returns a mapping: { process_name: { carrier: share, ... }, ... }
    """
    shares: Dict[str, Dict[str, float]] = {}
    for proc in recipes:
        MJ_by_carrier: Dict[str, float] = {}
        for c, amt in proc.inputs.items():
            if c == 'Electricity':
                MJ_by_carrier[c] = float(amt)
            elif c in energy_content:
                MJ_by_carrier[c] = float(amt) * float(energy_content[c])
        total = sum(MJ_by_carrier.values())
        if total > 1e-12:
            shares[proc.name] = {c: mj / total for c, mj in MJ_by_carrier.items()}
        else:
            shares[proc.name] = {}
    return shares


from .costs import analyze_energy_costs, analyze_material_costs
from .transforms import (
    apply_fuel_substitutions,
    apply_dict_overrides,
    apply_recipe_overrides,
)

# Transforms/overrides (gas routing implemented locally)

# Reference helpers (local implementations)

DEFAULT_PRODUCER_PRIORITY = (
    # Core downstream/defaults
    "Continuous Casting (R)",
    "Hot Rolling",
    "Cold Rolling",
    "Basic Oxygen Furnace",
    "Electric Arc Furnace",
    # Additional priorities aligned with API legacy to ensure deterministic picks
    "Bypass Raw→IP3",
    "Bypass CR→IP3",
    "Nitrogen Production",
    "Oxygen Production",
    "Dolomite Production",
    "Burnt Lime Production",
    "Coke Production",
    "Natural gas from Market",
    "LPG from Market",
    "Biomethane from Market",
    "Hydrogen (Methane reforming) from Market",
    "Hydrogen (Electrolysis) from Market",
)


def _build_producers_index(recipes: List[Process]) -> Dict[str, List[Process]]:
    prod: Dict[str, List[Process]] = {}
    for r in recipes:
        for m in r.outputs:
            prod.setdefault(m, []).append(r)
    return prod


def _build_routes_from_picks(
    recipes: List[Process],
    demand_mat: str,
    picks_by_material: Dict[str, str],
    pre_mask: Optional[Dict[str, int]] = None,
    pre_select: Optional[Dict[str, int]] = None,
    fallback_materials: Optional[Set[str]] = None,
) -> Dict[str, int]:
    producers = _build_producers_index(recipes)
    pre_mask = dict(pre_mask or {})
    pre_select = dict(pre_select or {})
    chosen: Dict[str, int] = {}

    def score(proc: Process):
        try:
            idx = DEFAULT_PRODUCER_PRIORITY.index(proc.name)
            return (0, idx, proc.name)
        except ValueError:
            return (1, 0, proc.name)

    from collections import deque
    q = deque([demand_mat])
    visited_mats: Set[str] = {demand_mat}

    while q:
        mat = q.popleft(); visited_mats.discard(mat)
        cand_all = producers.get(mat, [])
        if not cand_all:
            continue
        allowed = [p for p in cand_all if pre_mask.get(p.name, 1) > 0]
        if not allowed:
            continue
        pick_name = picks_by_material.get(mat, "")
        pick = next((p for p in allowed if p.name == pick_name), None) if pick_name else None
        if pick is None:
            enabled = [p for p in allowed if pre_select.get(p.name, 1) > 0] or allowed
            enabled.sort(key=score)
            pick = enabled[0]
        chosen[pick.name] = 1
        for r in cand_all:
            if r.name != pick.name:
                chosen[r.name] = 0
        for im in pick.inputs.keys():
            if im not in visited_mats:
                q.append(im); visited_mats.add(im)
    return chosen


def _ensure_fallback_processes(
    recipes: List[Process],
    production_routes: Dict[str, int],
    fallback_materials: Optional[Set[str]],
) -> None:
    if not fallback_materials:
        return
    fallback_set = {str(m).strip() for m in fallback_materials if str(m).strip()}
    existing = {r.name for r in recipes}
    for mat in fallback_set:
        proc_name = f"External {mat} (auto)"
        if proc_name not in existing:
            recipes.append(Process(proc_name, inputs={}, outputs={mat: 1.0}))
            existing.add(proc_name)
        production_routes.setdefault(proc_name, 1)


__all__ = [
    # calcs
    "expand_energy_tables_for_active",
    "derive_energy_shares",
    "analyze_energy_costs",
    "analyze_material_costs",
    # transforms
    "apply_fuel_substitutions",
    "apply_dict_overrides",
    "apply_recipe_overrides",
]
