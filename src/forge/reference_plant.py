"""
Reference plant primitives used by gas routing and recovery.

The Streamlit app lets users move the accounting boundary, but emission factors
for electricity and process gas remain anchored to a deterministic reference
plant. This module provides the lightweight dataclasses used to represent those
metrics so they can be cached or persisted independent of scenario choices.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Optional

from forge.steel_model_core import (
    STAGE_MATS,
    build_route_mask,
    build_pre_for_route,
    build_routes_interactive,
    calculate_balance_matrix,
    calculate_energy_balance,
    compute_fixed_plant_elec_model,
    expand_energy_tables_for_active,
)


@dataclass(frozen=True)
class ReferencePlantConfig:
    """Declarative description of the fixed reference chain."""

    route_preset: str = "BF-BOF"
    stage_ref: str = "IP3"
    demand_qty: float = 1000.0


@dataclass(frozen=True)
class ReferencePlantMetrics:
    """
    Anchors for emission factor blending.

    Attributes:
        inside_elec_ref: Total in-mill electricity (MJ) for the reference chain.
        total_gas_consumption: Total process-gas consumption (MJ) for the reference chain.
        util_efficiency: Utility plant efficiency (MJ electricity produced per MJ gas).
    """

    inside_elec_ref: float
    total_gas_consumption: float
    util_efficiency: float


DEFAULT_REFERENCE_CONFIG = ReferencePlantConfig()


def make_reference_metrics(
    inside_elec_ref: float,
    total_gas_consumption: float,
    util_efficiency: Optional[float] = None,
) -> ReferencePlantMetrics:
    """
    Helper used by callers that pre-compute the reference metrics elsewhere
    (e.g., during a preprocessing step or cached on disk).
    """
    util_eff = 0.0 if util_efficiency is None else float(util_efficiency)
    return ReferencePlantMetrics(
        inside_elec_ref=float(inside_elec_ref),
        total_gas_consumption=float(total_gas_consumption),
        util_efficiency=util_eff,
    )


def compute_inside_elec_reference_for_share(
    recipes,
    energy_int,
    energy_shares,
    energy_content,
    params,
    route_key: str,
    demand_qty: float,
    stage_ref: str = "IP3",
) -> float:
    """
    Return the fixed in-mill electricity demand (MJ) for the deterministic reference chain.
    """
    inside_elec_ref, _, _ = compute_fixed_plant_elec_model(
        recipes,
        energy_int,
        energy_shares,
        energy_content,
        params,
        route_key=route_key,
        demand_qty=demand_qty,
        stage_ref=stage_ref,
    )
    return float(inside_elec_ref)


def compute_inside_gas_reference_for_share(
    recipes,
    energy_int,
    energy_shares,
    energy_content,
    params,
    route_key: str,
    demand_qty: float,
    stage_ref: str = "IP3",
    stage_lookup=None,
    gas_carrier=None,
    fallback_materials=None,
    **_,
) -> float:
    """
    Compute total plant-level gas consumption (MJ) for the deterministic reference chain.
    """
    # Build copies to avoid mutating live objects
    recipes_ref = copy.deepcopy(recipes)
    energy_int_ref = dict(energy_int)
    energy_shares_ref = {k: dict(v) for k, v in energy_shares.items()}

    pre_mask_ref = build_route_mask(route_key, recipes_ref)
    try:
        pre_select_ref, pre_mask_from_prebuilder, _ = build_pre_for_route(route_key)
        if pre_mask_from_prebuilder:
            pre_mask_ref.update(pre_mask_from_prebuilder)
    except Exception:
        pre_select_ref = {}

    demand_mat_ref = STAGE_MATS[stage_ref]
    final_demand_ref = {demand_mat_ref: float(demand_qty)}
    production_routes_ref = build_routes_interactive(
        recipes_ref,
        demand_mat_ref,
        pre_select=pre_select_ref,
        pre_mask=pre_mask_ref,
        interactive=False,
    )

    balance_ref, prod_ref = calculate_balance_matrix(recipes_ref, final_demand_ref, production_routes_ref)
    if balance_ref is None:
        return 0.0

    active_procs_ref = [proc for proc, runs in prod_ref.items() if runs > 1e-9]
    expand_energy_tables_for_active(active_procs_ref, energy_shares_ref, energy_int_ref)

    energy_ref = calculate_energy_balance(prod_ref, energy_int_ref, energy_shares_ref)
    if "Gas" not in energy_ref.columns:
        return 0.0
    rows = [r for r in energy_ref.index if r not in ("TOTAL", "Utility Plant")]
    return float(energy_ref.loc[rows, "Gas"].clip(lower=0).sum())
