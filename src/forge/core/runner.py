"""Core runner: deterministic compute given a resolved scenario.

This module provides a single entrypoint that accepts a fully-specified
scenario (no UI heuristics, no file I/O), runs the core calculations
(balance → energy → gas-routing/credits → emissions → optional LCI), and
returns structured results.

The API layer is responsible for loading YAML, applying user overrides,
resolving route picks/masks, and shaping a CoreScenario for execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Iterable
from functools import partial
import inspect

import pandas as pd

from .models import Process
from .routing import STAGE_MATS, build_route_mask
from .engine import (
    calculate_balance_matrix,
    calculate_energy_balance,
    calculate_emissions,
)
from .compute import (
    expand_energy_tables_for_active,
    calculate_internal_electricity,
    apply_gas_routing_and_credits,
    calculate_lci,
)
from .costs import analyze_energy_costs, analyze_material_costs


@dataclass
class CoreScenario:
    """Resolved inputs the core needs to compute results.

    Required:
      - recipes: complete list with expressions already evaluated
      - energy_int: per-run energy intensity (MJ)
      - energy_shares: energy carrier shares by process
      - energy_content: carrier lower heating values (MJ/unit)
      - params: object (namespace) with any tuning fields used by compute
      - production_routes: {process: 0/1} fully resolved mask
      - demand_material: final demanded material name
      - demand_qty: demand quantity (kg) for demand_material
      - mkt_cfg: market config (used by emissions and outside-mill detection)
      - energy_efs: {carrier: gCO2e/MJ}
      - process_efs: {process: kgCO2e/t output}

    Optional features:
      - route_preset, stage_ref: keys for reference helpers (gas, electricity)
      - gas_config, gas_routing: gas routing inputs for credits/blending
      - fallback_materials: which materials are treated as external-only
      - allow_direct_onsite: chemistry whitelist to allow direct emissions onsite
      - outside_mill_procs: processes treated as outside the mill (grid power only)
      - enable_lci: whether to compute LCI outputs (default False)
    """
    # Core data
    recipes: List[Process]
    energy_int: Dict[str, float]
    energy_shares: Dict[str, Dict[str, float]]
    energy_content: Dict[str, float]
    params: Any
    production_routes: Dict[str, int]
    demand_material: str
    demand_qty: float
    mkt_cfg: Dict[str, Any]
    energy_efs: Dict[str, float]
    process_efs: Dict[str, float]

    # Optional configuration
    route_preset: str = "auto"
    stage_ref: str = "IP3"
    gas_config: Dict[str, Any] = field(default_factory=dict)
    gas_routing: Dict[str, Any] = field(default_factory=dict)
    fallback_materials: Set[str] | None = None
    allow_direct_onsite: Set[str] | None = None
    outside_mill_procs: Set[str] | None = None
    enable_lci: bool = False
    # Optional cost inputs
    energy_prices: Optional[Dict[str, float]] = None
    material_prices: Optional[Dict[str, float]] = None
    external_purchase_rows: Optional[Iterable[str]] = None


@dataclass
class CoreResults:
    production_routes: Dict[str, int]
    prod_levels: Dict[str, float]
    balance_matrix: pd.DataFrame
    energy_balance: pd.DataFrame
    emissions: Optional[pd.DataFrame]
    total_co2e_kg: Optional[float]
    lci: Optional[pd.DataFrame] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    energy_efs_out: Dict[str, float] = field(default_factory=dict)
    total_cost: Optional[float] = None
    material_cost: Optional[float] = None


def _robust_call_calculate_emissions(calc_fn, **kwargs):
    """Call calculate_emissions with only the parameters it accepts."""
    sig = inspect.signature(calc_fn)
    usable = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return calc_fn(**usable)


def _compute_inside_gas_reference_for_share(
    *,
    recipes: List[Process],
    energy_int: Dict[str, float],
    energy_shares: Dict[str, Dict[str, float]],
    energy_content: Dict[str, float],
    params: Any,
    route_key: str,
    demand_qty: float,
    stage_ref: str = "IP3",
    stage_lookup: Optional[Dict[str, str]] = None,
    gas_carrier: str = "Gas",
    fallback_materials: Optional[Set[str]] = None,
) -> float:
    """Plant-level gas consumption for the entire route to a reference stage."""
    pre_mask = build_route_mask(route_key, recipes)
    stage_map = stage_lookup or STAGE_MATS
    if stage_ref not in stage_map:
        return 0.0
    demand_mat = stage_map[stage_ref]

    # Deterministic build to the reference material
    production_routes_full = {r.name: pre_mask.get(r.name, 1) for r in recipes}
    final_demand_full = {demand_mat: demand_qty}

    import copy as _copy
    recipes_full = _copy.deepcopy(recipes)
    bm_full, prod_full = calculate_balance_matrix(
        recipes_full, final_demand_full, production_routes_full
    )
    if bm_full is None:
        return 0.0
    energy_balance_full = calculate_energy_balance(prod_full, energy_int, energy_shares)
    if gas_carrier in energy_balance_full.columns:
        return float(energy_balance_full[gas_carrier].sum())
    return 0.0


def run_core_scenario(scn: CoreScenario) -> CoreResults:
    """Execute the deterministic compute for a resolved scenario."""
    # 1) Balance
    final_demand = {scn.demand_material: float(scn.demand_qty)}
    balance_matrix, prod_levels = calculate_balance_matrix(
        scn.recipes, final_demand, scn.production_routes
    )
    if balance_matrix is None:
        return CoreResults(
            production_routes=scn.production_routes,
            prod_levels={},
            balance_matrix=pd.DataFrame(),
            energy_balance=pd.DataFrame(),
            emissions=None,
            total_co2e_kg=None,
            lci=None,
            meta={"error": "Material balance failed"},
        )

    # 2) Energy balance and internal electricity
    active = [p for p, r in prod_levels.items() if r > 1e-9]
    expand_energy_tables_for_active(active, scn.energy_shares, scn.energy_int)
    recipes_dict = {r.name: r for r in scn.recipes}
    internal_elec = calculate_internal_electricity(prod_levels, recipes_dict, scn.params)
    energy_balance = calculate_energy_balance(prod_levels, scn.energy_int, scn.energy_shares)

    # 3) Gas routing + credit, EF blending
    gas_reference_fn = partial(
        _compute_inside_gas_reference_for_share,
        stage_lookup=STAGE_MATS,
        fallback_materials=scn.fallback_materials or set(),
    )
    eb_adj, e_efs, gas_meta = apply_gas_routing_and_credits(
        energy_balance=energy_balance,
        recipes=scn.recipes,
        prod_levels=prod_levels,
        params=scn.params,
        energy_shares=scn.energy_shares,
        energy_int=scn.energy_int,
        energy_content=scn.energy_content,
        e_efs=scn.energy_efs,
        scenario={
            "gas_routing": scn.gas_routing,
            "route_preset": scn.route_preset,
            "demand_qty": scn.demand_qty,
            "stage_ref": scn.stage_ref,
            "gas_config": scn.gas_config,
            "fallback_materials": list(scn.fallback_materials or []),
        },
        credit_on=True,
        compute_inside_gas_reference_fn=gas_reference_fn,
    )

    # Build a variant of energy balance for emissions: use BF base intensity
    eb_for_emissions = eb_adj.copy()
    try:
        if "Blast Furnace" in eb_for_emissions.index:
            bf_runs = float(prod_levels.get("Blast Furnace", 0.0))
            bf_base = float(getattr(scn.params, "bf_base_intensity", scn.energy_int.get("Blast Furnace", 0.0)))
            bf_sh = scn.energy_shares.get("Blast Furnace", {}) or {}
            for carrier in eb_for_emissions.columns:
                if carrier == "Electricity":
                    continue
                share = float(bf_sh.get(carrier, 0.0))
                eb_for_emissions.loc["Blast Furnace", carrier] = bf_runs * bf_base * share
            if "TOTAL" in eb_for_emissions.index:
                eb_for_emissions.loc["TOTAL"] = eb_for_emissions.drop(index="TOTAL").sum()
    except Exception:
        eb_for_emissions = eb_adj

    # 4) Emissions (robust signature)
    emissions = _robust_call_calculate_emissions(
        calculate_emissions,
        mkt_cfg=scn.mkt_cfg,
        prod_lvl=prod_levels,
        prod_level=prod_levels,
        energy_balance=eb_for_emissions,
        energy_df=eb_for_emissions,
        e_efs=e_efs,
        energy_efs=e_efs,
        process_emissions_table=scn.process_efs,
        process_efs=scn.process_efs,
        internal_elec=internal_elec,
        final_demand=final_demand,
        total_gas_MJ=gas_meta.get("total_process_gas_MJ", 0.0),
        EF_process_gas=gas_meta.get("EF_process_gas", 0.0),
        internal_fraction_plant=gas_meta.get("f_internal", 0.0),
        ef_internal_electricity=gas_meta.get("ef_internal_electricity", 0.0),
        outside_mill_procs=scn.outside_mill_procs,
        allow_direct_onsite=scn.allow_direct_onsite,
    )

    total_co2 = None
    try:
        if emissions is not None and not emissions.empty:
            if "TOTAL" not in emissions.index and "TOTAL CO2e" in emissions.columns:
                emissions.loc["TOTAL"] = emissions.sum()
            if "TOTAL" in emissions.index and "TOTAL CO2e" in emissions.columns:
                total_co2 = float(emissions.loc["TOTAL", "TOTAL CO2e"]) * 1000.0
            elif "TOTAL CO2e" in emissions.columns:
                total_co2 = float(emissions["TOTAL CO2e"].sum()) * 1000.0
    except Exception:
        total_co2 = None

    # 5) Optional LCI
    lci_df = None
    if scn.enable_lci:
        try:
            lci_df = calculate_lci(
                prod_level=prod_levels,
                recipes=scn.recipes,
                energy_balance=eb_adj,
                electricity_internal_fraction=float(gas_meta.get("f_internal", 0.0)),
                gas_internal_fraction=float(gas_meta.get("f_internal_gas", 0.0)),
                natural_gas_carrier=str(gas_meta.get("natural_gas_carrier", "Gas")),
                process_gas_carrier=str(gas_meta.get("process_gas_carrier", "Process Gas")),
            )
        except Exception:
            lci_df = None

    meta = {
        "route_preset": scn.route_preset,
        "stage_ref": scn.stage_ref,
        **gas_meta,
    }

    # 6) Optional costs
    total_cost = None
    material_cost = None
    try:
        if scn.energy_prices:
            total_cost = analyze_energy_costs(eb_adj, scn.energy_prices)
    except Exception:
        total_cost = None
    try:
        if scn.material_prices:
            material_cost = analyze_material_costs(
                balance_matrix, scn.material_prices, external_rows=scn.external_purchase_rows
            )
    except Exception:
        material_cost = None

    return CoreResults(
        production_routes=scn.production_routes,
        prod_levels=prod_levels,
        balance_matrix=balance_matrix,
        energy_balance=eb_adj,
        emissions=emissions,
        total_co2e_kg=total_co2,
        lci=lci_df,
        meta=meta,
        energy_efs_out=e_efs,
        total_cost=total_cost,
        material_cost=material_cost,
    )


__all__ = [
    "CoreScenario",
    "CoreResults",
    "run_core_scenario",
]
