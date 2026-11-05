"""Core runner: deterministic compute given a resolved scenario.

This module provides a single entrypoint that accepts a fully-specified
scenario (no UI heuristics, no file I/O), runs the core calculations
(balance → energy → gas-routing/credits → emissions), and returns structured
results.

The API layer is responsible for loading YAML, applying user overrides,
resolving route picks/masks, and shaping a CoreScenario for execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Iterable, Tuple
from functools import partial
import inspect
import os

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
)
from .gas import (
    calculate_internal_electricity,
    apply_gas_routing_and_credits,
    compute_inside_energy_reference_for_share,
)
from .costs import analyze_energy_costs, analyze_material_costs


def _env_flag_truthy(var_name: str) -> bool:
    try:
        raw = os.environ.get(var_name, "")
    except Exception:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _costs_enabled() -> bool:
    return _env_flag_truthy("FORGE_ENABLE_COSTS")


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
    material_credit_map: Dict[str, Tuple[str, float]] | None = None
    # Optional cost inputs
    energy_prices: Optional[Dict[str, float]] = None
    material_prices: Optional[Dict[str, float]] = None
    external_purchase_rows: Optional[Iterable[str]] = None
    costed_elsewhere: Optional[Dict[str, str]] = None


@dataclass
class CoreResults:
    production_routes: Dict[str, int]
    prod_levels: Dict[str, float]
    balance_matrix: pd.DataFrame
    energy_balance: pd.DataFrame
    emissions: Optional[pd.DataFrame]
    total_co2e_kg: Optional[float]
    meta: Dict[str, Any] = field(default_factory=dict)
    energy_efs_out: Dict[str, float] = field(default_factory=dict)
    total_cost: Optional[float] = None
    material_cost: Optional[float] = None


def _robust_call_calculate_emissions(calc_fn, **kwargs):
    """Call calculate_emissions with only the parameters it accepts."""
    sig = inspect.signature(calc_fn)
    usable = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return calc_fn(**usable)


def _process_output_emissions_from_recipes(recipes: List[Process], prod_levels: Dict[str, float]) -> Dict[str, float]:
    """
    Compute direct emissions from explicit gas outputs in recipes (kg CO2e).
    Currently targets aluminum electrolysis processes with CO2/CO/Other outputs.
    """
    totals: Dict[str, float] = {}
    targets = {"Electrolysis (prebaked)", "Electrolysis (Soderberg)"}
    for proc in recipes:
        if proc.name not in targets:
            continue
        outs = getattr(proc, "outputs", {}) or {}
        def _get(key: str) -> float:
            try:
                return float(outs.get(key, 0.0) or 0.0)
            except Exception:
                return 0.0
        co2 = _get("CO2")
        co = _get("CO")
        other = _get("Other")
        factor = co2 + 0.7 * co + other
        if abs(factor) <= 1e-12:
            continue
        runs = float(prod_levels.get(proc.name, 0.0))
        if runs > 1e-12:
            # factor is per run (per kg of output); convert to kg and scale by runs
            totals[proc.name] = runs * factor * 1000.0
    return totals


def _conversion_emissions(
    recipes: List[Process],
    prod_levels: Dict[str, float],
    energy_content: Dict[str, float],
    energy_efs: Dict[str, float],
    processes: Iterable[str],
    exclude_carriers: Iterable[str],
) -> tuple:
    """Net emissions (g) of fuel-conversion processes: fuel in minus fuel out.

    A coke plant turns coking coal (a recipe input, never on the energy
    balance) into coke (a recipe output, charged where it burns). Charging the
    coal at its life-cycle EF and crediting the coke at its own EF keeps the
    coal's carbon -- and its upstream emissions -- in the inventory exactly
    once. Recovered gas is excluded here; it is credited by the gas logic.
    """
    recipes_dict = {r.name: r for r in recipes}
    skip = set(exclude_carriers or [])
    grams: Dict[str, float] = {}
    diag: Dict[str, Dict[str, float]] = {}

    def _fuel_side(flows: Dict[str, Any]) -> tuple:
        mj = g = 0.0
        for carrier, qty in (flows or {}).items():
            if carrier in skip:
                continue
            try:
                q = float(qty or 0.0)
                lhv = float(energy_content.get(carrier, 0.0) or 0.0)
                ef = float(energy_efs.get(carrier, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if q > 0 and lhv > 0 and ef > 0:
                mj += q * lhv
                g += q * lhv * ef
        return mj, g

    for name in processes or []:
        runs = float(prod_levels.get(name, 0.0) or 0.0)
        proc = recipes_dict.get(name)
        if runs <= 1e-12 or proc is None:
            continue
        feed_mj, feed_g = _fuel_side(proc.inputs)
        prod_mj, prod_g = _fuel_side(proc.outputs)
        grams[name] = runs * (feed_g - prod_g)
        diag[name] = {"feed_MJ": runs * feed_mj, "product_MJ": runs * prod_mj, "net_g": grams[name]}
    return grams, diag


def run_core_scenario(scn: CoreScenario) -> CoreResults:
    """Execute the deterministic compute for a resolved scenario."""
    # 1) Balance
    final_demand = {scn.demand_material: float(scn.demand_qty)}
    balance_matrix, prod_levels = calculate_balance_matrix(
        scn.recipes, final_demand, scn.production_routes, scn.material_credit_map
    )
    if balance_matrix is None:
        return CoreResults(
            production_routes=scn.production_routes,
            prod_levels={},
            balance_matrix=pd.DataFrame(),
            energy_balance=pd.DataFrame(),
            emissions=None,
            total_co2e_kg=None,
            meta={"error": "Material balance failed"},
        )

    # 2) Energy balance and internal electricity
    active = [p for p, r in prod_levels.items() if r > 1e-9]
    expand_energy_tables_for_active(active, scn.energy_shares, scn.energy_int)
    recipes_dict = {r.name: r for r in scn.recipes}
    internal_elec = calculate_internal_electricity(prod_levels, recipes_dict, scn.params)
    energy_balance = calculate_energy_balance(prod_levels, scn.energy_int, scn.energy_shares)

    # 3) Gas routing + credit, EF blending
    reference_fn = partial(
        compute_inside_energy_reference_for_share,
        stage_lookup=STAGE_MATS,
        fallback_materials=scn.fallback_materials or set(),
        material_credit_map=scn.material_credit_map,
        exclude_processes=set(scn.outside_mill_procs or ()),
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
            "outside_mill_procs": list(scn.outside_mill_procs or []),
        },
        compute_inside_reference_fn=reference_fn,
    )

    eb_for_emissions = eb_adj.copy()

    process_output_emissions = _process_output_emissions_from_recipes(scn.recipes, prod_levels)

    gas_cfg = scn.gas_config or {}
    conversion_procs = list(gas_cfg.get("conversion_processes") or [])
    gas_carriers = {gas_cfg.get("process_gas_carrier") or "Process Gas"}
    gas_carriers.update((gas_cfg.get("process_gases") or {}).keys())
    gas_carriers.update(
        str(s.get("carrier")) for s in (gas_cfg.get("process_gas_sources") or []) if s.get("carrier")
    )
    conversion_emissions, conversion_diag = _conversion_emissions(
        scn.recipes, prod_levels, scn.energy_content, e_efs, conversion_procs, gas_carriers
    )
    # A coke plant whose coal is not charged must not be credited for its gas.
    gas_credit_exempt = {"Coke Production"} - set(conversion_procs)

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
        gas_credit_details=gas_meta.get("gas_credit_details", {}),
        gas_credit_exempt=gas_credit_exempt,
        flared_gas_MJ=gas_meta.get("flared_gas_MJ", 0.0),
        conversion_emissions=conversion_emissions,
        internal_fraction_plant=gas_meta.get("f_internal", 0.0),
        ef_internal_electricity=gas_meta.get("ef_internal_electricity", 0.0),
        outside_mill_procs=scn.outside_mill_procs,
        allow_direct_onsite=scn.allow_direct_onsite,
        process_output_emissions=process_output_emissions,
    )
    gas_meta = dict(gas_meta)
    gas_meta["conversion"] = conversion_diag
    gas_meta["conversion_feed_MJ"] = float(sum(d["feed_MJ"] for d in conversion_diag.values()))
    gas_meta["conversion_product_MJ"] = float(sum(d["product_MJ"] for d in conversion_diag.values()))

    total_co2 = None
    try:
        if emissions is not None and not emissions.empty:
            if "TOTAL" not in emissions.index and "TOTAL CO2e" in emissions.columns:
                emissions.loc["TOTAL"] = emissions.sum()
            if "TOTAL" in emissions.index and "TOTAL CO2e" in emissions.columns:
                total_co2 = float(emissions.loc["TOTAL", "TOTAL CO2e"])  # already kg
            elif "TOTAL CO2e" in emissions.columns:
                total_co2 = float(emissions["TOTAL CO2e"].sum())  # already kg
    except Exception:
        total_co2 = None

    meta = {
        "route_preset": scn.route_preset,
        "stage_ref": scn.stage_ref,
        **gas_meta,
    }

    # 5) Optional costs
    total_cost = None
    material_cost = None
    if _costs_enabled():
        try:
            if scn.energy_prices:
                total_cost = analyze_energy_costs(eb_adj, scn.energy_prices)
        except Exception:
            total_cost = None
        try:
            if scn.material_prices:
                material_cost = analyze_material_costs(
                    balance_matrix, scn.material_prices, external_rows=scn.external_purchase_rows,
                    costed_elsewhere=scn.costed_elsewhere,
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
