"""
Gas routing and credit application helpers.

The public entry point is intentionally side-effect free: callers provide the
current energy balance along with scenario metadata and (optionally) callbacks
that return reference-plant metrics. This keeps the logic decoupled from the
core route-building code while preserving the existing accounting behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Dict, Iterable, Optional

import pandas as pd

from forge.models import Process

ReferenceFn = Callable[..., float]


def _blend_ef(shares: Dict[str, float], emission_factors: Dict[str, float]) -> float:
    """Blend energy-carrier emission factors, skipping electricity."""
    fuels = [(carrier, share) for carrier, share in shares.items() if carrier != "Electricity" and share > 0]
    if not fuels:
        return 0.0
    denominator = sum(share for _, share in fuels) or 1e-12
    return sum(share * float(emission_factors.get(carrier, 0.0)) for carrier, share in fuels) / denominator


def _roles_for(process_roles: Dict[str, Iterable[str]], proc_name: str) -> set[str]:
    """Normalise descriptor-defined process roles to lowercase strings."""
    roles = process_roles.get(proc_name, set())
    if isinstance(roles, dict):
        iterable = roles.keys()
    elif isinstance(roles, (list, tuple, set)):
        iterable = roles
    elif roles:
        iterable = [roles]
    else:
        iterable = []
    return {str(role).lower() for role in iterable}


def apply_gas_routing_and_credits(
    energy_balance: pd.DataFrame,
    recipes: Iterable[Process],
    prod_levels: Dict[str, float],
    params: SimpleNamespace,
    energy_shares: Dict[str, Dict[str, float]],
    energy_int: Dict[str, float],
    energy_content: Dict[str, Dict[str, float]],
    e_efs: Dict[str, float],
    scenario: Dict[str, object],
    credit_on: bool,
    compute_inside_gas_reference_fn: Optional[ReferenceFn] = None,
    compute_inside_elec_reference_fn: Optional[ReferenceFn] = None,
) -> tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Adjust the energy balance, blend emission factors, and return diagnostics.

    Args:
        energy_balance: Current energy balance dataframe.
        recipes / prod_levels: Active recipes and production runs.
        params / energy_*: Scenario-adjusted inputs.
        e_efs: Emission factor mapping (a defensive copy is returned).
        scenario: Metadata assembled by the API (gas routing, descriptor info, etc.).
        credit_on: Whether to apply process-gas → electricity credits.
        compute_inside_*: Optional callbacks for fixed reference-plant metrics.
    """
    recipes_dict = {recipe.name: recipe for recipe in recipes}
    gas_config = scenario.get("gas_config", {}) or {}
    process_roles = scenario.get("process_roles", {}) or {}
    fallback_materials = set(scenario.get("fallback_materials", []))

    process_gas_carrier = gas_config.get("process_gas_carrier") or "Process Gas"
    natural_gas_carrier = gas_config.get("natural_gas_carrier") or "Gas"
    utility_process_name = gas_config.get("utility_process") or "Utility Plant"

    # ---- Process-gas volumes -------------------------------------------------------------
    coke_recipe = recipes_dict.get("Coke Production")
    gas_coke_MJ = (
        float(prod_levels.get("Coke Production", 0.0)) *
        float((coke_recipe.outputs.get(process_gas_carrier, 0.0) if coke_recipe else 0.0))
    )

    try:
        bf_adj = float(getattr(params, "bf_adj_intensity", 0.0))
        bf_base = float(getattr(params, "bf_base_intensity", 0.0))
        gas_bf_MJ = (bf_adj - bf_base) * float(prod_levels.get("Blast Furnace", 0.0))
    except Exception:
        gas_bf_MJ = 0.0

    total_process_gas_MJ = float(gas_coke_MJ + gas_bf_MJ)

    EF_coke_gas = _blend_ef(energy_shares.get("Coke Production", {}), e_efs)
    EF_bf_gas = _blend_ef(energy_shares.get("Blast Furnace", {}), e_efs)
    if total_process_gas_MJ <= 1e-9:
        EF_process_gas = float(e_efs.get(process_gas_carrier, 0.0))
    else:
        EF_process_gas = (
            (EF_coke_gas * (gas_coke_MJ / max(1e-12, total_process_gas_MJ))) +
            (EF_bf_gas * (gas_bf_MJ / max(1e-12, total_process_gas_MJ)))
        )

    gas_source_names = gas_config.get("gas_sources")
    if not gas_source_names:
        gas_source_names = [
            name for name in recipes_dict
            if "gas_source" in _roles_for(process_roles, name)
        ]
    gas_source_names = list(dict.fromkeys(gas_source_names))

    gas_sources_MJ = 0.0
    ef_weighted = 0.0
    weight_sum = 0.0
    gas_source_details: Dict[str, float] = {}
    for src in gas_source_names:
        proc = recipes_dict.get(src)
        if not proc:
            continue
        gas_output = float(proc.outputs.get(process_gas_carrier, 0.0) or 0.0)
        if gas_output <= 0:
            continue
        runs = float(prod_levels.get(src, 0.0) or 0.0)
        contribution = runs * gas_output
        if contribution <= 1e-12:
            continue
        gas_sources_MJ += contribution
        gas_source_details[src] = contribution
        shares = energy_shares.get(src, {})
        ef_source = _blend_ef(shares, e_efs) or float(e_efs.get(process_gas_carrier, 0.0))
        ef_weighted += ef_source * contribution
        weight_sum += contribution

    if gas_sources_MJ > 0 and abs(gas_sources_MJ - total_process_gas_MJ) <= 1e-6:
        total_process_gas_MJ = float(gas_sources_MJ)
        EF_process_gas = (ef_weighted / weight_sum) if weight_sum > 0 else float(e_efs.get(process_gas_carrier, 0.0))
        gas_coke_MJ = float(gas_source_details.get("Coke Production", gas_coke_MJ))
        gas_bf_MJ = float(gas_source_details.get("Blast Furnace", gas_bf_MJ))
    elif gas_sources_MJ <= 0 and total_process_gas_MJ <= 1e-9:
        EF_process_gas = float(e_efs.get(process_gas_carrier, 0.0))

    utility_recipe = recipes_dict.get(utility_process_name)
    util_eff = float(utility_recipe.outputs.get("Electricity", 0.0)) if utility_recipe else 0.0
    if util_eff <= 0 and utility_process_name != "Utility Plant":
        fallback_recipe = recipes_dict.get("Utility Plant")
        util_eff = float(fallback_recipe.outputs.get("Electricity", 0.0)) if fallback_recipe else 0.0

    # ---- Scenario routing fractions ------------------------------------------------------
    gas_routing = scenario.get("gas_routing", {}) or {}
    default_direct = gas_config.get("default_direct_use_fraction", 0.5)
    direct_use_fraction = gas_routing.get("direct_use_fraction", default_direct)
    if direct_use_fraction is None:
        direct_use_fraction = default_direct
    direct_use_fraction = max(0.0, min(1.0, float(direct_use_fraction)))

    electricity_fraction = gas_routing.get("electricity_fraction")
    if electricity_fraction is None:
        electricity_fraction = max(0.0, 1.0 - direct_use_fraction)
    else:
        electricity_fraction = max(0.0, min(1.0, float(electricity_fraction)))
    if direct_use_fraction + electricity_fraction > 1.0:
        electricity_fraction = max(0.0, 1.0 - direct_use_fraction)

    direct_use_gas_MJ = total_process_gas_MJ * direct_use_fraction
    electricity_gas_MJ = total_process_gas_MJ * electricity_fraction
    internal_elec_MJ = electricity_gas_MJ * util_eff

    # ---- Reference metrics ---------------------------------------------------------------
    route_key = str(scenario.get("route_preset", "") or "")
    stage_ref = str(scenario.get("stage_ref", "IP3"))
    demand_qty = float(scenario.get("demand_qty", 1000.0))

    total_gas_consumption_plant = 0.0
    if compute_inside_gas_reference_fn is not None:
        try:
            total_gas_consumption_plant = float(
                compute_inside_gas_reference_fn(
                    recipes=recipes,
                    energy_int=energy_int,
                    energy_shares=energy_shares,
                    energy_content=energy_content,
                    params=params,
                    route_key=route_key,
                    demand_qty=demand_qty,
                    stage_ref=stage_ref,
                    stage_lookup=scenario.get("stage_lookup"),
                    gas_carrier=scenario.get("gas_carrier"),
                    fallback_materials=scenario.get("fallback_materials"),
                )
            )
        except Exception:
            total_gas_consumption_plant = 0.0

    f_internal_gas = (
        min(1.0, direct_use_gas_MJ / max(total_gas_consumption_plant, 1e-9))
        if total_gas_consumption_plant > 1e-9
        else 0.0
    )
    ef_natural_gas = float(e_efs.get(natural_gas_carrier, 0.0))
    ef_gas_blended = f_internal_gas * EF_process_gas + (1 - f_internal_gas) * ef_natural_gas

    inside_elec_ref = float(scenario.get("inside_elec_ref", 0.0))
    if inside_elec_ref <= 0.0 and compute_inside_elec_reference_fn is not None:
        try:
            inside_elec_ref = float(
                compute_inside_elec_reference_fn(
                    recipes=recipes,
                    energy_int=energy_int,
                    energy_shares=energy_shares,
                    energy_content=energy_content,
                    params=params,
                    route_key=route_key,
                    demand_qty=demand_qty,
                    stage_ref=stage_ref,
                )
            )
        except Exception:
            inside_elec_ref = 0.0

    f_internal = (
        min(1.0, internal_elec_MJ / max(inside_elec_ref, 1e-9))
        if inside_elec_ref > 1e-9
        else 0.0
    )
    ef_internal_electricity = (EF_process_gas / util_eff) if util_eff > 1e-9 else 0.0

    # ---- Apply credits / mutate balance --------------------------------------------------
    adjusted_balance = energy_balance.copy()
    if credit_on:
        from forge.steel_model_core import adjust_energy_balance  # local import to avoid cycle

        adjusted_balance = adjust_energy_balance(adjusted_balance, internal_elec_MJ)
        if direct_use_gas_MJ > 0 and total_gas_consumption_plant > 1e-9:
            if natural_gas_carrier not in adjusted_balance.columns:
                adjusted_balance[natural_gas_carrier] = 0.0
            if process_gas_carrier not in adjusted_balance.columns:
                adjusted_balance[process_gas_carrier] = 0.0
            for proc_name in adjusted_balance.index:
                current = float(adjusted_balance.loc[proc_name, natural_gas_carrier])
                if current <= 0:
                    continue
                reduction = current * f_internal_gas
                adjusted_balance.loc[proc_name, natural_gas_carrier] = current - reduction
                adjusted_balance.loc[proc_name, process_gas_carrier] += reduction
    else:
        internal_elec_MJ = 0.0
        direct_use_gas_MJ = 0.0
        electricity_gas_MJ = 0.0
        total_process_gas_MJ = 0.0
        gas_source_details.clear()

    updated_e_efs = dict(e_efs)
    updated_e_efs[natural_gas_carrier] = ef_gas_blended
    updated_e_efs[process_gas_carrier] = EF_process_gas

    meta = {
        "total_process_gas_MJ": total_process_gas_MJ,
        "gas_coke_MJ": gas_coke_MJ,
        "gas_bf_MJ": gas_bf_MJ,
        "gas_sources_MJ": gas_sources_MJ,
        "gas_source_details": gas_source_details,
        "direct_use_gas_MJ": direct_use_gas_MJ,
        "electricity_gas_MJ": electricity_gas_MJ,
        "total_gas_consumption_plant": total_gas_consumption_plant,
        "f_internal_gas": f_internal_gas,
        "ef_gas_blended": ef_gas_blended,
        "EF_coke_gas": EF_coke_gas,
        "EF_bf_gas": EF_bf_gas,
        "EF_process_gas": EF_process_gas,
        "util_eff": util_eff,
        "direct_use_fraction": direct_use_fraction,
        "electricity_fraction": electricity_fraction,
        "f_internal": f_internal,
        "ef_internal_electricity": ef_internal_electricity,
        "process_gas_carrier": process_gas_carrier,
        "natural_gas_carrier": natural_gas_carrier,
        "utility_process": utility_process_name,
        "fallback_materials": list(fallback_materials),
    }

    return adjusted_balance, updated_e_efs, meta
