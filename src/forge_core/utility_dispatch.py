# utility_dispatch.py
# Pure dispatch of process gas -> steam / direct heat / power / export / flare
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class DispatchParams:
    dispatch_priority: List[str]  # e.g., ["steam", "direct", "power"]
    boiler_eff: float             # gas -> steam (MJ_steam / MJ_gas)
    power_eff: float              # gas -> elec (MJ_elec / MJ_gas)
    allow_export: bool
    allow_flare: bool


@dataclass
class Demands:
    steam_MJ: float            # target steam demand in MJ (as steam)
    direct_heat_MJ: float      # target direct-fuel heat demand in MJ (as heat/fuel)
    electricity_MJ: float      # target electricity demand in MJ (as electricity)


@dataclass
class DispatchResult:
    # gas allocation by end use (gas basis, MJ_gas)
    gas_to_steam_MJ: float
    gas_to_direct_MJ: float
    gas_to_power_MJ: float
    gas_export_MJ: float
    gas_flare_MJ: float

    # production (useful energy basis)
    produced_steam_MJ: float
    produced_electricity_MJ: float

    # shortfalls to be met by market (useful energy basis)
    shortfall_steam_MJ: float
    shortfall_direct_MJ: float
    shortfall_electricity_MJ: float

    # internal electricity mix
    ef_internal_electricity_g_per_MJ: float  # weighted only from gas_to_power
    internal_electricity_fraction: float     # produced_elec / demand (clamped to [0,1])

    # bookkeeping
    gas_available_MJ_by_source: Dict[str, float]
    gas_used_for_power_by_source: Dict[str, float]


def _draw_from_pool(
    pool: Dict[str, float], amount: float
) -> Tuple[float, Dict[str, float]]:
    """
    Draw `amount` MJ from a multi-source gas pool proportionally by current shares.
    Returns (actual_draw, draw_by_source dict) and mutates `pool` in place.
    """
    total = sum(pool.values())
    if total <= 0.0 or amount <= 0.0:
        return 0.0, {k: 0.0 for k in pool}

    draw = min(amount, total)
    draw_by_source: Dict[str, float] = {}
    if total > 0.0:
        for k, v in pool.items():
            share = 0.0 if total == 0.0 else v / total
            take = share * draw
            draw_by_source[k] = take
            pool[k] = max(0.0, v - take)
    return draw, draw_by_source


def dispatch(
    gas_sources: Dict[str, Dict[str, float]],
    # gas_sources = { "BF": {"MJ": 4213.4, "ef_g_per_MJ": 95.0}, "COG": {...}, ... }
    demands: Demands,
    params: DispatchParams,
) -> DispatchResult:
    # build mutable pool
    pool = {k: max(0.0, v.get("MJ", 0.0)) for k, v in gas_sources.items()}
    ef_by_source = {k: max(0.0, v.get("ef_g_per_MJ", 0.0)) for k, v in gas_sources.items()}
    total_gas = sum(pool.values())

    # targets (useful energy)
    steam_target = max(0.0, demands.steam_MJ)
    direct_target = max(0.0, demands.direct_heat_MJ)
    elec_target = max(0.0, demands.electricity_MJ)

    # effs
    eta_b = max(1e-12, params.boiler_eff)
    eta_p = max(1e-12, params.power_eff)

    # accumulators (gas basis)
    g2steam = 0.0
    g2direct = 0.0
    g2power = 0.0
    used_for_power_by_source: Dict[str, float] = {k: 0.0 for k in pool}

    # helper to allocate a named leg by gas requirement
    def _alloc(end_use: str, useful_target: float) -> float:
        nonlocal g2steam, g2direct, g2power, used_for_power_by_source
        if useful_target <= 0.0:
            return 0.0

        if end_use == "steam":
            gas_need = useful_target / eta_b
            drawn, _ = _draw_from_pool(pool, gas_need)
            g2steam += drawn
            return drawn  # gas basis
        elif end_use == "direct":
            gas_need = useful_target  # 1:1 (fuel MJ == useful heat MJ at this layer)
            drawn, _ = _draw_from_pool(pool, gas_need)
            g2direct += drawn
            return drawn
        elif end_use == "power":
            gas_need = useful_target / eta_p
            drawn, by_src = _draw_from_pool(pool, gas_need)
            g2power += drawn
            for s, x in by_src.items():
                used_for_power_by_source[s] = used_for_power_by_source.get(s, 0.0) + x
            return drawn
        else:
            return 0.0

    # allocate by policy
    remaining_steam = steam_target
    remaining_direct = direct_target
    remaining_elec = elec_target

    for leg in params.dispatch_priority:
        if leg == "steam" and remaining_steam > 0.0:
            drawn_gas = _alloc("steam", remaining_steam)
            produced = drawn_gas * eta_b
            remaining_steam = max(0.0, remaining_steam - produced)
        elif leg == "direct" and remaining_direct > 0.0:
            drawn_gas = _alloc("direct", remaining_direct)
            produced = drawn_gas  # 1:1
            remaining_direct = max(0.0, remaining_direct - produced)
        elif leg == "power" and remaining_elec > 0.0:
            drawn_gas = _alloc("power", remaining_elec)
            produced = drawn_gas * eta_p
            remaining_elec = max(0.0, remaining_elec - produced)

    # export/flare remaining pool
    remaining_gas_pool = sum(pool.values())
    gas_export = 0.0
    gas_flare = 0.0
    if remaining_gas_pool > 0.0:
        if params.allow_export:
            gas_export = remaining_gas_pool
        elif params.allow_flare:
            gas_flare = remaining_gas_pool
        else:
            # if neither allowed, leave it as flare by convention (non-negative, explicit)
            gas_flare = remaining_gas_pool
        # zero-out pool
        for k in list(pool.keys()):
            pool[k] = 0.0

    # productions
    produced_steam = g2steam * eta_b
    produced_elec = g2power * eta_p

    # shortfalls (useful energy)
    shortfall_steam = max(0.0, steam_target - produced_steam)
    shortfall_direct = max(0.0, direct_target - g2direct)
    shortfall_elec = max(0.0, elec_target - produced_elec)

    # ef_internal_electricity (only from gas routed to power)
    if produced_elec > 0.0 and g2power > 0.0:
        # weighted by gas to power composition
        num = 0.0
        for s, gas_MJ in used_for_power_by_source.items():
            num += gas_MJ * ef_by_source.get(s, 0.0)
        ef_internal = num / produced_elec  # gCO2 per MJ_electricity
    else:
        ef_internal = 0.0

    internal_frac = 0.0 if elec_target <= 0.0 else min(1.0, produced_elec / elec_target)

    # conservation check (gas basis)
    lhs = total_gas
    rhs = g2steam + g2direct + g2power + gas_export + gas_flare
    tol = max(1e-9, 0.001 * max(1.0, lhs))  # 0.1% absolute tolerance
    assert abs(lhs - rhs) <= tol, (
        f"[utility_dispatch] Gas conservation failed: "
        f"available={lhs:.6f}, used={rhs:.6f} (Δ={lhs-rhs:.6f})"
    )

    return DispatchResult(
        gas_to_steam_MJ=g2steam,
        gas_to_direct_MJ=g2direct,
        gas_to_power_MJ=g2power,
        gas_export_MJ=gas_export,
        gas_flare_MJ=gas_flare,
        produced_steam_MJ=produced_steam,
        produced_electricity_MJ=produced_elec,
        shortfall_steam_MJ=shortfall_steam,
        shortfall_direct_MJ=shortfall_direct,
        shortfall_electricity_MJ=shortfall_elec,
        ef_internal_electricity_g_per_MJ=ef_internal,
        internal_electricity_fraction=internal_frac,
        gas_available_MJ_by_source={k: gas_sources[k].get("MJ", 0.0) for k in gas_sources},
        gas_used_for_power_by_source=used_for_power_by_source,
    )
