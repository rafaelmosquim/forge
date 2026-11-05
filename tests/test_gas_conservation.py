"""Recovered process gas must be charged exactly once.

These guard the allocation in forge.core.gas: before it, a fixed 50/50
direct/power split ignored the gas that recipes already burn, so the plant was
charged for up to 43% more gas than it made under one setting and lost up to
40% of it under another, and a narrower reporting boundary dropped ~0.3 t/t of
blast-furnace carbon.
"""
import itertools
import math

import pandas as pd
import pytest

from forge.core.gas import allocate_gas_pool
from forge.steel_core_api_v2 import OUTSIDE_MILL_PROCS, RouteConfig, ScenarioInputs, run_scenario


# ---------------------------------------------------------------- pool (unit)

@pytest.mark.parametrize(
    "gas, native, ng, elec, eff, loss, flex",
    [
        (9000, 4500, 2000, 3500, 0.34, 0.10, 0.61),   # typical
        (9000, 12000, 2000, 3500, 0.34, 0.10, 0.61),  # native burners exceed the pool
        (9000, 1000, 500, 100, 0.34, 0.0, 1.0),       # power capped by demand -> overflow
        (9000, 1000, 500, 3500, 0.0, 0.05, 0.5),      # no utility plant
        (0, 1000, 500, 3500, 0.34, 0.10, 0.61),       # no gas (DRI/EAF)
    ],
)
def test_pool_spends_exactly_what_is_made(gas, native, ng, elec, eff, loss, flex):
    p = allocate_gas_pool(gas, native, ng, elec, eff, loss, flex)
    spent = p["native_used_MJ"] + p["substituted_MJ"] + p["power_gas_MJ"] + p["flared_MJ"]
    assert math.isclose(spent, gas, rel_tol=1e-12, abs_tol=1e-9)
    assert p["native_used_MJ"] <= native + 1e-9
    assert p["substituted_MJ"] <= flex * ng + 1e-9
    assert p["internal_elec_MJ"] <= max(elec, 0.0) + 1e-9
    assert 0.0 <= p["f_native"] <= 1.0 and 0.0 <= p["f_internal_gas"] <= 1.0 and 0.0 <= p["f_internal"] <= 1.0


# ------------------------------------------------------- full model (integration)

def _run(data_dir, stage="Finished", route="BF-BOF", routing=None):
    scn = ScenarioInputs(
        country_code=None,
        scenario={"gas_routing": dict(routing or {})},
        route=RouteConfig(route_preset=route, stage_key=stage, demand_qty=1000.0),
    )
    out = run_scenario(str(data_dir), scn)
    assert out is not None and out.emissions is not None
    return out


def _gas_ledger(out):
    """(made, charged) in MJ: burned in processes + via internal power + flared.

    Outside-mill lines are charged grid power, so only mill rows draw on the
    internal electricity.
    """
    m = out.meta
    eb = out.energy_balance
    pg, util = m["process_gas_carrier"], m["utility_process"]
    rows = [r for r in eb.index if r not in ("TOTAL", util)]
    mill = [r for r in rows if r not in set(OUTSIDE_MILL_PROCS)]
    burned = float(eb.loc[rows, pg].clip(lower=0).sum()) if pg in eb.columns else 0.0
    gross_elec = float(eb.loc[mill, "Electricity"].clip(lower=0).sum())
    via_power = m["f_internal"] * gross_elec / m["util_eff"] if m["util_eff"] else 0.0
    return m["total_process_gas_MJ"], burned + via_power + m.get("flared_gas_MJ", 0.0)


@pytest.mark.integration
@pytest.mark.parametrize("flex, loss", list(itertools.product((0.0, 0.61, 0.95), (0.0, 0.10, 0.19))))
def test_bf_bof_full_plant_charges_every_mj_once(data_dir, flex, loss):
    """Default product picks run past the reference stage into outside-mill
    finishing lines; those must not draw recovered gas or internal power."""
    out = _run(data_dir, routing={"flexible_share": flex, "loss_fraction": loss})
    made, charged = _gas_ledger(out)
    assert made > 0
    assert math.isclose(charged, made, rel_tol=1e-9), f"made {made:.1f} MJ, charged {charged:.1f} MJ"


@pytest.mark.integration
def test_process_inventories_do_not_depend_on_boundary(data_dir):
    """Each process carries the same emissions at the crude and finished boundary."""
    fin = _run(data_dir, stage="Finished").emissions["TOTAL CO2e"].drop("TOTAL", errors="ignore")
    crude = _run(data_dir, stage="CastCrude").emissions["TOTAL CO2e"].drop("TOTAL", errors="ignore")
    common = crude.index.intersection(fin.index)
    assert "Blast Furnace" in common and "Process Gas Flaring" in common
    diff = (fin[common] - crude[common]).abs()
    assert float(diff.max()) < 1e-6, diff[diff > 1e-6].to_dict()


@pytest.mark.integration
@pytest.mark.parametrize("route", ["DRI-EAF", "EAF-Scrap"])
def test_routes_without_gas_recovery_buy_their_fuel(data_dir, route):
    """No recovered gas on these routes: nothing may be booked as (free) process gas."""
    out = _run(data_dir, route=route)
    eb = out.energy_balance
    pg = out.meta["process_gas_carrier"]
    if pg in eb.columns:
        rows = [r for r in eb.index if r != "TOTAL"]
        assert float(eb.loc[rows, pg].abs().sum()) < 1e-9
    assert out.meta.get("flared_gas_MJ", 0.0) == 0.0


@pytest.mark.integration
def test_coke_ovens_charged_for_the_coal_they_convert(data_dir):
    out = _run(data_dir)
    conv = out.meta["conversion"]["Coke Production"]
    # coal/coke energy ratio 1.40 across 9 Brazilian plant-years
    assert 1.3 < conv["feed_MJ"] / conv["product_MJ"] < 1.5
    assert conv["net_g"] > 0


# The mill through hot rolling, without outside-mill finishing lines.
HOT_ROLLED_PICKS = {
    "Burnt Lime": "Burnt Lime Production", "Cast Steel (IP1)": "Continuous Casting (R)",
    "Dolomite": "Dolomite Production", "Finished Products": "No Coating",
    "Intermediate Process 3": "Bypass Raw→IP3", "Liquid Steel": "Regular Steel",
    "Manufactured Feed (IP4)": "Direct use of Basic Steel Products",
    "Nitrogen": "Nitrogen Production", "Oxygen": "Oxygen Production",
    "Raw Products (types)": "Hot Rolling",
}


@pytest.mark.integration
def test_bf_bof_total_within_brazilian_plant_range(data_dir):
    """Plant fuel purchases imply 2.44-2.70 t CO2/t before carbonates (ABM
    balances of Tubarao, CSN and Cubatao, which roll most of their output)."""
    scn = ScenarioInputs(
        country_code=None,
        scenario={"energy_int_floor": {"Blast Furnace": 11.0}},
        route=RouteConfig(route_preset="BF-BOF", stage_key="Finished", demand_qty=1000.0,
                          picks_by_material=dict(HOT_ROLLED_PICKS), pre_select_soft={}),
    )
    t = float(run_scenario(str(data_dir), scn).total_co2e_kg) / 1000.0
    assert 2.4 < t < 2.9, t
