"""Blast-furnace fleet: relining windows derived from unit facts by one rule.

datasets/steel/plants_units.csv holds what is known about each furnace (Global
Energy Monitor, Global Blast Furnace Tracker, April 2024): status, capacity,
start-up year, most recent completed relining and any planned relining. The
transition model needs, per coke-based furnace, its next relining window and
its end of life. Both follow from the facts by one rule, applied to every unit:

    window = the planned relining, if GEM lists one; otherwise the most recent
             relining, or start-up for a furnace never relined, plus one
             campaign; never earlier than the base year
    EOL    = max(start-up + lifespan, window)

Windows then recur every campaign (paper_scenarios.relining_due). Charcoal
furnaces do not transition in the scenarios (EOL = window = 2101).

scripts/build_fleet_from_gem.py writes plants_units.csv and plants.csv.
"""
from __future__ import annotations

import pandas as pd

BASE_YEAR = 2025
NO_TRANSITION = 2101
KEEP_STATUSES = ("operating", "mothballed")


def windows_from_units(
    units: pd.DataFrame,
    campaign: int = 20,
    lifespan: int = 50,
    base_year: int = BASE_YEAR,
    honor_planned: bool = True,
    keep_statuses: tuple = KEEP_STATUSES,
) -> pd.DataFrame:
    """Return the units kept, with Relining, EOL and the rule that set the window."""
    u = units[units["status"].isin(keep_statuses) & (pd.to_numeric(units["capacity"]) > 0)].copy()
    rel, eol, rule = [], [], []
    for _, r in u.iterrows():
        if r["fuel"] == "charcoal":
            rel.append(NO_TRANSITION); eol.append(NO_TRANSITION); rule.append("charcoal: no transition")
            continue
        if honor_planned and pd.notna(r.get("planned_relining")):
            w, why = int(r["planned_relining"]), "GEM planned relining"
        elif pd.notna(r.get("last_relining")):
            w, why = int(r["last_relining"]) + campaign, f"last relining + {campaign}"
        else:
            w, why = int(r["start"]) + campaign, f"never relined: start-up + {campaign}"
        w = max(w, base_year)
        rel.append(w)
        eol.append(max(int(r["start"]) + lifespan, w))
        rule.append(why)
    u["Relining"], u["EOL"], u["rule"] = rel, eol, rule
    return u.reset_index(drop=True)


def relining_due(fleet: pd.DataFrame, year: int, interval: int = 20) -> pd.Series:
    """Boolean mask: furnaces with a relining window in ``year``.

    "Relining" is the next window; windows recur every ``interval`` years, so a
    furnace relined on coal before the transition gets another window one
    campaign later.
    """
    rel = pd.to_numeric(fleet["Relining"], errors="coerce")
    return (rel <= year) & (((year - rel) % interval) == 0)


def to_plants_frame(fleet: pd.DataFrame) -> pd.DataFrame:
    """The columns paper_scenarios reads (plants.csv)."""
    f = fleet.reset_index(drop=True)
    return pd.DataFrame({
        "BF": range(1, len(f) + 1),
        "Capacity": pd.to_numeric(f["capacity"]).round(0).astype(int),
        "EOL": f["EOL"].astype(int),
        "Relining": f["Relining"].astype(int),
        "Fuel": f["fuel"],
    })


__all__ = ["windows_from_units", "relining_due", "to_plants_frame", "BASE_YEAR", "KEEP_STATUSES"]
