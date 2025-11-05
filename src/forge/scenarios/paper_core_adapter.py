# -*- coding: utf-8 -*-
"""
Adapter helpers to power paper_scenarios directly from the core model.

Goals
- Keep existing paper_scenarios.py logic for BF replacement and visuals intact.
- Replace static CSV lookups with programmatic calls into the core for each
  route/year/config, returning the same kind of emission factors and LCIs.

Usage (inside your paper_scenarios.py)
    from forge.scenarios.paper_core_adapter import (
        ScenarioKnobs, compute_route_result, compute_mix_lci
    )

    knobs = ScenarioKnobs(
        snapshot_year=2035,
        improvement_pct_per_year=1.0,   # 1%/yr
        dri_mix="Blue",                 # or "Green"
        final_picks={                   # lock final processing
            "Manufactured Feed (IP4)": "Stamping/calendering/lamination",
            "Finished Products": "No Coating",
        },
        charcoal_share_schedule=None,   # optional {2030:0.0, 2040:0.3, 2045:0.6}
        charcoal_global_substitute=False, # set True to map Coal/Coke→Charcoal globally
    )

    # Per-route EF + LCI
    out_bf = compute_route_result("BF-BOF", knobs)
    out_dri = compute_route_result("DRI-EAF", knobs)
    out_eaf = compute_route_result("EAF-Scrap", knobs)

    # Blend LCIs by weights from your BF replacement logic
    lci_mix = compute_mix_lci({"BF-BOF": 0.4, "DRI-EAF": 0.4, "EAF-Scrap": 0.2}, knobs)

Notes
- Demand is fixed at 1000 (consistent with the app/CLI).
- FORGE_ENABLE_LCI is set automatically on first call.
- Return values include both raw and yield-adjusted CO2, plus LCI DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional, Any

import os
import pandas as pd

from forge.steel_core_api_v2 import (
    RouteConfig,
    ScenarioInputs,
    run_scenario,
)


DEFAULT_DATA_DIR = "datasets/steel/likely"


@dataclass(frozen=True)
class ScenarioKnobs:
    snapshot_year: int
    improvement_pct_per_year: float = 0.0   # e.g., 1.0 for 1%/yr
    dri_mix: str = "Blue"                   # "Blue" | "Green"
    final_picks: Dict[str, str] = field(default_factory=dict)
    country_code: Optional[str] = None

    # Charcoal options (choose one, or none)
    charcoal_share_schedule: Optional[Dict[int, float]] = None  # {year: frac 0..1}
    charcoal_global_substitute: bool = False  # map Coal/Coke→Charcoal globally

    # Optional BF cap on base intensity (before adjustment)
    bf_base_intensity_cap: Optional[float] = None  # e.g., 11.0

    data_dir: str = DEFAULT_DATA_DIR

    def to_scenario_dict(self) -> Dict[str, Any]:
        scn: Dict[str, Any] = {
            "snapshot_year": int(self.snapshot_year),
            "energy_int_schedule": {
                "rate_pct_per_year": float(self.improvement_pct_per_year),
            },
            "dri_mix": str(self.dri_mix),
        }
        # Optional: BF cap on base intensity
        if self.bf_base_intensity_cap is not None:
            scn.setdefault("energy_int", {})["Blast Furnace"] = float(self.bf_base_intensity_cap)
        # Optional: charcoal expansion schedule (shares applied at BF)
        if isinstance(self.charcoal_share_schedule, dict) and self.charcoal_share_schedule:
            scn["charcoal_expansion"] = "Expansion"
            # Accept 0..1 or 0..100 values
            sched = {}
            for k, v in self.charcoal_share_schedule.items():
                try:
                    y = int(k)
                    f = float(v)
                    if f > 1.0:
                        f /= 100.0
                    sched[str(y)] = max(0.0, min(1.0, f))
                except Exception:
                    continue
            if sched:
                scn["charcoal_share_schedule"] = sched
        # Optional: global substitution (coarse switch)
        if self.charcoal_global_substitute:
            scn.setdefault("fuel_substitutions", {}).update({
                "Coal": "Charcoal",
                "Coke": "Charcoal",
            })
        return scn


def _ensure_lci_enabled() -> None:
    if os.environ.get("FORGE_ENABLE_LCI") not in ("1", "true", "True"):  # idempotent
        os.environ["FORGE_ENABLE_LCI"] = "1"


@lru_cache(maxsize=256)
def _compute_route_result_cached(route_preset: str, knobs_key: tuple) -> Dict[str, Any]:
    knobs: ScenarioKnobs = knobs_key[0]  # single-object tuple
    _ensure_lci_enabled()
    # Build RouteConfig
    rc = RouteConfig(
        route_preset=route_preset,
        stage_key="Finished",
        stage_role=None,
        demand_qty=1000.0,
        picks_by_material=dict(knobs.final_picks or {}),
        pre_select_soft={},
    )
    scn = knobs.to_scenario_dict()
    si = ScenarioInputs(
        country_code=knobs.country_code,
        scenario=scn,
        route=rc,
    )
    out = run_scenario(knobs.data_dir, si)
    # Normalize outputs
    raw = float(getattr(out, "total_co2e_kg", 0.0) or 0.0)
    return {
        "summary": {
            "route": route_preset,
            "raw_co2e_kg": raw,
            "total_co2e_kg": raw * (1.0 / 0.85),  # apply reported yield divisor
        },
        "lci": getattr(out, "lci", None),
    }


def compute_route_result(route_preset: str, knobs: ScenarioKnobs) -> Dict[str, Any]:
    """Return compact summary + LCI for a single route under the provided knobs."""
    key = (knobs,)
    return _compute_route_result_cached(route_preset, key)


def _weight_lci(df: Optional[pd.DataFrame], weight: float) -> Optional[pd.DataFrame]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    lci = df.copy()
    if "Amount" not in lci.columns:
        return None
    lci["Amount"] = pd.to_numeric(lci["Amount"], errors="coerce").fillna(0.0) * float(weight)
    return lci


def _accumulate_lci(acc: Optional[pd.DataFrame], add: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if add is None:
        return acc
    if acc is None:
        acc = add
    else:
        acc = pd.concat([acc, add], ignore_index=True)
    keys = [
        c for c in ["Process", "Output", "Flow", "Input", "Category", "ValueUnit", "Unit"]
        if c in acc.columns
    ]
    if not keys or "Amount" not in acc.columns:
        return acc
    grouped = acc.groupby(keys, dropna=False, as_index=False)["Amount"].sum()
    return grouped.loc[:, keys + ["Amount"]]


def compute_mix_lci(weights_by_route: Dict[str, float], knobs: ScenarioKnobs) -> pd.DataFrame:
    """Blend LCIs across routes with given weights (auto-normalized)."""
    total = sum(max(0.0, float(w)) for w in weights_by_route.values()) or 1.0
    acc: Optional[pd.DataFrame] = None
    for route, w in weights_by_route.items():
        res = compute_route_result(route, knobs)
        lci = res.get("lci")
        acc = _accumulate_lci(acc, _weight_lci(lci, max(0.0, float(w)) / total))
    return acc if isinstance(acc, pd.DataFrame) else pd.DataFrame()


__all__ = [
    "ScenarioKnobs",
    "compute_route_result",
    "compute_mix_lci",
]


# -------- Convenience for paper_scenarios.py --------

def compute_ef_for_routes(
    routes: list[str],
    knobs: ScenarioKnobs,
    *,
    use_yield_adjusted: bool = False,
) -> dict[tuple[str, str], float]:
    """Return EF lookup mapping (Route, ConfigLabel) -> EF (tCO2/t).

    - EF is numerically raw_co2e_kg / 1000 (kg/kg), i.e., tCO2/t steel.
    - ConfigLabel encodes the knobs for later reference; recommended pattern:
        f"y{knobs.snapshot_year}_{knobs.dri_mix}_{knobs.improvement_pct_per_year:.0f}pct"
    - If use_yield_adjusted=True, uses total_co2e_kg (yield-adjusted) instead.
    """
    label = f"y{int(knobs.snapshot_year)}_{knobs.dri_mix}_{int(knobs.improvement_pct_per_year)}pct"
    ef_map: dict[tuple[str, str], float] = {}
    for route in routes:
        res = compute_route_result(route, knobs)
        s = res.get("summary") or {}
        raw = float(s.get("total_co2e_kg" if use_yield_adjusted else "raw_co2e_kg") or 0.0)
        ef = raw / 1000.0  # kg per 1000 kg → t/t numerically
        ef_map[(route, label)] = ef
    return ef_map


def build_ef_table(
    routes: list[str],
    knobs: ScenarioKnobs,
    *,
    config_label: str | None = None,
    use_yield_adjusted: bool = False,
) -> pd.DataFrame:
    """Build a DataFrame like the legacy routes.csv with columns:
        Route, Config, Emissions
    """
    label = config_label or f"y{int(knobs.snapshot_year)}_{knobs.dri_mix}_{int(knobs.improvement_pct_per_year)}pct"
    rows = []
    for route in routes:
        res = compute_route_result(route, knobs)
        s = res.get("summary") or {}
        raw = float(s.get("total_co2e_kg" if use_yield_adjusted else "raw_co2e_kg") or 0.0)
        ef = raw / 1000.0
        rows.append({"Route": route, "Config": label, "Emissions": ef})
    return pd.DataFrame(rows, columns=["Route", "Config", "Emissions"])


def write_ef_csv(df: pd.DataFrame, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
