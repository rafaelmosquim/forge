#!/usr/bin/env python3
"""Collect FGV EF lookup tables into a single 2025 summary.

This script scans all ``results/fgv/*/tables/ef_lookup.csv`` files and
builds a consolidated ``results/fgv/ef_summary.csv`` containing one row
per (label, route, config, product_config) for year 2025.

It is intended as a lightweight archive of baseline (2025) emission
factors for each FGV portfolio (regular + high-alloy), across all routes
that have been run.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set

import yaml

# Ensure repo root and src/ on sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forge.steel_core_api_v2 import RouteConfig, ScenarioInputs, run_scenario  # noqa: E402


FGV_ROOT = ROOT / "results" / "fgv"
DATA_DIR = Path(os.getenv("FORGE_DATA_DIR", "datasets/steel/likely"))
YEAR = 2025
BASE_YEAR = 2025
DEMAND_FALLBACK = 1000.0


def _load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _merge(a: dict | None, b: dict | None) -> dict:
    if not b:
        return dict(a or {})
    out = dict(a or {})
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _guess_alloy_config(label: str) -> str:
    """Best-effort guess of alloy config from the FGV label."""
    name = (label or "").lower()
    if "high" in name and "regular" not in name:
        return "high-alloy"
    if "regular" in name:
        return "regular"
    return ""


def _spec_path_for_label(label: str) -> Path | None:
    """Map FGV label to its portfolio spec path."""
    name = (label or "").lower()
    if name.startswith("fgv_regular"):
        return ROOT / "configs" / "fgv_regular.yml"
    if name.startswith("fgv_high"):
        return ROOT / "configs" / "fgv_high.yml"
    return None


def _blend_components_for_spec(spec_path: Path) -> Tuple[List[Tuple[str, float, dict, dict]], float, str]:
    """Return list of (run_name, share, picks, scenario_defaults), demand_qty, stage_key."""
    spec = _load_yaml(spec_path)
    defaults = spec.get("defaults") or {}
    default_route = defaults.get("route") or {}
    demand_qty = float(default_route.get("demand_qty") or DEMAND_FALLBACK)
    stage_key = default_route.get("stage_key") or "Finished"
    default_picks = defaults.get("picks_by_material") or {}
    default_scenario = defaults.get("scenario") or {}

    runs = spec.get("runs") or []
    run_map = {r.get("name"): (r.get("picks_by_material") or {}) for r in runs if isinstance(r, dict)}
    blends = spec.get("blends") or []
    blend = None
    for b in blends:
        if isinstance(b, dict) and str(b.get("name", "")).strip() == "finished_portfolio":
            blend = b
            break
    if blend is None and blends:
        blend = blends[0]
    components = blend.get("components", []) if blend else []

    items: List[Tuple[str, float, dict, dict]] = []
    for comp in components:
        try:
            share = float(comp.get("share", 0.0) or 0.0)
        except Exception:
            continue
        run_name = comp.get("run")
        if not run_name or run_name not in run_map:
            continue
        picks = _merge(default_picks, run_map[run_name])
        scn = dict(default_scenario)
        items.append((run_name, share, picks, scn))
    return items, demand_qty, stage_key


def _scenario_for_bf_config(config: str) -> dict:
    """Load BF-BOF route defaults for a given config (Coke vs Charcoal)."""
    cfg = (config or "").strip().upper()
    if cfg == "CHARCOAL":
        fname = "BF_BOF_charcoal.yml"
    else:
        fname = "BF_BOF_coal.yml"
    path = DATA_DIR / "scenarios" / fname
    if not path.exists():
        return {}
    return _load_yaml(path)


def _compute_bf_product_rows(bf_targets: Set[Tuple[str, str, str]]) -> List[Dict[str, str]]:
    """Compute 2025 EF (tCO2/t) for each individual FGV product on BF-BOF (Coke/Charcoal)."""
    rows: List[Dict[str, str]] = []
    for label, country_code, config in sorted(bf_targets):
        spec_path = _spec_path_for_label(label)
        if spec_path is None or not spec_path.exists():
            continue
        components, demand_qty, stage_key = _blend_components_for_spec(spec_path)
        if not components:
            continue

        denom = max(1e-9, demand_qty / 1000.0)  # convert demand kg → tonnes
        scn_route = _scenario_for_bf_config(config)
        code_norm = (country_code or "").strip().upper()
        alloy = _guess_alloy_config(label)

        for run_name, _share, picks, scenario_defaults in components:
            scenario = _merge(scn_route, scenario_defaults)
            rc = RouteConfig(
                route_preset="BF-BOF",
                stage_key=stage_key,
                stage_role=None,
                demand_qty=float(demand_qty),
                picks_by_material=dict(picks or {}),
                pre_select_soft={},
            )
            si = ScenarioInputs(
                country_code=code_norm,
                scenario=dict(scenario or {}),
                route=rc,
            )
            out = run_scenario(str(DATA_DIR), si)
            total_kg = float(getattr(out, "total_co2e_kg", 0.0) or 0.0)
            ef_kg_per_t = total_kg / denom  # kg CO2 per tonne product
            ef_tco2_per_t = ef_kg_per_t / 1000.0

            rows.append(
                {
                    "fgv_label": label,
                    "alloy_config": alloy,
                    "route": "BF-BOF",
                    "config": config,
                    "year": str(YEAR),
                    "base_year": str(BASE_YEAR),
                    # Explicitly mark these as baseline (no improvement schedule)
                    "annual_improvement": "0.0",
                    "country_code": code_norm,
                    "product_config": f"run:{run_name}",
                    "ef_tco2_per_t": f"{ef_tco2_per_t}",
                }
            )
    return rows


def _load_rows_for_path(ef_path: Path) -> List[Dict[str, str]]:
    label = ef_path.parent.parent.name  # results/fgv/<label>/tables/ef_lookup.csv
    alloy = _guess_alloy_config(label)
    rows: List[Dict[str, str]] = []

    with ef_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Keep only 2025 entries (baseline year)
            year_raw = (row.get("year") or "").strip()
            try:
                year_val = int(float(year_raw))
            except Exception:
                continue
            if year_val != 2025:
                continue

            out_row: Dict[str, str] = {
                "fgv_label": label,
                "alloy_config": alloy,
                "route": row.get("route", ""),
                "config": row.get("config", ""),
                "year": year_raw,
                "base_year": (row.get("base_year") or "").strip(),
                "annual_improvement": (row.get("annual_improvement") or "").strip(),
                "country_code": (row.get("country_code") or "").strip(),
                "product_config": (row.get("product_config") or "").strip(),
                "ef_tco2_per_t": (row.get("ef_tco2_per_t") or "").strip(),
            }
            rows.append(out_row)
    return rows


def main() -> None:
    if not FGV_ROOT.exists():
        print(f"[fgv-summary] FGV root not found at {FGV_ROOT} – nothing to do.")
        return

    all_rows: List[Dict[str, str]] = []
    bf_targets: Set[Tuple[str, str, str]] = set()
    for ef_path in sorted(FGV_ROOT.glob("*/tables/ef_lookup.csv")):
        try:
            rows = _load_rows_for_path(ef_path)
        except Exception as e:
            print(f"[fgv-summary] Failed to read {ef_path}: {e}")
            continue
        all_rows.extend(rows)
        # Collect BF-BOF baseline portfolio entries for which we want per-product EFs
        for r in rows:
            if (r.get("route") == "BF-BOF") and (r.get("product_config") == "portfolio"):
                key = (
                    r.get("fgv_label", ""),
                    r.get("country_code", ""),
                    r.get("config", ""),
                )
                bf_targets.add(key)

    if not all_rows:
        print("[fgv-summary] No EF rows for 2025 found under results/fgv – summary not written.")
        return

    # Augment with per-product EFs for BF-BOF (Coke + Charcoal), baseline year 2025
    if bf_targets:
        try:
            bf_rows = _compute_bf_product_rows(bf_targets)
            all_rows.extend(bf_rows)
        except Exception as e:
            print(f"[fgv-summary] Failed to compute BF-BOF per-product EFs: {e}")

    all_rows.sort(
        key=lambda r: (
            r.get("fgv_label", ""),
            r.get("route", ""),
            r.get("config", ""),
            r.get("country_code", ""),
            r.get("product_config", ""),
        )
    )

    out_path = FGV_ROOT / "ef_summary.csv"
    fieldnames = [
        "fgv_label",
        "alloy_config",
        "route",
        "config",
        "year",
        "base_year",
        "annual_improvement",
        "country_code",
        "product_config",
        "ef_tco2_per_t",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[fgv-summary] EF summary written to {out_path}")


if __name__ == "__main__":
    main()
