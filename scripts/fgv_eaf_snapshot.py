"""Run FGV portfolio specs on the Scrap-EAF route for year 2025 and summarize EF.

This is a lightweight helper that:
  - Computes the blended EF (tCO2/t) for each FGV portfolio (regular + high) on EAF-Scrap.
  - Writes a minimal ef_lookup.csv under results/fgv/<label>/tables/.
  - Updates results/fgv/summary.csv with one row per (grid code, label, route).

No dependency on the paper scenario flow; only year 2025 is computed.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# Ensure repo root and src/ on sys.path for imports
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forge.steel_core_api_v2 import RouteConfig, ScenarioInputs, run_scenario  # noqa: E402


DATA_DIR = Path(os.getenv("FORGE_DATA_DIR", "datasets/steel/likely"))
YEAR = 2025
BASE_YEAR = 2025
ANNUAL_IMPROVEMENT = 0.0
ROUTE = "EAF-Scrap"
CONFIG = "Scrap"
DEMAND_FALLBACK = 1000.0
BLEND_NAME = "finished_portfolio"


@dataclass(frozen=True)
class PortfolioSpec:
    label: str
    spec_path: Path
    country_code: str


def _norm_code(code: str | None) -> str:
    return (str(code or "").strip().upper())


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


def _load_electricity_map() -> Dict[str, float]:
    path = DATA_DIR / "electricity_intensity.yml"
    mapping: Dict[str, float] = {}
    if not path.exists():
        return mapping
    payload = _load_yaml(path)
    entries = payload.get("electricity_intensity", payload)
    if isinstance(entries, list):
        for row in entries:
            if not isinstance(row, dict):
                continue
            code = str(row.get("code") or row.get("country") or "").strip()
            val = row.get("intensity")
            if code and val is not None:
                try:
                    mapping[code.upper()] = float(val)
                except Exception:
                    continue
    elif isinstance(entries, dict):
        for code, val in entries.items():
            try:
                mapping[str(code).upper()] = float(val)
            except Exception:
                continue
    return mapping


def _blend_components(spec: dict) -> Tuple[List[Tuple[str, float, dict, dict]], float, str]:
    """Return list of (run_name, share, picks, scenario_defaults), demand_qty, stage_key."""
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
        if isinstance(b, dict) and str(b.get("name", "")).strip() == BLEND_NAME:
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


def _blend_runs(spec: dict) -> Tuple[List[Tuple[float, dict, dict]], float, str]:
    """Return list of (share, picks, scenario_defaults), demand_qty, stage_key."""
    components, demand_qty, stage_key = _blend_components(spec)
    items: List[Tuple[float, dict, dict]] = []
    for _run_name, share, picks, scn in components:
        items.append((share, picks, scn))
    return items, demand_qty, stage_key


def _compute_ef_for_portfolio(spec_path: Path, country_code: str) -> float:
    spec = _load_yaml(spec_path)
    items, demand_qty, stage_key = _blend_runs(spec)
    if not items:
        raise RuntimeError(f"No blend components found in {spec_path}")

    ef_total = 0.0  # kg CO2 weighted sum
    denom = max(1e-9, demand_qty / 1000.0)  # convert demand kg → tonnes
    for share, picks, scenario_defaults in items:
        rc = RouteConfig(
            route_preset=ROUTE,
            stage_key=stage_key,
            stage_role=None,
            demand_qty=float(demand_qty),
            picks_by_material=dict(picks or {}),
            pre_select_soft={},
        )
        si = ScenarioInputs(
            country_code=_norm_code(country_code),
            scenario=dict(scenario_defaults or {}),
            route=rc,
        )
        out = run_scenario(str(DATA_DIR), si)
        total_kg = float(getattr(out, "total_co2e_kg", 0.0) or 0.0)
        ef_kg_per_t = total_kg / denom  # kg CO2 per tonne product
        ef_total += share * ef_kg_per_t
    # Convert to tCO2/t (per user request) after blending
    return ef_total / 1000.0


def _compute_efs_for_products(spec_path: Path, country_code: str) -> List[dict]:
    """Compute EF (tCO2/t) for each individual product (blend component)."""
    spec = _load_yaml(spec_path)
    components, demand_qty, stage_key = _blend_components(spec)
    if not components:
        raise RuntimeError(f"No blend components found in {spec_path}")

    denom = max(1e-9, demand_qty / 1000.0)  # convert demand kg → tonnes
    code_norm = _norm_code(country_code)
    rows: List[dict] = []

    for run_name, _share, picks, scenario_defaults in components:
        rc = RouteConfig(
            route_preset=ROUTE,
            stage_key=stage_key,
            stage_role=None,
            demand_qty=float(demand_qty),
            picks_by_material=dict(picks or {}),
            pre_select_soft={},
        )
        si = ScenarioInputs(
            country_code=code_norm,
            scenario=dict(scenario_defaults or {}),
            route=rc,
        )
        out = run_scenario(str(DATA_DIR), si)
        total_kg = float(getattr(out, "total_co2e_kg", 0.0) or 0.0)
        ef_kg_per_t = total_kg / denom  # kg CO2 per tonne product
        ef_tco2_per_t = ef_kg_per_t / 1000.0
        rows.append(
            {
                "route": ROUTE,
                "config": CONFIG,
                "year": YEAR,
                "base_year": BASE_YEAR,
                "annual_improvement": ANNUAL_IMPROVEMENT,
                "country_code": code_norm,
                # Tag each row with the underlying run name for traceability
                "product_config": f"run:{run_name}",
                "ef_tco2_per_t": ef_tco2_per_t,
            }
        )
    return rows


def _write_ef_lookup(label: str, rows: List[dict]) -> None:
    base_dir = Path("results") / "fgv" / label / "tables"
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / "ef_lookup.csv"
    fieldnames = [
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
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"[fgv] ef_lookup written to {out_path}")


def _update_summary(entries: List[dict]) -> None:
    summary_path = Path("results") / "fgv" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    existing: List[dict] = []
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                existing.append(row)

    rows = existing + entries
    dedup: Dict[Tuple[str, str, str, str, str], dict] = {}
    for row in rows:
        key = (
            row.get("electricity_code") or "",
            row.get("output_label") or "",
            row.get("route") or "",
            row.get("config") or "",
            str(row.get("year") or ""),
        )
        dedup[key] = row

    fieldnames = [
        "electricity_code",
        "electricity_ef_gco2_per_kwh",
        "output_label",
        "route",
        "config",
        "year",
        "ef_tco2_per_t",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(dedup.keys()):
            writer.writerow(dedup[key])
    print(f"[fgv] summary updated at {summary_path}")


def main() -> None:
    specs = [
        PortfolioSpec("fgv_regular_br_eaf", Path("configs/fgv_regular.yml"), "BRA"),
        PortfolioSpec("fgv_regular_br_low_eaf", Path("configs/fgv_regular.yml"), "BRA (low-carbon)"),
        PortfolioSpec("fgv_regular_br_high_eaf", Path("configs/fgv_regular.yml"), "BRA (high-carbon)"),
        PortfolioSpec("fgv_high_br_eaf", Path("configs/fgv_high.yml"), "BRA"),
        PortfolioSpec("fgv_high_br_low_eaf", Path("configs/fgv_high.yml"), "BRA (low-carbon)"),
        PortfolioSpec("fgv_high_br_high_eaf", Path("configs/fgv_high.yml"), "BRA (high-carbon)"),
    ]

    elec_map = _load_electricity_map()
    summary_entries: List[dict] = []

    for spec in specs:
        ef = _compute_ef_for_portfolio(spec.spec_path, spec.country_code)
        code_norm = _norm_code(spec.country_code)
        # Portfolio-level EF (blended)
        portfolio_row = {
            "route": ROUTE,
            "config": CONFIG,
            "year": YEAR,
            "base_year": BASE_YEAR,
            "annual_improvement": ANNUAL_IMPROVEMENT,
            "country_code": code_norm,
            "product_config": "portfolio",
            "ef_tco2_per_t": ef,
        }
        # Per-product EFs for each blend component (all for 2025 baseline)
        product_rows = _compute_efs_for_products(spec.spec_path, spec.country_code)
        _write_ef_lookup(spec.label, [portfolio_row, *product_rows])

        summary_entries.append(
            {
                "electricity_code": code_norm,
                "electricity_ef_gco2_per_kwh": elec_map.get(code_norm, ""),
                "output_label": spec.label,
                "route": ROUTE,
                "config": CONFIG,
                "year": YEAR,
                "ef_tco2_per_t": ef,
            }
        )

    _update_summary(summary_entries)


if __name__ == "__main__":
    main()
