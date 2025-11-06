# -*- coding: utf-8 -*-
"""
Monte Carlo runner using triangular sampling between min/likely/max datasets.

Purpose
- Run a fixed configuration (route + picks or portfolio basket) many times,
  sampling uncertain data from three dataset variants:
    - min_dir: optimistic_low (min)
    - mode_dir: likely (mode)
    - max_dir: pessimistic_high (max)

What varies
- energy_int, energy_matrix, energy_content, emission_factors are sampled
  elementwise via numpy.random.triangular(min, mode, max).

How to use (examples)
  # Simple picks (stamping/no coat)
  python3 -m forge.scenarios.monte_carlo_tri \
      --min datasets/steel/optimistic_low \
      --mode datasets/steel/likely \
      --max datasets/steel/pessimistic_high \
      --route BF-BOF \
      --n 500 \
      --out results/mc_finished

  # Portfolio basket (route-agnostic down to IP4/Finishing)
  python3 -m forge.scenarios.monte_carlo_tri \
      --min datasets/steel/optimistic_low \
      --mode datasets/steel/likely \
      --max datasets/steel/pessimistic_high \
      --route BF-BOF \
      --portfolio configs/finished_steel_portfolio.yml \
      --blend finished_portfolio \
      --n 200 \
      --out results/mc_finished

Outputs
- Writes CSV at <out_dir>/mc_summary.csv with per-sample totals and EF.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from forge.steel_core_api_v2 import RouteConfig, ScenarioInputs, run_scenario
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def _load_yaml_map(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    # unwrap single-key files like { processes: {...} }
    if isinstance(data, dict) and len(data) == 1:
        key, val = next(iter(data.items()))
        if isinstance(val, dict):
            return val
    return data if isinstance(data, dict) else {}


def _tri(a: float, c: float, b: float) -> float:
    # numpy.triangular(left, mode, right)
    try:
        return float(np.random.triangular(float(a), float(c), float(b)))
    except Exception:
        return float(c)


def _tri_map(min_map: Dict[str, Any], mode_map: Dict[str, Any], max_map: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = set(min_map.keys()) | set(mode_map.keys()) | set(max_map.keys())
    for k in keys:
        vmin = min_map.get(k)
        vmod = mode_map.get(k)
        vmax = max_map.get(k)
        # nested dict → recurse
        if isinstance(vmin, dict) or isinstance(vmod, dict) or isinstance(vmax, dict):
            out[k] = _tri_map(vmin or {}, vmod or {}, vmax or {})
            continue
        # scalars → sample
        try:
            a = float(vmin) if vmin is not None else float(vmod or 0.0)
            c = float(vmod) if vmod is not None else a
            b = float(vmax) if vmax is not None else c
            out[k] = _tri(a, c, b)
        except Exception:
            out[k] = vmod
    return out


def _renormalize_energy_matrix(em: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for proc, shares in (em or {}).items():
        row = dict(shares or {})
        # consider numeric carriers only
        numeric = {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
        s = sum(max(0.0, x) for x in numeric.values())
        if s > 1e-12:
            for k in numeric:
                row[k] = max(0.0, float(row[k])) / s
        out[proc] = row
    return out


def _sample_overrides(min_dir: Path, mode_dir: Path, max_dir: Path) -> Dict[str, Any]:
    def L(d: Path, name: str) -> Dict[str, Any]:
        return _load_yaml_map(d / name)

    emap_min = L(min_dir, "energy_matrix.yml")
    emap_mod = L(mode_dir, "energy_matrix.yml")
    emap_max = L(max_dir, "energy_matrix.yml")
    econtent = _tri_map(L(min_dir, "energy_content.yml"), L(mode_dir, "energy_content.yml"), L(max_dir, "energy_content.yml"))
    eint = _tri_map(L(min_dir, "energy_int.yml"), L(mode_dir, "energy_int.yml"), L(max_dir, "energy_int.yml"))
    eefs = _tri_map(L(min_dir, "emission_factors.yml"), L(mode_dir, "emission_factors.yml"), L(max_dir, "emission_factors.yml"))

    emap = _tri_map(emap_min, emap_mod, emap_max)
    emap = _renormalize_energy_matrix(emap)

    return {
        "energy_int": eint,
        "energy_matrix": emap,
        "energy_content": econtent,
        "emission_factors": eefs,
    }


def _merge(a: Dict[str, Any], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not b:
        return a
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _run_single(data_dir: Path, route: str, picks: Dict[str, str], scenario: Dict[str, Any]) -> float:
    rc = RouteConfig(
        route_preset=route,
        stage_key="Finished",
        stage_role=None,
        demand_qty=1000.0,
        picks_by_material=dict(picks or {}),
        pre_select_soft={},
    )
    si = ScenarioInputs(country_code=None, scenario=scenario, route=rc)
    out = run_scenario(str(data_dir), si)
    return float(getattr(out, "total_co2e_kg", 0.0) or 0.0)


def _load_portfolio(spec_path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]], list[Tuple[str, float]]]:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    defaults = spec.get("defaults") or {}
    runs = spec.get("runs") or []
    blends = spec.get("blends") or []
    run_picks = {r.get("name"): (r.get("picks_by_material") or {}) for r in runs if isinstance(r, dict)}
    # pick first blend
    blend = None
    for b in blends:
        if isinstance(b, dict):
            blend = b; break
    comps = []
    if blend:
        for c in blend.get("components", []):
            try:
                comps.append((c.get("run"), float(c.get("share", 0.0) or 0.0)))
            except Exception:
                continue
    return defaults, run_picks, comps


def run_mc(
    min_dir: Path,
    mode_dir: Path,
    max_dir: Path,
    base_dir: Path,
    route: str,
    n: int,
    out_dir: Path,
    picks: Optional[Dict[str, str]] = None,
    scenario_defaults: Optional[Dict[str, Any]] = None,
    portfolio: Optional[Path] = None,
    *,
    plot: bool = True,
) -> Path:
    np.random.seed()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        sampled = _sample_overrides(min_dir, mode_dir, max_dir)
        scn = _merge(sampled, scenario_defaults)
        if portfolio and portfolio.exists():
            defaults, run_picks, comps = _load_portfolio(portfolio)
            scn = _merge(scn, defaults.get("scenario") or {})
            total = 0.0
            total_w = 0.0
            for run_name, share in comps:
                pp = dict(defaults.get("picks_by_material") or {})
                pp.update(run_picks.get(run_name, {}))
                val = _run_single(base_dir, route, pp, scn)
                total += float(share) * val
                total_w += float(share)
            total = total / total_w if total_w > 0 else total
        else:
            total = _run_single(base_dir, route, picks or {}, scn)
        rows.append({"sample": i+1, "total_co2e_kg": total, "ef_kg_per_unit": total/1000.0})
    out_path = out_dir / "mc_summary.csv"
    import csv
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample", "total_co2e_kg", "ef_kg_per_unit"])
        w.writeheader(); w.writerows(rows)

    # Optional plots and stats
    if plot and rows:
        ef = np.array([float(r["ef_kg_per_unit"]) for r in rows], dtype=float)
        mean = float(np.mean(ef))
        median = float(np.median(ef))
        p05 = float(np.percentile(ef, 5))
        p95 = float(np.percentile(ef, 95))

        title_suffix = f"route={route}, n={n}"
        if portfolio and portfolio.exists():
            title_suffix += f", basket={portfolio.name}"

        # Histogram
        plt.figure(figsize=(8,5))
        plt.hist(ef, bins=40, color="#4C72B0", alpha=0.85, edgecolor="black")
        for x, color, lbl, ls in [
            (median, "#D62728", "median", "-"),
            (p05, "#2CA02C", "p05", "--"),
            (p95, "#2CA02C", "p95", "--"),
        ]:
            plt.axvline(x, color=color, linestyle=ls, linewidth=1.8, label=f"{lbl}: {x:.4f}")
        plt.title(f"MC EF Histogram ({title_suffix})")
        plt.xlabel("EF (kg CO₂ per unit)")
        plt.ylabel("Frequency")
        plt.legend()
        hist_path = out_dir / "mc_ef_hist.png"
        plt.tight_layout(); plt.savefig(hist_path, dpi=160); plt.close()

        # ECDF
        xs = np.sort(ef)
        ys = np.linspace(0, 1, len(xs), endpoint=True)
        plt.figure(figsize=(8,5))
        plt.plot(xs, ys, color="#4C72B0", linewidth=2)
        for x, color, lbl, ls in [
            (median, "#D62728", "median", "-"),
            (p05, "#2CA02C", "p05", "--"),
            (p95, "#2CA02C", "p95", "--"),
        ]:
            plt.axvline(x, color=color, linestyle=ls, linewidth=1.4)
        plt.title(f"MC EF ECDF ({title_suffix})")
        plt.xlabel("EF (kg CO₂ per unit)")
        plt.ylabel("Cumulative probability")
        ecdf_path = out_dir / "mc_ef_ecdf.png"
        plt.tight_layout(); plt.savefig(ecdf_path, dpi=160); plt.close()

        # Stats JSON
        stats_path = out_dir / "mc_stats.json"
        with stats_path.open("w", encoding="utf-8") as fh:
            json.dump({"mean": mean, "median": median, "p05": p05, "p95": p95, "n": n}, fh, indent=2)
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Monte Carlo (triangular) runner across dataset min/mode/max.")
    p.add_argument("--min", dest="min_dir", required=True)
    p.add_argument("--mode", dest="mode_dir", required=True)
    p.add_argument("--max", dest="max_dir", required=True)
    p.add_argument("--base", dest="base_dir", default="datasets/steel/likely")
    p.add_argument("--route", dest="route", default="BF-BOF")
    p.add_argument("--n", dest="n", type=int, default=200)
    p.add_argument("--out", dest="out_dir", default="results/mc")
    p.add_argument("--picks", dest="picks", help="YAML/JSON file with picks_by_material (simple mode).")
    p.add_argument("--portfolio", dest="portfolio", help="Portfolio YAML for downstream basket.")
    p.add_argument("--no-plot", action="store_true", help="Disable PNG plots (hist/cdf).")
    args = p.parse_args(argv)

    min_dir = Path(args.min_dir)
    mode_dir = Path(args.mode_dir)
    max_dir = Path(args.max_dir)
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)

    picks = None
    if args.picks:
        with Path(args.picks).open("r", encoding="utf-8") as fh:
            if args.picks.endswith(('.yml', '.yaml')):
                picks = yaml.safe_load(fh) or {}
            else:
                picks = json.load(fh) or {}
            if not isinstance(picks, dict):
                raise ValueError("picks file must be a mapping")

    portfolio = Path(args.portfolio) if args.portfolio else None
    out_path = run_mc(
        min_dir=min_dir,
        mode_dir=mode_dir,
        max_dir=max_dir,
        base_dir=base_dir,
        route=args.route,
        n=args.n,
        out_dir=out_dir,
        picks=picks,
        scenario_defaults=None,
        portfolio=portfolio,
        plot=not args.no_plot,
    )
    print(f"Monte Carlo summary written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
