"""Minimal CLI to exercise the refactored core engine.

Examples
  python -m forge.cli.engine_cli \
    --data datasets/steel/likely --route BF-BOF --stage Finished --country BRA --demand 1000 --lci
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import json
import sys

import pandas as pd

# Ensure package resolvable in local dev
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.steel_core_api_v2 import RouteConfig, ScenarioInputs, run_scenario


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a single scenario via refactored engine and print a summary")
    p.add_argument("--data", required=True, help="Dataset directory (e.g., datasets/steel/likely)")
    p.add_argument("--route", default="BF-BOF", help="Route preset: BF-BOF | DRI-EAF | EAF-Scrap | External")
    p.add_argument("--stage", default="Finished", help="Stage key (e.g., Finished, IP3, Cast)")
    p.add_argument("--demand", type=float, default=1000.0, help="Demand quantity at stage (kg)")
    p.add_argument("--country", default=None, help="Grid country code for electricity EF (e.g., BRA)")
    p.add_argument("--lci", action="store_true", help="Enable LCI and write CSVs to output dir")
    p.add_argument("--out", default="results/engine_demo", help="Output directory for CSVs")
    args = p.parse_args(argv)

    if args.lci:
        os.environ.setdefault("FORGE_ENABLE_LCI", "1")

    scn = ScenarioInputs(
        country_code=args.country,
        scenario={},
        route=RouteConfig(route_preset=args.route, stage_key=args.stage, demand_qty=args.demand),
    )

    out = run_scenario(args.data, scn)

    # Summary
    print("=== Scenario Summary ===")
    print(f"Route: {args.route}  Stage: {args.stage}  Demand: {args.demand}")
    print(f"Total CO2e (kg): {out.total_co2e_kg:.4f}")
    if out.emissions is not None and not out.emissions.empty:
        top = out.emissions.sort_values("TOTAL CO2e", ascending=False).head(5)
        print("Top emitters:")
        for proc, row in top.iterrows():
            print(f"  {proc:30s}  {float(row['TOTAL CO2e']):,.3f} t")

    # Optionally write CSVs
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if out.energy_balance is not None and not out.energy_balance.empty:
        out.energy_balance.to_csv(out_dir / "energy_balance.csv")
    if out.emissions is not None and not out.emissions.empty:
        out.emissions.to_csv(out_dir / "emissions.csv")
    if args.lci and out.lci is not None and not out.lci.empty:
        out.lci.to_csv(out_dir / "lci.csv", index=False)

    # Write a lightweight manifest for reproducibility
    try:
        import subprocess as sp
        sha = sp.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        sha = None
    manifest = {
        "data": str(Path(args.data).resolve()),
        "route": args.route,
        "stage": args.stage,
        "demand": args.demand,
        "country": args.country,
        "enable_lci": bool(args.lci),
        "git_sha": sha,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
