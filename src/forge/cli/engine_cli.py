"""Minimal CLI to exercise the refactored core engine.

Examples
  python -m forge.cli.engine_cli \
    --data datasets/steel/likely --route BF-BOF --stage Finished --country BRA --demand 1000
"""
from __future__ import annotations

import argparse
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
    p.add_argument("--out", default="results/engine_demo", help="Output directory for CSVs")
    p.add_argument("--show-gas-meta", action="store_true", help="Print process-gas emission factor diagnostics")
    args = p.parse_args(argv)

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

    if args.show_gas_meta and out.meta:
        gas_keys = [
            "EF_process_gas",
            "ef_gas_blended",
            "EF_coke_gas",
            "EF_bf_gas",
            "total_process_gas_MJ",
            "direct_use_gas_MJ",
            "electricity_gas_MJ",
            "gas_coke_MJ",
            "gas_bf_MJ",
            "f_internal_gas",
        ]
        print("Process gas diagnostics:")
        for key in gas_keys:
            if key in out.meta:
                print(f"  {key:22s}: {out.meta.get(key)}")
        details = out.meta.get("gas_source_details") or {}
        if details:
            print("  Source breakdown (MJ):")
            for proc, mj in details.items():
                print(f"    {proc:25s} {mj}")
        credits = out.meta.get("gas_credit_details") or {}
        if credits:
            print("  Credit allocation (MJ):")
            for proc, info in credits.items():
                du = info.get("direct_use_MJ", 0.0)
                elec = info.get("electricity_MJ", 0.0)
                print(f"    {proc:25s} direct={du:.2f}  elec={elec:.2f}")

    # Optionally write CSVs
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if out.energy_balance is not None and not out.energy_balance.empty:
        out.energy_balance.to_csv(out_dir / "energy_balance.csv")
    if out.emissions is not None and not out.emissions.empty:
        out.emissions.to_csv(out_dir / "emissions.csv")

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
        "git_sha": sha,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
