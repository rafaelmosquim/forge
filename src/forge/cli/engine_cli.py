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
from typing import Any

import yaml

# Ensure package resolvable in local dev
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.core.latex import render_run_tables_to_latex
from forge.steel_core_api_v2 import RouteConfig, ScenarioInputs, run_scenario


def _load_any(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _resolve_dataset_relative_path(dataset_dir: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    # First try under <dataset>/scenarios/
    candidate = (dataset_dir / "scenarios" / p).resolve()
    if candidate.exists():
        return candidate
    return p.resolve()



def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a single scenario via refactored engine and print a summary")
    p.add_argument("--data", required=True, help="Dataset directory (e.g., datasets/steel/likely)")
    p.add_argument("--route", default="BF-BOF", help="Route preset: BF-BOF | DRI-EAF | EAF-Scrap | External")
    p.add_argument("--stage", default="Finished", help="Stage key (e.g., Finished, IP3, Cast)")
    p.add_argument("--stage-role", default=None, help="Optional stage role key (validation/crude/etc)")
    p.add_argument("--demand", type=float, default=1000.0, help="Demand quantity at stage (kg)")
    p.add_argument("--country", default=None, help="Grid country code for electricity EF (e.g., BRA)")
    p.add_argument("--scenario", default=None, help="Optional scenario YAML/JSON path (or filename under <data>/scenarios/).")
    p.add_argument("--picks", default=None, help="Optional YAML/JSON picks_by_material path.")
    p.add_argument("--out", default="results/engine_demo", help="Output directory for CSVs")
    p.add_argument("--show-gas-meta", action="store_true", help="Print process-gas emission factor diagnostics")
    p.add_argument("--latex", action="store_true", help="Write LaTeX tabular outputs for balance/energy/emissions.")
    p.add_argument("--latex-decimals", type=int, default=3, help="Decimal places for LaTeX outputs (default: 3).")
    p.add_argument("--latex-zero-tol", type=float, default=1e-9, help="Values with abs(x) < tol print as 0 (default: 1e-9).")
    p.add_argument(
        "--latex-emissions-unit",
        default="kg",
        choices=["kg", "t"],
        help="Unit for LaTeX emissions table values (kg or t CO2e, default: kg).",
    )
    args = p.parse_args(argv)

    dataset_dir = Path(args.data)
    scenario: dict[str, Any] = {}
    scenario_path: Path | None = None
    if args.scenario:
        scenario_path = _resolve_dataset_relative_path(dataset_dir, args.scenario)
        if not scenario_path.exists():
            print(f"Scenario file not found: {scenario_path}", file=sys.stderr)
            return 2
        payload = _load_any(scenario_path)
        if isinstance(payload, dict):
            scenario = payload

    picks_by_material: dict[str, Any] = {}
    picks_path: Path | None = None
    if args.picks:
        picks_path = Path(args.picks).expanduser().resolve()
        payload = _load_any(picks_path)
        if payload is None:
            picks_by_material = {}
        elif isinstance(payload, dict):
            picks_by_material = payload
        else:
            print(f"Expected mapping in picks file {picks_path}, got {type(payload).__name__}.", file=sys.stderr)
            return 2

    scn = ScenarioInputs(
        country_code=args.country,
        scenario=scenario,
        route=RouteConfig(
            route_preset=args.route,
            stage_key=args.stage,
            stage_role=args.stage_role,
            demand_qty=args.demand,
            picks_by_material=picks_by_material,
        ),
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
            print(f"  {proc:30s}  {float(row['TOTAL CO2e']):,.3f} kg")

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

    if out.balance_matrix is not None and not out.balance_matrix.empty:
        out.balance_matrix.to_csv(out_dir / "balance_matrix.csv")
    if out.energy_balance is not None and not out.energy_balance.empty:
        out.energy_balance.to_csv(out_dir / "energy_balance.csv")
    if out.emissions is not None and not out.emissions.empty:
        out.emissions.to_csv(out_dir / "emissions.csv")

    if args.latex:
        emissions_scale = 1.0 if args.latex_emissions_unit == "kg" else 0.001
        tables = render_run_tables_to_latex(
            balance_matrix=out.balance_matrix,
            energy_balance=out.energy_balance,
            emissions=out.emissions,
            decimals=int(args.latex_decimals),
            zero_tol=float(args.latex_zero_tol),
            emissions_scale=emissions_scale,
        )
        for key, latex in tables.items():
            if key == "emissions":
                latex = f"% Units: {args.latex_emissions_unit} CO2e\n" + latex
            (out_dir / f"{key}.tex").write_text(latex, encoding="utf-8")

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
        "stage_role": args.stage_role,
        "demand": args.demand,
        "country": args.country,
        "scenario_file": str(scenario_path) if scenario_path else None,
        "picks_file": str(picks_path) if picks_path else None,
        "git_sha": sha,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
