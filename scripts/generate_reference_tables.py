#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forge.core.latex import render_run_tables_to_latex
from forge.descriptor import load_sector_descriptor
from forge.steel_core_api_v2 import RouteConfig, ScenarioInputs, run_scenario


ROUTE_SCENARIO_FILES = {
    "BF-BOF": "BF_BOF_coal.yml",
    "DRI-EAF": "DRI_EAF.yml",
    "EAF-Scrap": "scrap_EAF.yml",
}

FINISHED_PICKS = {
    "Manufactured Feed (IP4)": "Stamping/calendering/lamination",
    "Finished Products": "No Coating",
}

ALUMINUM_FINISHED_PICKS = {
    "Metallurgical Aluminum": "Metallurgical Aluminum from Series 1",
    "Basic Aluminum Products": "Raw Aluminum (rolled)",
    "Manufactured Aluminum Products": "Direct use of Basic Aluminum Products",
    "Finished Aluminum Products": "No Coating",
}

FINISHED_PICKS_BY_SECTOR = {
    "steel": FINISHED_PICKS,
    "aluminum": ALUMINUM_FINISHED_PICKS,
}


def _load_any(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _sanitize_name(text: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in (text or "").strip())
    safe = "_".join([t for t in safe.split("_") if t])
    return safe or "run"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Run a single steel scenario and emit LaTeX tabular outputs for the "
            "material balance matrix, energy balance, and emissions balance."
        )
    )
    p.add_argument("--data", default="datasets/steel/likely", help="Dataset directory (default: datasets/steel/likely)")
    p.add_argument("--route", default="BF-BOF", help="Route preset (default: BF-BOF)")
    p.add_argument("--stage", default="Cast", help="Stage key (default: Cast)")
    p.add_argument("--stage-role", default=None, help="Optional stage role key (validation/crude/etc)")
    p.add_argument("--demand", type=float, default=1000.0, help="Demand quantity at stage in kg (default: 1000)")
    p.add_argument("--country", default="BRA", help="Grid country code for electricity EF (default: BRA)")
    p.add_argument(
        "--scenario",
        default=None,
        help=(
            "Scenario YAML/JSON path. If omitted, uses the route default under "
            "<data>/scenarios/ when available."
        ),
    )
    p.add_argument("--picks", default=None, help="Optional YAML/JSON picks_by_material path.")
    p.add_argument("--out-dir", default="results/reference_tables", help="Directory for .tex outputs.")
    p.add_argument("--name", default=None, help="Optional output name prefix (default: auto from route/stage/country).")
    p.add_argument("--decimals", type=int, default=3, help="Decimal places for numeric cells (default: 3).")
    p.add_argument("--zero-tol", type=float, default=1e-9, help="Values with abs(x) < tol print as 0 (default: 1e-9).")
    p.add_argument(
        "--emissions-unit",
        default="kg",
        choices=["kg", "t"],
        help="Unit for emissions table values (kg or t CO2e, default: kg).",
    )
    args = p.parse_args(argv)

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 2

    sector_key: str | None = None
    try:
        descriptor = load_sector_descriptor(data_dir)
        sector_key = str(getattr(descriptor, "key", "") or "").strip().lower() or None
    except Exception:
        sector_key = None

    scenario: dict[str, Any] = {}
    scenario_path: Path | None = None
    if args.scenario:
        scenario_path = Path(args.scenario).expanduser()
        if not scenario_path.is_absolute():
            candidate = (data_dir / "scenarios" / scenario_path).resolve()
            if candidate.exists():
                scenario_path = candidate
            else:
                scenario_path = scenario_path.resolve()
        if not scenario_path.exists():
            print(f"Scenario file not found: {scenario_path}", file=sys.stderr)
            return 2
        payload = _load_any(scenario_path)
        if isinstance(payload, dict):
            scenario = payload
    else:
        scen_name = ROUTE_SCENARIO_FILES.get(str(args.route))
        if scen_name:
            candidate = (data_dir / "scenarios" / scen_name).resolve()
            if candidate.exists():
                scenario_path = candidate
                payload = _load_any(candidate)
                if isinstance(payload, dict):
                    scenario = payload

    picks_by_material: dict[str, Any] = {}
    if args.picks:
        picks_path = Path(args.picks).expanduser().resolve()
        payload = _load_any(picks_path)
        if payload is None:
            picks_by_material = {}
        elif isinstance(payload, dict):
            picks_by_material = payload
        else:
            raise ValueError(f"Expected mapping in picks file {picks_path}, got {type(payload).__name__}")
    elif str(args.stage).strip().lower() == "finished":
        if sector_key and sector_key in FINISHED_PICKS_BY_SECTOR:
            picks_by_material = dict(FINISHED_PICKS_BY_SECTOR[sector_key])
        else:
            picks_by_material = dict(FINISHED_PICKS)

    name = args.name
    if not name:
        name = f"{args.route}_{args.stage}_{args.country}"
    prefix = _sanitize_name(name)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    out = run_scenario(str(data_dir), scn)

    emissions_scale = 1.0 if args.emissions_unit == "kg" else 0.001
    tables = render_run_tables_to_latex(
        balance_matrix=getattr(out, "balance_matrix", None),
        energy_balance=getattr(out, "energy_balance", None),
        emissions=getattr(out, "emissions", None),
        decimals=int(args.decimals),
        zero_tol=float(args.zero_tol),
        emissions_scale=emissions_scale,
    )
    if not tables:
        print("No tables produced (empty run outputs).", file=sys.stderr)
        return 3

    manifest = {
        "data_dir": str(data_dir.resolve()),
        "route": args.route,
        "stage": args.stage,
        "stage_role": args.stage_role,
        "demand": args.demand,
        "country": args.country,
        "scenario_file": str(scenario_path) if scenario_path else None,
        "emissions_unit": args.emissions_unit,
        "outputs": {},
    }

    for key, latex in tables.items():
        path = out_dir / f"{prefix}_{key}.tex"
        if key == "emissions":
            latex = f"% Units: {args.emissions_unit} CO2e\n" + latex
        path.write_text(latex, encoding="utf-8")
        manifest["outputs"][key] = str(path)
        print(f"Wrote {key} LaTeX to {path}")

    (out_dir / f"{prefix}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
