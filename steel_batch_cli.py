#!/usr/bin/env python3
"""
Steel model batch runner CLI.

This utility executes one or multiple scenarios through ``run_scenario`` without
going through the Streamlit UI. Scenarios can be supplied directly on the command
line or through a YAML/JSON spec that describes a batch. Typical workflow:

    python steel_batch_cli.py run --spec configs/batch.yml --output results.csv

Spec files can be either a list of runs or a mapping containing ``defaults`` and
``runs``. Each run entry supports:

    name: Optional label for summaries/logs
    data_dir: Override for the model data directory (defaults to CLI --data-dir)
    scenario_file: Path to a YAML/JSON scenario definition
    scenario: Inline mapping merged on top of scenario_file contents
    overrides:
      energy_int.Blast Furnace: 14.5   # dotted path assignments
    route:
      route_preset: auto
      stage_key: Finished
      demand_qty: 1000
    picks_by_material: path/to/picks.yml  # or inline dict
    pre_select_soft: {}
    country_code: BR
    log_dir: logs/steel

When no spec is provided, a single run can be invoked with ``--scenario`` and
optional ``--set`` overrides (e.g. ``--set emission_factors.Electricity=0.45``).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from steel_core_api_v2 import (
    RouteConfig,
    ScenarioInputs,
    run_scenario,
    write_run_log,
)

DEFAULT_DATA_DIR = Path("datasets/steel/likely")


# -----------------------------
# Utilities
# -----------------------------

def _coerce_jsonish(value: str) -> Any:
    """Parse CLI override strings into basic Python types."""
    text = value.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries, returning the mutated base.
    Scalars in ``update`` overwrite values from ``base``.
    """
    for key, value in (update or {}).items():
        if (
            isinstance(value, dict)
            and isinstance(base.get(key), dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _apply_path_override(target: Dict[str, Any], dotted_path: str, value: Any) -> None:
    """Assign ``value`` into ``target`` following a dotted path (e.g. a.b.c)."""
    tokens = [t for t in dotted_path.split(".") if t]
    if not tokens:
        raise ValueError("Override path cannot be empty.")
    cursor = target
    for token in tokens[:-1]:
        if token not in cursor or not isinstance(cursor[token], dict):
            cursor[token] = {}
        cursor = cursor[token]
    cursor[tokens[-1]] = value


def _load_mapping(path: Path) -> Dict[str, Any]:
    """Load a JSON or YAML mapping from disk."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _load_any(path: Path) -> Any:
    """Load arbitrary JSON/YAML content."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _resolve_path(raw: Optional[str | Path], base: Path) -> Optional[Path]:
    if raw is None:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _ensure_dict(value: Any, base_dir: Path) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, Path)):
        resolved = _resolve_path(value, base_dir)
        if resolved is None:
            raise ValueError(f"Could not resolve path for '{value}'.")
        return _load_mapping(resolved)
    raise TypeError(f"Expected dict or path-like, got {type(value).__name__}")


def _scenario_from_spec(spec: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    scenario: Dict[str, Any] = {}
    scenario_file = spec.get("scenario_file")
    if scenario_file:
        scenario_path = _resolve_path(scenario_file, base_dir)
        if scenario_path is None:
            raise ValueError(f"Could not resolve scenario_file: {scenario_file}")
        raw = _load_any(scenario_path)
        if raw:
            if not isinstance(raw, dict):
                raise ValueError(f"Scenario file {scenario_path} must contain a mapping.")
            scenario = raw
    inline = spec.get("scenario")
    if inline:
        if not isinstance(inline, dict):
            raise ValueError("Inline 'scenario' entry must be a dict.")
        _deep_merge(scenario, inline)
    overrides = spec.get("overrides", {})
    if overrides:
        if isinstance(overrides, dict):
            for path_str, val in overrides.items():
                _apply_path_override(scenario, path_str, val)
        elif isinstance(overrides, Iterable) and not isinstance(overrides, (str, bytes)):
            for item in overrides:
                if not isinstance(item, str):
                    raise ValueError("List overrides must contain strings like key=value.")
                if "=" not in item:
                    raise ValueError(f"Override '{item}' must contain '='.")
                key, raw_val = item.split("=", 1)
                _apply_path_override(scenario, key, _coerce_jsonish(raw_val))
        else:
            raise TypeError("Overrides must be a dict or list of assignments.")
    return scenario


def _load_dict_option(spec: Dict[str, Any], key: str, base_dir: Path) -> Dict[str, Any]:
    value = spec.get(key)
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, Path)):
        return _ensure_dict(value, base_dir)
    raise TypeError(f"Expected dict or path-like for '{key}', got {type(value).__name__}")


def _float_or(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _choose_run_name(spec: Dict[str, Any], scenario_path: Optional[Path], idx: int) -> str:
    name = spec.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    if scenario_path:
        return scenario_path.stem
    return f"run_{idx+1}"


@dataclass
class RunPlan:
    name: str
    data_dir: Path
    scenario: Dict[str, Any]
    route: Dict[str, Any]
    picks_by_material: Dict[str, Any]
    pre_select_soft: Dict[str, Any]
    country_code: Optional[str]
    log_dir: Optional[Path]


def _merge_defaults(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def _enumerate_run_specs(
    spec_data: Any,
    spec_path: Optional[Path],
    cli_defaults: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Normalize spec content into a list of per-run dictionaries.
    """
    if spec_data is None:
        return []
    base_dir = spec_path.parent if spec_path else Path.cwd()
    if isinstance(spec_data, list):
        runs_raw = spec_data
        defaults: Dict[str, Any] = {}
    elif isinstance(spec_data, dict):
        if "runs" in spec_data:
            runs_raw = spec_data.get("runs") or []
        else:
            # Allow a top-level run definition
            runs_raw = [spec_data]
        defaults = spec_data.get("defaults") or {}
    else:
        raise ValueError("Spec must be a list or mapping.")

    if not isinstance(runs_raw, list):
        raise ValueError("'runs' must be a list of run definitions.")

    plans: List[Dict[str, Any]] = []
    for idx, raw in enumerate(runs_raw):
        if not isinstance(raw, dict):
            raise ValueError("Each run entry must be a dict.")
        combined = _merge_defaults(defaults, raw)
        combined = _merge_defaults(cli_defaults, combined)
        combined["_spec_base_dir"] = base_dir
        combined["_index"] = idx
        plans.append(combined)
    return plans


def _build_run_plan(raw: Dict[str, Any]) -> RunPlan:
    base_dir: Path = raw.get("_spec_base_dir", Path.cwd())
    idx: int = raw.get("_index", 0)
    scenario_path = _resolve_path(raw.get("scenario_file"), base_dir)
    scenario = _scenario_from_spec(raw, base_dir)
    picks = _load_dict_option(raw, "picks_by_material", base_dir)
    pre_select = _load_dict_option(raw, "pre_select_soft", base_dir)
    route_spec = raw.get("route", {})
    if not isinstance(route_spec, dict):
        raise TypeError("'route' entry must be a dict when provided.")
    for key in ("route_preset", "stage_key", "stage_role", "demand_qty"):
        if key in raw and key not in route_spec:
            route_spec[key] = raw[key]
    name = _choose_run_name(raw, scenario_path, idx)
    data_dir = raw.get("data_dir")
    resolved_data_dir = _resolve_path(data_dir, base_dir) if data_dir else None
    if resolved_data_dir is None:
        resolved_data_dir = DEFAULT_DATA_DIR.resolve()
    country = raw.get("country_code")
    log_dir = _resolve_path(raw.get("log_dir"), base_dir) if raw.get("log_dir") else None
    return RunPlan(
        name=name,
        data_dir=resolved_data_dir,
        scenario=scenario,
        route=route_spec,
        picks_by_material=picks,
        pre_select_soft=pre_select,
        country_code=country,
        log_dir=log_dir,
    )


def _plans_from_spec(spec_path: Optional[Path], cli_defaults: Dict[str, Any]) -> List[RunPlan]:
    if spec_path is None:
        return []
    spec_data = _load_any(spec_path)
    run_specs = _enumerate_run_specs(spec_data, spec_path, cli_defaults)
    return [_build_run_plan(raw) for raw in run_specs]


def _single_run_plan(args: argparse.Namespace) -> RunPlan:
    base_dir = Path.cwd()
    scenario_spec: Dict[str, Any] = {
        "scenario_file": args.scenario,
        "scenario": {},
        "overrides": {},
    }
    overrides = {}
    for assignment in args.set or []:
        if "=" not in assignment:
            raise ValueError(f"Override '{assignment}' must contain '='.")
        key, raw_val = assignment.split("=", 1)
        overrides[key] = _coerce_jsonish(raw_val)
    if overrides:
        scenario_spec["overrides"] = overrides
    picks: Dict[str, Any] = {}
    if args.picks:
        picks = _ensure_dict(args.picks, base_dir)
    pre_select: Dict[str, Any] = {}
    if args.pre_select:
        pre_select = _ensure_dict(args.pre_select, base_dir)
    route = {
        "route_preset": args.route_preset,
        "stage_key": args.stage_key,
        "stage_role": args.stage_role,
        "demand_qty": args.demand,
    }
    name = args.name
    scenario_path = Path(args.scenario).expanduser() if args.scenario else None
    if not name:
        if scenario_path:
            name = scenario_path.stem
        else:
            name = "run"
    data_dir = args.data_dir
    resolved_data_dir = _resolve_path(data_dir, base_dir) if data_dir else DEFAULT_DATA_DIR.resolve()
    log_dir = _resolve_path(args.log_dir, base_dir) if args.log_dir else None
    scenario = _scenario_from_spec(scenario_spec, base_dir)
    return RunPlan(
        name=name,
        data_dir=resolved_data_dir,
        scenario=scenario,
        route=route,
        picks_by_material=picks,
        pre_select_soft=pre_select,
        country_code=args.country_code,
        log_dir=log_dir,
    )


def _build_route_cfg(plan: RunPlan) -> RouteConfig:
    route = dict(plan.route or {})
    route_preset = route.get("route_preset", "auto")
    stage_key = route.get("stage_key", "Finished")
    demand_qty = _float_or(route.get("demand_qty", 1000.0), 1000.0)
    stage_role = route.get("stage_role")
    picks = dict(plan.picks_by_material or {})
    pre_select = dict(plan.pre_select_soft or {})
    return RouteConfig(
        route_preset=route_preset,
        stage_key=stage_key,
        stage_role=stage_role,
        demand_qty=demand_qty,
        picks_by_material=picks,
        pre_select_soft=pre_select,
    )


def _summarize_result(plan: RunPlan, route_cfg: RouteConfig, result) -> Dict[str, Any]:
    total = getattr(result, "total_co2e_kg", None)
    demand_qty = float(route_cfg.demand_qty) if route_cfg.demand_qty is not None else float("nan")
    gross_ef = None
    if (
        total is not None
        and demand_qty
        and math.isfinite(demand_qty)
        and demand_qty != 0.0
    ):
        try:
            gross_ef = float(total) / float(demand_qty)
        except (ZeroDivisionError, TypeError):
            gross_ef = None
    summary: Dict[str, Any] = {
        "name": plan.name,
        "data_dir": str(plan.data_dir),
        "route_preset": route_cfg.route_preset,
        "stage_key": route_cfg.stage_key,
        "stage_role": route_cfg.stage_role or "",
        "demand_qty": route_cfg.demand_qty,
        "country_code": plan.country_code or "",
        "total_co2e_kg": float(total) if total is not None else None,
        "gross_ef_kg_per_unit": float(gross_ef) if gross_ef is not None else None,
    }
    cost = getattr(result, "total_cost", None)
    if cost is not None:
        summary["total_cost"] = cost
    material_cost = getattr(result, "material_cost", None)
    if material_cost is not None:
        summary["material_cost"] = material_cost
    return summary


def _write_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".json", ".jsonl"}:
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return
    if suffix == ".csv":
        fieldnames: List[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fieldnames=fieldnames, f=fh)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        return
    raise ValueError(f"Unsupported output format for '{path}'. Use .csv or .json.")


def _log_payload(plan: RunPlan, route_cfg: RouteConfig, result_summary: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "name": plan.name,
        "data_dir": str(plan.data_dir),
        "country_code": plan.country_code,
        "route": asdict(route_cfg),
        "scenario": plan.scenario,
        "results": result_summary,
        "meta": meta,
    }
    return payload


def run_batch(
    plans: List[RunPlan],
    show_meta: bool = False,
    log_dir_default: Optional[Path] = None,
    fail_fast: bool = False,
) -> Tuple[List[Dict[str, Any]], int]:
    summaries: List[Dict[str, Any]] = []
    failures = 0
    for plan in plans:
        route_cfg = _build_route_cfg(plan)
        scenario_payload = copy.deepcopy(plan.scenario)
        scn_inputs = ScenarioInputs(
            country_code=plan.country_code,
            scenario=scenario_payload,
            route=route_cfg,
        )
        try:
            result = run_scenario(str(plan.data_dir), scn_inputs)
            summary = _summarize_result(plan, route_cfg, result)
            summaries.append(summary)
            total = summary.get("total_co2e_kg")
            gross = summary.get("gross_ef_kg_per_unit")
            def _fmt(value: Any) -> str:
                if value is None:
                    return "n/a"
                try:
                    return f"{float(value):,.4f}"
                except (TypeError, ValueError):
                    return str(value)
            print(
                f"{plan.name}: total_co2e={_fmt(total)} kg; "
                f"gross_EF={_fmt(gross)} kg/unit"
            )
            if show_meta and getattr(result, "meta", None):
                print(json.dumps(result.meta, indent=2))
            log_dir = plan.log_dir or log_dir_default
            if log_dir:
                payload = _log_payload(plan, route_cfg, summary, getattr(result, "meta", {}))
                try:
                    path_written = write_run_log(str(log_dir), payload)
                    print(f"  ↳ log written to {path_written}")
                except Exception as exc:
                    print(f"  ! failed to write log in {log_dir}: {exc}", file=sys.stderr)
        except Exception as exc:
            failures += 1
            message = f"{plan.name}: FAILED – {exc}"
            print(message, file=sys.stderr)
            if fail_fast:
                raise
            summaries.append({
                "name": plan.name,
                "error": str(exc),
            })
    return summaries, failures


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch runner for the steel model core API.",
    )
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Execute one or more scenarios.")
    run_parser.add_argument("--spec", type=Path, help="YAML/JSON file describing scenario runs.")
    run_parser.add_argument("--scenario", type=str, help="Scenario YAML/JSON for a single run (shortcut instead of --spec).")
    run_parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Data directory (default: datasets/steel/likely).")
    run_parser.add_argument("--route", dest="route_preset", default="auto", help="Route preset (default: auto).")
    run_parser.add_argument("--stage-key", default="Finished", help="Stage key to demand (default: Finished).")
    run_parser.add_argument("--stage-role", default=None, help="Optional stage role (validation/crude/etc).")
    run_parser.add_argument("--demand", type=float, default=1000.0, help="Demand quantity at stage (default: 1000).")
    run_parser.add_argument("--country-code", default=None, help="Country code override for electricity.")
    run_parser.add_argument("--set", action="append", help="Override scenario value using dotted path, e.g. energy_int.BlastFurnace=14.5")
    run_parser.add_argument("--picks", type=str, help="Path to YAML/JSON with picks_by_material.")
    run_parser.add_argument("--pre-select", type=str, help="Path to YAML/JSON with pre_select_soft.")
    run_parser.add_argument("--output", type=Path, help="Optional CSV/JSON summary output path.")
    run_parser.add_argument("--log-dir", type=str, help="Directory to store detailed JSON logs per run.")
    run_parser.add_argument("--show-meta", action="store_true", help="Print the meta payload returned by run_scenario.")
    run_parser.add_argument("--name", type=str, help="Friendly name for a single run.")
    run_parser.add_argument("--countries", nargs="+", help="List of country codes to cycle through (overrides --country-code).")
    run_parser.add_argument("--fail-fast", action="store_true", help="Abort on first failure.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 1
    cli_defaults = {}
    if args.data_dir:
        cli_defaults["data_dir"] = args.data_dir
    if args.route_preset:
        cli_defaults.setdefault("route", {})["route_preset"] = args.route_preset
    if args.stage_key:
        cli_defaults.setdefault("route", {})["stage_key"] = args.stage_key
    if args.stage_role:
        cli_defaults.setdefault("route", {})["stage_role"] = args.stage_role
    if args.demand is not None:
        cli_defaults.setdefault("route", {})["demand_qty"] = args.demand
    if args.country_code:
        cli_defaults["country_code"] = args.country_code
    spec_plans: List[RunPlan] = []
    if args.spec:
        spec_plans = _plans_from_spec(Path(args.spec), cli_defaults)
    plans: List[RunPlan]
    if spec_plans:
        plans = spec_plans
    else:
        if not args.scenario:
            print("Provide either --spec or --scenario for a single run.", file=sys.stderr)
            return 2
        try:
            base_plan = _single_run_plan(args)
        except Exception as exc:
            print(f"Failed to build run plan: {exc}", file=sys.stderr)
            return 2
        if args.countries:
            plans = []
            base_route = copy.deepcopy(base_plan.route)
            base_picks = copy.deepcopy(base_plan.picks_by_material)
            base_pre_select = copy.deepcopy(base_plan.pre_select_soft)
            base_scenario = base_plan.scenario
            for idx, code in enumerate(args.countries):
                country_code = (code or "").strip() or None
                plan_name = base_plan.name
                if len(args.countries) > 1:
                    suffix = (country_code or f"run{idx+1}").replace(" ", "")
                    plan_name = f"{base_plan.name}_{suffix}"
                plans.append(
                    RunPlan(
                        name=plan_name,
                        data_dir=base_plan.data_dir,
                        scenario=copy.deepcopy(base_scenario),
                        route=copy.deepcopy(base_route),
                        picks_by_material=copy.deepcopy(base_picks),
                        pre_select_soft=copy.deepcopy(base_pre_select),
                        country_code=country_code,
                        log_dir=base_plan.log_dir,
                    )
                )
        else:
            plans = [base_plan]
    log_dir_default = _resolve_path(args.log_dir, Path.cwd()) if args.log_dir else None
    try:
        summaries, failures = run_batch(
            plans,
            show_meta=args.show_meta,
            log_dir_default=log_dir_default,
            fail_fast=args.fail_fast,
        )
    except Exception as exc:
        print(f"Execution aborted: {exc}", file=sys.stderr)
        return 3
    if args.output:
        try:
            _write_summary(args.output, summaries)
            print(f"Summary written to {args.output}")
        except Exception as exc:
            print(f"Failed to write summary: {exc}", file=sys.stderr)
            return 4
    return 0 if failures == 0 else 5


if __name__ == "__main__":
    sys.exit(main())
