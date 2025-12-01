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
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from forge.steel_core_api_v2 import (
    RouteConfig,
    ScenarioInputs,
    RunOutputs,
    run_scenario,
    write_run_log,
)

DEFAULT_DATA_DIR = Path("datasets/steel/likely")
REPORTED_YIELD_FACTOR = 0.85  # Core outputs "raw" CO2; divide by 0.85 to gross up totals
REPORTED_YIELD_DIVISOR = 1.0 / REPORTED_YIELD_FACTOR  # ≈1.17647×


# -----------------------------
# Utilities
# -----------------------------

def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.4f}"
    except (TypeError, ValueError):
        return str(value)

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
    """Resolve a path relative to spec base, with project-root fallback.

    Behavior:
    - Absolute paths are returned as-is.
    - Relative paths are resolved against the spec base directory first.
    - If that does not exist, fall back to project root (ROOT_DIR) for common
      repo-relative references like 'datasets/...', 'configs/...', etc.
    """
    if raw is None:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    # First try relative to the spec file
    p1 = (base / candidate).resolve()
    if p1.exists():
        return p1
    # Fallback: resolve relative to the repository root
    # Compute repository root as three levels above this file (…/src/forge/cli → repo)
    try:
        repo_root = Path(__file__).resolve().parents[3]
    except Exception:
        repo_root = Path.cwd()
    p2 = (repo_root / candidate).resolve()
    return p2


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


def _expand_grid_runs(spec_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate run entries from a compact 'grid' definition inside the spec.

    Expected shape (all keys optional, sensible defaults inferred):

      grid:
        name_prefix: paper
        years: [2030, 2040]
        scenario: [BAU, Conservative, Aggressive]
        utilization_rate: [65, 80, 100]
        intensity_improvement: [0, 1, 2]
        charcoal_expansion: [None, Expansion]
        dri_mix: [Blue, Green]
        base_demand: 1000

      grid_policies:
        scenario_route_logic:
          BAU: {2030: BF-BOF, 2040: BF-BOF}
          Conservative: {2030: BF-BOF, 2040: DRI-EAF}
          Aggressive: {2030: DRI-EAF, 2040: DRI-EAF}
        dri_mix_definitions:
          Blue:
            2030: { Gas: 1.0 }
            2040: { Gas: 0.70, Biomethane: 0.10, "Green hydrogen": 0.20 }
          Green:
            2030: { Gas: 0.70, Biomethane: 0.10, "Green hydrogen": 0.20 }
            2040: { Gas: 0.40, Biomethane: 0.20, "Green hydrogen": 0.40 }
        charcoal_share_schedule: {2030: 0.0, 2035: 0.2, 2040: 0.4, 2045: 0.6}

    Returns a list of per-run dicts compatible with the 'runs' entries.
    """
    if not isinstance(spec_data, dict):
        return []
    grid = spec_data.get("grid")
    if not isinstance(grid, dict):
        return []
    pol = spec_data.get("grid_policies") or {}
    # Axes
    years = list(grid.get("years", [2030, 2040]))
    scenarios = list(grid.get("scenario", ["BAU", "Conservative", "Aggressive"]))
    utils = list(grid.get("utilization_rate", [65, 80, 100]))
    improves = list(grid.get("intensity_improvement", [0, 1, 2]))
    char_opts = list(grid.get("charcoal_expansion", ["None", "Expansion"]))
    mixes = list(grid.get("dri_mix", ["Blue", "Green"]))
    prefix = str(grid.get("name_prefix", "paper")).strip() or "paper"
    # Always run with demand 1000 by default (clean output); utilization does not scale demand
    base_demand = 1000.0

    # Policies
    route_logic = pol.get("scenario_route_logic") or {
        "BAU": {2030: "BF-BOF", 2040: "BF-BOF"},
        "Conservative": {2030: "BF-BOF", 2040: "DRI-EAF"},
        "Aggressive": {2030: "DRI-EAF", 2040: "DRI-EAF"},
    }
    dri_defs = pol.get("dri_mix_definitions")  # optional; default handled in core
    charcoal_sched = pol.get("charcoal_share_schedule")  # optional

    def _route_for(scn: str, year: int) -> str:
        m = route_logic.get(scn) or {}
        # lenient: accept string keys
        if str(year) in m:
            return m[str(year)]
        if year in m:
            return m[year]
        # nearest <=
        try:
            keys = sorted({int(k) for k in m.keys()})
            chosen = None
            for k in keys:
                if k <= year:
                    chosen = k
            if chosen is None and keys:
                chosen = min(keys)
            if chosen is not None:
                return m.get(chosen) or m.get(str(chosen))
        except Exception:
            pass
        # fallback
        return "BF-BOF"

    runs: List[Dict[str, Any]] = []
    for year in years:
        try:
            y = int(year)
        except Exception:
            continue
        for scn in scenarios:
            scn_str = str(scn)
            rp = _route_for(scn_str, y)
            for util in utils:
                # Utilization is retained for naming, but demand is fixed at 1000
                try:
                    u = float(util)
                except Exception:
                    u = 100.0
                demand_qty = base_demand
                for imp in improves:
                    try:
                        imp_rate = float(imp)
                    except Exception:
                        imp_rate = 0.0
                    for char_mode in char_opts:
                        char_str = str(char_mode)
                        for mix in mixes:
                            mix_str = str(mix)
                            name = f"{prefix}_{scn_str}_{int(u)}u_{int(imp_rate)}imp_{char_str}_{mix_str}_{y}"
                            entry: Dict[str, Any] = {
                                "name": name.replace(" ", ""),
                                "route": {
                                    "route_preset": rp,
                                    "stage_key": grid.get("stage_key", "Finished"),
                                    "demand_qty": demand_qty,
                                },
                                "scenario": {
                                    "snapshot_year": y,
                                    "energy_int_schedule": {"rate_pct_per_year": imp_rate},
                                    "charcoal_expansion": "Expansion" if char_str.lower() == "expansion" else "None",
                                    "dri_mix": mix_str,
                                },
                            }
                            if dri_defs:
                                entry["scenario"]["dri_mix_definitions"] = dri_defs
                            if charcoal_sched:
                                entry["scenario"]["charcoal_share_schedule"] = charcoal_sched
                            runs.append(entry)
    return runs


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


@dataclass
class RunRecord:
    plan: RunPlan
    route_cfg: RouteConfig
    result: RunOutputs
    summary: Dict[str, Any]


@dataclass
class BlendComponent:
    run_name: str
    share: float
    label: Optional[str] = None


@dataclass
class BlendPlan:
    name: str
    components: List[BlendComponent]
    notes: Optional[str] = None
    log_dir: Optional[Path] = None


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


def _parse_share_value(raw: Any) -> float:
    if raw is None:
        raise ValueError("Blend component missing 'share'.")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError("Blend component share cannot be empty.")
        if text.endswith("%"):
            text = text[:-1].strip()
            if not text:
                raise ValueError("Blend component share cannot be '%' without a value.")
            return float(text) / 100.0
        return float(text)
    return float(raw)


def _build_blend_plan(raw: Dict[str, Any], run_lookup: Dict[str, RunPlan], base_dir: Path) -> BlendPlan:
    if not isinstance(raw, dict):
        raise ValueError("Each blend entry must be a dict.")
    name_val = raw.get("name")
    if not isinstance(name_val, str) or not name_val.strip():
        raise ValueError("Blend entry requires a non-empty 'name'.")
    name = name_val.strip()
    components_raw = raw.get("components") or raw.get("mix") or raw.get("portfolio")
    if not isinstance(components_raw, list) or not components_raw:
        raise ValueError(f"Blend '{name}' must define a non-empty 'components' list.")
    components: List[BlendComponent] = []
    for idx, item in enumerate(components_raw):
        if not isinstance(item, dict):
            raise ValueError(f"Blend '{name}' component #{idx+1} must be a dict.")
        run_name_val = item.get("run") or item.get("name")
        if not isinstance(run_name_val, str) or not run_name_val.strip():
            raise ValueError(f"Blend '{name}' component #{idx+1} missing 'run' name.")
        run_name = run_name_val.strip()
        if run_name not in run_lookup:
            raise ValueError(f"Blend '{name}' references unknown run '{run_name}'.")
        try:
            share = _parse_share_value(item.get("share", item.get("fraction", item.get("weight"))))
        except ValueError as exc:
            raise ValueError(f"Blend '{name}' component '{run_name}': {exc}") from exc
        if share < 0:
            raise ValueError(f"Blend '{name}' component '{run_name}' has negative share {share}.")
        label = item.get("label")
        if isinstance(label, str):
            label = label.strip() or None
        components.append(BlendComponent(run_name=run_name, share=share, label=label))
    notes = raw.get("notes") or raw.get("description")
    if isinstance(notes, str):
        notes = notes.strip() or None
    log_dir_raw = raw.get("log_dir")
    log_dir = _resolve_path(log_dir_raw, base_dir) if log_dir_raw else None
    return BlendPlan(name=name, components=components, notes=notes, log_dir=log_dir)


def _plans_from_spec(
    spec_path: Optional[Path],
    cli_defaults: Dict[str, Any],
) -> Tuple[List[RunPlan], List[BlendPlan]]:
    if spec_path is None:
        return [], []
    spec_data = _load_any(spec_path)
    # Optionally expand grid into concrete run entries and merge with existing 'runs'
    if isinstance(spec_data, dict) and "grid" in spec_data:
        grid_runs = _expand_grid_runs(spec_data)
        orig_runs = spec_data.get("runs") or []
        combined = list(orig_runs) + grid_runs
        spec_data = {**spec_data, "runs": combined}
    run_specs = _enumerate_run_specs(spec_data, spec_path, cli_defaults)
    plans = [_build_run_plan(raw) for raw in run_specs]
    blend_plans: List[BlendPlan] = []
    if isinstance(spec_data, dict):
        blends_raw = spec_data.get("blends") or spec_data.get("portfolios") or []
        if blends_raw:
            if not isinstance(blends_raw, list):
                raise ValueError("'blends' must be a list of blend definitions.")
            base_dir = spec_path.parent
            run_lookup = {}
            for plan in plans:
                if plan.name in run_lookup:
                    raise ValueError(f"Duplicate run name '{plan.name}' in spec; blend references are ambiguous.")
                run_lookup[plan.name] = plan
            blend_plans = [_build_blend_plan(raw_blend, run_lookup, base_dir) for raw_blend in blends_raw]
    return plans, blend_plans


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
    demand_qty = _float_or(route_cfg.demand_qty, 0.0)
    raw_total = None
    total_with_yield = None
    raw_ef = None
    gross_ef = None
    if total is not None:
        try:
            raw_total = float(total)
            total_with_yield = raw_total * REPORTED_YIELD_DIVISOR
            if demand_qty > 0:
                raw_ef = raw_total / demand_qty
                gross_ef = total_with_yield / demand_qty
        except (TypeError, ValueError):
            raw_total = None
            total_with_yield = None
    summary: Dict[str, Any] = {
        "name": plan.name,
        "data_dir": str(plan.data_dir),
        "route_preset": route_cfg.route_preset,
        "stage_key": route_cfg.stage_key,
        "stage_role": route_cfg.stage_role or "",
        "country_code": plan.country_code or "",
        "raw_co2e_kg": raw_total if raw_total is not None else None,
        "total_co2e_kg": total_with_yield if total_with_yield is not None else None,
        "raw_ef_kg_per_unit": raw_ef,
        "gross_ef_kg_per_unit": gross_ef,
    }
    if total is not None:
        try:
            total_val = float(total)
            summary["raw_co2e_kg"] = total_val
            summary["total_co2e_kg"] = total_val * REPORTED_YIELD_DIVISOR
            if demand_qty > 0:
                summary["raw_ef_kg_per_unit"] = total_val / demand_qty
                summary["gross_ef_kg_per_unit"] = (total_val * REPORTED_YIELD_DIVISOR) / demand_qty
        except (TypeError, ValueError):
            pass
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


def _write_lci_csv(df: Optional[pd.DataFrame], out_dir: Path, name: str) -> Optional[Path]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    path = out_dir / f"{safe_name}.csv"
    try:
        df.to_csv(path, index=False)
        return path
    except Exception:
        return None


def _compute_blend_result(
    blend: BlendPlan,
    record_map: Dict[str, RunRecord],
) -> Tuple[Dict[str, Any], RunOutputs]:
    def _weight_numeric_frame(df: Optional[pd.DataFrame], weight: float) -> Optional[pd.DataFrame]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        numeric = df.select_dtypes(include="number").copy()
        if numeric.empty:
            return None
        return numeric.mul(weight)

    def _accumulate_numeric(
        accumulator: Optional[pd.DataFrame],
        addition: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        if addition is None:
            return accumulator
        if accumulator is None:
            return addition
        return accumulator.add(addition, fill_value=0.0)

    def _weight_lci(df: Optional[pd.DataFrame], weight: float) -> Optional[pd.DataFrame]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        lci = df.copy()
        if 'Amount' not in lci.columns:
            return None
        try:
            lci['Amount'] = pd.to_numeric(lci['Amount'], errors='coerce').fillna(0.0) * float(weight)
        except Exception:
            return None
        return lci

    def _accumulate_lci(acc: Optional[pd.DataFrame], add: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if add is None:
            return acc
        if acc is None:
            acc = add
        else:
            acc = pd.concat([acc, add], ignore_index=True)
        # Group by non-numeric identity columns and sum Amount
        keys = [
            c for c in ['Process', 'Output', 'Flow', 'Input', 'Category', 'ValueUnit', 'Unit']
            if c in acc.columns
        ]
        if not keys or 'Amount' not in acc.columns:
            return acc
        grouped = (
            acc.groupby(keys, dropna=False, as_index=False)['Amount']
               .sum()
        )
        # Reorder columns for consistency
        cols = keys + ['Amount']
        return grouped.loc[:, cols]

    components_payload = []
    weighted_routes: defaultdict[str, float] = defaultdict(float)
    weighted_levels: defaultdict[str, float] = defaultdict(float)
    total_share = 0.0
    entries: List[Tuple[BlendComponent, RunRecord]] = []
    for component in blend.components:
        record = record_map.get(component.run_name)
        if record is None:
            raise ValueError(f"Blend '{blend.name}' references run '{component.run_name}' which did not complete successfully.")
        entries.append((component, record))
        total_share += component.share
    if total_share <= 0.0:
        raise ValueError(f"Blend '{blend.name}' has non-positive total share ({total_share}).")
    demand_qty = 0.0
    total_co2e = 0.0
    total_cost = 0.0
    material_cost = 0.0
    cost_valid = True
    material_cost_valid = True
    energy_balance: Optional[pd.DataFrame] = None
    emissions_df: Optional[pd.DataFrame] = None
    balance_df: Optional[pd.DataFrame] = None
    lci_df: Optional[pd.DataFrame] = None

    route_set = set()
    for component in blend.components:
        record = record_map[component.run_name]
        weight = component.share / total_share
        summary = record.summary
        component_raw = summary.get("raw_co2e_kg")
        if component_raw is None:
            raise ValueError(f"Blend '{blend.name}' component '{component.run_name}' missing total CO2e result.")
        total_co2e += weight * float(component_raw)
        demand_qty += weight * float(record.route_cfg.demand_qty)
        if record.route_cfg.route_preset:
            route_set.add(record.route_cfg.route_preset)
        comp_cost = getattr(record.result, "total_cost", None)
        if comp_cost is None:
            cost_valid = False
        else:
            total_cost += weight * float(comp_cost)
        comp_material_cost = getattr(record.result, "material_cost", None)
        if comp_material_cost is None:
            material_cost_valid = False
        else:
            material_cost += weight * float(comp_material_cost)

        contribution = _weight_numeric_frame(record.result.energy_balance, weight)
        energy_balance = _accumulate_numeric(energy_balance, contribution)

        contribution = _weight_numeric_frame(record.result.emissions, weight)
        emissions_df = _accumulate_numeric(emissions_df, contribution)

        contribution = _weight_numeric_frame(record.result.balance_matrix, weight)
        balance_df = _accumulate_numeric(balance_df, contribution)

        contribution_lci = _weight_lci(record.result.lci, weight)
        lci_df = _accumulate_lci(lci_df, contribution_lci)

        for proc, flag in record.result.production_routes.items():
            try:
                weighted_routes[proc] += weight * float(flag)
            except (TypeError, ValueError):
                continue
        for proc, level in record.result.prod_levels.items():
            try:
                weighted_levels[proc] += weight * float(level)
            except (TypeError, ValueError):
                continue

        components_payload.append({
            "run": record.plan.name,
            "label": component.label or record.plan.name,
            "normalized_share": weight,
            "input_share": component.share,
            "raw_co2e_kg": component_raw,
            "total_co2e_kg": float(component_raw) * REPORTED_YIELD_DIVISOR if component_raw is not None else None,
            "raw_ef_kg_per_unit": summary.get("raw_ef_kg_per_unit"),
            "gross_ef_kg_per_unit": summary.get("gross_ef_kg_per_unit"),
            "demand_qty": record.route_cfg.demand_qty,
        })

    result = RunOutputs(
        production_routes=dict(weighted_routes),
        prod_levels=dict(weighted_levels),
        energy_balance=energy_balance if energy_balance is not None else pd.DataFrame(),
        emissions=emissions_df,
        total_co2e_kg=total_co2e,
        total_cost=total_cost if cost_valid else None,
        material_cost=material_cost if material_cost_valid else None,
        balance_matrix=balance_df,
        lci=lci_df,
        meta={
            "blend": {
                "name": blend.name,
                "notes": blend.notes,
                "components": components_payload,
                "demand_qty": demand_qty,
            }
        },
    )
    summary: Dict[str, Any] = {
        "name": blend.name,
        "is_blend": True,
        "blend": True,
        "raw_co2e_kg": total_co2e,
        "total_co2e_kg": total_co2e * REPORTED_YIELD_DIVISOR,
        "raw_ef_kg_per_unit": (total_co2e / demand_qty) if demand_qty > 0 else None,
        "gross_ef_kg_per_unit": (total_co2e * REPORTED_YIELD_DIVISOR / demand_qty) if demand_qty > 0 else None,
    }
    if route_set:
        summary["route_preset"] = route_set.pop() if len(route_set) == 1 else ",".join(sorted(route_set))
    if blend.notes:
        summary["notes"] = blend.notes
    summary["raw_co2e_kg"] = total_co2e
    summary["total_co2e_kg"] = total_co2e * REPORTED_YIELD_DIVISOR
    return summary, result


def _run_blends(
    blends: List[BlendPlan],
    records: List[RunRecord],
    log_dir_default: Optional[Path] = None,
    write_lci: bool = False,
    lci_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    if not blends:
        return []
    record_map = {record.plan.name: record for record in records}
    summaries: List[Dict[str, Any]] = []
    for blend in blends:
        summary, result = _compute_blend_result(blend, record_map)
        summaries.append(summary)
        print(f"[blend] {blend.name}: raw={_fmt_float(summary.get('raw_co2e_kg'))} kg; adjusted={_fmt_float(summary.get('total_co2e_kg'))} kg")
        log_dir = blend.log_dir or log_dir_default
        if log_dir:
            payload = {
                "name": blend.name,
                "blend": True,
                "components": summary.get("components", []),
                "notes": blend.notes,
                "results": summary,
                "meta": result.meta,
            }
            try:
                path_written = write_run_log(str(log_dir), payload)
                print(f"  ↳ blend log written to {path_written}")
            except Exception as exc:
                print(f"  ! failed to write blend log in {log_dir}: {exc}", file=sys.stderr)
        if write_lci:
            out_dir = lci_dir if lci_dir is not None else Path("results/lci").resolve()
            path_written = _write_lci_csv(getattr(result, 'lci', None), out_dir, blend.name)
            if path_written:
                print(f"  ↳ blend LCI written to {path_written}")
    return summaries


def run_batch(
    plans: List[RunPlan],
    show_meta: bool = False,
    log_dir_default: Optional[Path] = None,
    fail_fast: bool = False,
    write_lci: bool = False,
    lci_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], int, List[RunRecord]]:
    summaries: List[Dict[str, Any]] = []
    failures = 0
    records: List[RunRecord] = []
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
            raw = summary.get("raw_co2e_kg")
            adj = summary.get("total_co2e_kg")
            print(f"{plan.name}: raw={_fmt_float(raw)} kg; adjusted={_fmt_float(adj)} kg")
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
            # Optionally write LCI CSV per run
            if write_lci:
                out_dir = lci_dir if lci_dir is not None else Path("results/lci").resolve()
                path_written = _write_lci_csv(getattr(result, 'lci', None), out_dir, plan.name)
                if path_written:
                    print(f"  ↳ LCI written to {path_written}")
            records.append(RunRecord(plan=plan, route_cfg=route_cfg, result=result, summary=summary))
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
    return summaries, failures, records


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
    run_parser.add_argument("--enable-lci", action="store_true", help="Enable LCI outputs (sets FORGE_ENABLE_LCI=1).")
    run_parser.add_argument("--write-lci", action="store_true", help="Write LCI CSV for each run and blend.")
    run_parser.add_argument("--lci-dir", type=str, default="results/lci", help="Directory for LCI CSV outputs (with --write-lci).")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 1
    # Optionally enable LCI calculations explicitly for CLI runs
    if getattr(args, "enable_lci", False):
        import os as _os
        _os.environ["FORGE_ENABLE_LCI"] = "1"
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
    blend_plans: List[BlendPlan] = []
    if args.spec:
        spec_plans, blend_plans = _plans_from_spec(Path(args.spec), cli_defaults)
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
        summaries, failures, records = run_batch(
            plans,
            show_meta=args.show_meta,
            log_dir_default=log_dir_default,
            fail_fast=args.fail_fast,
            write_lci=bool(getattr(args, 'write_lci', False)),
            lci_dir=_resolve_path(args.lci_dir, Path.cwd()) if getattr(args, 'lci_dir', None) else None,
        )
    except Exception as exc:
        print(f"Execution aborted: {exc}", file=sys.stderr)
        return 3
    blend_summaries: List[Dict[str, Any]] = []
    if blend_plans:
        try:
            blend_summaries = _run_blends(
                blend_plans,
                records,
                log_dir_default=log_dir_default,
                write_lci=bool(getattr(args, 'write_lci', False)),
                lci_dir=_resolve_path(args.lci_dir, Path.cwd()) if getattr(args, 'lci_dir', None) else None,
            )
        except Exception as exc:
            print(f"Blend calculation failed: {exc}", file=sys.stderr)
            return 6
    all_summaries = list(summaries) + blend_summaries
    if args.output:
        try:
            _write_summary(args.output, all_summaries)
            print(f"Summary written to {args.output}")
        except Exception as exc:
            print(f"Failed to write summary: {exc}", file=sys.stderr)
            return 4
    return 0 if failures == 0 else 5


if __name__ == "__main__":
    sys.exit(main())
