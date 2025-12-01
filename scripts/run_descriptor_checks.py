"""
Ad-hoc smoke test to exercise descriptor-driven scenario runs.

Executes a steel scenario (baseline BF-BOF) and the aluminum scenario.
Not part of the normal test suite; intended for manual verification.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from steel_core_api_v2 import RouteConfig, ScenarioInputs, run_scenario


def run_case(data_dir: str, scenario_name: str, stage_key: str = "Finished"):
    base = Path(data_dir)
    scenario_path = base / "scenarios" / scenario_name
    if not scenario_path.exists():
        scenario_path = base / scenario_name
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    cfg = RouteConfig(route_preset="auto", stage_key=stage_key, demand_qty=1000.0)
    scn = ScenarioInputs(country_code=None, scenario=scenario, route=cfg)
    result = run_scenario(data_dir, scn)
    print(
        f"{data_dir}:{scenario_name} → CO2e={result.total_co2e_kg}, "
        f"has emissions={result.emissions is not None and not result.emissions.empty}"
    )
    meta_path = Path(f"tmp_{Path(data_dir).name}_{Path(scenario_name).stem}_meta.json")
    meta_path.write_text(json.dumps(result.meta, indent=2), encoding="utf-8")
    print(f"meta written to {meta_path}")


if __name__ == "__main__":
    run_case("datasets/steel/likely", "BF_BOF_coal.yml")
    run_case("datasets/aluminum/baseline", "primary-al.yml", stage_key="primary")
