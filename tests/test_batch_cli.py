import os
from pathlib import Path

from forge.cli.steel_batch_cli import run_batch, RunPlan


def make_plan(name: str, data_dir: Path, route_preset: str = 'BF-BOF', stage_key: str = 'Finished', demand: float = 1000.0, country: str | None = None) -> RunPlan:
    return RunPlan(
        name=name,
        data_dir=data_dir,
        scenario={},
        route={"route_preset": route_preset, "stage_key": stage_key, "demand_qty": demand},
        picks_by_material={},
        pre_select_soft={},
        country_code=country,
        log_dir=None,
    )


def test_run_batch_single(tmp_path, repo_root):
    data_dir = (repo_root / 'datasets' / 'steel' / 'likely').resolve()
    assert data_dir.exists()
    plan = make_plan('unit_single', data_dir)
    summaries, failures, records = run_batch([plan], show_meta=False, log_dir_default=tmp_path)
    assert failures == 0
    assert len(summaries) == 1
    assert records and records[0].summary.get('total_co2e_kg') is not None


def test_run_batch_multi_country(tmp_path, repo_root):
    data_dir = (repo_root / 'datasets' / 'steel' / 'likely').resolve()
    p1 = make_plan('unit_bra', data_dir, country='BRA')
    p2 = make_plan('unit_usa', data_dir, country='USA')
    summaries, failures, records = run_batch([p1, p2], show_meta=False, log_dir_default=tmp_path)
    assert failures == 0
    assert len(summaries) == 2


def test_run_batch_writes_logs(tmp_path, repo_root):
    data_dir = (repo_root / 'datasets' / 'steel' / 'likely').resolve()
    plan = make_plan('unit_log', data_dir)
    summaries, failures, records = run_batch([plan], show_meta=False, log_dir_default=tmp_path)
    assert failures == 0
    logs = list(tmp_path.glob('run_*.json'))
    # Logs may be nested if function uses a subfolder; glob recursively
    if not logs:
        logs = list(tmp_path.rglob('run_*.json'))
    assert logs, 'expected a run log JSON to be written'

