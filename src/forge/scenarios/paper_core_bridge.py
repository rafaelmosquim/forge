# -*- coding: utf-8 -*-
"""
Paper scenarios → core bridge (non-destructive).

Use these helpers from your existing paper_scenarios.py to execute the
scenario grid directly against the core model with consistent LCI and
blending. This leaves your paper_scenarios.py logic intact.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from forge.cli import steel_batch_cli as batch


def _ensure_lci(enabled: bool) -> None:
    if enabled:
        os.environ["FORGE_ENABLE_LCI"] = "1"


def run_from_spec(
    spec_path: str | Path,
    *,
    enable_lci: bool = True,
    write_lci: bool = False,
    lci_dir: str | Path | None = None,
    summary_out: str | Path | None = None,
    log_dir: str | Path | None = None,
    fail_fast: bool = False,
) -> List[Dict[str, Any]]:
    """Execute the steel model for all runs/blends in a spec, with optional LCI.

    Returns a list of compact summary rows (raw/adjusted CO2, etc.).
    """
    _ensure_lci(enable_lci)
    spec_path = Path(spec_path).resolve()
    plans, blend_plans = batch._plans_from_spec(spec_path, cli_defaults={})

    summaries, failures, records = batch.run_batch(
        plans,
        show_meta=False,
        log_dir_default=Path(log_dir).resolve() if log_dir else None,
        fail_fast=fail_fast,
        write_lci=write_lci,
        lci_dir=Path(lci_dir).resolve() if lci_dir else None,
    )
    if failures and fail_fast:
        raise RuntimeError(f"{failures} runs failed (fail_fast)")

    blend_summaries: List[Dict[str, Any]] = []
    if blend_plans:
        blend_summaries = batch._run_blends(
            blend_plans,
            records,
            log_dir_default=Path(log_dir).resolve() if log_dir else None,
            write_lci=write_lci,
            lci_dir=Path(lci_dir).resolve() if lci_dir else None,
        )

    all_rows = list(summaries) + list(blend_summaries)

    if summary_out:
        out_path = Path(summary_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        batch._write_summary(out_path, all_rows)

    return all_rows


def run_paper_grid_default(
    *,
    spec: str | Path = "configs/paper_grid.yml",
    enable_lci: bool = True,
    write_lci: bool = False,
    out_dir: str | Path = "results",
) -> List[Dict[str, Any]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_from_spec(
        spec_path=spec,
        enable_lci=enable_lci,
        write_lci=write_lci,
        lci_dir=out_dir / "lci",
        summary_out=out_dir / "paper_grid.csv",
        log_dir=out_dir / "paper_logs",
    )


__all__ = [
    "run_from_spec",
    "run_paper_grid_default",
]

