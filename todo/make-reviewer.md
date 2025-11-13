# Reviewer One-Command Repro plan

Goal: give reviewers a single, reliable command that runs representative workloads and produces compact, inspectable outputs. Keep it minimal: no manual venv steps required; Docker alternative included.

## Make targets to add

Add these targets to `Makefile` so reviewers can run `make reviewer` (and optional verification) with zero setup beyond Git + Python (or Docker).

```make
# --- Reviewer helpers (self-contained venv) ---
VENV := .venv
PY_VENV := $(VENV)/bin/python3
PIP_VENV := $(VENV)/bin/pip

reviewer-setup:
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP_VENV) install -r requirements.txt

reviewer: reviewer-setup
	$(PY_VENV) -m forge.cli.steel_batch_cli run \
	  --spec configs/finished_steel_portfolio.yml \
	  --data-dir datasets/steel/likely \
	  --output results/finished_portfolio_bf.json
	$(PY_VENV) -m forge.cli.steel_batch_cli run \
	  --spec configs/finished_steel_portfolio_eaf.yml \
	  --data-dir datasets/steel/likely \
	  --output results/finished_portfolio_eaf.json
	$(PY_VENV) -m forge.cli.steel_batch_cli run \
	  --spec configs/finished_steel_portfolio_bf_charcoal.yml \
	  --data-dir datasets/steel/likely \
	  --output results/finished_portfolio_bf_charcoal.json
	$(PY_VENV) scripts/summarize_portfolios.py results/finished_portfolio_*.json -o results/summary.json
	@echo "Done. See results/summary.json"

# Optional: verify against a committed expected summary with tolerances
reviewer-verify: reviewer
	$(PY_VENV) - <<'PY'
import json, math
exp = json.load(open('docs/reviewer_expected_summary.json'))
act = json.load(open('results/summary.json'))
for e in exp:
    a = next((x for x in act if x['route'] == e['route']), None)
    assert a, f"Missing route {e['route']}"
    ae = float(a.get('total_co2e_kg') or 0)
    ee = float(e.get('total_co2e_kg') or 0)
    assert math.isclose(ae, ee, rel_tol=1e-3, abs_tol=1.0), f"Route {e['route']}: {ae} vs {ee}"
print('reviewer verification OK')
PY

# Docker-only variant (no local Python/venv needed)
reviewer-docker: docker-build
	docker run --rm -v "$(PWD)/results:/app/results" forge:paper bash -lc "\
	  python3 -m forge.cli.steel_batch_cli run --spec configs/finished_steel_portfolio.yml --data-dir datasets/steel/likely --output results/finished_portfolio_bf.json && \
	  python3 -m forge.cli.steel_batch_cli run --spec configs/finished_steel_portfolio_eaf.yml --data-dir datasets/steel/likely --output results/finished_portfolio_eaf.json && \
	  python3 -m forge.cli.steel_batch_cli run --spec configs/finished_steel_portfolio_bf_charcoal.yml --data-dir datasets/steel/likely --output results/finished_portfolio_bf_charcoal.json && \
	  python3 scripts/summarize_portfolios.py results/finished_portfolio_*.json -o results/summary.json && \
	  cat results/summary.json"
```

Notes:
- Uses the existing CLI `forge.cli.steel_batch_cli` and summarizer `scripts/summarize_portfolios.py`.
- Outputs land in `results/`:
  - `results/finished_portfolio_bf.json`
  - `results/finished_portfolio_eaf.json`
  - `results/finished_portfolio_bf_charcoal.json`
  - `results/summary.json` (compact table for eyeballing)
- The `reviewer-verify` target expects a committed `docs/reviewer_expected_summary.json` with the same shape as the summarizer’s output.

## Quick usage (for README / submission)

- Local (no manual setup):
  - `make reviewer`
  - Inspect: `cat results/summary.json`
  - Optional: `make reviewer-verify`

- Docker (no local Python):
  - `make reviewer-docker`

## Pre-requisites and behavior

- Local run: requires Python ≥ 3.10 and `make`. The target bootstraps `.venv` and installs `requirements.txt`.
- Docker run: requires Docker; image built via existing `make docker-build` (used implicitly by `reviewer-docker`).
- Determinism: totals are stable; verification uses `abs_tol=1.0` kg and `rel_tol=1e-3` as guardrails.
- Runtime: a few minutes on a typical laptop; creates `results/cache_*` folders for memoization.

## Optional polish (nice-to-have)

- Commit `docs/reviewer_expected_summary.json` (the canonical numbers for the submission snapshot).
- Include a short “Reviewer quick start” in `README.md` pointing to `make reviewer` and where outputs appear.
- If desired, add `reviewer-test` to run `pytest -q` after the reviewer run as an extra proof point.
- Consider adjusting the Dockerfile default `CMD` to `src/forge/apps/streamlit_app.py` so `docker run forge:paper` launches the UI correctly (optional; unrelated to reviewer flow).

