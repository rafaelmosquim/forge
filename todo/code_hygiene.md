# Code Hygiene Backlog

Focused cleanup tasks to keep the Python packages, CI, and docs accurate and low-noise.

## Tests & Tooling
- [ ] Replace `tests/test_core_compute_invariant.py` with coverage that targets `steel_core_api_v2.run_scenario` instead of the deprecated `forge_core` module.
- [ ] Enable Ruff’s autofix or formatting stages (e.g., `ruff check --fix` + `ruff format`) in CI once the codebase is compliant, ensuring failures remain actionable.
- [ ] Wire pytest markers/env vars (`FORGE_ENABLE_NUMERIC_TESTS`) into CI so numeric snapshots run at least nightly, not just locally.
- [ ] Add missing tests for `steel_batch_cli.run_batch`, including spec parsing, multi-country fan-out, and log writing failure handling.

## Streamlit & Core Modules
- [ ] Trim unused imports/duplicate helpers in `streamlit_app.py` (e.g., redundant `Path`, double YAML imports) to reduce lint noise.
- [ ] Split `streamlit_app.py` into UI components (sector gate, scenario view, charts) to keep the module manageable and testable.
- [ ] Audit `steel_model_core.py` for legacy CLI code paths (argparse entry point, Plotly Sankey plotting) and consider moving them into a dedicated utility module.
- [ ] Add type annotations + `mypy` configuration for `sector_descriptor.py` and `scenario_resolver.py` to catch descriptor drift early.

## Documentation Accuracy
- [ ] Ensure README “Quick start” stays synced with the actual CLI arguments (e.g., route/stage switches) and mention all supported sectors.
- [ ] Document the descriptor architecture (stage menu, route presets) in developer docs so new contributors understand the data-driven approach.
- [ ] Update `CONTRIBUTING.md` with the enforced lint/test steps from CI and describe expected outputs for validation runs.

## CI / Automation
- [ ] Add a scheduled workflow that runs `steel_batch_cli.py` across canonical scenarios, uploading artifacts for regression review.
- [ ] Emit coverage reports (e.g., `pytest --cov`) and upload to Codecov or a GitHub artifact to watch trends.
- [ ] Introduce pre-commit hooks (ruff, mypy, pytest -m smoke) to catch hygiene issues before they reach CI.

## Tracking & Visibility
- [ ] Publish CI/CD expectations in `docs/` (how to run lint/tests, when to update baselines).
- [ ] Maintain a “known tech debt” table (maybe `docs/tech_debt.md`) pointing to these hygiene todos and linking to issues/owners.
