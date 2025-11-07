# Code Hygiene Backlog

Focused cleanup tasks to keep the Python packages, CI, and docs accurate and low-noise.

Status legend: [x] done • [ ] pending

## Tests & Tooling
- [x] Migrate tests to the public API (`steel_core_api_v2.run_scenario`) and refactored core modules (no `steel_model_core` imports).
- [x] Add coverage to CI (`pytest --cov`) and upload artifact.
- [x] Add nightly numeric workflow (gates via `FORGE_ENABLE_NUMERIC_TESTS`).
- [x] Add pre-commit with ruff (format/fix), YAML checks, mypy, and pytest smoke.
 - [x] Replace/retire any remaining deprecated tests (e.g., `test_core_compute_invariant.py`) if still present.
 - [x] Add missing tests for `steel_batch_cli.run_batch` (spec parsing, multi-country fan-out, log writing).

## Streamlit & Core Modules
- [x] Point Streamlit app to refactored core imports (models/io/transforms/routing/viz).
- [x] Remove legacy core imports in the app.
- [ ] Split `streamlit_app.py` into smaller UI components (sector gate, scenario, charts) for testability.
- [ ] Add type annotations + mypy coverage for descriptor and routing helpers.

## Documentation Accuracy
- [x] Update README with Make targets and Engine CLI quickstart (refactored API usage).
- [ ] Document descriptor architecture (stage menu, route presets) for contributors.
- [ ] Update `CONTRIBUTING.md` with lint/test steps and expected outputs for validation runs.

## CI / Automation
- [x] Emit coverage and upload artifact.
- [x] Nightly numeric snapshot job.
- [ ] Scheduled batch run across canonical scenarios (store artifacts for regressions).
- [ ] Add HTML coverage and publish as build artifact or Pages.

## Tracking & Visibility
- [ ] Publish CI/CD expectations in `docs/` (how to run lint/tests, snapshot policy).
- [ ] Maintain a “known tech debt” document (`docs/tech_debt.md`) linking to these todos with owners.
