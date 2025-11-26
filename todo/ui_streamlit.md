# UI & Streamlit Experience

Planned enhancements to the interactive interface so researchers can navigate multi-sector datasets with less friction.

Status legend: [x] done • [ ] pending

## Sector & Dataset Selection
- [ ] Persist the sector choice across sessions (cookie or query param) so users returning to the app bypass the gate when desired.
- [ ] Surface dataset metadata (source, QA date, notes) directly in the sidebar—pull from the forthcoming `meta.yml`.
- [ ] Add search/filter when more than ~4 dataset variants exist, ensuring aluminum/steel additions remain manageable.
- [x] Ensure app calls refactored API/core (no legacy imports).

## Scenario & Route UX
- [x] Render downstream picks per descriptor stage (including aluminum) instead of hard-coded steel buckets.
- [ ] Provide preset buttons for canonical scenarios (Validation, Baseline, Low/High) that pre-populate picks and route masks.
- [ ] Expose descriptor-fed tooltips explaining why certain processes are disabled for a given route preset.
- [ ] Allow exporting/importing pick bundles directly from the UI to stay in sync with batch workflows.

## Results & Reporting
- [ ] Offer downloadable CSV/Parquet outputs for balance matrices, energy tables, and emissions without leaving the UI.
- [ ] Embed provenance metadata (git SHA, dataset hash) in the UI results card for citation.
- [x] Keep downstream controls sector-aware (steel vs aluminum) and avoid leaking steel-only UI into aluminum.
- [x] Show EF metrics with/without Coke Production for quick sensitivity (two headline metrics).
- [x] Hide zero-only columns in the Energy Balance table to slim the view.

## Performance & Reliability
- [ ] Cache parsed YAML/descriptor data per dataset to reduce app reload time.
- [ ] Add in-app diagnostics panel surfacing run_scenario errors with actionable tips.
- [ ] Implement headless smoke-test for Streamlit endpoints to catch regressions.
- [x] Ensure `process_gases.yml` is applied before recipe evaluation in the app (fixes missing `process_gases` errors).

## Stage Unambiguity
- [x] Rename Crude Steel `stage_id` to `CastCrude` (was `Cast`) to disambiguate CLI usage while keeping functionality identical to Validation (as‑cast stop).
