# UI & Streamlit Experience

Planned enhancements to the interactive interface so researchers can navigate multi-sector datasets with less friction.

## Sector & Dataset Selection
- [ ] Persist the sector choice across sessions (cookie or query param) so users returning to the app bypass the gate when desired.
- [ ] Surface dataset metadata (source, QA date, notes) directly in the sidebar—pull from the forthcoming `meta.yml`.
- [ ] Add search/filter when more than ~4 dataset variants exist, ensuring aluminum/steel additions remain manageable.

## Scenario & Route UX
- [ ] Replace the current per-material radio forest with a structured “recipe graph” view showing AND/OR nodes and current picks.
- [ ] Provide preset buttons for canonical scenarios (Validation, Baseline, Low/High) that pre-populate picks and route masks.
- [ ] Expose descriptor-fed tooltips explaining why certain processes are disabled for a given route preset.
- [ ] Allow exporting/importing pick bundles directly from the UI to stay in sync with `steel_batch_cli` workflows.

## Results & Reporting
- [ ] Offer downloadable CSV/Parquet outputs for balance matrices, energy tables, and emissions without leaving the UI.
- [ ] Embed the new provenance metadata (git SHA, dataset hash) in the UI results card for quick citation.
- [ ] Add comparison mode to plot multiple route runs side-by-side (e.g., BF-BOF vs DRI-EAF).

## Performance & Reliability
- [ ] Cache parsed YAML/descriptor data per dataset to reduce app reload time.
- [ ] Add in-app diagnostics panel that surfaces errors from `run_scenario` with actionable remediation tips rather than raw tracebacks.
- [ ] Implement smoke-test script (headless) that hits the Streamlit endpoints to catch gating regressions before release.
