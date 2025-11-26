# Aluminum Dataset Improvements

Work items to bring the aluminum sector pack up to the same quality as the steel datasets.

Status: in progress (UI + gas routing + downstream stages wired)

## Data Quality & Validation
- [ ] Reconcile current aluminum model outputs with empirical references (World Aluminum data, peer-reviewed LCAs) to diagnose the “values still wrong” note from Oct 24 summary.
- [ ] Build a validation scenario (analogous to Steel “Validation (as cast)”) with expected intensities and add pytest coverage that enforces tolerances.
- [ ] Audit recipes, energy matrices, and emission factors for unit consistency (MJ vs kWh, kg vs tonne) and document conversions.

## Descriptor & Scenario Parity
- [x] Expand `datasets/aluminum/baseline/sector.yml` to match the richer stage/route metadata used by steel, ensuring Streamlit menus and CLI presets behave consistently.
- [ ] Populate descriptor aliases, feed modes, and process roles so `scenario_resolver` can apply the same automation (route masks, feed clamps).
- [ ] Create canonical scenario YAMLs (baseline, low-carbon, 100% scrap) plus resolved variants for testing.

## Data Provenance & Documentation
- [ ] Add a `meta.yml` describing data sources, licenses, and QA owners; cite these in README/docs.
- [ ] Provide transformation notebooks or scripts that show how raw references were processed into YAML values.
- [ ] Document assumptions (e.g., anode/cathode efficiencies, recycled content) in a dedicated section of the JOSS paper or docs site.

## Tooling & UX
- [x] Update Streamlit downstream UI to surface aluminum stages (remelt → alloying/basic/manufactured/finished) with selectable options.
- [ ] Ensure `steel_batch_cli.py` (soon multi-sector) can locate aluminum datasets via `--data-dir datasets/aluminum/baseline`.
- [ ] Capture aluminum runs in `results_validation_by_country.json` (or a new results file) so output comparisons cover both sectors.
