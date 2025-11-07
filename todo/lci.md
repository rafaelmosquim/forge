# LCI tasks

Status legend: [x] done • [ ] pending

## Implementation & Core
- [x] Implement `calculate_lci` in refactored core (forge.core.lci) and route it through the API.
- [x] Split Electricity and Gas into internal vs grid/natural components in LCI inputs when meta is available.
- [ ] Add numeric parity tests for LCI on key processes (BF, EAF) with documented tolerances.

## Documentation & Schema
- [ ] Document LCI table schema in README/docs (columns, units, per‑kg normalization, carrier splits).
- [ ] Provide example CSV and a short tutorial on interpreting per‑process LCI rows.

## UX
- [ ] Expose LCI CSV download in Streamlit (when enabled) alongside balance/emissions.
- [ ] Add a toggle in UI/CLI profiles to turn LCI on/off and indicate potential run‑time cost.
