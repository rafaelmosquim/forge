# Open Science Roadmap

Comprehensive backlog to keep FORGE aligned with transparency, reproducibility, and FAIR data principles.

## 1. Reproducible Execution & Environments
- [ ] Publish resolver-locked dependencies (`requirements-lock.txt`, optional Conda/UV lock) plus documentation describing the relationship to `pyproject.toml` / `requirements.txt`.
- [ ] Ship a container image (Dockerfile or Apptainer) that bundles datasets, CLI tools, and pinned Python. Exercise it in CI for smoke-testing release artifacts.
- [ ] Add a `make reproduce-validation` (or `python -m forge.reproduce`) command that seeds randomness, runs the Likely/BRA validation scenario, and records provenance (git SHA, dataset hash, CLI args).
- [ ] Capture hardware/software fingerprints (OS, Python version, BLAS vendor) in `steel_core_api_v2.run_scenario` metadata and surface them inside exported logs.

## 2. Transparent & FAIR Data Management
- [ ] Provide `meta.yml` files inside each `datasets/<sector>/<variant>/` describing source, license, QA date, owner, and processing steps; surface this metadata in the Streamlit sidebar.
- [ ] Automate dataset archiving to Zenodo/OSF concurrent with software releases and embed the resulting DOIs into scenario outputs.
- [ ] Write a YAML schema guide (`docs/data_dictionary.md`) that explains each field in recipes, energy tables, and emission factors.
- [ ] Add a `datasets/README.md` that outlines versioning policies, source attribution requirements, and a contact for corrections.

## 3. Automated Validation & Regression Proofs
- [ ] Extend pytest coverage to hit the descriptor-aware `run_scenario` path with golden JSON fixtures per sector/route; block merges on these checks instead of manual UI runs.
- [ ] Promote `run_descriptor_checks.py` into a CLI subcommand that CI can execute (nightly or on-demand), storing balance/emission artifacts for regression review.
- [ ] Maintain the Table 1 validation baselines as versioned CSV/Parquet and load them directly in snapshot tests with documented tolerances.
- [ ] Cover Streamlit-only helpers (sector gate, route matching, pick resolution) with UI-focused tests (e.g., `pytest-streamlit` components) to prevent silent regressions.

## 4. Discoverability & Documentation
- [ ] Turn `docs/` into a mkdocs or Sphinx site that hosts weekly summaries, API docs, and tutorial notebooks; publish via GitHub Pages for citable references.
- [ ] Update `CONTRIBUTING.md` with dataset citation expectations, links to open science checklists (e.g., NASA TOPS), and instructions for declaring provenance.
- [ ] Produce narrative notebooks demonstrating how to extend FORGE to new sectors, capturing assumptions and decision logs as an “open lab notebook.”
- [ ] Write a “Reproduce the JOSS paper” guide chaining environment setup, dataset choice, CLI commands, and expected outputs.

## 5. Community & Governance
- [ ] Publish a governance memo describing maintainer roles, review expectations, and release cadence.
- [ ] Add GitHub issue/PR templates that prompt contributors to document data sources, verification steps, and whether outputs feed publications.
- [ ] Maintain a public roadmap (Projects board) tracking user requests, dataset needs, and prioritized features to keep planning transparent.

## 6. Open Results & Archival Quality
- [ ] Convert `results*.json` into tidy CSV/Parquet releases with embedded schema + units, host them alongside the Zenodo software DOI, and cite them in the README.
- [ ] Automate changelog, DOI, and metadata updates during release tagging so every artifact references an immutable snapshot.
- [ ] Attach checksums and SBOMs to each release so downstream users can verify integrity.
- [ ] Register the repository with long-term preservation partners (Software Heritage, institutional repositories) for durability beyond GitHub.
