# FORGE — Flexible Optimization of Routes for GHG & Energy

**FORGE** is a transparent, YAML-driven steel plant model that computes **cradle-to-gate** energy use and GHG emissions across multiple routes (BF-BOF, DRI-EAF, EAF-scrap) and downstream options. It supports route locking by scenario, on-site electricity crediting, sensitivity analysis, and Monte Carlo uncertainty.

[![CI](https://github.com/rafaelmosquim/forge/actions/workflows/ci.yml/badge.svg)](https://github.com/rafaelmosquim/forge/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17246738.svg)](https://doi.org/10.5281/zenodo.17246738)


> **Reference this release:**  
> This paper uses **FORGE v1.0.2** — Zenodo DOI: **10.5281/zenodo.17279849**  
> Concept DOI (always points to latest): **10.5281/zenodo.17145189**

## How to cite
Mosquim, R., Lima, P. S. P., Pastre, L., & Seabra, J. (2025).
*FORGE — Flexible Optimization of Routes for GHG & Energy* (v1.0.2).
Zenodo. https://doi.org/10.5281/zenodo.17279849

For general reference to the project (latest version):  
https://doi.org/10.5281/zenodo.17145189
---

**AI assistance:** Portions of the Python were generated from author-written prompts; model design, review, validation, and all architectural decisions are human-led.


## Features
- Route-locked scenarios (BF-BOF, DRI-EAF, EAF-scrap, External)
- Clean stage boundaries: **Pig iron**, **Crude steel**, **Finished**
- Explicit recipes + energy matrix (shares), energy contents, and carrier EFs
- On-site utility-plant electricity crediting (no double counting)
- 1-D sensitivity sweeps + Monte Carlo with pinned seeds
- Streamlit UI for picks (AND/OR recipe graph) and charts

## Optional LCI outputs
- Life-cycle inventory (LCI) post-processing is experimental and **off by default**.
- Set `FORGE_ENABLE_LCI=1` (or `true/yes/on`) before running the CLI/API/UI to enable LCI tables.
- When the flag is unset, LCI calculations are skipped and the API returns `lci=None`.

## Data bundles
- Datasets now live under `datasets/<sector>/<variant>/` (e.g., `datasets/steel/likely`, `datasets/aluminum/baseline`).
- The Streamlit app opens with a sector gate so you can select *Steel* or *Aluminum* before choosing a dataset variant.

## Install
Requires **Python ≥ 3.10**.

```bash
git clone https://github.com/rafaelmosquim/forge.git
cd forge
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start (Streamlit)
```bash
streamlit run streamlit_app.py
```
In the main UI:
1. Landing screen → Sector = Steel, click Continue
2. Sidebar → Dataset = Likely, Grid = BRA, Product = Validation (as-cast)
2. Sidebar → pick a Route (BF-BOF / DRI-EAF / EAF-scrap)
3. Main → Tab = Main Model → Run model
4. Reported Crude steel (as-cast) CO₂e matches Table 1 in the paper.

## CLI batch runner
When you already know the scenario and route selections you can execute the model headlessly:

```bash
python steel_batch_cli.py run \
  --scenario datasets/steel/likely/scenarios/BF_BOF_coal.yml \
  --route BF-BOF \
  --stage-key Cast \
  --stage-role validation \
  --country-code BRA
```

`steel_batch_cli.py --help` lists additional options for supplying batches, override files, and pick lists.

## Tests
Run the lightweight consistency checks:

```bash
pytest
```

## License
MIT
