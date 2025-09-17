# FORGE — Flexible Optimization of Routes for GHG & Energy
[![DOI: v0.99](https://zenodo.org/badge/DOI/10.5281/zenodo.17145190.svg)](https://doi.org/10.5281/zenodo.17145190)
[![DOI: latest](https://zenodo.org/badge/DOI/10.5281/zenodo.17145189.svg)](https://doi.org/10.5281/zenodo.17145189)


**FORGE** is a transparent, YAML-driven steel plant model that computes **cradle-to-gate** energy use and GHG emissions across multiple routes (BF-BOF, DRI-EAF, EAF-scrap) and downstream options. It supports route locking by scenario, on-site electricity crediting, sensitivity analysis, and Monte Carlo uncertainty.

> **Reference this release:**  
> This paper uses **FORGE v0.99** — Zenodo DOI: **10.5281/zenodo.17145190**  
> Latest releases (concept DOI): **10.5281/zenodo.17145189**

## How to cite
If you use FORGE, please cite the archived release:
Mosquim, R. *FORGE — Flexible Optimization of Routes for GHG & Energy* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.<VERSION>
---

## Features
- Route-locked scenarios (BF-BOF, DRI-EAF, EAF-Scrap, External)
- Clean stage boundaries: **Pig iron**, **Crude steel**, **Finished**
- Explicit recipes + energy matrix (shares), energy contents, and carrier EFs
- On-site utility-plant electricity crediting (no double counting)
- 1-D sensitivity sweeps + Monte Carlo with pinned seeds
- Streamlit UI for picks (AND/OR recipe graph) and charts

## Install
Requires **Python ≥ 3.10**.

```bash
git clone https://github.com/USER/forge.git
cd forge
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
