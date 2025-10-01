# FORGE — Flexible Optimization of Routes for GHG & Energy

**FORGE** is a transparent, YAML-driven steel plant model that computes **cradle-to-gate** energy use and GHG emissions across multiple routes (BF-BOF, DRI-EAF, EAF-scrap) and downstream options. It supports route locking by scenario, on-site electricity crediting, sensitivity analysis, and Monte Carlo uncertainty.

> **Reference this release:**  
> This paper uses **FORGE v1.0.0** — Zenodo DOI: **10.5281/zenodo.17192803**  
> Concept DOI (always points to latest): **10.5281/zenodo.17145189**

## How to cite
If you use FORGE, please cite the archived release:
Mosquim, R. *FORGE — Flexible Optimization of Routes for GHG & Energy* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.17192803

For general reference to the project (latest version):  
https://doi.org/10.5281/zenodo.17145189
---

**AI assistance:** Portions of the Python were generated from author-written prompts; model design, review, validation, and all architectural decisions are human-led.


## Features
- Route-locked scenarios (BF-BOF, DRI-EAF, EAF-Scrap, External)
- Clean stage boundaries: **Pig iron**, **Crude steel**, **Finished**
- Explicit recipes + energy matrix (shares), energy contents, and carrier EFs
- On-site utility-plant electricity crediting (no double counting)
- 1-D sensitivity sweeps + Monte Carlo with pinned seeds
- Streamlit UI for picks (AND/OR recipe graph) and charts

## Install
Requires **Python ≥ 3.10**.
streamlit run streamlit_app.py
# In the UI (sidebar): Dataset = Likely → Grid = BRA → Product = Validation (as cast)
# In the UI (sidebar): Route: should be selected for each main route
# In the UI (main): Tab = Main Model → Run model button
# Values are displayed and match Table 1 in the paper.


```bash
git clone https://github.com/rafaelmosquim/forge.git
cd forge
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
