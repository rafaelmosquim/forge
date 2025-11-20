---
title: 'FORGE: Flexible Optimization of Routes for GHG and Energy'
tags:
  - Python
  - steel production
  - emissions modeling
  - life cycle assessment
  - industrial decarbonization
authors:
  - name: Rafael Mosquim
    orcid: 0000-0002-8636-6649
    affiliation: 1
  - name: Paulo Sergio Pinheiro Lima
    orcid: 0009-0002-7474-6342  
    affiliation: 1
  - name: Leonardo Pastre
    orcid: 0009-0003-8892-1460
    affiliation: 1
  - name: Joaquim Seabra
    orcid: 0000-0002-1463-7104
    affiliation: 1
affiliations:
  - name: UNICAMP - Faculdade de Engenharia Mecânica, Brazil
    index: 1
date: 2025-11-19
bibliography: paper.bib
---

# Summary

FORGE is a Python-based model for assessing energy consumption and greenhouse gas
emissions in industrial production routes, with an initial focus on steel. It enables
comparative analysis of multiple steelmaking pathways (BF–BOF, DRI–EAF, EAF–scrap) and
products (pig iron, cast steel, automotive steel) under customizable process
parameters, energy carriers, and emission factors. The underlying logic was first
prototyped in Excel and then translated and generalized into a descriptor-driven
Python engine; an aluminium dataset already exists, but this paper concentrates on
the steel implementation.

# Statement of Need

The steel industry accounts for approximately 7–9% of global CO₂ emissions
[@worldsteel2023]. Academic works usually fall into three main categories: life cycle
analysis of a single plant; top-down, black-box calculations using sectoral energy
consumption and carrier shares; or bottom-up analyses using proprietary data. These
approaches have major shortcomings, which FORGE addresses by providing:

- Transparent, YAML-driven process configuration
- End-to-end integration between processes and plant-level energy and emission balances
- Multiple route modelling with scenario locking
- Interactive sensitivity analysis
- Monte Carlo uncertainty quantification
- Streamlit-based user interface

Unlike commercial LCA software, FORGE is open-source and specifically designed for
steel production analysis, making it accessible for researchers and policymakers. Its
extensive customization allows researchers to tailor process assumptions to their own
contexts. The software can also be used as a simulation tool for industry or
academia.

# Model overview

FORGE represents an industrial plant as a directed graph of process “recipes” loaded
from YAML files. Each recipe describes material inputs and outputs for a process
step, and the core engine solves the resulting material, energy, and emissions
balances for a user‑specified demand at a chosen stage.

At the heart of the model, `calculate_balance_matrix` walks upstream from the
final-demand material, selecting one enabled producer per material and treating any
remaining requirement as an external purchase. This produces a material balance table
and per‑process production levels, which are combined with energy‑intensity tables and
carrier shares to build a complete energy balance. Emissions are computed by
multiplying carrier‑specific energy use by emission factors and adding
process‑specific direct emissions, with a consistent split between onsite processes
and market purchases.

Route choice and sector‑specific assumptions are kept out of the core and encoded in
data. A sector descriptor (`sector.yml`) defines stage boundaries, route presets,
process roles, fallback materials, and gas configuration. These settings drive a
deterministic route builder that turns route presets and user picks into a binary
`production_routes` mask, ensuring that the balance engine always sees at most one
producer per material and can fall back to external purchases when needed.

For steel, FORGE includes detailed logic for blast‑furnace, basic oxygen furnace and coke‑oven gas recovery. Process gases can be used directly or converted to internal electricity in a utility plant; the model derives plant‑level blends for gas and electricity and applies credits for avoided external purchases, while also computing blended emission factors for gas and electricity use.

All of this is orchestrated by a thin Python API (`run_scenario`) that loads
datasets, applies scenario overrides, builds routes, and calls the core engine. The
same API underpins a minimal command‑line interface for batch runs and a Streamlit
web application that exposes route selection, parameter overrides, sensitivity
analysis, and Monte Carlo studies without requiring users to edit YAML or write code.

# Validation

To validate FORGE against industry benchmarks, we configured the model with the "Likely" dataset and Brazil grid electricity (country code: BRA), then computed emissions at the crude steel stage ("Validation (as-cast)") boundary. Table 1 compares these results with Worldsteel 2023 industry averages [@worldsteel2023].


**Table 1. Model validation: FORGE vs. Worldsteel 2023 (tonne CO₂e per tonne crude steel).**

| Route      | FORGE | Worldsteel 2023 |
|------------|-------|-----------------|
| BF–BOF     | 2.216 | 2.32            |
| DRI–EAF    | 0.971 | 1.43            |
| EAF–Scrap  | 0.208 | 0.70            |

*Notes: FORGE simulations used the “Likely” dataset and Brazil grid electricity factor. Boundary: crude steel (as-cast), selected via “Validation (as-cast)” option.*

The validation shows strong agreement for BF-BOF. The EAF-scrap and DRI-EAF routes show significantly lower emissions in FORGE due to Brazil's renewable-heavy electricity grid, compared to the global average grid mix reflected in Worldsteel data.

## Model Configuration

Validation simulations were performed with the following FORGE settings:
- **Data selection**: "Likely" dataset (baseline scenario)
- **Electricity grid**: Brazil (country code BRA) 
- **System boundary**: Crude steel ("Validation (as-cast)" stage)
- **Demand quantity**: 1000 kg steel

The "Validation (as-cast)" stage boundary ensures consistent comparison by stopping the model after continuous casting, matching the crude steel reporting boundary used in industry benchmarks. This stage uses fixed pre-selections to enable reproducible 
validation by independent users.

# Acknowledgements

First author would like to thank the funding from Fundação de Desenvolvimento da Pesquisa (FUNDEP), Project 27192*57 - Linha V Mover ''Do berço ao Portão''

## Development Process

The codebase was written with AI assistance under human design, validation, 
and benchmarking against a previously validated Excel model.

## Installation
```bash
git clone https://github.com/rafaelmosquim/forge.git
cd forge
pip install -r requirements.txt
streamlit run streamlit_app.py
```

# Availability
FORGE is open-source and available under the MIT license. The latest version is available at:
- **Source code**: https://github.com/rafaelmosquim/forge
- **Archived release (v1.0.2):** https://doi.org/10.5281/zenodo.17279849
- **Documentation**: Included in repository

# References
