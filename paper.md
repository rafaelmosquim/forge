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
date: 2025-10-06
bibliography: paper.bib
---

# Summary

FORGE is a Python-based model for assessing energy consumption and greenhouse gas 
emissions in steel production routes. It enables comparative analysis of different 
steelmaking pathways (BF-BOF, DRI-EAF, EAF-scrap) and products (Pig Iron, Cast Steel, Automotive steel) with customizable process parameters, energy carriers, and emission factors.

# Statement of Need

The steel industry accounts for approximately 7-9% of global CO₂ emissions [@worldsteel2023]. Academic works usually fall into three main categories: Life Cycle Analysis of a single plant, top-down, black box calculations using total sectorial energy consumption and carrier share, or bottom-up analysies using proprietary data. These approaches have major shortcomings which FORGE addresses by providing:

- Transparent, YAML-driven process configuration
- End-to-end integration between processes and total energy and emission balances
- Multiple route modeling with scenario locking  
- Interactive sensitivity analysis
- Monte Carlo uncertainty quantification
- Streamlit-based user interface

Unlike commercial LCA software, FORGE is open-source and specifically designed for 
steel production analysis, making it accessible for researchers and policymakers. Its extensive customization allows researchers to easily tailor processes to their realities. The software can also be used as a simulation tool for industry or researchers alike. 

# Features

## Core Capabilities
- **Route Analysis**: BF-BOF, DRI-EAF, EAF-scrap, and external steel routes
- **Stage Boundaries**: Pig iron, crude steel, and finished product analysis
- **Energy Balance**: Comprehensive energy carrier tracking with process gas crediting
- **Emissions Calculation**: CO₂e emissions across scope 1, 2, and upstream processes

## Technical Implementation  
- YAML-based data configuration for transparency
- Recipe-based material and energy flows
- Interactive parameter sweeps and Monte Carlo analysis
- Streamlit web interface for accessibility

## Design Philosophy

FORGE employs an interactive, web-based architecture to handle the inherent complexity of steel production pathway selection. The model's recipe graph contains numerous ambiguous producer choices (e.g., cast steel can be made either flat or long, and cold rolling may or may not be applied after hot rolling) that require user guidance. The Streamlit interface provides an intuitive way to resolve these ambiguities while maintaining transparency in the underlying calculations.

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
