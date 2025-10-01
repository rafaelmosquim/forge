---
title: 'FORGE: Flexible Optimization of Routes for GHG & Energy'
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
date: 2025
bibliography: paper.bib
---

# Summary

FORGE is a Python-based model for assessing energy consumption and greenhouse gas 
emissions in steel production routes. It enables comparative analysis of different 
steelmaking pathways (BF-BOF, DRI-EAF, EAF-scrap) with customizable process parameters, 
energy carriers, and emission factors.

# Statement of Need

The steel industry accounts for approximately 7-9% of global CO₂ emissions [@worldsteel2023]. 
Decarbonization requires tools that can model complex production routes and their 
environmental impacts. FORGE addresses this need by providing:

- Transparent, YAML-driven configuration
- Multiple route modeling with scenario locking  
- Interactive sensitivity analysis
- Monte Carlo uncertainty quantification
- Streamlit-based user interface

Unlike commercial LCA software, FORGE is open-source and specifically designed for 
steel production analysis, making it accessible for researchers and policymakers.

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

To validate FORGE against industry benchmarks, we configured the model with the "Likely" dataset and Brazil grid electricity (country code: BRA), then computed emissions at the crude steel stage ("Validation (as cast") boundary. Table 1 compares these results with Worldsteel 2023 industry averages [@worldsteel2023].

Table 1 compares FORGE results (Likely dataset, Brazil grid, crude steel boundary) with Worldsteel 2023 benchmarks.

**Table 1. Model validation: FORGE vs. Worldsteel 2023 (tonne CO₂ per tonne crude steel).**

| Route      | FORGE | Worldsteel 2023 |
|------------|-------|-----------------|
| BF–BOF     | 2.216 | 2.32            |
| DRI–EAF    | 0.971 | 1.43            |
| EAF–Scrap  | 0.208 | 0.70            |

*Notes: FORGE simulations used the “Likely” dataset and Brazil grid electricity factor. Boundary: crude steel (as-cast), selected via “Validation (as cast)” option.*

The validation shows strong agreement for BF-BOF. The EAF-Scrap and DRI-EAF routes routes show significantly lower emissions in FORGE due to Brazil's renewable-heavy electricity grid (84% renewable in 2023), compared to the global average grid mix reflected in Worldsteel data.

## Model Configuration

Validation simulations were performed with the following FORGE settings:
- **Data selection**: "Likely" dataset (baseline scenario)
- **Electricity grid**: Brazil (country code BRA) 
- **System boundary**: Crude steel ("Validation (as cast)" stage)
- **Demand quantity**: 1000 kg steel

The "Validation (as cast)" stage boundary ensures consistent comparison by stopping the model after continuous casting, matching the crude steel reporting boundary used in industry benchmarks. This stage uses fixed pre-selections to enable reproducible 
validation by independent users.

# Acknowledgements

First author would like to thank the funding from \textit{Fundação de Desenvolvimento da Pesquisa} (FUNDEP), Project 27192*57 - Linha V Mover ''Do berço ao Portão''

The authors designed the modeling logic and data structures based on a prior Excel implementation. Large language models were used as coding assistants to translate stepwise specifications into Python snippets. Each generated change was reviewed, tested, and integrated by the authors, who iterated the prompts from failing tests and model discrepancies. All architectural choices, algorithms, and validation procedures are human contributions.

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
- **Archived release**: [@mosquim2025forge]
- **Documentation**: Included in repository

# References
