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
emissions in steel production routes. It enables comparative analysis of different 
steelmaking pathways (BF-BOF, DRI-EAF, EAF-scrap) and products (Pig Iron, Cast Steel, Automotive steel) with customizable process parameters, energy carriers, and emission factors. FORGE logic was built in Excel and then translated/expanded into Python. While FORGE can calculate emissions for any industrial process, and an Aluminum data-set exists, implementantion is still in an early phase, so this paper will deal mostly with steel. 

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

# Model overview
The main functions are found in engine.py, with auxiliary calculations and definitions split into separate modules, for better maintenance. For steel production, main auxiliaries are the process gas recovery logic and the route disambiguation, as the core model needs a single producer per process, or it does not run. 


## Core (engine.py)
The core functions in FORGE are found in engine.py. "calculate_balance_matrix" is the main function and the heart of the model. It works by being fed with a final demand product, set by the user. From this final demand the model works upstream throught the graph to map all possible producers, defined via recipes.yml; choosing only one producer per process (based on production_routes in ). If more than one producer is available for that product, a an error is raised ("Ambiguous producer for '{mat}'"). If no internal producer is found, it treates that product as an outside purchase. The function returns a material balance with all material inputs needed to fulfil that final demand. It also produces "prod_level", which accounts for the number of runs each process needs to run to produce that final output; With the balance matrix constructed, total energy usage is obtained by multiplying production level for all processes by its energy demand per process, set in energy_int.yml. This energy demand is split per carrier, with information from energy_matrix.yml; Finally, with per carrier total demand, emissions are obtained by multiplying per carrier energy demand by its emission factor. This energy emission is then added to process emissions later (function as a inside/outside mill logic to handle that).

## Steel specific logic (compute.py)

FORGE began as a steel model and the production process is very complex. As such, some auxiliary functions are needed.

### Gas recovery
For gas recovery, the main logic is that some processes (ie. Coke Making, Blast Furnance and Basic Oxygen Furnace) produce gases that are recovered and reused in the plant. This reuse can be done in two ways: as gas, or to feed an utility plant to produce electricity. To obtain a single emission factor for all gas and electricity used, a blend for gas and another for electricity are made. The blends need to be fixed, regardless of plant boundary chosen by the user, which acts as an accounting, not physical choice. So a reference, fixed plant, is run, to obtain this fixed share of internal vs. external ratio. As the gas recovery reduces external purchases, credits are applied for process gas producers. 

## Route disambiguation (compute.py, builder.py)
The main configuration is done in recipes.yml, which contains all possible producers for all products, and the core model is configured to be a pure calculation engine only. As such, route disambiguation (ie. Pig Iron can be produced by the Blast Furnace or Direct Reduction, but a real plant only produces via one of these two) needs to be done elsewhere. The disambiguation is done in core.compute._build_route_from_picks, which resolves user selection and combines with some pre-masks, which is drive by scenarios.builder.build_core_scenario. The pure calculation logic of core functions is deliberate, to allow FORGE to calculate energy and emission balances for any industrial process. 


## Data loading (io.py)
FORGE relies entirely on data-sets configured via .yml files. This is done to make data editing user friedly. There are separate functions for ymls configured in nested (say process a has energy carriers x, y and z) or not nested ways (energy intensity of process A is x MJ per run). A more complex data loader for recipes is needed as some values inside it are variables (like process_gas), not constants. This makes changing parameters easier later. 

## Sector descriptor (sector.yml)
FORGE was expanded to be able to deal with any industrial process, and configuration is done via data input only. As such, information about route masks and material mappings are defined in sector.yml. This makes adding a new task a data-driven process, with no need to touch core logic. 

## CLI runs
A light Command-line Interface exists to allow FORGE to produce batch runs, or reproduce some pre-defined route, to be used as validation. This allows scenarios to be configured in ymls and be automated. 

## APP UI
FORGE also have a UI suite, a streamlit app. There the user can select the industrial process (Steel or aluminum so far), then select main data-sets (likely, optimistic and pessiminstic) and routes (ie. BF-BOF, DRI-EAF). The UI then surfaces emission intensity for that given choice, as well as CSVs for energy balance, emission per process, and production runs. User can also customize routes, such as selecting utilities to be produced in house or purchased, as well as downstream, optional treatments. There are also tabs for specific parameter overrides, a simple sensitivity analysis and Monte Carlo, so users can stress test the model without touching code or editing yml files. 

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
