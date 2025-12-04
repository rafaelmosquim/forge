# FORGE Methods Paper – Working TODO

This file tracks what the current `forge_paper.tex` already does well and what still needs to be done to make it a solid methods paper.

## 1. High‑level goals

- Position the paper explicitly as a **methods/infrastructure** paper for FORGE.
- Make the **plant→sector consistency gap** (LCA vs top‑down) the central motivation.
- Demonstrate that FORGE is:
  - Physically consistent from process → plant → sector.
  - Descriptor/YAML driven and sector‑extensible (steel + aluminum).
  - Reproducible and scriptable (Makefile, CLI, MC).

## 2. What is already strong

- Clear narrative that FORGE bridges plant‑LCA and top‑down models.
- Good modular description of architecture (`core`, descriptors, API, UI).
- Detailed math for:
  - Material balance.
  - Energy balance.
  - Gas routing and internal electricity.
  - Emissions (carrier and process‑level, blended electricity).
- “Data Configuration” and YAML‑driven pipeline are well articulated.
- Gas routing description is a distinctive technical contribution.

## 3. Missing or weak pieces

### 3.1 Front matter

- [ ] Add a proper **abstract**:
  - Problem/gap (plant‑consistent vs aggregate models).
  - What FORGE is and what is new.
  - How it is demonstrated (steel + aluminum, scenarios).
  - 2–3 key takeaways.
- [ ] In the Introduction, add a short **“This paper contributes”** list (3–4 bullets).
- [ ] Remove “this document sketches / starting point” language; write as final paper.

### 3.2 Structure and sectioning

Reorganize into a clearer methods structure (mostly by re‑labeling/moving existing text):

- [ ] **Introduction**
  - Problem statement, gap, contributions.
- [ ] **Model overview**
  - System boundaries, stages, routes at a conceptual level.
- [ ] **Data and configuration pipeline**
  - Sector descriptors (`sector.yml`).
  - Core datasets (`recipes.yml`, `energy_int.yml`, `energy_matrix.yml`, `emission_factors.yml`, etc.).
  - Scenario YAMLs (overrides, route options).
- [ ] **Core algorithms**
  - Material balance (formulation + link to implementation).
  - Energy balance.
  - Gas routing and internal electricity.
  - Emissions (including special handling for coke, process gas).
- [ ] **Implementation and software**
  - Core/API/UI layers.
  - Batch CLI and Makefile profiles.
  - Testing / regression checks.
- [ ] **Demonstration and validation**
  - Compact but concrete results (see next section).
- [ ] **Discussion and limitations**
  - What FORGE can/cannot do; future sectors and extensions.

### 3.3 Demonstration and validation

Need explicit evidence that the method works in practice:

- [ ] **Reproduction / alignment**:
  - Show that FORGE reproduces a known plant or published EF for at least one steel case (table: reference vs FORGE).
- [ ] **Scenario propagation example**:
  - Example where changing BF energy intensity or gas routing:
    - Changes process‑level balances.
    - Changes plant totals.
    - Changes an aggregate metric (e.g., tCO₂e/t steel at Finished).
- [ ] **Aluminum sector demonstration**:
  - One short example showing that aluminum runs through the same descriptor + YAML stack and core pipeline as steel, highlighting that it is a fully integrated sector with a current baseline dataset (min/max variants still to come).
- [ ] Add a small **“Results / Demonstration”** section tying the above into 1–2 figures/tables.

### 3.4 Related work and comparisons

- [ ] Add a **related‑work / comparison** subsection:
  - Explicitly name representative tools:
    - Process‑based LCA / plant models.
    - Top‑down energy/sector models (TIMES, REMIND‑like, or sector modules).
    - Any existing steel‑specific open models.
  - Clarify what they lack:
    - Route locking.
    - Plant‑consistent propagation from process to sector.
    - Gas routing and internal electricity.
    - Sector‑agnostic descriptors/YAML.
- [ ] Optional but helpful: a **comparison table** with rows = models and columns = key features, showing where FORGE sits.

### 3.5 Readability and narrative flow

- [ ] Add 1–2 sentence “why this matters” intros/outros to dense method sections (material balance, gas routing, emissions) to anchor the reader.
- [ ] Clean up informal phrases (“starting point”, “will be explained later”) and unify tense (present, assertive).
- [ ] Tighten aluminum section to clearly describe it as a fully integrated second sector with a somewhat narrower process scope and only a baseline dataset (minimum and maximum datasets as future work), rather than as a mere proof-of-concept.

### 3.6 Figures and examples

Plan at least:

- [ ] **Architecture schematic**:
  - Datasets + descriptors → scenario resolution → core runner → outputs.
- [ ] **Route / system diagram**:
  - BF‑BOF route with process gas flows and stage boundaries (Pig iron / Crude / Finished).
- [ ] **YAML/CLI example figure**:
  - Small snippet of a scenario file or batch spec, annotated to show how a user defines a run.
- [ ] **Validation / demonstration figure(s)**:
  - Example of how a scenario change propagates (bar or Sankey style, or EF comparison).

### 3.7 Software and reproducibility

- [ ] Add a short **Software and reproducibility** section:
  - Repository URL and versioning (tag for the paper).
  - How to set up environment (venv + `requirements.txt`).
  - How to regenerate key results:
    - `make finished`, `make paper`, `make aluminum`, `make mc-as-cast`, `make mc-finished`.
  - Note on tests (`pytest`) and regression checks.

## 4. Later / nice-to-have

- [ ] Optional: small sensitivity analysis example using the Monte Carlo machinery in a methods‑appropriate way (e.g., robustness of a route EF to key parameters).
- [ ] Optional: short “Outlook” paragraph on future sectors, coupling with energy system models, or cross‑material analysis (steel + aluminum).***
