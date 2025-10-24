# Development Summary • Week of 2025-10-22

## October 22, 2025

- **Core refactor & safety improvements**  
  Major restructuring pushed calculations into `steel_model_core`, added safer evaluation guards, and introduced logging hooks. Setup was modernized with `pyproject.toml`, CI workflow, and a gas integration test scaffold for regression coverage.

- **Aluminum dataset assembled**  
  Added a complete aluminum data suite (recipes, parameters, emissions, prices, intensities) while keeping Streamlit changes minimal to avoid disrupting existing steel flows (`dont touch aluminum yet`).

- **New automation scaffolding**  
  `run_descriptor_checks.py` was added as a scriptable smoke test that wires `ScenarioInputs` into the generalized API, offering a quick regression pass for descriptor-driven scenarios without launching the UI.

## October 23, 2025

- **Sector descriptor generalization**  
  Introduced descriptor-driven configuration via new modules:  
  • `sector_descriptor.py` defines typed dataclasses and parsing logic for route/stage metadata, lifting hard-coded assumptions out of the core.  
  • `scenario_resolver.py` translates descriptor inputs into runtime masks, stage maps, and feed modes, replacing bespoke per-route branching and simplifying future sector additions.  
  Companion YAMLs across each data pack were expanded to feed these modules.

- **Validation-stage enforcement**  
  Iterative fixes ensured the validation product locks auxiliaries to market supply, applied stage-driven route masks, and refined upstream radio behaviour. Descriptor YAMLs and debug snapshots were updated in tandem.

## October 24, 2025

- **Final polish on validation & routes**  
  Additional adjustments stabilized upstream radio handling, updated change log entries, and captured richer debug snapshots for balance/emission inputs (`radio works for upstream`, `all working; aluminum values still wrong`).

- **Batch automation tooling**  
  Added the `steel_batch_cli.py` utility to run scenarios without the UI, emit JSON/CSV summaries, and log payloads. Extended it with multi-country fan-out support and defensive deep copies to prevent run-to-run state leakage—offering a scriptable alternative to the Streamlit front-end for sweeps and regression capture.
- **Dataset consolidation**  
  Migrated legacy root-level folders (`data/`, `data_min/`, etc.) into `datasets/<sector>/<variant>/`, paving the way for a multi-sector UI gate and cleaner CLI defaults.
- **Streamlit sector gate**  
  The app now opens on a lightweight FORGE-branded landing screen where users select their industrial sector before entering the main sidebar. This resolves duplicate dataset prompts and makes the multi-sector workflow explicit across the UI.

---

_Prepared for repository oversight to track major changes introduced between Oct 22–24, 2025._
