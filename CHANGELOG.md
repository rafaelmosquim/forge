# Changelog

## Unreleased
- Core refactor: no `steel_model_core` dependency
  - Introduced `forge.core.engine` (engine trio): `calculate_balance_matrix`, `calculate_energy_balance`, `calculate_emissions`.
  - Split helpers into `forge.core.compute` (gas routing, intensity adjustments, reference helpers), `forge.core.io`, `forge.core.models`, `forge.core.routing`, `forge.core.transforms`, `forge.core.costs`, and `forge.core.viz`.
  - Moved descriptor code to `forge.descriptor.{sector_descriptor,scenario_resolver}`; app and API updated.
  - Streamlit app now imports exclusively from refactored core modules.

- Unified API path
  - All runs (paper/finished/MC/engine CLI) funnel through `forge.steel_core_api_v2.run_scenario`.
  - Added internal-electricity/gas reference helpers in core for consistent plant‑wide splits.

- New Engine CLI and Make targets
  - `src/forge/cli/engine_cli.py` for single scenario runs; writes `manifest.json` (commit SHA, args, dataset path).
  - Make targets: `finished`, `paper`, `parallel`, `engine-smoke`.

- Paper pipeline hygiene
  - Deterministic output roots under `results/<label>/{figs,tables}`.
  - Portfolio label derived from portfolio spec; safe overwrite by design.

- Monte Carlo scenario driver
  - Triangular sampling across min/mode/max datasets; stage clamping by stage key; country sweep.
  - Headless plotting and per‑run stats JSONs.

- CI/tooling improvements
  - Added coverage collection and artifact upload; nightly numeric workflow.
  - Added mypy type check (non‑blocking) for core/descriptor.
  - Pre‑commit with ruff (fix/format), YAML checks, mypy, and pytest smoke.
  - Tests updated to refactored API/core; added new light tests for transforms, viz, descriptor, gas routing, engine imports, and batch CLI.

- Descriptor‑driven behavior
  - Validation (as‑cast) clamps auxiliaries to market; other stages respect user picks.
  - Route masks, process roles, gas config, and fallback materials applied consistently.

- Miscellaneous
  - Added provenance manifest to Engine CLI outputs.
  - Removed unused legacy CLI wrapper.
  - Introduced `FORGE_ENABLE_COSTS` feature flag (default off) so cost calculations stay hidden while code paths remain available.
  - Removed LCI feature surface (no LCI compute/exports in API/UI/CLI).
  - Eliminated base/adjusted intensity hacks and BF base-intensity emission rewrites; emissions now run on the same adjusted balance.
  - Process‑gas EF blending excludes Electricity and process‑gas carriers (attribution only to primary fuels).
  - Streamlit: headline EF metrics now show with/without Coke Production; Energy Balance view hides zero‑only columns.
  - Datasets: cleaned energy matrices (removed FLF/RLF columns); renamed Crude Steel `stage_id` to `CastCrude` to disambiguate CLI stage_key.
  - Reproducibility: `make reproduce-validation` runs Likely/BRA Validation (as‑cast) via Engine CLI and writes artifacts; `run_scenario` meta now includes an environment fingerprint (`meta['env']`).

## v1.0.2 — 2025-10-06
- JOSS submission snapshot.
- Availability section updated (archive DOI placeholder).
- Citations refreshed; minor doc polish.
- **No code changes.**

## v1.0.1
- Minor documentation/metadata updates.
- **Archival note:** The **Zenodo v1.0.1** record differs from the **GitHub `v1.0.1` tag** in docs/metadata only. The GitHub tag is the canonical code snapshot.

## v1.0.0
- Dataset switcher with three energy-intensity sets: Likely, Low, High.
- Semantics: Low = optimistic/best-case; High = pessimistic/worst-case.
- Validation “as cast” boundary to match the paper’s setup.

## v0.99
- Initial public release.
