# Changelog

## Unreleased
- Repackaged the core modules under `src/forge` with package metadata so app, CLI, and tests share the same import path.
- Updated Streamlit app, CLI entrypoints, and test harness to reference `forge.*` modules after the reorg.
- Ensured helper scripts export `PYTHONPATH` when invoking the relocated CLI tools.
- Generalized core/app via sector descriptors (multi-industry datasets).
- Added dataset-specific `sector.yml` configs and descriptor loader module.
- Enabled fallback external supplies to avoid loops when internal utilities absent.
- Updated cost/emissions calculations to honor descriptor metadata.
- Validation (as-cast) stage now clamps auxiliaries to market purchases while other stages respect user picks.
- Restored deterministic internal-electricity reference so validation routing doesn’t skew emission factors.
- Upstream clamp logic relaxed so radios control onsite vs market choices outside validation.
- Restored BF-BOF charcoal scenario behaviour and clarified scenario labels; kept biomethane optional by lowering its priority.

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
