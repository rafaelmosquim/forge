# Changelog

## Unreleased
- Generalized core/app via sector descriptors (multi-industry datasets).
- Added dataset-specific `sector.yml` configs and descriptor loader module.
- Enabled fallback external supplies to avoid loops when internal utilities absent.
- Updated cost/emissions calculations to honor descriptor metadata.

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
