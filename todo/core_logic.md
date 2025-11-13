# Core Logic Notes

## Process Gas Circularity
- Legacy Excel model only had a single 'Process Gas' carrier, so they lowered BF/Coke intensity ("base") and added back credits manually to avoid circular references.
- In our graph solver we can make process gas flows explicit by using distinct carriers:
  - BF and Coke emit source-specific carriers (e.g., `BF Process Gas`, `Coke Process Gas`).
  - A dedicated utility/router node blends those into a generic `Process Gas (Internal)` after routing decisions.
  - Any consumer (even BF/Coke auxiliaries) draws from the generic carrier, keeping the graph acyclic without intensity hacks.
- Long-term fix: eliminate the base/adjusted intensity swap and always use the real (adjusted) intensity, then apply explicit credits after emissions (similar to how electricity credit works). This would make energy/emission accounting transparent and remove the need for dual intensity definitions.

## Status / Next Steps
- [x] Model recovered gas manifold explicitly (sources → router → generic sinks) so BF/Coke can consume internal gas without circular dependencies (implemented in `apply_gas_routing_and_credits`).
- [x] Remove base/adjusted intensity swap; use real intensities and handle credits explicitly (no BF/Coke base-intensity rewrites; credit routing lives in `apply_gas_routing_and_credits`).
- [x] Compute process‑gas EF from primary fuels only (exclude Electricity and the process‑gas carrier from EF blending).
- [ ] Optional: add a fossil‑only toggle for process‑gas EF (exclude biogenic carriers like Charcoal when desired).
