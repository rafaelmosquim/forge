"""Core computations and transforms (wrappers).

This module re-exports compute/transformation functions from the legacy
`forge.steel_model_core` module to decouple call sites from the monolith path.
"""
from __future__ import annotations

from forge import steel_model_core as _core

# Calculations
calculate_balance_matrix = _core.calculate_balance_matrix
expand_energy_tables_for_active = getattr(_core, "expand_energy_tables_for_active", lambda *a, **k: None)
calculate_internal_electricity = getattr(_core, "calculate_internal_electricity", lambda *a, **k: None)
calculate_energy_balance = _core.calculate_energy_balance
adjust_energy_balance = getattr(_core, "adjust_energy_balance", lambda *a, **k: None)
analyze_energy_costs = getattr(_core, "analyze_energy_costs", lambda *a, **k: None)
analyze_material_costs = getattr(_core, "analyze_material_costs", lambda *a, **k: None)
calculate_emissions = _core.calculate_emissions
calculate_lci = getattr(_core, "calculate_lci", lambda *a, **k: None)

# Transforms/overrides
apply_fuel_substitutions = getattr(_core, "apply_fuel_substitutions", lambda *a, **k: None)
apply_dict_overrides = getattr(_core, "apply_dict_overrides", lambda *a, **k: None)
apply_recipe_overrides = getattr(_core, "apply_recipe_overrides", lambda *a, **k: None)
apply_gas_routing_and_credits = getattr(_core, "apply_gas_routing_and_credits", lambda *a, **k: (None, None, {}))

# Specific intensity adjustment helpers (if present)
adjust_blast_furnace_intensity = getattr(_core, "adjust_blast_furnace_intensity", lambda *a, **k: None)
adjust_process_gas_intensity = getattr(_core, "adjust_process_gas_intensity", lambda *a, **k: None)

# Reference helpers
compute_inside_elec_reference_for_share = getattr(_core, "compute_inside_elec_reference_for_share", lambda *a, **k: 0.0)
compute_inside_gas_reference_for_share = getattr(_core, "compute_inside_gas_reference_for_share", lambda *a, **k: 0.0)

__all__ = [
    # calcs
    "calculate_balance_matrix",
    "expand_energy_tables_for_active",
    "calculate_internal_electricity",
    "calculate_energy_balance",
    "adjust_energy_balance",
    "analyze_energy_costs",
    "analyze_material_costs",
    "calculate_emissions",
    "calculate_lci",
    # transforms
    "apply_fuel_substitutions",
    "apply_dict_overrides",
    "apply_recipe_overrides",
    "apply_gas_routing_and_credits",
    "adjust_blast_furnace_intensity",
    "adjust_process_gas_intensity",
    # refs
    "compute_inside_elec_reference_for_share",
    "compute_inside_gas_reference_for_share",
]
