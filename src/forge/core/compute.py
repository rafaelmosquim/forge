"""Core computations and transforms.

This module starts as wrappers around the legacy monolith and incrementally
duplicates selected functions so callers can depend on `forge.core.compute`
without the monolith. Where we provide local implementations, they take
precedence; otherwise we delegate to `forge.steel_model_core`.
"""
from __future__ import annotations

from forge import steel_model_core as _core
from typing import Dict, List
import os
from .models import Process

# -----------------------
# Local implementations
# -----------------------

def _env_flag_truthy(var_name: str) -> bool:
    try:
        raw = os.environ.get(var_name, "")
    except Exception:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _debug_print(*args, **kwargs) -> None:
    if _env_flag_truthy("FORGE_DEBUG_IO"):
        print(*args, **kwargs)


def expand_energy_tables_for_active(active_names, energy_shares, energy_int):
    """Let variant names like 'Continuous Casting (R)' reuse base rows.

    Mutates energy_shares/energy_int in-place by copying entries for each
    active process variant from its base name (text before " (").
    """
    def base(n: str) -> str:
        return n.split(" (")[0]

    for n in list(active_names or []):
        b = base(n)
        if n not in energy_shares and b in energy_shares:
            energy_shares[n] = dict(energy_shares[b])
        if n not in energy_int and b in energy_int:
            energy_int[n] = energy_int[b]


def calculate_internal_electricity(prod_level: Dict[str, float], recipes_dict: Dict[str, Process], params) -> float:
    """Compute internal electricity from recovered gases (BF delta + Coke gas).

    Utility Plant efficiency is taken as the Electricity output per MJ gas
    from the 'Utility Plant' recipe.
    """
    util_eff = 0.0
    if 'Utility Plant' in recipes_dict:
        util_eff = recipes_dict['Utility Plant'].outputs.get('Electricity', 0.0)

    internal_elec = 0.0

    # BF top-gas delta between adjusted and base intensities
    bf_runs = float(prod_level.get('Blast Furnace', 0.0))
    if bf_runs > 0 and hasattr(params, 'bf_base_intensity') and hasattr(params, 'bf_adj_intensity'):
        bf_delta = params.bf_adj_intensity - params.bf_base_intensity
        gf = bf_runs * bf_delta
        _debug_print(f"DBG gas BF: runs={bf_runs:.3f}, delta={bf_delta:.2f} MJ/run → {gf:.1f} MJ")
        internal_elec += gf * util_eff

    # Coke-oven gas
    cp_runs = float(prod_level.get('Coke Production', 0.0))
    if 'Coke Production' in recipes_dict:
        gas_per_run_cp = recipes_dict['Coke Production'].outputs.get('Process Gas', 0.0)
        if cp_runs > 0 and gas_per_run_cp > 0:
            gf_cp = cp_runs * gas_per_run_cp
            _debug_print(f"DBG gas Coke: runs={cp_runs:.3f}, gas_per_run={gas_per_run_cp:.2f} MJ/run → {gf_cp:.1f} MJ")
            internal_elec += gf_cp * util_eff

    return internal_elec


def adjust_energy_balance(energy_df, internal_elec):
    """Apply internal electricity credit to energy balance DataFrame.

    - Subtract internal_elec from TOTAL's Electricity
    - Add/overwrite a 'Utility Plant' row with negative Electricity equal to
      the internal_elec (export from recovered gases)
    """
    df = energy_df.copy()
    if 'Electricity' not in df.columns:
        df['Electricity'] = 0.0
    # TOTAL row may not exist; defensively create if missing
    if 'TOTAL' not in df.index:
        try:
            df.loc['TOTAL'] = df.sum(numeric_only=True)
        except Exception:
            df.loc['TOTAL'] = 0.0
    df.loc['TOTAL', 'Electricity'] = float(df.loc['TOTAL'].get('Electricity', 0.0)) - float(internal_elec)
    df.loc['Utility Plant'] = 0.0
    df.loc['Utility Plant', 'Electricity'] = -float(internal_elec)
    return df


def derive_energy_shares(recipes: List[Process], energy_content: Dict[str, float]):
    """Derive energy carrier shares per process from recipe inputs.

    Electricity is taken directly (MJ units). Fuels are converted using
    lower heating values provided in `energy_content`.
    Returns a mapping: { process_name: { carrier: share, ... }, ... }
    """
    shares: Dict[str, Dict[str, float]] = {}
    for proc in recipes:
        MJ_by_carrier: Dict[str, float] = {}
        for c, amt in proc.inputs.items():
            if c == 'Electricity':
                MJ_by_carrier[c] = float(amt)
            elif c in energy_content:
                MJ_by_carrier[c] = float(amt) * float(energy_content[c])
        total = sum(MJ_by_carrier.values())
        if total > 1e-12:
            shares[proc.name] = {c: mj / total for c, mj in MJ_by_carrier.items()}
        else:
            shares[proc.name] = {}
    return shares

# Calculations
calculate_balance_matrix = _core.calculate_balance_matrix
# Use our local implementations (names shadow the getattr fallbacks)
# expand_energy_tables_for_active defined above
# calculate_internal_electricity defined above
calculate_energy_balance = _core.calculate_energy_balance
from .costs import analyze_energy_costs, analyze_material_costs
from .lci import calculate_lci
from .transforms import (
    apply_fuel_substitutions,
    apply_dict_overrides,
    apply_recipe_overrides,
    adjust_blast_furnace_intensity,
    adjust_process_gas_intensity,
)
calculate_emissions = _core.calculate_emissions
calculate_lci = getattr(_core, "calculate_lci", lambda *a, **k: None)

# Transforms/overrides (gas routing still delegated for now)
apply_gas_routing_and_credits = getattr(_core, "apply_gas_routing_and_credits", lambda *a, **k: (None, None, {}))

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
    "derive_energy_shares",
    "analyze_energy_costs",
    "analyze_material_costs",
    "calculate_emissions",
    "calculate_lci",
    # transforms
    "apply_fuel_substitutions",
    "apply_dict_overrides",
    "apply_recipe_overrides",
    "adjust_blast_furnace_intensity",
    "adjust_process_gas_intensity",
    "apply_gas_routing_and_credits",
    # refs
    "compute_inside_elec_reference_for_share",
    "compute_inside_gas_reference_for_share",
]
