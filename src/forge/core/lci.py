"""Life Cycle Inventory (LCI) computation utilities.

Duplicated from the monolith so call sites can import from `forge.core.lci`.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Optional

import pandas as pd


def calculate_lci(
    prod_level: Dict[str, float],
    recipes: Iterable,
    energy_balance: Optional[pd.DataFrame] = None,
    electricity_internal_fraction: Optional[float] = None,
    gas_internal_fraction: Optional[float] = None,
    natural_gas_carrier: str = "Gas",
    process_gas_carrier: str = "Process Gas",
):
    """Build a life cycle inventory with per-process flows normalized per unit of the primary output.

    Args:
        prod_level: Mapping of process name → runs solved by the material balance.
        recipes: Iterable of Process objects (need .name/.inputs/.outputs).
        energy_balance: Optional DataFrame with per-process carrier consumption (MJ).
        electricity_internal_fraction: Share of electricity supplied internally (0–1).
        gas_internal_fraction: Share of gas supplied by process gas (0–1).
        natural_gas_carrier: Column name representing purchased gas in the energy balance.
        process_gas_carrier: Column name representing recovered process gas in the energy balance.

    Returns:
        pandas.DataFrame with columns [Process, Output, Flow, Input, Category, ValueUnit, Amount, Unit].
        Amounts are expressed per kg of the primary output.
    """
    entries = []
    recipes_dict = {r.name: r for r in recipes}

    energy_df = energy_balance if isinstance(energy_balance, pd.DataFrame) else None
    energy_carriers = set(energy_df.columns) if energy_df is not None else set()

    for proc, runs in (prod_level or {}).items():
        if runs <= 1e-9:
            continue
        recipe = recipes_dict.get(proc)
        if not recipe:
            continue

        outputs_totals = OrderedDict()
        for out, qty in (recipe.outputs or {}).items():
            try:
                per_run = float(qty)
            except (TypeError, ValueError):
                continue
            total = runs * per_run
            if abs(total) > 1e-9:
                outputs_totals[out] = total

        if not outputs_totals:
            continue

        primary_output = None
        primary_total = 0.0
        for out, total in outputs_totals.items():
            if out != process_gas_carrier:
                primary_output = out
                primary_total = total
                break
        if primary_output is None:
            primary_output, primary_total = next(iter(outputs_totals.items()))

        if abs(primary_total) <= 1e-9:
            continue

        denom = primary_total

        material_inputs = OrderedDict()
        for mat, qty in (recipe.inputs or {}).items():
            try:
                total = runs * float(qty)
            except (TypeError, ValueError):
                continue
            if abs(total) > 1e-9:
                material_inputs[mat] = total

        energy_inputs = OrderedDict()
        if energy_df is not None and proc in energy_df.index:
            row = energy_df.loc[proc]
            handled = set()

            elec_mj = float(row.get('Electricity', 0.0) or 0.0)
            if abs(elec_mj) > 1e-9:
                handled.add('Electricity')
                if electricity_internal_fraction is not None:
                    share_internal = max(0.0, min(1.0, float(electricity_internal_fraction)))
                    share_grid = 1.0 - share_internal
                    if abs(elec_mj * share_grid) > 1e-9:
                        energy_inputs["Electricity (Grid)"] = elec_mj * share_grid
                    if abs(elec_mj * share_internal) > 1e-9:
                        energy_inputs["Electricity (In-house)"] = elec_mj * share_internal
                else:
                    energy_inputs['Electricity'] = elec_mj

            nat_val = float(row.get(natural_gas_carrier, 0.0) or 0.0) if natural_gas_carrier else 0.0
            proc_val = float(row.get(process_gas_carrier, 0.0) or 0.0) if process_gas_carrier else 0.0
            if natural_gas_carrier:
                handled.add(natural_gas_carrier)
            if process_gas_carrier:
                handled.add(process_gas_carrier)
            gas_total = nat_val + proc_val

            if abs(gas_total) > 1e-9:
                if gas_internal_fraction is not None:
                    share_process = max(0.0, min(1.0, float(gas_internal_fraction)))
                    share_natural = 1.0 - share_process
                    if abs(gas_total * share_natural) > 1e-9:
                        energy_inputs[f"{natural_gas_carrier} (Natural)"] = gas_total * share_natural
                    if abs(gas_total * share_process) > 1e-9:
                        energy_inputs[f"{process_gas_carrier} (Internal)"] = gas_total * share_process
                else:
                    if natural_gas_carrier and abs(nat_val) > 1e-9:
                        energy_inputs[natural_gas_carrier] = nat_val
                    if (
                        process_gas_carrier
                        and process_gas_carrier != natural_gas_carrier
                        and abs(proc_val) > 1e-9
                    ):
                        energy_inputs[process_gas_carrier] = proc_val

            for carrier, mj in row.items():
                if carrier in handled or carrier == 'TOTAL':
                    continue
                if abs(mj) > 1e-9:
                    energy_inputs[str(carrier)] = float(mj)

        # Avoid double counting: if a material also appears as energy carrier, keep in energy only
        for special in ("Coke", "Coal"):
            if special in material_inputs and special in energy_inputs:
                material_inputs.pop(special, None)

        for mat, total in material_inputs.items():
            amount = total / denom
            entries.append({
                "Process": proc,
                "Output": primary_output,
                "Flow": "Input",
                "Input": mat,
                "Category": "Material",
                "ValueUnit": "kg",
                "Amount": amount,
                "Unit": f"kg per kg {primary_output}",
            })

        for carrier, total in energy_inputs.items():
            amount = total / denom
            entries.append({
                "Process": proc,
                "Output": primary_output,
                "Flow": "Input",
                "Input": carrier,
                "Category": "Energy",
                "ValueUnit": "MJ",
                "Amount": amount,
                "Unit": f"MJ per kg {primary_output}",
            })

        for out_name, total in outputs_totals.items():
            ratio = total / denom
            value_unit = "MJ" if (out_name in energy_carriers or out_name == process_gas_carrier) else "kg"
            entries.append({
                "Process": proc,
                "Output": primary_output,
                "Flow": "Output",
                "Input": out_name,
                "Category": "Output",
                "ValueUnit": value_unit,
                "Amount": ratio,
                "Unit": f"{value_unit} per kg {primary_output}",
            })

    df = pd.DataFrame(
        entries,
        columns=["Process", "Output", "Flow", "Input", "Category", "ValueUnit", "Amount", "Unit"],
    )
    if not df.empty:
        df.sort_values(["Process", "Output", "Flow", "Category", "Input"], inplace=True)
    return df


__all__ = ["calculate_lci"]

