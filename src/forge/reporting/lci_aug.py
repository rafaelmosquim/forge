"""LCI augmentation and debug reporting utilities.

This module adds presentation-oriented rows to a base LCI DataFrame and prints
diagnostics for emission factors. It mirrors the prior inline augmentation in
the API to avoid behavior changes while separating responsibilities.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd

from forge.core.io import load_data_from_yaml


def augment_lci_and_debug(
    *,
    lci_df: pd.DataFrame,
    prod_levels: Dict[str, float],
    recipes: list,
    energy_balance: pd.DataFrame,
    energy_shares: Dict[str, Dict[str, float]],
    energy_content: Dict[str, float],
    params: Any,
    gas_meta: Dict[str, Any],
    e_efs: Dict[str, float],
    meta: Dict[str, Any],
    base_path: str,
    natural_gas_carrier: str,
    process_gas_carrier: str,
    f_internal: float,
) -> pd.DataFrame:
    """Augment the LCI table with per-process totals and credits; print EF debug."""
    # Augment LCI with per-process emission totals (kg CO2e per kg output)
    try:
        gp_path = os.path.join(base_path, 'ghg_protocol.yml')
        gp_raw = load_data_from_yaml(gp_path) or {}
        ghg_map = gp_raw.get('emission_ghg', gp_raw) or {}
        # Convert to kg/MJ for YAML-sourced EFs (which are g/MJ)
        ef_ghg_per_mj = {str(k): float(v) / 1000.0 for k, v in ghg_map.items() if v is not None}

        # Internal EFs for LCI (in-house gas and electricity treated as zero for totals)
        ef_internal_electricity = 0.0
        ef_process_gas = float(gas_meta.get('EF_process_gas', 0.0) or 0.0)  # used only for credit line

        nat_gas_label = gas_meta.get('natural_gas_carrier', natural_gas_carrier) or 'Gas'
        proc_gas_label = gas_meta.get('process_gas_carrier', process_gas_carrier) or 'Process Gas'

        def _ef_for_lci_input(input_name: str) -> float:
            name = str(input_name)
            if name == 'Electricity (Grid)':
                return float(ef_ghg_per_mj.get('Electricity', 0.0))
            if name == 'Electricity (In-house)':
                return 0.0
            if name == f'{nat_gas_label} (Natural)':
                return float(ef_ghg_per_mj.get(nat_gas_label, ef_ghg_per_mj.get('Gas', 0.0)))
            if name == f'{proc_gas_label} (Internal)':
                return 0.0
            base = name.split(' (')[0]
            return float(ef_ghg_per_mj.get(base, 0.0))

        if isinstance(lci_df, pd.DataFrame) and not lci_df.empty:
            additions = []
            output_additions = []
            trace_additions = []
            coke_custom_rows = []
            bf_pg_output_by_out: Dict[str, float] = {}

            for (proc, out), sub in lci_df.groupby(['Process', 'Output'], dropna=False):
                proc_name = str(proc)
                # Coke Production special handling
                if proc_name == 'Coke Production':
                    try:
                        runs_cp = float(prod_levels.get('Coke Production', 0.0) or 0.0)
                        r_cp = next((r for r in recipes if r.name == 'Coke Production'), None)
                        coke_per_run = float((r_cp.outputs.get('Coke', 1.0)) if r_cp else 1.0)
                        coal_per_run = float((r_cp.inputs.get('Coal', 0.0)) if r_cp else 0.0)
                        n2_per_run = float((r_cp.inputs.get('Nitrogen', 0.0)) if r_cp else 0.0)
                        coke_total_kg = runs_cp * coke_per_run
                        coal_total_kg = runs_cp * coal_per_run
                        lhv_coke = float(energy_content.get('Coke', 0.0) or 0.0)
                        lhv_coal = float(energy_content.get('Coal', 0.0) or 0.0)
                        denom_MJ = coke_total_kg * lhv_coke
                        coal_energy_MJ = coal_total_kg * lhv_coal
                        sh_cp = energy_shares.get('Coke Production', {}) or {}
                        sh_elec = float(sh_cp.get('Electricity', 0.0) or 0.0)
                        coal_per_MJprod = (coal_energy_MJ / denom_MJ) if denom_MJ > 1e-12 else 0.0
                        elec_per_MJprod = (((coal_energy_MJ / max(coke_total_kg, 1e-12)) * sh_elec) / max(lhv_coke, 1e-12))
                        elec_grid_per_MJprod = elec_per_MJprod * (1.0 - float(f_internal or 0.0))
                        elec_in_per_MJprod = elec_per_MJprod * float(f_internal or 0.0)
                        n2_total_kg = runs_cp * n2_per_run
                        n2_per_MJprod = (n2_total_kg / denom_MJ) if denom_MJ > 1e-12 else 0.0

                        out_label = str(out) if str(out) else 'Coke'
                        coke_custom_rows.extend([
                            {'Process': 'Coke Production', 'Output': out_label, 'Flow': 'Input', 'Input': 'Coal (Energy)', 'Category': 'Energy', 'ValueUnit': 'MJ', 'Amount': float(coal_per_MJprod), 'Unit': 'MJ per MJ Coke'},
                            {'Process': 'Coke Production', 'Output': out_label, 'Flow': 'Input', 'Input': 'Electricity (Grid)', 'Category': 'Energy', 'ValueUnit': 'MJ', 'Amount': float(elec_grid_per_MJprod), 'Unit': 'MJ per MJ Coke'},
                            {'Process': 'Coke Production', 'Output': out_label, 'Flow': 'Input', 'Input': 'Electricity (In-house)', 'Category': 'Energy', 'ValueUnit': 'MJ', 'Amount': float(elec_in_per_MJprod), 'Unit': 'MJ per MJ Coke'},
                            {'Process': 'Coke Production', 'Output': out_label, 'Flow': 'Input', 'Input': 'Nitrogen', 'Category': 'Material', 'ValueUnit': 'kg', 'Amount': float(n2_per_MJprod), 'Unit': 'kg per MJ Coke'},
                            {'Process': 'Coke Production', 'Output': out_label, 'Flow': 'Output', 'Input': 'Coke (Energy)', 'Category': 'Output', 'ValueUnit': 'MJ', 'Amount': 1.0, 'Unit': 'MJ per MJ Coke'},
                            {'Process': 'Coke Production', 'Output': out_label, 'Flow': 'Output', 'Input': proc_gas_label, 'Category': 'Output', 'ValueUnit': 'MJ', 'Amount': float((gas_meta.get('gas_coke_MJ', 0.0) or 0.0) / max(denom_MJ, 1e-12)), 'Unit': 'MJ per MJ Coke'},
                            {'Process': 'Coke Production', 'Output': out_label, 'Flow': 'Output', 'Input': 'Óleo Alcatrão', 'Category': 'Output', 'ValueUnit': 'kg', 'Amount': float(((30.0 / 1000.0) * (coal_energy_MJ / max(lhv_coal, 1e-12))) / max(denom_MJ, 1e-12)), 'Unit': 'kg per MJ Coke'},
                            {'Process': 'Coke Production', 'Output': out_label, 'Flow': 'Output', 'Input': 'Óleo Leve Bruto', 'Category': 'Output', 'ValueUnit': 'kg', 'Amount': float(((8.0 / 1000.0) * (coal_energy_MJ / max(lhv_coal, 1e-12))) / max(denom_MJ, 1e-12)), 'Unit': 'kg per MJ Coke'},
                            {'Process': 'Coke Production', 'Output': out_label, 'Flow': 'Output', 'Input': 'Amônia Anidra', 'Category': 'Output', 'ValueUnit': 'kg', 'Amount': float(((3.0 / 1000.0) * (coal_energy_MJ / max(lhv_coal, 1e-12))) / max(denom_MJ, 1e-12)), 'Unit': 'kg per MJ Coke'},
                        ])

                        ef_coal = float(ef_ghg_per_mj.get('Coal', 0.0))
                        ef_elec = float(ef_ghg_per_mj.get('Electricity', 0.0))
                        total_kg_per_kg = coal_per_MJprod * ef_coal + elec_grid_per_MJprod * ef_elec
                        additions.append({'Process': 'Coke Production', 'Output': out_label, 'Flow': 'Emissions', 'Input': 'CO2e (LCI)', 'Category': 'Emissions', 'ValueUnit': 'kgCO2e', 'Amount': float(total_kg_per_kg), 'Unit': 'kg CO2e per MJ Coke'})

                        try:
                            pg_amount_per_MJ = float((gas_meta.get('gas_coke_MJ', 0.0) or 0.0) / max(denom_MJ, 1e-12))
                        except Exception:
                            pg_amount_per_MJ = 0.0
                        ef_pg_coke = float(gas_meta.get('EF_coke_gas', gas_meta.get('EF_process_gas', 0.0)) or 0.0) / 1000.0
                        if pg_amount_per_MJ > 0 and ef_pg_coke >= 0:
                            additions.append({'Process': 'Coke Production', 'Output': out_label, 'Flow': 'Emissions', 'Input': 'CO2e (Process Gas)', 'Category': 'Emissions', 'ValueUnit': 'kgCO2e', 'Amount': - float(pg_amount_per_MJ * ef_pg_coke), 'Unit': 'kg CO2e per MJ Coke'})

                        coke_out_mj_per_mj = 1.0
                        ef_coke = float(ef_ghg_per_mj.get('Coke', 0.0) or 0.0)
                        if ef_coke >= 0:
                            additions.append({'Process': 'Coke Production', 'Output': out_label, 'Flow': 'Emissions', 'Input': 'CO2e (Coke)', 'Category': 'Emissions', 'ValueUnit': 'kgCO2e', 'Amount': - float(coke_out_mj_per_mj * ef_coke), 'Unit': 'kg CO2e per MJ Coke'})

                        net_amount = float(total_kg_per_kg) - float(pg_amount_per_MJ * ef_pg_coke) - float(coke_out_mj_per_mj * ef_coke)
                        additions.append({'Process': 'Coke Production', 'Output': out_label, 'Flow': 'Emissions', 'Input': 'CO2e (Net)', 'Category': 'Emissions', 'ValueUnit': 'kgCO2e', 'Amount': float(net_amount), 'Unit': 'kg CO2e per MJ Coke'})
                    except Exception:
                        pass
                    continue

                # Generic path
                sub_energy = sub[(sub['Flow'] == 'Input') & (sub['Category'] == 'Energy')]
                if sub_energy.empty:
                    continue
                total_kg_per_kg = 0.0
                for _, row in sub_energy.iterrows():
                    try:
                        amt_mj_per_kg = float(row.get('Amount', 0.0) or 0.0)
                    except Exception:
                        amt_mj_per_kg = 0.0
                    ef_per_mj = _ef_for_lci_input(str(row.get('Input', '')))
                    total_kg_per_kg += amt_mj_per_kg * ef_per_mj

                additions.append({'Process': proc, 'Output': out, 'Flow': 'Emissions', 'Input': 'CO2e (LCI)', 'Category': 'Emissions', 'ValueUnit': 'kgCO2e', 'Amount': total_kg_per_kg, 'Unit': f'kg CO2e per kg {out}'})

                if str(proc) == 'Blast Furnace':
                    try:
                        bf_delta = max(0.0, float(getattr(params, 'bf_adj_intensity', 0.0)) - float(getattr(params, 'bf_base_intensity', 0.0)))
                    except Exception:
                        bf_delta = 0.0
                    if bf_delta > 0.0:
                        bf_pg_output_by_out[str(out)] = bf_delta
                        output_additions.append({'Process': 'Blast Furnace', 'Output': out, 'Flow': 'Output', 'Input': proc_gas_label, 'Category': 'Output', 'ValueUnit': 'MJ', 'Amount': bf_delta, 'Unit': f'MJ per kg {out}'})
                    try:
                        r_bf = next((r for r in recipes if r.name == 'Blast Furnace'), None)
                        pig_per_run = float((r_bf.outputs.get('Pig Iron', 1.0)) if r_bf else 1.0)
                        escoria_per_kg = 0.330 / max(pig_per_run, 1e-12)
                    except Exception:
                        escoria_per_kg = 0.0
                    if escoria_per_kg > 0.0:
                        output_additions.append({'Process': 'Blast Furnace', 'Output': out, 'Flow': 'Output', 'Input': 'Escória', 'Category': 'Output', 'ValueUnit': 'kg', 'Amount': escoria_per_kg, 'Unit': f'kg per kg {out}'})

            # Process gas credit lines
            pg_additions = []
            net_additions = []

            def _add_pg_credit_for_process(proc_name: str):
                sub_proc = lci_df[lci_df['Process'] == proc_name]
                if sub_proc.empty:
                    return
                if proc_name == 'Blast Furnace':
                    ef_pg_local = float(gas_meta.get('EF_bf_gas', gas_meta.get('EF_process_gas', 0.0)) or 0.0) / 1000.0
                elif proc_name == 'Coke Production':
                    if coke_custom_rows:
                        return
                    ef_pg_local = float(gas_meta.get('EF_coke_gas', gas_meta.get('EF_process_gas', 0.0)) or 0.0) / 1000.0
                else:
                    ef_pg_local = float(gas_meta.get('EF_process_gas', 0.0) or 0.0) / 1000.0
                for out_val, sub_out in sub_proc.groupby('Output', dropna=False):
                    mj_per_kg = float(bf_pg_output_by_out.get(str(out_val), 0.0) or 0.0) if proc_name == 'Blast Furnace' else 0.0
                    if mj_per_kg <= 0.0:
                        mask = (
                            (sub_out['Flow'] == 'Input') & (sub_out['Category'] == 'Energy') & (sub_out['Input'] == f"{proc_gas_label} (Internal)")
                        )
                        mj_per_kg = float(sub_out.loc[mask, 'Amount'].sum()) if not sub_out.loc[mask].empty else 0.0
                    if mj_per_kg <= 0:
                        continue
                    pg_additions.append({'Process': proc_name, 'Output': str(out_val), 'Flow': 'Emissions', 'Input': 'CO2e (Process Gas)', 'Category': 'Emissions', 'ValueUnit': 'kgCO2e', 'Amount': - mj_per_kg * ef_pg_local, 'Unit': f'kg CO2e per kg {out_val}'})
                    total_rows = [r for r in additions if r['Process'] == proc_name and r['Output'] == str(out_val) and r['Input'] == 'CO2e (LCI)']
                    total_val = float(total_rows[0]['Amount']) if total_rows else 0.0
                    net_additions.append({'Process': proc_name, 'Output': str(out_val), 'Flow': 'Emissions', 'Input': 'CO2e (Net)', 'Category': 'Emissions', 'ValueUnit': 'kgCO2e', 'Amount': total_val - (mj_per_kg * ef_pg_local), 'Unit': f'kg CO2e per kg {out_val}'})

            _add_pg_credit_for_process('Blast Furnace')
            _add_pg_credit_for_process('Coke Production')

            if coke_custom_rows:
                try:
                    lci_df = lci_df[lci_df['Process'] != 'Coke Production']
                except Exception:
                    pass
                lci_df = pd.concat([lci_df, pd.DataFrame(coke_custom_rows)], ignore_index=True)

            if additions or pg_additions or net_additions or output_additions or trace_additions:
                to_add = additions + pg_additions + net_additions + output_additions + trace_additions
                lci_df = pd.concat([lci_df, pd.DataFrame(to_add)], ignore_index=True)
                try:
                    _order_map = {'CO2e (LCI)': 0, 'CO2e (Process Gas)': 1, 'CO2e (Coke)': 2, 'CO2e (Net)': 3}
                    lci_df['InputOrder'] = lci_df['Input'].map(_order_map)
                    lci_df.loc[lci_df['Flow'] != 'Emissions', 'InputOrder'] = -1
                    lci_df['InputOrder'] = lci_df['InputOrder'].fillna(50)
                    lci_df.sort_values(["Process", "Output", "Flow", "Category", "InputOrder", "Input"], inplace=True)
                    lci_df.drop(columns=['InputOrder'], inplace=True)
                except Exception:
                    lci_df.sort_values(["Process", "Output", "Flow", "Category", "Input"], inplace=True)
    except Exception:
        # Non-fatal augmentation failure
        pass

    # Debug prints for emission factors (units: g CO2e/MJ)
    print("\n=== EMISSION FACTOR DEBUG ===")
    elec_grid = float(e_efs.get('Electricity', 0.0) or 0.0)
    elec_internal = float(gas_meta.get('ef_internal_electricity', meta.get('ef_internal_electricity', 0.0)) or 0.0)
    elec_f_int = float(gas_meta.get('f_internal', meta.get('f_internal', 0.0)) or 0.0)
    elec_used = float(gas_meta.get('ef_electricity_used', elec_f_int * elec_internal + (1 - elec_f_int) * elec_grid))
    print("ELECTRICITY:")
    print(f"  Grid EF: {elec_grid:.2f} g CO2e/MJ")
    print(f"  Internal EF: {elec_internal:.2f} g CO2e/MJ")
    print(f"  Internal Fraction: {elec_f_int:.3f}")
    print(f"  Used EF (blended): {elec_used:.2f} g CO2e/MJ")

    gas_grid = float(gas_meta.get('ef_nat_gas_grid', e_efs.get('Gas', 0.0)) or 0.0)
    gas_proc = float(gas_meta.get('EF_process_gas', 0.0) or 0.0)
    gas_f_int = float(gas_meta.get('f_internal_gas', 0.0) or 0.0)
    gas_used = float(gas_meta.get('ef_gas_blended', e_efs.get('Gas', 0.0)) or 0.0)
    print("\nGAS:")
    print(f"  Natural Gas EF (grid): {gas_grid:.2f} g CO2e/MJ")
    print(f"  Process Gas EF (internal): {gas_proc:.2f} g CO2e/MJ")
    print(f"  Internal Fraction: {gas_f_int:.3f}")
    print(f"  Used EF (blended): {gas_used:.2f} g CO2e/MJ")

    print(f"\nPROCESS GAS BREAKDOWN:")
    print(f"  Coke Gas: {gas_meta.get('gas_coke_MJ',0.0):.1f} MJ (EF: {gas_meta.get('EF_coke_gas',0.0):.1f})")
    print(f"  BF Gas: {gas_meta.get('gas_bf_MJ',0.0):.1f} MJ (EF: {gas_meta.get('EF_bf_gas',0.0):.1f})")
    print(f"  Total: {gas_meta.get('total_process_gas_MJ',0.0):.1f} MJ (EF: {gas_meta.get('EF_process_gas',0.0):.1f})")
    print(f"  Direct Use: {gas_meta.get('direct_use_gas_MJ',0.0):.1f} MJ")
    print(f"  Electricity: {gas_meta.get('electricity_gas_MJ',0.0):.1f} MJ")
    print("============================\n")

    return lci_df


__all__ = [
    'augment_lci_and_debug',
]

