# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 11:26:02 2025

@author: rafae
"""

# -*- coding: utf-8 -*-
"""
Final script to calculate a material balance matrix.
This version uses an explicit market configuration file (mkt_config.yml)
to determine the correct emissions calculation logic for each process.

This updated version includes co-product energy recovery from the
blast furnace and coke making processes, internal electricity generation,
and a system expansion calculation to determine the net emissions
credit/debit, as per ISO 14044 guidelines.

Created on Tue Aug 05 2025
@author: rafae
"""

import yaml
import pandas as pd
from collections import defaultdict, deque
import os
import json
from types import SimpleNamespace

# ===================================================================
#                           Data Models
# ===================================================================

class Process:
    """Represents a single recipe with its inputs and outputs."""
    __slots__ = ('name', 'inputs', 'outputs')
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs

# ===================================================================
#                         Configuration Loaders
# ===================================================================

def load_parameters(filepath):
    """Loads a YAML file and converts it into a nested object."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            params_dict = yaml.safe_load(f)
            # Convert to namespace object for easy access (e.g., params.lca_parameters.ef_grid)
            params = json.loads(json.dumps(params_dict), 
                                object_hook=lambda d: SimpleNamespace(**d))
            return params
    except FileNotFoundError:
        print(f"FATAL ERROR: The parameters file '{filepath}' was not found.")
        return None

def load_recipes_from_yaml(filepath, params):
    """Loads all recipes and evaluates formulas."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            recipe_data = yaml.safe_load(f) or []
    except FileNotFoundError:
        print(f"FATAL ERROR: The recipe file '{filepath}' was not found.")
        return None
        
    recipes = []
    # Create a dictionary from the params namespace for use in eval()
    param_vars = vars(params)
    # If nested parameter objects exist (like lca_parameters), add their attributes to the eval context.
    # Iterate over a copy of the items to prevent "dictionary changed size during iteration" error.
    for key, value in list(vars(params).items()):
        if isinstance(value, SimpleNamespace):
            param_vars.update(vars(value))

    for item in recipe_data:
        process_name = item.get('process', '').strip()
        if not process_name: continue

        inputs = {
            material.strip(): eval(str(formula), param_vars)
            for material, formula in item.get('inputs', {}).items()
        }
        outputs = {k.strip(): v for k, v in item.get('outputs', {}).items()}
        recipes.append(Process(process_name, inputs, outputs))
        
    return recipes

def load_market_config(filepath):
    """Loads the market configuration file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or []
            # Convert list of dicts to a single dict for easy lookup
            return {item['name'].strip(): item['value'] for item in config_data}
    except FileNotFoundError:
        print(f"FATAL ERROR: The market config file '{filepath}' was not found.")
        return None
    except (KeyError, TypeError):
        print(f"FATAL ERROR: The file '{filepath}' has an incorrect format. Expected a list of dicts with 'name' and 'value' keys.")
        return None

def load_data_from_yaml(filepath, default_value=0):
    """A robust, generic function to load other data from a YAML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: The data file '{filepath}' was not found. Using empty data.")
        return {}
    
    # Handle cases where the data is nested under a single top-level key
    if isinstance(data, dict) and len(data) == 1:
        data = next(iter(data.values())) or {}
    
    if not isinstance(data, dict):
        print(f"Warning: Data in '{filepath}' is not a dictionary. Returning empty data.")
        return {}

    return {str(k).strip(): (v if v is not None else default_value) for k, v in data.items()}

# ===================================================================
#                       Calculation Functions
# ===================================================================

def adjust_blast_furnace_intensity(energy_int_data, energy_shares_data, params):
    """Adjusts blast furnace energy intensity based on process gas usage."""
    try:
        # The parameter is negative because it represents an energy credit
        process_gas_credit = abs(params.process_gas) 
    except AttributeError:
        print("ERROR: 'process_gas' not found in parameters. Skipping BF intensity adjustment.")
        return
    
    if "Blast Furnace" not in energy_int_data:
        return
    
    base_intensity = energy_int_data["Blast Furnace"]
    
    bf_energy_shares = energy_shares_data.get("Blast Furnace", {})
    relevant_carriers = ['Gas', 'Coal', 'Coke', 'Charcoal']
    carrier_sum = sum(bf_energy_shares.get(c, 0) for c in relevant_carriers)
    
    # The credit reduces the net energy input
    denominator = 1 - (process_gas_credit * carrier_sum)
    if abs(denominator) > 1e-6:
        adjusted_intensity = base_intensity / denominator
        energy_int_data["Blast Furnace"] = adjusted_intensity
        print(f"Adjusted Blast Furnace intensity from {base_intensity:.2f} to {adjusted_intensity:.2f} MJ/t")


def calculate_balance_matrix(recipes, final_demand, production_routes):
    """Calculates material balance, returning the matrix, production levels, and external inputs."""
    all_materials, producers_of = set(), defaultdict(list)
    recipes_dict = {r.name: r for r in recipes}
    for r in recipes:
        all_materials.update(r.inputs.keys())
        all_materials.update(r.outputs.keys())
        for m in r.outputs: producers_of[m].append(r)
            
    demand_item = list(final_demand.keys())[0]
    if not producers_of.get(demand_item): return None, None, None
        
    required, production_level = defaultdict(float, final_demand), defaultdict(float)
    queue, processed_in_queue = deque(list(final_demand.keys())), set(list(final_demand.keys()))
    
    while queue:
        material = queue.popleft()
        producers = producers_of.get(material)
        if not producers: continue
        
        amount_needed = required[material]
        if amount_needed <= 1e-9: continue
        
        active_producers = [p for p in producers if production_routes.get(p.name, 0) > 0]
        if not active_producers: continue
            
        for process in active_producers:
            share, produced_per_run = production_routes.get(process.name, 0), process.outputs.get(material, 0)
            if produced_per_run == 0: continue
            
            runs = (amount_needed * share) / produced_per_run
            production_level[process.name] += runs
            
            for in_mat, in_amt in process.inputs.items():
                required[in_mat] += runs * in_amt
                if producers_of.get(in_mat) and in_mat not in queue and in_mat not in processed_in_queue:
                    queue.append(in_mat)
        processed_in_queue.add(material)
        required[material] = 0
        
    external_inputs = defaultdict(float)
    for mat, amt in required.items():
        if amt > 1e-9 and not producers_of.get(mat): external_inputs[mat] = amt
            
    sorted_materials = sorted(list(all_materials))
    matrix_data, row_names = [], []
    for name, level in production_level.items():
        if level > 0:
            recipe = recipes_dict[name]
            row = [(recipe.outputs.get(m, 0) - recipe.inputs.get(m, 0)) * level for m in sorted_materials]
            matrix_data.append(row)
            row_names.append(name)
            
    if not matrix_data: return None, None, None
    
    matrix_data.append([external_inputs.get(m, 0) for m in sorted_materials])
    row_names.append("External Inputs")
    matrix_data.append([-final_demand.get(m, 0) for m in sorted_materials])
    row_names.append("Final Demand")
    
    balance_df = pd.DataFrame(matrix_data, index=row_names, columns=sorted_materials)
    return balance_df.loc[:, (balance_df.abs() > 1e-9).any(axis=0)], production_level, external_inputs


def calculate_energy_balance(production_level, energy_int_data, energy_shares_data):
    """Calculates the energy consumption per process and energy carrier."""
    if not energy_int_data or not energy_shares_data: return None
    
    energy_int = pd.Series(energy_int_data).fillna(0)
    energy_shares = pd.DataFrame.from_dict(energy_shares_data, orient='index').fillna(0)
    energy_df = energy_shares.multiply(energy_int, axis='index')
    
    matrix_data, process_names = [], []
    for process, runs in production_level.items():
        if process in energy_df.index and runs > 0:
            row = [runs * val for val in energy_df.loc[process]]
            matrix_data.append(row)
            process_names.append(process)

    if not matrix_data: return None
    
    balance_df = pd.DataFrame(matrix_data, index=process_names, columns=energy_df.columns)
    balance_df.loc['TOTAL'] = balance_df.sum()
    return balance_df

def calculate_emissions(mkt_config, production_level, energy_balance_df, energy_emission_factors, process_emission_factors, expansion_debits=None):
    """
    Calculates total emissions, using the market config file to split the logic.
    This version is updated to include the system expansion debit/credit.
    """
    if production_level is None: return None
    
    emissions_data = []
    for process_name, level in production_level.items():
        if level <= 1e-9: continue

        row = {'Process': process_name, 'Energy Emissions': 0, 'Direct Emissions': 0, 'Expansion Debit': 0}
        process_type = mkt_config.get(process_name)

        if process_type == 1: # Type 1: "From Market" process (direct emissions only)
            factor = process_emission_factors.get(process_name, 0)
            row['Direct Emissions'] = level * factor
        elif process_type == 2: # Type 2: Internal manufacturing process
            # Energy emissions
            if energy_balance_df is not None and process_name in energy_balance_df.index:
                energy_row = energy_balance_df.loc[process_name]
                for carrier, consumption in energy_row.items():
                    factor = energy_emission_factors.get(carrier, 0)
                    row['Energy Emissions'] += consumption * factor
            # Direct process emissions
            factor = process_emission_factors.get(process_name, 0)
            row['Direct Emissions'] += level * factor
        else:
            print(f"Warning: Process '{process_name}' not found in mkt_config.yml or has invalid type. Emissions not calculated.")
            
        emissions_data.append(row)

    if not emissions_data: return None

    emissions_df = pd.DataFrame(emissions_data).set_index('Process')
    
    # Add system expansion debit/credit (which is in kg) after converting other emissions to kg
    emissions_df_kg = emissions_df / 1000.0
    
    if expansion_debits:
        for process, debit_kg in expansion_debits.items():
            if process in emissions_df_kg.index:
                emissions_df_kg.loc[process, 'Expansion Debit'] = debit_kg

    emissions_df_kg['TOTAL CO2e'] = emissions_df_kg['Energy Emissions'] + emissions_df_kg['Direct Emissions'] + emissions_df_kg['Expansion Debit']
    emissions_df_kg.loc['TOTAL'] = emissions_df_kg.sum()
    
    return emissions_df_kg

# ===================================================================
#         NEW: Co-Product Energy & Emission Credit Functions
# ===================================================================

def calculate_recovered_energy(external_inputs, energy_balance_df, recipes, production_level, params):
    """Calculates the recoverable energy from Coke Making and Blast Furnace."""
    try:
        lca = params.lca_parameters
        
        # Energy from Coke Making (process energy + coal chemical energy)
        coke_recipe = next((r for r in recipes if r.name == "Coke Making"), None)
        if not coke_recipe:
            print("Warning: 'Coke Making' recipe not found.")
            return None
        
        # Find the name for coking coal in the inputs, assuming it contains 'Coal'
        coal_input_name = next((k for k in coke_recipe.inputs if 'Coal' in k), None)
        if not coal_input_name:
            print("Warning: 'Coking Coal' not found in Coke Making inputs. Cannot calculate coke energy recovery.")
            mass_coking_coal_kg = 0
        else:
            coke_runs = production_level.get("Coke Making", 0)
            mass_coking_coal_kg = coke_recipe.inputs[coal_input_name] * coke_runs

        # Total energy consumed by the coking process (from energy balance)
        energy_coke_process_mj = energy_balance_df.loc['Coke Making'].sum() if 'Coke Making' in energy_balance_df.index else 0
        
        # Chemical energy from the input coal
        energy_coal_chemical_mj = mass_coking_coal_kg * lca.lhv_coking_coal_mj_per_kg
        
        total_coke_energy_input_mj = energy_coke_process_mj + energy_coal_chemical_mj
        recovered_energy_coke_mj = total_coke_energy_input_mj * lca.coke_making_recovery_factor

        # Energy from Blast Furnace (process energy)
        energy_bf_process_mj = energy_balance_df.loc['Blast Furnace'].sum() if 'Blast Furnace' in energy_balance_df.index else 0
        recovered_energy_bf_mj = energy_bf_process_mj * abs(params.process_gas) # Use the same factor as intensity adjustment
        
        total_recovered = recovered_energy_coke_mj + recovered_energy_bf_mj
        if total_recovered < 1e-6:
             print("Warning: Total recovered energy is near zero. Expansion calculation may not be meaningful.")
             return None

        return {
            "coke_mj": recovered_energy_coke_mj,
            "bf_mj": recovered_energy_bf_mj,
            "total_mj": total_recovered
        }
    except Exception as e:
        print(f"ERROR calculating recovered energy: {e}")
        return None

def calculate_electricity_and_expansion(recovered_energy, final_demand, params):
    """
    Calculates the electricity balance, system expansion debit/credit,
    and final weighted electricity emission factor.
    """
    try:
        lca = params.lca_parameters
        final_demand_tonnes = sum(final_demand.values()) / 1000.0

        # 1. Total Plant Electricity Demand
        total_elec_demand_kwh = lca.plant_elec_demand_kwh_per_tonne * final_demand_tonnes

        # 2. Internal Electricity Generation
        total_recovered_energy_gj = recovered_energy['total_mj'] / 1000.0
        GJ_TO_KWH = 277.778
        internal_elec_gen_kwh = total_recovered_energy_gj * lca.utility_plant_efficiency * GJ_TO_KWH

        # 3. Electricity Balance
        grid_elec_purchased_kwh = max(0, total_elec_demand_kwh - internal_elec_gen_kwh)

        # 4. System Expansion Calculation (Credit/Debit)
        emissions_from_cog_kg = (recovered_energy['coke_mj'] / 1000.0) * lca.ef_cog_combustion_kg_per_gj
        emissions_from_bfg_kg = (recovered_energy['bf_mj'] / 1000.0) * lca.ef_bfg_combustion_kg_per_gj
        total_internal_gen_emissions_kg = emissions_from_cog_kg + emissions_from_bfg_kg

        avoided_grid_emissions_kg = internal_elec_gen_kwh * lca.grid_emission_factor_kg_per_kwh
        
        # Net impact: Incurred - Avoided. A positive value is a debit (bad), negative is a credit (good).
        net_expansion_impact_kg = total_internal_gen_emissions_kg - avoided_grid_emissions_kg
        
        # 5. Allocate Debit/Credit back to the producing processes
        # A positive net_expansion_impact is a debit that gets added to the process emissions
        debit_coke_kg = net_expansion_impact_kg * (recovered_energy['coke_mj'] / recovered_energy['total_mj'])
        debit_bf_kg = net_expansion_impact_kg * (recovered_energy['bf_mj'] / recovered_energy['total_mj'])
        debits = {'Coke Making': debit_coke_kg, 'Blast Furnace': debit_bf_kg}

        # 6. Final Weighted-Average Emission Factor for Electricity
        emissions_from_grid_purchase_kg = grid_elec_purchased_kwh * lca.grid_emission_factor_kg_per_kwh
        total_electricity_emissions_kg = total_internal_gen_emissions_kg + emissions_from_grid_purchase_kg
        final_weighted_ef_elec_kg_per_kwh = total_electricity_emissions_kg / total_elec_demand_kwh if total_elec_demand_kwh > 0 else 0

        return {
            "total_elec_demand_kwh": total_elec_demand_kwh,
            "internal_elec_gen_kwh": internal_elec_gen_kwh,
            "grid_elec_purchased_kwh": grid_elec_purchased_kwh,
            "net_expansion_impact_kg": net_expansion_impact_kg,
            "debits": debits,
            "final_weighted_ef_elec_kg_per_kwh": final_weighted_ef_elec_kg_per_kwh
        }
    except Exception as e:
        print(f"ERROR in electricity and expansion calculation: {e}")
        return None

# ===================================================================
#                           MAIN EXECUTION
# ===================================================================
if __name__ == "__main__":
    # --- Load all configurations ---
    params = load_parameters(os.path.join('data', 'parameters.yml'))
    if not params: exit()
    
    recipes = load_recipes_from_yaml(os.path.join('data', 'recipes.yml'), params)
    production_routes = load_data_from_yaml(os.path.join('data', 'route_config.yml'))
    mkt_config = load_market_config(os.path.join('data', 'mkt_config.yml'))
    energy_emission_factors = load_data_from_yaml(os.path.join('data', 'emission_factors.yml'))
    process_emission_factors = load_data_from_yaml(os.path.join('data', 'process_emissions.yml'))
    energy_int_data = load_data_from_yaml(os.path.join('data', 'energy_int.yml'))
    energy_shares_data = load_data_from_yaml(os.path.join('data', 'energy_matrix.yml'))

    # --- Pre-calculation Adjustments ---
    if all([energy_int_data, energy_shares_data, hasattr(params, 'process_gas')]):
        adjust_blast_furnace_intensity(energy_int_data, energy_shares_data, params)

    if all((params, recipes, production_routes, mkt_config)):
        final_demand = {"Finished Steel": 1000.0}
        balance_matrix, production_level = calculate_balance_matrix(recipes, final_demand, production_routes)
        
        if balance_matrix is not None:
            recipes_dict = {r.name: r for r in recipes}
            internal_electricity = calculate_internal_electricity(production_level, recipes_dict, params)
            balance_matrix = adjust_electricity_demand(balance_matrix, internal_electricity)
            energy_balance = calculate_energy_balance(production_level, energy_int_data, energy_shares_data)
            
            # Only ONE call to calculate_emissions is needed
            emissions_table = calculate_emissions(
                mkt_config,
                production_level,
                energy_balance, 
                energy_emission_factors, 
                process_emission_factors,
                internal_electricity,
                final_demand
            )
            
            if emissions_table is not None:
                product_name = list(final_demand.keys())[0]
                total_emissions = emissions_table.loc['TOTAL', 'TOTAL CO2e']
                print("\n-----------------------------------------------------------------------")
                print(f"Total emission for {final_demand[product_name]} units of {product_name} is {total_emissions:.2f} kg CO2e")
                print("-----------------------------------------------------------------------")
    else:
        print("\nModel execution failed. A required configuration file may be missing or have an incorrect format.")