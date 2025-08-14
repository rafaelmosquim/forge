# -*- coding: utf-8 -*-
"""
Final script to calculate a material balance matrix.
This version uses a single, robust function for all YAML data loading.

Created on Sun Aug  3 17:25:00 2025
@author: rafae
"""

import numpy as np
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
#                        Configuration Loaders
# ===================================================================

def load_data_from_yaml(filepath, default_value=0):
    """
    A robust, generic function to load data from a YAML configuration file.
    - Handles a single top-level key (e.g., 'processes', 'routes').
    - Handles empty files and null (~) values, replacing them with a default.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"FATAL ERROR: The data file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"FATAL Error reading or parsing '{filepath}': {e}")
        return None

    # If the data is nested inside a single top-level key, extract it.
    if isinstance(data, dict) and len(data) == 1:
        data = next(iter(data.values())) or {}
    
    if not isinstance(data, dict):
        print(f"Warning: Data in '{filepath}' is not a dictionary. Returning empty.")
        return {}

    # Clean the final data: replace None values with the default and strip key whitespace
    cleaned_data = {str(key).strip(): (value if value is not None else default_value) 
                    for key, value in data.items()}
    
    return cleaned_data

def load_parameters(filepath):
    """Loads a YAML file and converts it into a nested object."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            params_dict = yaml.safe_load(f)
            return json.loads(json.dumps(params_dict), object_hook=lambda d: SimpleNamespace(**d))
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
    param_vars = vars(params)
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

# ===================================================================
#                         Calculation Functions
# ===================================================================

def calculate_balance_matrix(recipes, final_demand, production_routes):
    """Calculates the material balance using share-based production routes."""
    all_materials = set()
    producers_of = defaultdict(list)
    recipes_dict = {r.name: r for r in recipes}
    for recipe in recipes:
        all_materials.update(recipe.inputs.keys())
        all_materials.update(recipe.outputs.keys())
        for material in recipe.outputs:
            producers_of[material].append(recipe)
    demand_item = list(final_demand.keys())[0]
    if not producers_of[demand_item]:
        print(f"FATAL ERROR: The demanded product '{demand_item}' is not an output in any recipe.")
        return None, None
    required = defaultdict(float, final_demand)
    production_level = defaultdict(float)
    queue = deque(list(final_demand.keys()))
    processed_in_queue = set(queue)
    while queue:
        material = queue.popleft()
        processed_in_queue.remove(material)
        producers = producers_of.get(material)
        if not producers: continue
        amount_needed = required[material]
        if amount_needed <= 1e-9: continue
        active_producers = [p for p in producers if production_routes.get(p.name, 0) > 0]
        if not active_producers:
            print(f"Warning: Material '{material}' is needed, but no active route found to produce it.")
            continue
        for process in active_producers:
            share = production_routes.get(process.name, 0)
            amount_from_this_process = amount_needed * share
            produced_per_run = process.outputs.get(material, 0)
            if produced_per_run == 0: continue
            runs = amount_from_this_process / produced_per_run
            production_level[process.name] += runs
            for in_material, in_amount in process.inputs.items():
                required[in_material] += runs * in_amount
                if producers_of.get(in_material) and in_material not in processed_in_queue:
                    queue.append(in_material)
                    processed_in_queue.add(in_material)
        required[material] = 0
    external_inputs = defaultdict(float)
    for material, amount in required.items():
        if amount > 1e-9 and not producers_of.get(material):
            external_inputs[material] = amount
    sorted_materials = sorted(list(all_materials))
    matrix_data, row_names = [], []
    for recipe_name, level in production_level.items():
        if level > 0:
            recipe = recipes_dict[recipe_name]
            row = [(recipe.outputs.get(m, 0) - recipe.inputs.get(m, 0)) * level for m in sorted_materials]
            matrix_data.append(row)
            row_names.append(recipe.name)
    if not matrix_data: return None, None
    matrix_data.append([external_inputs.get(m, 0) for m in sorted_materials])
    row_names.append("External Inputs")
    matrix_data.append([-final_demand.get(m, 0) for m in sorted_materials])
    row_names.append("Final Demand")
    balance_df = pd.DataFrame(matrix_data, index=row_names, columns=sorted_materials)
    balance_df = balance_df.loc[:, (balance_df.abs() > 1e-9).any(axis=0)]
    balance_df = balance_df.loc[(balance_df.abs() > 1e-9).any(axis=1)]
    return balance_df, production_level

def calculate_energy_balance(production_level, energy_int_data, energy_shares_data):
    """Calculates energy consumption based on production levels."""
    if not energy_int_data or not energy_shares_data: return None
    
    energy_int_series = pd.Series(energy_int_data).fillna(0)
    energy_shares_df = pd.DataFrame.from_dict(energy_shares_data, orient='index').fillna(0)
    energy_shares_df = energy_shares_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    energy_df = energy_shares_df.multiply(energy_int_series, axis='index')
    all_carriers = sorted(energy_df.columns.tolist())
    
    energy_dict = energy_df.to_dict('index')

    matrix_data, process_names = [], []
    for process, runs in production_level.items():
        if process in energy_dict and runs > 0:
            process_energy = energy_dict[process]
            row = [runs * process_energy.get(carrier, 0) for carrier in all_carriers]
            matrix_data.append(row)
            process_names.append(process)

    if not matrix_data: return None
    
    balance_df = pd.DataFrame(matrix_data, index=process_names, columns=all_carriers)
    balance_df.loc['TOTAL'] = balance_df.sum()
    
    return balance_df

def calculate_emissions(energy_balance_df, energy_emission_factors, production_level, process_emission_factors):
    """Calculates total emissions from both energy use and direct process emissions."""
    if energy_balance_df is None: return None

    # Energy-Related Emissions
    aligned_energy_factors = pd.Series(energy_emission_factors).reindex(energy_balance_df.columns).fillna(0)
    emissions_df = energy_balance_df.multiply(aligned_energy_factors, axis='columns') / 1000.0 # g->kg

    # Direct Process Emissions
    process_emissions_series = pd.Series(production_level).map(process_emission_factors).fillna(0)
    emissions_df['Process Emissions'] = process_emissions_series

    # Final Total
    emissions_df['TOTAL CO2e'] = emissions_df.sum(axis=1)
    return emissions_df

# ===================================================================
#                           MAIN EXECUTION
# ===================================================================
if __name__ == "__main__":
    # --- Load all configurations ---
    params = load_parameters(os.path.join('data', 'parameters.yml'))
    recipes = load_recipes_from_yaml(os.path.join('data', 'recipes.yml'), params)
    production_routes = load_data_from_yaml(os.path.join('data', 'route_config.yml'))
    energy_emission_factors = load_data_from_yaml(os.path.join('data', 'emission_factors.yml'))
    process_emission_factors = load_data_from_yaml(os.path.join('data', 'process_emissions.yml'))
    energy_int_data = load_data_from_yaml(os.path.join('data', 'energy_int.yml'))
    energy_shares_data = load_data_from_yaml(os.path.join('data', 'energy_matrix.yml'))

    # --- Run Calculations ---
    if all((params, recipes, production_routes)):
        final_demand = {"Regular Steel": 1000.0}
        balance_matrix, production_level = calculate_balance_matrix(recipes, final_demand, production_routes)
        
        if balance_matrix is not None:
            print("\n--- Calculated Balance Matrix ---")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 120)
            print(balance_matrix.round(3))
            
            if production_level:
                # --- Energy Balance ---
                energy_balance = calculate_energy_balance(production_level, energy_int_data, energy_shares_data)
                if energy_balance is not None:
                    print("\n--- Energy Balance (MJ) ---")
                    print(energy_balance.round(3))
                    
                    # --- Emissions Calculation ---
                    emissions_table = calculate_emissions(
                        energy_balance, 
                        energy_emission_factors, 
                        production_level,
                        process_emission_factors
                    )
                    if emissions_table is not None:
                        print("\n--- Emissions (kg CO2e) ---")
                        print(emissions_table.round(3))
                        
                        # --- ADDED SUMMARY LINE ---
                        product_name = list(final_demand.keys())[0]
                        total_emissions = emissions_table.loc['TOTAL', 'TOTAL CO2e']
                        print("\n-----------------------------------------------------------------------")
                        print(f"Total emission for {final_demand[product_name]} units of {product_name} is {total_emissions:.2f} kg CO2e")
                        print("-----------------------------------------------------------------------")
        else:
            print("\nCould not generate the balance matrix. Please check error messages.")