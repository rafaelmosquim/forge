# -*- coding: utf-8 -*-
"""
Final script to calculate a material balance matrix.
This version handles all configurations via YAML files, including direct process emissions.

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

class Process:
    """Represents a single recipe with its inputs and outputs."""
    __slots__ = ('name', 'inputs', 'outputs')
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs

def load_energy_data(energy_int_file, energy_matrix_file):
    """Loads and combines energy data from YAML files."""
    try:
        with open(energy_int_file, 'r', encoding='utf-8') as f:
            energy_int_data = yaml.safe_load(f).get('processes', {}) or {}
    except Exception as e:
        print(f"FATAL Error reading or parsing '{energy_int_file}': {e}")
        return {}, []

    try:
        with open(energy_matrix_file, 'r', encoding='utf-8') as f:
            energy_shares_data = yaml.safe_load(f).get('processes', {}) or {}
    except Exception as e:
        print(f"FATAL Error reading or parsing '{energy_matrix_file}': {e}")
        return {}, []

    energy_int_series = pd.Series(energy_int_data, name="total_energy").fillna(0)
    energy_shares_df = pd.DataFrame.from_dict(energy_shares_data, orient='index').fillna(0)
    energy_shares_df = energy_shares_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    energy_df = energy_shares_df.multiply(energy_int_series, axis='index')
    all_carriers = sorted(energy_df.columns.tolist())
    energy_dict = energy_df.to_dict('index')
    return energy_dict, all_carriers

def calculate_energy_balance(production_level, energy_dict, carriers):
    """Calculates energy consumption based on production levels."""
    if not energy_dict or not carriers: return None
    matrix_data = []
    process_names = []
    for process, runs in production_level.items():
        if process in energy_dict and runs > 0:
            process_energy = energy_dict[process]
            row = [runs * process_energy.get(carrier, 0) for carrier in carriers]
            matrix_data.append(row)
            process_names.append(process)
    if not matrix_data: return None
    energy_df = pd.DataFrame(matrix_data, index=process_names, columns=carriers)
    energy_df.loc['TOTAL'] = energy_df.sum()
    return energy_df

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
            recipe_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: The recipe file '{filepath}' was not found.")
        return None
    recipes = []
    param_vars = vars(params)
    for item in recipe_data:
        inputs = {
            material: eval(str(formula), param_vars)
            for material, formula in item.get('inputs', {}).items()
        }
        outputs = item.get('outputs', {})
        recipes.append(Process(item['process'], inputs, outputs))
    return recipes

def load_routes_from_yaml(filepath):
    """
    Loads a route configuration from a YAML file,
    correctly handling nesting and null (~) values.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        # Default to an empty dictionary if the file is empty
        nested_data = {}

        # Check if data is nested inside a single top-level key (like 'processes:' or 'routes:')
        if isinstance(data, dict) and len(data) == 1:
            nested_data = next(iter(data.values())) or {}
        elif isinstance(data, dict):
             nested_data = data

        # Clean the final data: replace None values (from ~) with 0
        # This is the key step that fixes the issue.
        cleaned_data = {key: (value if value is not None else 0) for key, value in nested_data.items()}
        return cleaned_data

    except FileNotFoundError:
        print(f"FATAL ERROR: The route configuration file '{filepath}' was not found.")
        return None
    
def load_energy_emission_factors(filepath):
    """Loads emission factors for energy carriers from a YAML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: The emission factors file '{filepath}' was not found.")
        return None

def load_process_emission_factors(filepath):
    """Loads direct emission factors for processes from a YAML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: The process emission factors file '{filepath}' was not found.")
        return None

def calculate_emissions(energy_balance_df, energy_emission_factors, production_level, process_emission_factors):
    """
    Calculates total emissions from both energy use and direct process emissions.
    Converts from g/MJ factors for energy.
    """
    # --- 1. Calculate Energy-Related Emissions ---
    # Align factors with the columns in the energy balance table
    aligned_energy_factors = pd.Series(energy_emission_factors).reindex(energy_balance_df.columns).fillna(0)
    # Multiply energy consumption (MJ) by emission factors (g/MJ)
    emissions_g_df = energy_balance_df.multiply(aligned_energy_factors, axis='columns')
    # Convert from grams to kilograms
    emissions_df = emissions_g_df / 1000.0

    # --- 2. Calculate Direct Process Emissions ---
    # Create a Series of process emissions based on production levels
    process_emissions_series = pd.Series(production_level).map(process_emission_factors).fillna(0)
    # Add this as a new column to the emissions table
    emissions_df['Process Emissions'] = process_emissions_series

    # --- 3. Calculate Final Total ---
    # Add a new column for the total emissions per process
    emissions_df['TOTAL CO2e'] = emissions_df.sum(axis=1)
    
    return emissions_df

def calculate_balance_matrix(recipes, final_demand, production_routes):
    """Calculates the material balance with a special diagnostic for debugging routes."""
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

        # --- DIAGNOSTIC BLOCK FOR ROUTE DEBUGGING ---
        if material == "Regular Steel":
            print("\n" + "="*25 + " DEBUGGING 'Regular Steel' ROUTE " + "="*25)
            print(f"Material Needed: '{material}'")
            producer_names = [p.name for p in producers]
            print(f"Producer(s) found in recipes.yml: {producer_names}")
            print(f"Routes available in route_config.yml: {list(production_routes.keys())[:10]}...") # Shows first 10 routes
            
            for p_name in producer_names:
                # To be extra sure, we check both stripped and raw names
                if p_name in production_routes:
                    print(f"  - ✅ SUCCESS: Match found for '{p_name}'. Route value: {production_routes.get(p_name)}")
                elif p_name.strip() in production_routes:
                    print(f"  - ❌ MISMATCH: Raw name '{p_name}' not found, but a version with stripped whitespace was found.")
                    print("     This suggests an invisible space character in one of your YAML files.")
                else:
                    print(f"  - ❌ FAILURE: No match found for '{p_name}' in the route_config.yml keys.")
            print("="*81 + "\n")
        # --- END DIAGNOSTIC BLOCK ---

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

# ===================================================================
#                           MAIN EXECUTION
# ===================================================================
if __name__ == "__main__":
    # --- Load all configurations ---
    param_filepath = os.path.join('data', 'parameters.yml')
    params = load_parameters(param_filepath)
    
    if params:
        recipe_filepath = os.path.join('data', 'recipes.yml')
        recipes = load_recipes_from_yaml(recipe_filepath, params)

        route_filepath = os.path.join('data', 'route_config.yml')
        production_routes = load_routes_from_yaml(route_filepath)

        energy_emission_factors_filepath = os.path.join('data', 'emission_factors.yml')
        energy_emission_factors = load_energy_emission_factors(energy_emission_factors_filepath)

        process_emission_factors_filepath = os.path.join('data', 'process_emissions.yml')
        process_emission_factors = load_process_emission_factors(process_emission_factors_filepath)


    # --- Run Calculations ---
    if params and recipes and production_routes:
        final_demand = {"Regular Steel": 1000.0}
        balance_matrix, production_level = calculate_balance_matrix(
            recipes, 
            final_demand, 
            production_routes.get('processes', {})  # Ensure proper route loading
    )
        
        if balance_matrix is not None:
            print("\n--- Calculated Balance Matrix ---")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 120)
            print(balance_matrix.round(3))
            
            if production_level:
                # --- Energy Balance ---
                energy_int_file = os.path.join('data', 'energy_int.yml')
                energy_matrix_file = os.path.join('data', 'energy_matrix.yml')
                energy_dict, carriers = load_energy_data(energy_int_file, energy_matrix_file)
                
                if energy_dict and carriers:
                    energy_balance = calculate_energy_balance(production_level, energy_dict, carriers)
                    if energy_balance is not None:
                        print("\n--- Energy Balance (MJ) ---")
                        print(energy_balance.round(3))
                        
                        # --- Emissions Calculation ---
                        if energy_emission_factors and process_emission_factors:
                            emissions_table = calculate_emissions(
                                energy_balance, 
                                energy_emission_factors, 
                                production_level,
                                process_emission_factors
                            )
                            if emissions_table is not None:
                                print("\n--- Emissions (kg CO2e) ---")
                                print(emissions_table.round(3))

        else:
            print("\nCould not generate the balance matrix. Please check error messages.")