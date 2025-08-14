
# -*- coding: utf-8 -*-
"""
Final script to calculate a material balance matrix.
This version handles complex production routes with specified shares,
loaded from an external route configuration CSV file.

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
    """
    Loads and combines energy data from the provided YAML file structure,
    correctly handling the 'processes' block and null (~) values.
    """
    try:
        # Load total energy values from the YAML file
        with open(energy_int_file, 'r', encoding='utf-8') as f:
            # Get the data from inside the 'processes' block
            energy_int_data = yaml.safe_load(f).get('processes', {})
            if energy_int_data is None:
                energy_int_data = {}

    except Exception as e:
        print(f"FATAL Error reading or parsing '{energy_int_file}': {e}")
        return {}, []

    try:
        # Load energy carrier shares from the YAML file
        with open(energy_matrix_file, 'r', encoding='utf-8') as f:
            # Get the data from inside the 'processes' block
            energy_shares_data = yaml.safe_load(f).get('processes', {})
            if energy_shares_data is None:
                energy_shares_data = {}

    except Exception as e:
        print(f"FATAL Error reading or parsing '{energy_matrix_file}': {e}")
        return {}, []

    # --- Use Pandas for the robust final calculation ---
    
    # Convert dictionaries to pandas objects.
    # The `fillna(0)` correctly handles the null (~) values from your YAML.
    energy_int_series = pd.Series(energy_int_data, name="total_energy").fillna(0)
    energy_shares_df = pd.DataFrame.from_dict(energy_shares_data, orient='index').fillna(0)
    
    # Ensure all share values are numeric, coercing any errors
    energy_shares_df = energy_shares_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Multiply the shares DataFrame by the intensity Series
    energy_df = energy_shares_df.multiply(energy_int_series, axis='index')

    # Get the list of all unique carriers
    all_carriers = sorted(energy_df.columns.tolist())

    # Convert the final, calculated DataFrame back to the required dictionary format
    energy_dict = energy_df.to_dict('index')

    return energy_dict, all_carriers

def calculate_energy_balance(production_level, energy_dict, carriers):
    """
    Calculates energy consumption based on production levels and energy data
    """
    if not energy_dict or not carriers:
        return None
    
    # Prepare matrix data
    matrix_data = []
    process_names = []
    
    for process, runs in production_level.items():
        if process in energy_dict and runs > 0:
            process_energy = energy_dict[process]
            # Create row for this process
            row = [runs * process_energy.get(carrier, 0) for carrier in carriers]
            matrix_data.append(row)
            process_names.append(process)
    
    if not matrix_data:
        return None
    
    # Create DataFrame
    energy_df = pd.DataFrame(
        matrix_data,
        index=process_names,
        columns=carriers
    )
    
    # Add total row
    energy_df.loc['TOTAL'] = energy_df.sum()
    
    return energy_df

def load_parameters(filepath):
    """Loads a YAML file and converts it into a nested object."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            params_dict = yaml.safe_load(f)
            params_obj = json.loads(json.dumps(params_dict), object_hook=lambda d: SimpleNamespace(**d))
            return params_obj
    except FileNotFoundError:
        print(f"FATAL ERROR: The parameters file '{filepath}' was not found.")
        return None

def load_recipes_from_yaml(filepath, params):
    """Loads all recipes and evaluates formulas using the provided parameters."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            recipe_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: The recipe file '{filepath}' was not found.")
        return None
        
    recipes = []
    param_vars = vars(params)
    for item in recipe_data:
        inputs = {}
        for material, formula in item.get('inputs', {}).items():
            try:
                inputs[material] = eval(str(formula), param_vars)
            except Exception as e:
                print(f"Error evaluating formula '{formula}' in process '{item['process']}': {e}")
                return None
        outputs = item.get('outputs', {})
        recipes.append(Process(item['process'], inputs, outputs))
        
    return recipes

def load_routes_from_csv(filepath):
    """Loads a route configuration from a CSV file into a dictionary."""
    try:
        df = pd.read_csv(filepath, delimiter=';', decimal=',', index_col=0)
        # Convert the 'Process' column to a dictionary
        return df['Process'].to_dict()
    except FileNotFoundError:
        print(f"FATAL ERROR: The route configuration file '{filepath}' was not found.")
        return None

def calculate_balance_matrix(recipes, final_demand, production_routes):
    """
    Calculates the material balance using share-based production routes.
    Returns:
        tuple: (balance_matrix DataFrame, production_level dictionary)
    """
    all_materials = set()
    producers_of = defaultdict(list)
    recipes_dict = {r.name: r for r in recipes} # For quick lookup

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

        # --- Share-based logic ---
        active_producers = [p for p in producers if production_routes.get(p.name, 0) > 0]
        
        if not active_producers:
             print(f"Warning: Material '{material}' is needed, but no active route found to produce it. Check route config.")
             continue

        for process in active_producers:
            share = production_routes.get(process.name, 0)
            amount_from_this_process = amount_needed * share

            produced_per_run = process.outputs.get(material, 0)
            if produced_per_run == 0: continue

            runs = amount_from_this_process / produced_per_run
            production_level[process.name] += runs
            
            # Add inputs from this specific process run to the requirements
            for in_material, in_amount in process.inputs.items():
                required[in_material] += runs * in_amount
                if producers_of.get(in_material) and in_material not in processed_in_queue:
                    queue.append(in_material)
                    processed_in_queue.add(in_material)
        
        required[material] = 0 # Mark the total requirement for this material as met

    external_inputs = defaultdict(float)
    for material, amount in required.items():
        if amount > 1e-9 and not producers_of.get(material):
            external_inputs[material] = amount

    # --- Build the Final Report DataFrame ---
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
#                          MAIN EXECUTION
# ===================================================================
if __name__ == "__main__":
    param_filepath = os.path.join('data', 'parameters.yml')
    params = load_parameters(param_filepath)
    
    if params:
        recipe_filepath = os.path.join('data', 'recipes.yml')
        recipes = load_recipes_from_yaml(recipe_filepath, params)

        # Load the desired production route from its CSV file
        route_filepath = os.path.join('data', 'route_config.csv')
        production_routes = load_routes_from_csv(route_filepath)

    if params and recipes and production_routes:
        # Define your final goal
        final_demand = {"Pig Iron": 1.0}
        
        # Run the calculation - single call that returns both results
        balance_matrix, production_level = calculate_balance_matrix(recipes, final_demand, production_routes)
        
        if balance_matrix is not None:
            print("\n--- Calculated Balance Matrix ---")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 120)
            print(balance_matrix.round(3))
            
            # Energy balance calculation
            if production_level:  # Only proceed if we have production levels
                # --- CHANGE THESE LINES ---
                # Point to the new YAML files instead of the old CSV files
                energy_int_file = os.path.join('data', 'energy_int.yml')
                energy_matrix_file = os.path.join('data', 'energy_matrix.yml')
                # --------------------------
                
                energy_dict, carriers = load_energy_data(energy_int_file, energy_matrix_file)
                
                if energy_dict and carriers:
                    energy_balance = calculate_energy_balance(production_level, energy_dict, carriers)
                    
                    if energy_balance is not None:
                        print("\n--- Energy Balance (MJ) ---")
                        print(energy_balance.round(3))
        else:
            print("\nCould not generate the balance matrix. Please check error messages.")
