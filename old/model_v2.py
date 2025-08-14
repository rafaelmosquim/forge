# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 17:50:38 2025

@author: rafae
"""

# -*- coding: utf-8 -*-
"""
Final script to calculate a material balance matrix.
This version handles alternative production routes by allowing the user
to specify which recipe to use when a material has multiple producers.

Created on Sun Aug  3 17:05:00 2025
@author: rafae
"""

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
        self.inputs = inputs    # Dictionary of materials consumed
        self.outputs = outputs  # Dictionary of materials produced

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

def calculate_balance_matrix(recipes, final_demand, production_routes):
    """
    Calculates the material balance, using the production_routes dictionary
    to resolve cases where a material has multiple producers.
    """
    all_materials = set()
    # MODIFIED: A material can have a LIST of producers.
    producers_of = defaultdict(list)

    for recipe in recipes:
        all_materials.update(recipe.inputs.keys())
        all_materials.update(recipe.outputs.keys())
        for material in recipe.outputs:
            producers_of[material].append(recipe)

    demand_item = list(final_demand.keys())[0]
    if not producers_of[demand_item]:
        print(f"FATAL ERROR: The demanded product '{demand_item}' is not an output in any of your recipes.")
        return None

    required = defaultdict(float, final_demand)
    production_level = defaultdict(float)
    queue = deque(list(final_demand.keys()))
    
    while queue:
        material = queue.popleft()
        
        producers = producers_of.get(material)
        if not producers: continue

        # --- LOGIC TO SELECT THE CORRECT PRODUCER ---
        process = None
        if len(producers) == 1:
            # If there's only one producer, use it.
            process = producers[0]
        elif material in production_routes:
            # If a route is specified, find that specific process.
            chosen_process_name = production_routes[material]
            for p in producers:
                if p.name == chosen_process_name:
                    process = p
                    break
            if not process:
                print(f"Warning: Specified producer '{chosen_process_name}' for '{material}' not found. Using first available.")
                process = producers[0]
        else:
            # If no route is specified for a choice, warn the user and default to the first.
            print(f"Warning: Material '{material}' has multiple producers. Defaulting to first recipe: '{producers[0].name}'.")
            process = producers[0]
        # ---------------------------------------------

        amount_needed = required[material]
        if amount_needed <= 1e-9: continue

        produced_per_run = process.outputs.get(material, 0)
        if produced_per_run == 0: continue

        runs = amount_needed / produced_per_run
        production_level[process.name] += runs
        required[material] -= runs * produced_per_run

        for in_material, in_amount in process.inputs.items():
            required[in_material] += runs * in_amount
            if producers_of.get(in_material) and in_material not in queue:
                queue.append(in_material)

    external_inputs = defaultdict(float)
    for material, amount in required.items():
        if amount > 1e-9:
            external_inputs[material] = amount

    # --- Build the Final Report DataFrame ---
    sorted_materials = sorted(list(all_materials))
    matrix_data, row_names = [], []
    
    for recipe in recipes:
        level = production_level.get(recipe.name, 0)
        if level > 0:
            row = [(recipe.outputs.get(m, 0) - recipe.inputs.get(m, 0)) * level for m in sorted_materials]
            matrix_data.append(row)
            row_names.append(recipe.name)

    if not matrix_data: return None

    matrix_data.append([external_inputs.get(m, 0) for m in sorted_materials])
    row_names.append("External Inputs")
    matrix_data.append([-final_demand.get(m, 0) for m in sorted_materials])
    row_names.append("Final Demand")

    balance_df = pd.DataFrame(matrix_data, index=row_names, columns=sorted_materials)
    balance_df = balance_df.loc[:, (balance_df.abs() > 1e-9).any(axis=0)]
    balance_df = balance_df.loc[(balance_df.abs() > 1e-9).any(axis=1)]

    return balance_df

# ===================================================================
#                           MAIN EXECUTION
# ===================================================================
if __name__ == "__main__":
    param_filepath = os.path.join('data', 'parameters.yml')
    params = load_parameters(param_filepath)

    if params:
        recipe_filepath = os.path.join('data', 'recipes.yml')
        recipes = load_recipes_from_yaml(recipe_filepath, params)

    if params and recipes:
        # 1. Define your final goal
        final_demand = {"Pig Iron": 1000.0}
        
        # 2. DEFINE YOUR PRODUCTION ROUTES HERE
        #    Tell the model which recipe to use when a choice exists.
        production_routes = {
            "Pig Iron": "Blast Furnace",
            "Electricity": "Electricity from Market"
            # Add other choices here, e.g., "Liquid Steel": "Basic Oxygen Furnace"
        }
        
        # 3. Run the calculation
        balance_matrix = calculate_balance_matrix(recipes, final_demand, production_routes)
        
        # 4. Print the resulting matrix
        if balance_matrix is not None:
            print("\n--- Calculated Balance Matrix ---")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 120)
            print(balance_matrix.round(3))
        else:
            print("\nCould not generate the balance matrix. Please check error messages.")