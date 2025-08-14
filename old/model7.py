# -*- coding: utf-8 -*-
"""
Final script to calculate a material balance matrix.
This version uses an explicit market configuration file (mkt_config.yml)
to determine the correct emissions calculation logic for each process.

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
            
            # Debug print to see raw loaded data
            print("\nDEBUG: Raw parameters from YAML:")
            print(params_dict)
            
            # Convert to namespace object
            params = json.loads(json.dumps(params_dict), 
                        object_hook=lambda d: SimpleNamespace(**d))
            
            # Debug print to verify process_gas
            print(f"\nDEBUG: process_gas value: {getattr(params, 'process_gas', 'NOT FOUND')}")
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
        
    # Load energy content data for formula evaluation
    energy_content_path = os.path.join(os.path.dirname(filepath), 'energy_content.yml')
    energy_content_data = load_data_from_yaml(energy_content_path, {})
    
    recipes = []
    param_vars = vars(params)
    context = {
        **param_vars,
        'energy_content': SimpleNamespace(**energy_content_data)
    }
    
    for item in recipe_data:
        process_name = item.get('process', '').strip()
        if not process_name: continue

        # Process inputs
        inputs = {}
        for material, formula in item.get('inputs', {}).items():
            material = material.strip()
            try:
                if isinstance(formula, str):
                    inputs[material] = eval(formula, context)
                else:
                    inputs[material] = formula
            except Exception as e:
                print(f"Error evaluating input '{material}' in {process_name}: {e}")
                inputs[material] = 0

        # Process outputs
        outputs = {}
        for material, formula in item.get('outputs', {}).items():
            material = material.strip()
            try:
                if isinstance(formula, str):
                    outputs[material] = eval(formula, context)
                else:
                    outputs[material] = formula
            except Exception as e:
                print(f"Error evaluating output '{material}' in {process_name}: {e}")
                outputs[material] = 0
                
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
    except KeyError:
        print(f"FATAL ERROR: The file '{filepath}' has an incorrect format. Expected 'name' and 'value' keys.")
        return None

def load_data_from_yaml(filepath, default_value=0):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: The data file '{filepath}' was not found. Using empty data.")
        return {}
    
    if isinstance(data, dict) and len(data) == 1:
        data = next(iter(data.values())) or {}
    
    cleaned_data = {}
    for k, v in data.items():
        key = str(k).strip()
        if v is None:
            cleaned_data[key] = default_value
        elif isinstance(v, str):
            try:
                # Try converting string numbers
                cleaned_data[key] = float(v) if '.' in v else int(v)
            except ValueError:
                cleaned_data[key] = v
        else:
            cleaned_data[key] = v
            
    return cleaned_data
# ===================================================================
#                       Calculation Functions
# ===================================================================

def adjust_blast_furnace_intensity(energy_int_data, energy_shares_data, params):
    """Adjusts blast furnace energy intensity"""
    try:
        # Get process_gas - now with proper error handling
        process_gas = params.process_gas  # Direct attribute access
    except AttributeError:
        print("ERROR: 'process_gas' not found in parameters")
        process_gas = 0  # Default value
    
    print(f"\nDEBUG: Using process_gas = {process_gas}")
    
    # Rest of the function remains the same...
    if "Blast Furnace" not in energy_int_data:
        return
    
    base_intensity = energy_int_data["Blast Furnace"]
    
    bf_energy_shares = energy_shares_data.get("Blast Furnace", {})
    relevant_carriers = ['Gas', 'Coal', 'Coke', 'Charcoal']
    carrier_sum = sum(bf_energy_shares.get(c, 0) for c in relevant_carriers)
    
    denominator = 1 + (process_gas * carrier_sum)
    if abs(denominator) > 1e-6:
        energy_int_data["Blast Furnace"] = base_intensity / denominator
        print(f"Adjusted Blast Furnace intensity from {base_intensity:.2f} to {energy_int_data['Blast Furnace']:.2f} MJ")

def calculate_balance_matrix(recipes, final_demand, production_routes):
    # This function remains unchanged
    all_materials, producers_of = set(), defaultdict(list)
    recipes_dict = {r.name: r for r in recipes}
    for r in recipes:
        all_materials.update(r.inputs.keys())
        all_materials.update(r.outputs.keys())
        for m in r.outputs: producers_of[m].append(r)
            
    demand_item = list(final_demand.keys())[0]
    if not producers_of.get(demand_item): return None, None
        
    required, production_level = defaultdict(float, final_demand), defaultdict(float)
    queue, processed_in_queue = deque(list(final_demand.keys())), set(list(final_demand.keys()))
    
    while queue:
        material = queue.popleft()
        processed_in_queue.remove(material)
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
                if producers_of.get(in_mat) and in_mat not in processed_in_queue:
                    queue.append(in_mat)
                    processed_in_queue.add(in_mat)
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
            
    if not matrix_data: return None, None
    
    matrix_data.append([external_inputs.get(m, 0) for m in sorted_materials])
    row_names.append("External Inputs")
    matrix_data.append([-final_demand.get(m, 0) for m in sorted_materials])
    row_names.append("Final Demand")
    
    balance_df = pd.DataFrame(matrix_data, index=row_names, columns=sorted_materials)
    return balance_df.loc[:, (balance_df.abs() > 1e-9).any(axis=0)], production_level


def calculate_energy_balance(production_level, energy_int_data, energy_shares_data):
    # This function remains unchanged
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

def calculate_emissions(mkt_config, production_level, energy_balance_df, 
                       energy_emission_factors, process_emission_factors,
                       internal_electricity, final_demand):
    """
    Calculates total emissions with special handling for electricity from grid vs internal production.
    """
    if production_level is None: 
        return None
    
    # Calculate electricity shares
    total_electricity_demand = abs(final_demand.get("Electricity", 0))
    if total_electricity_demand > 1e-6:
        internal_share = min(internal_electricity / total_electricity_demand, 1.0)
        grid_share = 1 - internal_share
    else:
        internal_share = 0
        grid_share = 1
    
    emissions_data = []
    for process_name, level in production_level.items():
        if level <= 1e-9: 
            continue

        row = {'Process': process_name, 'Energy Emissions': 0, 'Direct Emissions': 0}
        process_type = mkt_config.get(process_name)

        # --- Special handling for electricity from market ---
        if process_name == "Electricity from Market":
            factor = process_emission_factors.get(process_name, 0)
            row['Direct Emissions'] = level * factor * grid_share
            emissions_data.append(row)
            continue
        
        # --- Normal process handling ---
        if process_type == 1: # Type 1: "From Market" process
            factor = process_emission_factors.get(process_name, 0)
            row['Direct Emissions'] = level * factor
        elif process_type == 2: # Type 2: Internal manufacturing process
            if energy_balance_df is not None and process_name in energy_balance_df.index:
                energy_row = energy_balance_df.loc[process_name]
                for carrier, consumption in energy_row.items():
                    factor = energy_emission_factors.get(carrier, 0)
                    row['Energy Emissions'] += consumption * factor
            
            # Add any non-energy direct process emissions
            factor = process_emission_factors.get(process_name, 0)
            row['Direct Emissions'] = level * factor
        
        emissions_data.append(row)

    if not emissions_data: 
        return None

    emissions_df = pd.DataFrame(emissions_data).set_index('Process')
    # Convert from g to kg
    emissions_df = emissions_df / 1000.0 
    
    emissions_df['TOTAL CO2e'] = emissions_df['Energy Emissions'] + emissions_df['Direct Emissions']
    emissions_df.loc['TOTAL'] = emissions_df.sum()
    
    # Add internal electricity credit
    if internal_share > 0:
        emissions_df.loc['Internal Electricity Credit'] = {
            'Energy Emissions': 0,
            'Direct Emissions': -internal_electricity * process_emission_factors.get("Electricity from Market", 0) / 1000.0,
            'TOTAL CO2e': -internal_electricity * process_emission_factors.get("Electricity from Market", 0) / 1000.0
        }
        emissions_df.loc['TOTAL'] = emissions_df.sum()
    
    return emissions_df

def calculate_internal_electricity(production_level, recipes_dict, params):
    """Calculate electricity from by-product gases"""
    internal_electricity = 0
    
    # Find processes that produce process gas
    for process_name, runs in production_level.items():
        if process_name not in recipes_dict:
            continue
            
        recipe = recipes_dict[process_name]
        if "Process Gas" in recipe.outputs:
            # Calculate gas production
            gas_produced = runs * recipe.outputs["Process Gas"]
            
            # Calculate electricity from gas (34% efficiency)
            internal_electricity += gas_produced * 0.34
    
    return internal_electricity

# Add this function to adjust electricity demand
def adjust_electricity_demand(balance_matrix, internal_electricity):
    """Adjust electricity demand based on internal production"""
    if "Electricity" not in balance_matrix.columns:
        return balance_matrix
        
    total_demand = balance_matrix.loc["Final Demand", "Electricity"]
    
    if abs(total_demand) > 1e-6:
        # Calculate new grid electricity needed
        grid_electricity = abs(total_demand) - internal_electricity
        if grid_electricity < 0:
            grid_electricity = 0
            
        # Adjust final demand
        balance_matrix.loc["Final Demand", "Electricity"] = -grid_electricity
        
        # Add internal production as positive entry
        balance_matrix.loc["Internal Production", "Electricity"] = internal_electricity
        
    return balance_matrix

# ===================================================================
#                           MAIN EXECUTION
# ===================================================================
if __name__ == "__main__":
    # --- Load all configurations ---
    params = load_parameters(os.path.join('data', 'parameters.yml'))
    recipes = load_recipes_from_yaml(os.path.join('data', 'recipes.yml'), params)
    production_routes = load_data_from_yaml(os.path.join('data', 'route_config.yml'))
    mkt_config = load_market_config(os.path.join('data', 'mkt_config.yml'))
    energy_emission_factors = load_data_from_yaml(os.path.join('data', 'emission_factors.yml'))
    process_emission_factors = load_data_from_yaml(os.path.join('data', 'process_emissions.yml'))
    energy_int_data = load_data_from_yaml(os.path.join('data', 'energy_int.yml'))
    energy_shares_data = load_data_from_yaml(os.path.join('data', 'energy_matrix.yml'))
    energy_content_data = load_data_from_yaml(os.path.join('data', 'energy_content.yml'))
    
    # --- Apply Blast Furnace Intensity Adjustment ---
    if all([energy_int_data, energy_shares_data, params]):
        adjust_blast_furnace_intensity(energy_int_data, energy_shares_data, params)

    # --- Run Calculations ---


    if all((params, recipes, production_routes, mkt_config)):
        final_demand = {"Finished Steel": 1000.0}
        balance_matrix, production_level = calculate_balance_matrix(recipes, final_demand, production_routes)
        
      
        # In main execution after calculating balance matrix:
        if balance_matrix is not None:
            # Calculate internal electricity from by-product gases
            recipes_dict = {r.name: r for r in recipes}
            internal_electricity = calculate_internal_electricity(production_level, recipes_dict, params)
            
            # Adjust electricity demand in balance matrix
            balance_matrix = adjust_electricity_demand(balance_matrix, internal_electricity)
            
            # Recalculate energy balance with adjusted electricity
            energy_balance = calculate_energy_balance(production_level, energy_int_data, energy_shares_data)
      
           # Calculate emissions with internal electricity handling
            emissions_table = calculate_emissions(
                mkt_config,
                production_level,
                energy_balance, 
                energy_emission_factors, 
                process_emission_factors,
                internal_electricity,
                final_demand
            )
            if production_level:
                energy_balance = calculate_energy_balance(production_level, energy_int_data, energy_shares_data)
                if energy_balance is not None:
                    #print("\n--- Energy Balance (MJ) ---")
                    #print(energy_balance.round(3))
                    
                    emissions_table = calculate_emissions(
                        mkt_config,
                        production_level,
                        energy_balance, 
                        energy_emission_factors, 
                        process_emission_factors
                    )
                    if emissions_table is not None:
                        #print("\n--- Emissions (kg CO2e) ---")
                        #print(emissions_table.round(3))
                        
                        product_name = list(final_demand.keys())[0]
                        total_emissions = emissions_table.loc['TOTAL', 'TOTAL CO2e']
                        print("\n-----------------------------------------------------------------------")
                        print(f"Total emission for {final_demand[product_name]} units of {product_name} is {total_emissions:.2f} kg CO2e")
                        print("-----------------------------------------------------------------------")
    else:
        print("\nModel execution failed. A required configuration file may be missing or have an incorrect format.")