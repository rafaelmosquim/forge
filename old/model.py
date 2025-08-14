# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 14:52:58 2025

@author: rafae
"""

import pandas as pd
from collections import defaultdict, deque

class Process:
    __slots__ = ('name', 'inputs', 'outputs')
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs    # Dict: material -> amount per unit run
        self.outputs = outputs  # Dict: material -> amount per unit run

def compute_unit_balance(processes, final_demand):
    # Identify all materials and their relationships
    all_materials = set()
    producer_of = {}
    
    for p in processes:
        all_materials.update(p.inputs.keys())
        all_materials.update(p.outputs.keys())
        for m in p.outputs:
            if m in producer_of:
                raise ValueError(f"Material '{m}' produced by multiple processes")
            producer_of[m] = p
    
    # Classify materials
    produced_materials = set(producer_of.keys())
    external_materials = all_materials - produced_materials
    consumed_materials = {m for p in processes for m in p.inputs}
    waste_materials = (produced_materials - consumed_materials) - set(final_demand.keys())
    
    # Initialize tracking structures
    required = defaultdict(float, final_demand)
    production_level = {p.name: 0.0 for p in processes}
    external_inputs = defaultdict(float)
    waste_outputs = defaultdict(float)
    
    # Backward propagation using topological sort (reverse BFS)
    queue = deque(m for m in final_demand if m in produced_materials)
    processed = set()
    
    while queue:
        material = queue.popleft()
        if material in processed or material in external_materials:
            continue
        processed.add(material)
        
        # Get producing process and required amount
        process = producer_of[material]
        amount_needed = required[material]
        if amount_needed <= 1e-10:
            continue
        
        # Calculate production runs
        runs = amount_needed / process.outputs[material]
        production_level[process.name] += runs
        
        # Process outputs
        for out_material, out_amount in process.outputs.items():
            total_out = runs * out_amount
            if out_material == material:
                required[out_material] -= total_out
            elif out_material in waste_materials:
                waste_outputs[out_material] += total_out
            else:
                required[out_material] -= total_out
                if out_material not in external_materials:
                    queue.append(out_material)
        
        # Process inputs
        for in_material, in_amount in process.inputs.items():
            total_in = runs * in_amount
            if in_material in external_materials:
                external_inputs[in_material] += total_in
            else:
                required[in_material] += total_in
                if in_material in produced_materials and in_material not in processed:
                    queue.append(in_material)
    
    # Build balance matrix
    sorted_materials = sorted(all_materials)
    matrix_data = []
    row_names = []
    
    # Process rows
    for p in processes:
        row = []
        for material in sorted_materials:
            input_val = p.inputs.get(material, 0.0) * production_level[p.name]
            output_val = p.outputs.get(material, 0.0) * production_level[p.name]
            row.append(output_val - input_val)
        matrix_data.append(row)
        row_names.append(p.name)
    
    # External inputs row
    external_row = [external_inputs.get(m, 0.0) for m in sorted_materials]
    matrix_data.append(external_row)
    row_names.append("External Inputs")
    
    # Final demand row
    final_demand_row = [-final_demand.get(m, 0.0) for m in sorted_materials]
    matrix_data.append(final_demand_row)
    row_names.append("Final Demand")
    
    # Waste outputs row
    waste_row = [-waste_outputs.get(m, 0.0) for m in sorted_materials]
    matrix_data.append(waste_row)
    row_names.append("Waste Outputs")
    
    balance_df = pd.DataFrame(matrix_data, index=row_names, columns=sorted_materials)
    
    return {
        'balance_matrix': balance_df,
        'production_levels': production_level,
        'external_inputs': dict(external_inputs),
        'waste_outputs': dict(waste_outputs)
    }

def compute_emissions(results, emission_factors, direct_emission_factors):
    emissions = {
        'indirect_emissions': defaultdict(float),
        'direct_emissions': defaultdict(float),
        'total': 0.0
    }
    
    # Calculate indirect emissions from external inputs
    for material, amount in results['external_inputs'].items():
        if material in emission_factors:
            e = amount * emission_factors[material]
            emissions['indirect_emissions'][material] = e
            emissions['total'] += e
    
    # Calculate direct emissions from processes
    for process_name, level in results['production_levels'].items():
        if process_name in direct_emission_factors:
            e = level * direct_emission_factors[process_name]
            emissions['direct_emissions'][process_name] = e
            emissions['total'] += e
    
    return emissions

# Example Usage
if __name__ == "__main__":
    # ===================================================================
    # 1. Define Material Composition Parameters
    # ===================================================================
    iron_content_pig_iron = 0.615  # 61.5% iron in pig iron
    feo3_sinter = 0.80  # 80% FeO3 in sinter
    feo3_pellet = 0.90  # 90% FeO3 in pellet
    feo3_lump = 0.879   # 87.9% FeO3 in lump ore

    # ===================================================================
    # 2. Define Blast Furnace Blend Proportions
    # ===================================================================
    blend_proportions = {
        'sinter': 0.80,  # 80%
        'pellet': 0.10,  # 10%
        'lump': 0.10     # 10%
    }

    # ===================================================================
    # 3. Calculate Material Requirements for Blast Furnace Blend
    # ===================================================================
    # Calculate weighted average FeO3 content of blend
    avg_feo3 = (
        blend_proportions['sinter'] * feo3_sinter +
        blend_proportions['pellet'] * feo3_pellet +
        blend_proportions['lump'] * feo3_lump
    )
    
    # Total blend required per kg pig iron
    blend_per_pig_iron = (1 / iron_content_pig_iron) * (feo3_lump / avg_feo3)
    
    # Calculate individual material requirements
    sinter_per_pig_iron = blend_per_pig_iron * blend_proportions['sinter']
    pellet_per_pig_iron = blend_per_pig_iron * blend_proportions['pellet']
    lump_per_pig_iron = blend_per_pig_iron * blend_proportions['lump']

    # ===================================================================
    # 4. Define Energy Parameters (MJ per kg output)
    # ===================================================================
    sintering_energy = 2.0  # MJ per kg sinter
    sintering_elec_share = 0.23
    sintering_coal_share = 0.77

    # ===================================================================
    # 5. Define Emission Factors
    # ===================================================================
    EMISSION_FACTORS = {
        'Limestone': 0.10,     # kg CO2 per kg limestone
        'Iron Ore': 0.05,      # kg CO2 per kg iron ore (mining)
        'Coal_MJ': 0.08,       # kg CO2 per MJ coal
        'Electricity_MJ': 0.12, # kg CO2 per MJ electricity
    }
    
    DIRECT_EMISSION_FACTORS = {
        'Sintering': 0.05,     # kg CO2 per kg sinter
        'Blast Furnace': 1.2,  # kg CO2 per kg pig iron
    }

    # ===================================================================
    # 6. Define Production Processes
    # ===================================================================
    processes = [
        Process("Sintering", 
                inputs={
                    "Iron Ore": feo3_sinter/feo3_lump,
                    "Limestone": 0.14,
                    "Electricity_MJ": sintering_energy * sintering_elec_share,
                    "Coal_MJ": sintering_energy * sintering_coal_share
                }, 
                outputs={"Sinter": 1.0}),
        
        Process("Pelletizing", 
                inputs={
                    "Iron Ore": feo3_pellet/feo3_lump,
                    "Electricity_MJ": 0.8  # Example value
                }, 
                outputs={"Pellet": 1.0}),
        
        Process("Blast Furnace", 
                inputs={
                    "Sinter": sinter_per_pig_iron,
                    "Pellet": pellet_per_pig_iron,
                    "Lump Ore": lump_per_pig_iron,
                    "Coke_MJ": 10.5  # Example value
                }, 
                outputs={"Pig Iron": 1.0}),
        
        Process("Limestone from market",
                inputs={},
                outputs={"Limestone": 1.0}
                )
    ]
    
    # ===================================================================
    # 7. Set Final Demand
    # ===================================================================
    final_demand = {"Pig Iron": 1000.0}  # 1000 kg of Pig Iron
    
    # ===================================================================
    # 8. Compute Material Balance
    # ===================================================================
    results = compute_unit_balance(processes, final_demand)
    balance_df = results['balance_matrix']
    
    # ===================================================================
    # 9. Compute Emissions
    # ===================================================================
    emissions = compute_emissions(results, EMISSION_FACTORS, DIRECT_EMISSION_FACTORS)
    
    # ===================================================================
    # 10. Print Results
    # ===================================================================
    print("Material Balance (kg):")
    print(balance_df)
    
    print("\nProduction Levels:")
    for process, level in results['production_levels'].items():
        print(f"{process}: {level:.2f} kg")
    
    print("\nExternal Inputs:")
    for material, amount in results['external_inputs'].items():
        print(f"{material}: {amount:.2f} {'kg' if '_MJ' not in material else 'MJ'}")
    
    print("\nBlast Furnace Blend Composition:")
    total_blend = (sinter_per_pig_iron + pellet_per_pig_iron + lump_per_pig_iron) * 1000
    print(f"Sinter: {sinter_per_pig_iron * 1000:.2f} kg ({sinter_per_pig_iron/blend_per_pig_iron*100:.1f}%)")
    print(f"Pellet: {pellet_per_pig_iron * 1000:.2f} kg ({pellet_per_pig_iron/blend_per_pig_iron*100:.1f}%)")
    print(f"Lump Ore: {lump_per_pig_iron * 1000:.2f} kg ({lump_per_pig_iron/blend_per_pig_iron*100:.1f}%)")
    print(f"Total Blend: {total_blend:.2f} kg per tonne pig iron")
    
    print("\nCO2 Emissions Breakdown:")
    print(f"{'Source':<20} | {'Emission (kg CO2)':>15}")
    print("-" * 40)
    for material, amount in emissions['indirect_emissions'].items():
        print(f"{'Indirect: ' + material:<20} | {amount:>15.2f}")
    for process, amount in emissions['direct_emissions'].items():
        print(f"{'Direct: ' + process:<20} | {amount:>15.2f}")
    print("-" * 40)
    print(f"{'TOTAL CO2 EMISSIONS':<20} | {emissions['total']:>15.2f} kg")