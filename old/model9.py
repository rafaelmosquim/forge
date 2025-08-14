# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 14:52:28 2025

@author: rafae
"""

# -*- coding: utf-8 -*-
"""
Final script to calculate a material balance, energy balance, and emissions,
including internal utility-plant electricity credit.
"""
import os
import yaml
import json
import pandas as pd
from collections import defaultdict, deque
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
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            params_dict = yaml.safe_load(f)
            print("DEBUG: Raw parameters from YAML:", params_dict)
            params = json.loads(json.dumps(params_dict), 
                        object_hook=lambda d: SimpleNamespace(**d))
            print(f"DEBUG: process_gas value: {getattr(params, 'process_gas', 'NOT FOUND')}")
            return params
    except FileNotFoundError:
        print(f"FATAL ERROR: Parameters file not found: {filepath}")
        return None


def load_data_from_yaml(filepath, default_value=0):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: Data file not found: {filepath}, using empty.")
        return {}
    if isinstance(data, dict) and len(data) == 1:
        data = next(iter(data.values())) or {}
    cleaned = {}
    for k, v in data.items():
        key = str(k).strip()
        if v is None:
            cleaned[key] = default_value
        elif isinstance(v, str):
            try:
                cleaned[key] = float(v) if '.' in v else int(v)
            except ValueError:
                cleaned[key] = v
        else:
            cleaned[key] = v
    return cleaned


def load_recipes_from_yaml(filepath, params):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            recipe_data = yaml.safe_load(f) or []
    except FileNotFoundError:
        print(f"FATAL ERROR: Recipe file not found: {filepath}")
        return None
    # Load energy_content for formulas
    ec_path = os.path.join(os.path.dirname(filepath), 'energy_content.yml')
    energy_content = load_data_from_yaml(ec_path)
    recipes = []
    context = {**vars(params), 'energy_content': energy_content}
    for item in recipe_data:
        name = item.get('process','').strip()
        if not name: continue
        inputs, outputs = {}, {}
        for mat, formula in item.get('inputs',{}).items():
            try:
                inputs[mat] = eval(formula, context) if isinstance(formula,str) else formula
            except Exception:
                inputs[mat] = 0
        for mat, formula in item.get('outputs',{}).items():
            try:
                outputs[mat] = eval(formula, context) if isinstance(formula,str) else formula
            except Exception:
                outputs[mat] = 0
        recipes.append(Process(name, inputs, outputs))
    return recipes


def load_market_config(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or []
        return {i['name'].strip(): i['value'] for i in cfg}
    except Exception as e:
        print(f"FATAL ERROR loading market config: {e}")
        return None

# ===================================================================
#                       Calculation Functions
# ===================================================================

def adjust_blast_furnace_intensity(energy_int, energy_shares, params):
    """
    Scales the BF intensity and saves both the original and adjusted values
    so top-gas = base_intensity – adjusted_intensity can be harvested.
    """
    try:
        pg = params.process_gas
    except AttributeError:
        pg = 0.0

    if 'Blast Furnace' not in energy_int:
        return

    # 1) Remember the pre-adjusted intensity
    base = energy_int['Blast Furnace']
    params.bf_base_intensity = base

    # 2) Compute the adjustment denominator
    shares = energy_shares.get('Blast Furnace', {})
    carriers = ['Gas', 'Coal', 'Coke', 'Charcoal']
    S = sum(shares.get(c, 0) for c in carriers)
    denom = 1 - pg * S

    # 3) Apply adjustment and save it
    adj = base / denom
    energy_int['Blast Furnace'] = adj
    params.bf_adj_intensity = adj

    print(f"Adjusted BF intensity: {base:.2f} → {adj:.2f} MJ/t steel (recovering {pg*100:.1f}% of carriers)")


def calculate_balance_matrix(recipes, final_demand, production_routes):
    all_m, producers = set(), defaultdict(list)
    recipes_dict = {r.name:r for r in recipes}
    for r in recipes:
        all_m |= set(r.inputs)|set(r.outputs)
        for m in r.outputs: producers[m].append(r)
    demand = list(final_demand.keys())[0]
    if demand not in producers: return None, None
    required = defaultdict(float, final_demand)
    prod_level = defaultdict(float)
    queue = deque([demand])
    seen = {demand}
    while queue:
        mat = queue.popleft(); seen.remove(mat)
        amt = required[mat]
        if amt<=1e-9: continue
        procs = [p for p in producers[mat] if production_routes.get(p.name,0)>0]
        if not procs: continue
        for p in procs:
            share = production_routes[p.name]
            out_amt = p.outputs.get(mat,0)
            if out_amt<=0: continue
            runs = amt*share/out_amt
            prod_level[p.name]+=runs
            for im, ia in p.inputs.items():
                required[im]+=runs*ia
                if im in producers and im not in seen:
                    queue.append(im); seen.add(im)
        required[mat]=0
    ext = {m:amt for m,amt in required.items() if amt>1e-9 and m not in producers}
    mats = sorted(all_m)
    data, rows = [], []
    for nm, lvl in prod_level.items():
        if lvl>0:
            rec = recipes_dict[nm]
            row = [ (rec.outputs.get(m,0)-rec.inputs.get(m,0))*lvl for m in mats ]
            data.append(row); rows.append(nm)
    # external
    data.append([ext.get(m,0) for m in mats]); rows.append('External Inputs')
    data.append([-final_demand.get(m,0) for m in mats]); rows.append('Final Demand')
    df = pd.DataFrame(data, index=rows, columns=mats)
    return df.loc[:,(df.abs()>1e-9).any()], prod_level


def calculate_energy_balance(production_level, energy_int, energy_shares):
    ei = pd.Series(energy_int).fillna(0)
    es = pd.DataFrame.from_dict(energy_shares, orient='index').fillna(0)
    df = es.multiply(ei, axis='index')
    data, procs = [], []
    for p, runs in production_level.items():
        if p in df.index and runs>0:
            data.append(runs*df.loc[p]); procs.append(p)
    bal = pd.DataFrame(data, index=procs, columns=df.columns)
    bal.loc['TOTAL'] = bal.sum()
    return bal


def adjust_energy_balance(energy_df, internal_elec):
    # subtract internal from grid draw
    energy_df.loc['TOTAL','Electricity'] -= internal_elec
    # add utility plant output
    energy_df.loc['Utility Plant'] = 0
    energy_df.loc['Utility Plant','Electricity'] = -internal_elec
    return energy_df


def calculate_internal_electricity(prod_level, recipes_dict, params):
    """
    Harvests all Process Gas outputs AND the top-gas from the BF adjustment,
    then converts the total MJ of gas into kWh via the Utility Plant recipe.
    """
    internal_elec = 0.0
    util_eff = recipes_dict['Utility Plant'].outputs.get('Electricity', 0)

    # --- 1) Any recipe that declares Process Gas in outputs ---
    for proc, runs in prod_level.items():
        recipe = recipes_dict.get(proc)
        if not recipe:
            continue
        gas_per_run = recipe.outputs.get('Process Gas', 0)
        if gas_per_run:
            internal_elec += runs * gas_per_run * util_eff

    # --- 2) Top-gas from Blast Furnace adjustment ---
    bf_runs = prod_level.get('Blast Furnace', 0.0)
    if bf_runs > 0 and hasattr(params, 'bf_base_intensity') and hasattr(params, 'bf_adj_intensity'):
        # MJ of gas = (base_intensity – adjusted_intensity) * runs
        gas_mj = (params.bf_base_intensity - params.bf_adj_intensity) * bf_runs
        internal_elec += gas_mj * util_eff

    return internal_elec


def calculate_emissions(
    mkt_cfg,
    prod_level,
    energy_df,
    energy_efs,
    process_efs,
    internal_elec,
    final_demand,
    total_gas_MJ,
    EF_process_gas
):
    """
    Calculates total emissions, splitting electricity between grid and internal,
    burning recovered process gas in‐plant at EF_process_gas/util_eff.
    """

    # 1) Compute grid vs. internal shares
    if energy_df is not None and 'Electricity' in energy_df.columns:
        total_process_elec = abs(energy_df.loc['TOTAL', 'Electricity'] + internal_elec)
        if total_process_elec > 1e-6:
            internal_share = min(internal_elec / total_process_elec, 1.0)
            grid_share     = 1.0 - internal_share
        else:
            internal_share, grid_share = 0.0, 1.0
    else:
        internal_share, grid_share = 0.0, 1.0
        # — DEBUG —
    print(f"[DBG] total_gas_MJ       = {total_gas_MJ:.2f} MJ")
    print(f"[DBG] EF_process_gas     = {EF_process_gas:.2f} gCO2/MJ_gas")
    if total_gas_MJ > 1e-9:
        util_eff      = internal_elec / total_gas_MJ
        ef_internal_e = EF_process_gas / util_eff
        print(f"[DBG] util_efficiency    = {util_eff:.3f}")
        print(f"[DBG] ef_internal_e      = {ef_internal_e:.2f} gCO2/MJ_elec")
    else:
        ef_internal_e = 0.0
        print("[DBG] No gas recovered, ef_internal_e = 0")

    ef_grid = energy_efs.get('Electricity', 0)
    print(f"[DBG] ef_grid            = {ef_grid:.2f} gCO2/MJ_elec")
    print(f"[DBG] internal_share     = {internal_share:.1%}")
    print(f"[DBG] grid_share         = {grid_share:.1%}")
    print("— end DBG —\n")

    # 2) Compute the internal‐electricity EF (gCO2 per MJ_elec)
    if total_gas_MJ > 1e-9:
        util_eff        = internal_elec / total_gas_MJ
        ef_internal_e   = EF_process_gas / util_eff
    else:
        ef_internal_e   = 0.0

    ef_grid = energy_efs.get('Electricity', 0)

    # 3) Build per‐process emissions rows
    rows = []
    for proc_name, runs in prod_level.items():
        if runs <= 1e-9:
            continue

        row = {'Process': proc_name, 'Energy Emissions': 0.0, 'Direct Emissions': 0.0}

        # a) Energy‐related emissions
        if energy_df is not None and proc_name in energy_df.index:
            for carrier, cons in energy_df.loc[proc_name].items():
                if carrier == 'Electricity':
                    # split between internal and grid EFs
                    row['Energy Emissions'] += cons * (
                        internal_share * ef_internal_e
                        + grid_share * ef_grid
                    )
                else:
                    # all other carriers at their own EF
                    row['Energy Emissions'] += cons * energy_efs.get(carrier, 0)

        # b) Direct process emissions
        row['Direct Emissions'] += runs * process_efs.get(proc_name, 0)

        rows.append(row)

    if not rows:
        return None

    # 4) Assemble DataFrame and total
    emissions_df = pd.DataFrame(rows).set_index('Process') / 1000.0
    emissions_df['TOTAL CO2e'] = (
        emissions_df['Energy Emissions'] + emissions_df['Direct Emissions']
    )
    emissions_df.loc['TOTAL'] = emissions_df.sum()

    return emissions_df


# ===================================================================
#                           MAIN EXECUTION
# ===================================================================
if __name__ == '__main__':
    base = os.path.join('data', '')
    # 1) Load configurations
    params      = load_parameters(os.path.join(base, 'parameters.yml'))
    recipes     = load_recipes_from_yaml(os.path.join(base, 'recipes.yml'), params)
    route_cfg   = load_data_from_yaml(os.path.join(base, 'route_config.yml'))
    mkt_cfg     = load_market_config(os.path.join(base, 'mkt_config.yml'))
    e_efs       = load_data_from_yaml(os.path.join(base, 'emission_factors.yml'))
    p_efs       = load_data_from_yaml(os.path.join(base, 'process_emissions.yml'))
    e_int       = load_data_from_yaml(os.path.join(base, 'energy_int.yml'))
    e_sh        = load_data_from_yaml(os.path.join(base, 'energy_matrix.yml'))


    # 2) Adjust BF intensity
    if params and e_int and e_sh:
        adjust_blast_furnace_intensity(e_int, e_sh, params)

    # Build recipe lookup
    recipes_dict = {r.name: r for r in recipes} if recipes else {}

    # 3) Check that all required configs are present
    if not all((params, recipes, route_cfg, mkt_cfg)):
        print("Model execution failed: missing configuration.")
    else:
        # 4) Material balance
        final_demand     = {'Finished Steel': 1000.0}
        balance_matrix, prod_lvl = calculate_balance_matrix(recipes, final_demand, route_cfg)
        print("\n>>> Production levels (runs > 0):")
        for proc, runs in prod_lvl.items():
            if runs > 1e-9:
                print(f"  {proc:<20s} → {runs:.3f}")
        if balance_matrix is None:
            print("Material balance failed")
        else:
            # 5) Initial internal electricity (from any Process Gas recipes)
            internal_elec = calculate_internal_electricity(prod_lvl, recipes_dict, params)

            # 6) Energy balance
            energy_balance = calculate_energy_balance(prod_lvl, e_int, e_sh)


            # 7) Repair BF row back to base intensity
            if 'Blast Furnace' in energy_balance.index and hasattr(params, 'bf_base_intensity'):
                bf_runs = prod_lvl.get('Blast Furnace', 0.0)
                base    = params.bf_base_intensity
                shares  = e_sh.get('Blast Furnace', {})
                for carrier in energy_balance.columns:
                    energy_balance.loc['Blast Furnace', carrier] = bf_runs * base * shares.get(carrier, 0.0)

            # 8) Inject utility‐plant credit
            energy_balance = adjust_energy_balance(energy_balance, internal_elec)

            # 9) Compute recovered‐gas volumes & EF, then recompute internal_elec
            gas_coke_MJ  = prod_lvl.get('Coke Production', 0) * recipes_dict['Coke Production'].outputs.get('Process Gas', 0)
            gas_bf_MJ = (params.bf_adj_intensity - params.bf_base_intensity) \
             * prod_lvl.get('Blast Furnace', 0)
            total_gas_MJ = gas_coke_MJ + gas_bf_MJ

            EF_coal       = e_efs.get('Coal', 0)
            avoided_coke_CO2 = gas_coke_MJ * EF_coal

            bf_shares     = e_sh.get('Blast Furnace', {})
            relevant      = ['Coal', 'Coke']
            S             = sum(bf_shares.get(c, 0) for c in relevant)
            EF_bf_gas     = (sum(bf_shares.get(c, 0) * e_efs.get(c, 0) for c in relevant) / S) if S else 0
            avoided_bf_CO2   = gas_bf_MJ * EF_bf_gas

            EF_process_gas  = ((avoided_coke_CO2 + avoided_bf_CO2) / total_gas_MJ) if total_gas_MJ else 0

            util_eff       = recipes_dict['Utility Plant'].outputs.get('Electricity', 0)
            internal_elec  = total_gas_MJ * util_eff

            # 10) Final emissions (now 9-arg call)
            emissions = calculate_emissions(
                mkt_cfg,
                prod_lvl,
                energy_balance,
                e_efs,
                p_efs,
                internal_elec,
                final_demand,
                total_gas_MJ,
                EF_process_gas
            )
            if emissions is not None:
                print("\n--- Emissions (kg CO₂e) ---")
                print(emissions.round(3))
                total = emissions.loc['TOTAL', 'TOTAL CO2e']
                print(f"\nTotal CO₂e for {final_demand['Finished Steel']} units: {total:.2f} kg")

