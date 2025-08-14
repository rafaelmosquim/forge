# -*- coding: utf-8 -*-
"""
Final script to calculate a material balance, energy balance, and emissions,
including internal utility-plant electricity credit.
"""
import os
import argparse, pathlib
import yaml
import json
import pandas as pd
from collections import defaultdict, deque
from types import SimpleNamespace
import math
import plotly.graph_objects as go
from pathlib import Path

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


def load_data_from_yaml(filepath, default_value=0, unwrap_single_key=True):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: Data file not found: {filepath}, using empty.")
        return {}

    # only unwrap when requested (for legacy files like {"processes": {...}})
    if unwrap_single_key and isinstance(data, dict) and len(data) == 1:
        data = next(iter(data.values())) or {}

    cleaned = {}
    if not isinstance(data, dict):
        return data  # allow non-dict YAML (e.g., plain list/string) when needed

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


def load_recipes_from_yaml(filepath, params, energy_int, energy_shares, energy_content):
    recipe_data = yaml.safe_load(open(filepath, 'r', encoding='utf-8')) or []
    
        
    # Fixed context keys: energy_int, energy_shares, energy_content
    context = {**vars(params),
               'energy_int': energy_int,
               'energy_shares': energy_shares,
               'energy_content': energy_content}
    
    recipes = []

    for item in recipe_data:
        name = item.get('process','').strip()
        if not name: continue
        inputs, outputs = {}, {}
        
        for mat, formula in item.get('inputs',{}).items():
            if isinstance(formula, str):
                try:
                    inputs[mat] = eval(formula, context)
                except Exception as e:
                    print(f"Error in {name} input {mat}: {e}")
                    inputs[mat] = 0
            else:
                inputs[mat] = formula
                
        for mat, formula in item.get('outputs',{}).items():
            if isinstance(formula, str):
                out_ctx = {**context, 'inputs': inputs}
                try:
                    outputs[mat] = eval(formula, out_ctx)
                except Exception as e:
                    print(f"Error in {name} output {mat}: {e}")
                    outputs[mat] = 0
            else:
                outputs[mat] = formula
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
    
#
def make_mass_sankey(prod_lvl, recipes_dict, min_flow=0.5, title="Mass Flow Sankey"):
    """
    Build a 3-layer Sankey (Material_in → Process → Material_out)
    using your existing prod_lvl and recipes_dict.
    Units: whatever your recipes use (e.g., kg per run) times runs.
    min_flow filters tiny links.
    """
    # Collect flows for processes that actually ran
    mats_in, mats_out, procs = set(), set(), set()
    links = []  # (source_label, target_label, value)

    for p, runs in prod_lvl.items():
        if runs <= 1e-12 or p not in recipes_dict:
            continue
        proc = recipes_dict[p]
        procs.add(p)

        # inputs: Material_in -> Process
        for m, amt in proc.inputs.items():
            val = runs * float(amt)
            if val >= min_flow:
                mats_in.add(m)
                links.append((f"[IN] {m}", f"[P] {p}", val))

        # outputs: Process -> Material_out
        for m, amt in proc.outputs.items():
            val = runs * float(amt)
            if val >= min_flow:
                mats_out.add(m)
                links.append((f"[P] {p}", f"[OUT] {m}", val))

    # Stable ordering: materials in (left), processes (middle), materials out (right)
    mat_in_labels  = sorted(f"[IN] {m}"  for m in mats_in)
    proc_labels    = sorted(f"[P] {p}"   for p in procs)
    mat_out_labels = sorted(f"[OUT] {m}" for m in mats_out)

    labels = mat_in_labels + proc_labels + mat_out_labels
    index  = {lab:i for i,lab in enumerate(labels)}

    # Build sankey sources/targets/values
    sources, targets, values, link_labels = [], [], [], []
    for s, t, v in links:
        sources.append(index[s])
        targets.append(index[t])
        values.append(v)
        link_labels.append(f"{s} → {t}: {v:,.3f}")

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels,
            pad=15,
            thickness=18
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels
        )
    )])
    fig.update_layout(title_text=title, font_size=12)
    return fig


def make_energy_sankey(energy_balance_df, min_MJ=10.0, title="Energy Flow Sankey"):
    """
    Build a 2-layer Sankey (Carrier → Process) from your energy_balance DF.
      - Expects rows as process names (exclude TOTAL),
      - Columns as carriers (Electricity, Gas, Coal, Coke, ...),
      - Values in MJ.
    min_MJ filters tiny links.
    """
    # Remove non-process rows if present
    df = energy_balance_df.copy()
    df = df.drop(index=[r for r in ["TOTAL"] if r in df.index], errors="ignore")

    # Collect carriers and processes actually used
    carriers = []
    for c in df.columns:
        if (df[c].abs() > min_MJ).any():
            carriers.append(c)

    procs = [p for p in df.index if (df.loc[p, carriers].abs() > min_MJ).any()]

    # Build nodes: carriers first (left), then processes (right)
    carrier_labels = [f"[E] {c}" for c in carriers]
    proc_labels    = [f"[P] {p}" for p in procs]
    labels = carrier_labels + proc_labels
    index  = {lab:i for i,lab in enumerate(labels)}

    sources, targets, values, link_labels = [], [], [], []

    for p in procs:
        for c in carriers:
            val = float(df.at[p, c])
            if abs(val) >= min_MJ:
                # Some rows can have negative values (e.g., Utility Plant exports electricity);
                # we only want *consumption* arrows into processes.
                if val > 0:
                    s = f"[E] {c}"
                    t = f"[P] {p}"
                    sources.append(index[s])
                    targets.append(index[t])
                    values.append(val)
                    link_labels.append(f"{c} → {p}: {val:,.1f} MJ")

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels,
            pad=15,
            thickness=18
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels
        )
    )])
    fig.update_layout(title_text=title, font_size=12)
    return fig   

def make_energy_to_process_sankey(energy_balance_df,
                                  emissions_df=None,
                                  title="Energy → Processes (ranked)",
                                  min_MJ=10.0,
                                  sort_by="emissions",   # "emissions" or "energy"
                                  exclude_carriers=("TOTAL", "Utility Plant")):
    """
    Simple Sankey: energy carriers (MJ) → processes.
    - Ranks processes by TOTAL CO2e if emissions_df provided, else by total MJ.
    - Hides tiny links < min_MJ.
    """
    import plotly.graph_objects as go
    import pandas as pd

    eb = energy_balance_df.copy()

    # carriers (left)
    carriers = [c for c in eb.columns if c not in exclude_carriers]

    # processes (rows), drop totals/utility rows if present
    procs = [r for r in eb.index if r not in ("TOTAL", "Utility Plant")]

    # ranking order
    if sort_by == "emissions" and emissions_df is not None and "TOTAL CO2e" in emissions_df.columns:
        order = (emissions_df.loc[procs, "TOTAL CO2e"]
                 .fillna(0).sort_values(ascending=False).index.tolist())
    else:
        order = (eb.loc[procs, carriers].sum(axis=1)
                 .sort_values(ascending=False).index.tolist())

    # node list
    nodes = carriers + order
    idx = {n:i for i,n in enumerate(nodes)}

    # links: carrier -> process
    src, tgt, val, hover = [], [], [], []
    for p in order:
        for c in carriers:
            mj = float(eb.at[p, c]) if (p in eb.index and c in eb.columns) else 0.0
            if mj > min_MJ:
                src.append(idx[c])
                tgt.append(idx[p])
                val.append(mj)
                # Put both energy and (if available) emissions in hover
                if emissions_df is not None and p in emissions_df.index:
                    co2 = float(emissions_df.at[p, "TOTAL CO2e"]) if "TOTAL CO2e" in emissions_df.columns else 0.0
                    hover.append(f"{c} → {p}<br>{mj:,.1f} MJ<br>{co2:,.1f} kg CO₂e")
                else:
                    hover.append(f"{c} → {p}<br>{mj:,.1f} MJ")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, pad=18, thickness=16),
        link=dict(source=src, target=tgt, value=val,
                  hovertemplate="%{customdata}<extra></extra>",
                  customdata=hover)
    ))
    fig.update_layout(title=title, font=dict(size=12), height=700)
    return fig    

def make_hybrid_sankey(energy_balance_df, emissions_df, title="Hybrid Sankey: Energy → Processes → CO₂",
                       min_MJ=10.0, min_kg=0.1, co2_scale=None, include_direct_and_energy_sinks=True):
    """
    Build a Sankey with:
      carriers (MJ) → processes → CO₂ (kg) sinks.
    CO₂ links are scaled by `co2_scale` so link widths share one number system.

    Parameters
    ----------
    energy_balance_df : pd.DataFrame
        Rows = processes (incl. 'TOTAL' row), Cols = carriers (e.g., Electricity, Gas, Coal, Coke, ...)
        Values in MJ. We’ll ignore 'TOTAL' and any row not in emissions_df.
    emissions_df : pd.DataFrame
        Index = processes (plus optional 'TOTAL'), columns must include:
          'Energy Emissions' (kg), 'Direct Emissions' (kg), 'TOTAL CO2e' (kg)
    min_MJ : float
        Hide tiny carrier→process energy links below this MJ.
    min_kg : float
        Hide tiny process→CO₂ links below this kg.
    co2_scale : float or None
        If None, autoscale so sum(MJ links) == sum(kg links * scale).
    include_direct_and_energy_sinks : bool
        If True, create two sinks ('CO₂ (energy)', 'CO₂ (direct)'). Otherwise aggregate to one 'CO₂'.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # --- clean copies ---
    eb = energy_balance_df.copy()
    em = emissions_df.copy()

    # rows to include: intersection of processes in energy & emissions (drop TOTAL/Utility Plant)
    proc_rows = [r for r in eb.index if r not in ("TOTAL",)]
    proc_rows = [r for r in proc_rows if r in em.index]
    proc_rows = [r for r in proc_rows if r != "Utility Plant"]  # optional: hide UP

    # carriers: all energy carriers in columns (ignore non-carriers if present)
    carriers = [c for c in eb.columns if c != "TOTAL"]

    # totals for autoscale
    total_energy_MJ = float(eb.loc[proc_rows, carriers].clip(lower=0).sum().sum())
    # emissions_df is already in kg (in your pipeline)
    total_co2_kg = float(em.loc[proc_rows, "TOTAL CO2e"].clip(lower=0).sum())
    if co2_scale is None:
        co2_scale = (total_energy_MJ / total_co2_kg) if total_co2_kg > 0 else 1.0

    # --- build node list ---
    carrier_nodes = carriers[:]  # left side
    process_nodes = proc_rows[:] # middle

    if include_direct_and_energy_sinks:
        sink_nodes = ["CO₂ (energy)", "CO₂ (direct)"]
    else:
        sink_nodes = ["CO₂"]

    nodes = carrier_nodes + process_nodes + sink_nodes

    node_index = {n: i for i, n in enumerate(nodes)}

    # --- links: carriers → processes (MJ) ---
    src, tgt, val, hover = [], [], [], []
    for p in process_nodes:
        for c in carriers:
            mj = float(eb.at[p, c]) if (p in eb.index and c in eb.columns) else 0.0
            if mj > min_MJ:
                src.append(node_index[c])
                tgt.append(node_index[p])
                val.append(mj)
                hover.append(f"{c} → {p}<br>{mj:,.1f} MJ")

    # --- links: processes → CO₂ sinks (kg, scaled to width) ---
    for p in process_nodes:
        e_kg = float(em.at[p, "Energy Emissions"]) if "Energy Emissions" in em.columns else 0.0
        d_kg = float(em.at[p, "Direct Emissions"]) if "Direct Emissions" in em.columns else 0.0

        if include_direct_and_energy_sinks:
            if e_kg > min_kg:
                src.append(node_index[p])
                tgt.append(node_index["CO₂ (energy)"])
                val.append(e_kg * co2_scale)
                hover.append(f"{p} → CO₂ (energy)<br>{e_kg:,.2f} kg (scaled ×{co2_scale:,.3g})")
            if d_kg > min_kg:
                src.append(node_index[p])
                tgt.append(node_index["CO₂ (direct)"])
                val.append(d_kg * co2_scale)
                hover.append(f"{p} → CO₂ (direct)<br>{d_kg:,.2f} kg (scaled ×{co2_scale:,.3g})")
        else:
            co2_kg = e_kg + d_kg
            if co2_kg > min_kg:
                src.append(node_index[p])
                tgt.append(node_index["CO₂"])
                val.append(co2_kg * co2_scale)
                hover.append(f"{p} → CO₂<br>{co2_kg:,.2f} kg (scaled ×{co2_scale:,.3g})")

    # --- figure ---
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=nodes,
                pad=18,
                thickness=16
            ),
            link=dict(
                source=src,
                target=tgt,
                value=val,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
            )
        )
    )
    fig.update_layout(title=title, font=dict(size=12), height=700)
    # Show the scale in the subtitle-ish area
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>CO₂ link width uses scale: 1 kg × {co2_scale:,.3g} = 1 width-unit</sup>"
        )
    )
    return fig
    
# ------------------------------------------------------------------
# Scenario-helper utilities
# ------------------------------------------------------------------
def apply_route_overrides(route_cfg, overrides):
    """Replace default route shares (0-1 floats) with scenario values."""
    for proc, share in overrides.items():
        route_cfg[proc] = share
    return route_cfg

def apply_fuel_substitutions(sub_map, energy_shares, energy_int,
                             energy_content, emission_factors, recipes=None):
    for old, new in sub_map.items():
        if old == new: 
            continue
        # move share; keep old key at 0 so formulas still see it
        for proc, shares in energy_shares.items():
            if old in shares:
                old_val = shares.get(old) or 0.0
                shares[new] = (shares.get(new) or 0.0) + old_val
                shares[old] = 0.0
        # optional recipe key renames (input only; avoid Charcoal→Charcoal loops)
        if recipes is not None:
            for r in recipes:
                if old in r.inputs:
                    r.inputs[new] = r.inputs.pop(old)
                # do NOT rename outputs of Coke Production
                if r.name != 'Coke Production' and old in r.outputs:
                    r.outputs[new] = r.outputs.pop(old)


def apply_dict_overrides(target_dict, overrides):
    """Shallow update helper (energy_int, energy_content, emission_factors …)."""
    target_dict.update(overrides)

def apply_recipe_overrides(recipes, overrides, params, energy_int, energy_shares, energy_content):
    if not overrides: 
        return
    by_name = {r.name: r for r in recipes}
    base_ctx = {**vars(params), 'energy_int': energy_int,
                'energy_shares': energy_shares, 'energy_content': energy_content}
    for name, spec in overrides.items():
        r = by_name.get(name)
        if not r:
            continue
        if 'inputs' in spec:
            new_in = {}
            for k, v in spec['inputs'].items():
                new_in[k] = eval(v, base_ctx) if isinstance(v, str) else v
            r.inputs = new_in
        if 'outputs' in spec:
            out_ctx = {**base_ctx, 'inputs': r.inputs}
            new_out = {}
            for k, v in spec['outputs'].items():
                new_out[k] = eval(v, out_ctx) if isinstance(v, str) else v
            r.outputs = new_out

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

def adjust_process_gas_intensity(proc_name, param_key, energy_int, energy_shares, params):
    pg = getattr(params, param_key, 0.0)
    if proc_name not in energy_int or pg<=0: 
        return
    base = energy_int[proc_name]
    # remember base
    safe = proc_name.replace(' ','_').lower()
    setattr(params, f"{safe}_base_intensity", base)
    # adjust
    shares = energy_shares.get(proc_name,{})
    S = sum(shares.get(c,0) for c in ['Gas','Coal','Coke','Charcoal'])
    adj = base/(1 - pg*S)
    energy_int[proc_name] = adj
    setattr(params, f"{safe}_adj_intensity", adj)
    print(f"Adjusted {proc_name}: {base:.2f}→{adj:.2f} MJ/run")
    
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


def calculate_energy_balance(prod_level, energy_int, energy_shares):
    # 1) DataFrame of shares
    es = pd.DataFrame.from_dict(energy_shares, orient='index').fillna(0.0)
    # 2) Series of intensities
    ei = pd.Series(energy_int).fillna(0.0)
    # 3) MJ per run for each carrier = share * intensity
    per_run = es.multiply(ei, axis='index')
    # 4) Only keep processes you actually ran
    runs = pd.Series(prod_level)
    # align: rows = intersection of es.index & runs.index
    common = per_run.index.intersection(runs.index)
    data = per_run.loc[common].multiply(runs, axis=0)
    # 5) Ensure every carrier column is present (including Electricity)
    all_carriers = sorted(es.columns.union(pd.Index(['Electricity'])))
    bal = pd.DataFrame(data, index=common, columns=all_carriers).fillna(0.0)
    # 6) Add TOTAL row
    bal.loc['TOTAL'] = bal.sum()
    return bal


def adjust_energy_balance(energy_df, internal_elec):
    # Guarantee an Electricity column exists
    if 'Electricity' not in energy_df.columns:
        energy_df['Electricity'] = 0.0

    # subtract internal from grid draw
    energy_df.loc['TOTAL', 'Electricity'] -= internal_elec

    # insert Utility Plant row
    energy_df.loc['Utility Plant'] = 0.0
    energy_df.loc['Utility Plant', 'Electricity'] = -internal_elec

    return energy_df


def calculate_internal_electricity(prod_level, recipes_dict, params):
    internal_elec = 0.0
    util_eff = recipes_dict['Utility Plant'].outputs.get('Electricity', 0)

    # 1) BF top‐gas
    bf_runs = prod_level.get('Blast Furnace', 0.0)
    if bf_runs>0 and hasattr(params,'bf_base_intensity'):
        bf_delta = params.bf_adj_intensity - params.bf_base_intensity
        gf = bf_runs * bf_delta
        print(f"DBG gas BF: runs={bf_runs}, delta={bf_delta:.2f} MJ/run → {gf:.1f} MJ")
        internal_elec += gf * util_eff

    # 2) Coke‐oven gas (recipe‐based approach)
    cp_runs = prod_level.get('Coke Production', 0.0)
    gas_per_run_cp = recipes_dict['Coke Production'].outputs.get('Process Gas', 0)
    if cp_runs > 0 and gas_per_run_cp > 0:
        gf_cp = cp_runs * gas_per_run_cp
        print(f"DBG gas Coke: runs={cp_runs}, gas_per_run={gas_per_run_cp:.2f} MJ/run → {gf_cp:.1f} MJ")
        internal_elec += gf_cp * util_eff

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

    
    if 'Coke Production' in emissions_df.index:
        emissions_df.loc['Coke Production', ['Energy Emissions','Direct Emissions','TOTAL CO2e']] = 0.0

        emissions_df.loc['TOTAL'] = emissions_df.sum()
    if 'TOTAL' not in emissions_df.index:
        emissions_df.loc['TOTAL'] = emissions_df.sum()
    
    return emissions_df

def derive_energy_shares(recipes, energy_content):
    """
    For each process in recipes, look at its inputs and compute:
      • Electricity (as-is, in MJ)
      • Any c in energy_content → amt * LHV (MJ)
    Skip everything else.
    Then normalize to get fractions summing to 1.0.
    """
    shares = {}
    for proc in recipes:
        MJ_by_carrier = {}
        for c, amt in proc.inputs.items():
            if c == 'Electricity':
                MJ_by_carrier[c] = amt
            elif c in energy_content:
                MJ_by_carrier[c] = amt * energy_content[c]
            # else: skip non-energy inputs (ore, limestone, etc.)

        total = sum(MJ_by_carrier.values())
        if total > 1e-9:
            shares[proc.name] = {c: mj/total for c, mj in MJ_by_carrier.items()}
        else:
            shares[proc.name] = {}
    return shares

# User-facing choices (keep names exactly as in your recipes)
PATH_CHOICES = {
    "raw_shaping": [
        "Hot Rolling",
        "Rod/bar/section Mill",
        "Ingot Casting",
    ],
    "cold_roll_optional": [
        "Cold Rolling",    # or None
    ],
    "treatment_optional": [
        "Steel Thermal Treatment",
        "Hot Dip Metal Coating",
        "Electrolytic Metal Coating",
        # or None
    ],
    "manufacturing": [
        "Direct use of Basic Steel Products (IP4)",       # Raw -> IP4
        "Casting/Extrusion/Conformation",                 # IP3 -> IP4
        "Stamping/calendering/lamination",                # IP3 -> IP4
        "Machining",                                      # IP3 -> IP4
    ],
    "final_coat": [
        "No Coating",
        "Hot Dip Metal Coating FP",
        "Electrolytic Metal Coating FP",
        "Organic or Sintetic Coating (painting)",
    ],
}

# Always-on upstream set (grade split + continuous casting)
ALWAYS_ON = [
    "Regular Steel",
    "Steel Refining (Low Alloy)",
    "Steel Refining (High Alloy)",
    "Continuous Casting (R)",
    "Continuous Casting (L)",
    "Continuous Casting (H)",
]

PASS_THROUGH_RAW_TO_IP3 = "Bypass Raw→IP3"

def resolve_name(name, recipes):
    """Return the exact recipe name matching `name` (supports simple aliases)."""
    if not name:
        return None
    names = {r.name for r in recipes}
    if name in names:
        return name
    aliases = {
        "Direct use of Basic Steel Products": "Direct use of Basic Steel Products (IP4)",
        "Bypass to IP3": "Bypass Raw→IP3",
    }
    if name in aliases and aliases[name] in names:
        return aliases[name]
    # fallback: base name (strip ' (..)')
    base = name.split(" (")[0]
    for n in names:
        if n.split(" (")[0] == base:
            return n
    raise ValueError(f"Process '{name}' not found in recipes.")

def build_onoff_from_path(path: dict, recipes) -> dict:
    """Turn on only the processes in the chosen path (uses resolve_name)."""
    from collections import defaultdict
    on = defaultdict(int)

    # --- grade selection (one CC variant) ---
    g = (path.get("grade") or "R").upper()
    if g == "R":
        on[resolve_name("Regular Steel", recipes)]            = 1
        on[resolve_name("Continuous Casting (R)", recipes)]   = 1
    elif g == "L":
        on[resolve_name("Steel Refining (Low Alloy)", recipes)] = 1
        on[resolve_name("Continuous Casting (L)", recipes)]     = 1
    elif g == "H":
        on[resolve_name("Steel Refining (High Alloy)", recipes)] = 1
        on[resolve_name("Continuous Casting (H)", recipes)]       = 1
    else:
        raise ValueError("grade must be R/L/H")

    # --- post-casting picks ---
    rs = resolve_name(path["raw_shaping"], recipes); on[rs] = 1
    cr = resolve_name(path.get("cold_roll"), recipes) if path.get("cold_roll") else None
    if cr: on[cr] = 1
    tr = resolve_name(path.get("treatment"), recipes) if path.get("treatment") else None
    if tr: on[tr] = 1
    mf = resolve_name(path["manufacturing"], recipes); on[mf] = 1
    fc = resolve_name(path["final_coat"], recipes);    on[fc] = 1

    # Need IP3?
    needs_ip3 = mf in {
        resolve_name("Casting/Extrusion/Conformation", recipes),
        resolve_name("Stamping/calendering/lamination", recipes),
        resolve_name("Machining", recipes),
    }
    made_ip3  = bool(cr) or bool(tr)  # CR or any treatment makes IP3 via your recipes
    if needs_ip3 and not made_ip3:
        # choose the right bypass depending on CR
        bypass = "Bypass CR→IP3" if cr else "Bypass Raw→IP3"
        on[resolve_name(bypass, recipes)] = 1

    return dict(on)

def expand_energy_tables_for_active(active_names, energy_shares, energy_int):
    """
    Let variant names like 'Continuous Casting (R)' reuse the energy rows
    of their base name 'Continuous Casting'. Mutates energy_shares/energy_int.
    """
    def base(n): 
        return n.split(" (")[0]
    for n in active_names:
        b = base(n)
        if n not in energy_shares and b in energy_shares:
            energy_shares[n] = dict(energy_shares[b])
        if n not in energy_int and b in energy_int:
            energy_int[n] = energy_int[b]

def validate_path(path: dict):
    # Basic guardrails so users can’t pick weird combos
    assert path["raw_shaping"] in PATH_CHOICES["raw_shaping"], "Invalid raw_shaping"
    assert (path.get("cold_roll") in (None, "Cold Rolling")), "Invalid cold_roll"
    assert (path.get("treatment") in ([None]+PATH_CHOICES["treatment_optional"])), "Invalid treatment"
    assert path["manufacturing"] in PATH_CHOICES["manufacturing"], "Invalid manufacturing"
    assert path["final_coat"] in PATH_CHOICES["final_coat"], "Invalid final_coat"
                

# ===================================================================
#                           MAIN EXECUTION
# ===================================================================
if __name__ == '__main__':

    import argparse, pathlib  # local import so this block is self-contained
    from types import SimpleNamespace

    # ---- tiny helpers (local) -------------------------------------
    def _recursive_ns_update(ns, patch):
        """Recursively apply dict 'patch' into a SimpleNamespace tree."""
        for k, v in (patch or {}).items():
            if isinstance(v, dict):
                cur = getattr(ns, k, None)
                if not isinstance(cur, SimpleNamespace):
                    cur = SimpleNamespace()
                    setattr(ns, k, cur)
                _recursive_ns_update(cur, v)
            else:
                setattr(ns, k, v)

    def _renorm_blend(ns):
        """Ensure blend shares sum to 1.0 after overrides (if present)."""
        try:
            b = ns.blend
            s = getattr(b, 'sinter', 0.0)
            p = getattr(b, 'pellet', 0.0)
            l = getattr(b, 'lump',   0.0)
            tot = s + p + l
            if tot and abs(tot - 1.0) > 1e-9:
                b.sinter = s / tot
                b.pellet = p / tot
                b.lump   = l / tot
        except AttributeError:
            pass

    def _apply_recipe_overrides(recipes, overrides, eval_ctx=None):
        """Patch only specified inputs/outputs for given processes."""
        if not overrides:
            return recipes
        by_name = {r.name: r for r in recipes}
        for proc, patch in overrides.items():
            r = by_name.get(proc)
            if not r:
                continue
            for sect in ('inputs', 'outputs'):
                if sect in patch and isinstance(patch[sect], dict):
                    for k, v in patch[sect].items():
                        if isinstance(v, str) and eval_ctx is not None:
                            try:
                                val = eval(v, eval_ctx)
                            except Exception:
                                continue
                        else:
                            val = v
                        getattr(r, sect)[k] = val
        return list(by_name.values())
    # ----------------------------------------------------------------

    base = os.path.join('data', '')

    # ---------- scenario ----------
    p = argparse.ArgumentParser()
    p.add_argument('-s', '--scenario', default='DRI_EAF.yml',
                   help='file name inside data/scenarios')
    args = p.parse_args()

    sc_path = pathlib.Path(base) / 'scenarios' / args.scenario
    print('[DBG] scenario file at:', sc_path)
    print('[DBG] exists?', sc_path.exists())
    scenario = load_data_from_yaml(sc_path, default_value=None, unwrap_single_key=False)
    print(f"[INFO] Scenario: {scenario.get('description','(no description)')}")
    


    # ---------- base configs (load ONCE) ----------
    energy_int      = load_data_from_yaml(os.path.join(base,'energy_int.yml'))
    energy_shares   = load_data_from_yaml(os.path.join(base,'energy_matrix.yml'))
    energy_content  = load_data_from_yaml(os.path.join(base,'energy_content.yml'))
    e_efs           = load_data_from_yaml(os.path.join(base,'emission_factors.yml'))
    route_cfg       = load_data_from_yaml(os.path.join(base,'route_config.yml'))
    params          = load_parameters      (os.path.join(base,'parameters.yml'))
    
    # ---------- fuel/content/intensity/scenario overrides ----------
    apply_fuel_substitutions(
        scenario.get('fuel_substitutions', {}),
        energy_shares, energy_int, energy_content, e_efs
    )
    apply_dict_overrides(energy_int,     scenario.get('energy_int', {}))
    apply_dict_overrides(energy_shares,  scenario.get('energy_matrix', {}))
    apply_dict_overrides(energy_content, scenario.get('energy_content', {}))
    apply_dict_overrides(e_efs,          scenario.get('emission_factors', {}))

    # Keep original keys present (no pops), and don't delete energy_content/EFs
    apply_fuel_substitutions(
        scenario.get('fuel_substitutions', {}),
        energy_shares, energy_int, energy_content, e_efs
    )

    apply_dict_overrides(energy_int,     scenario.get('energy_int', {}))
    apply_dict_overrides(energy_shares,  scenario.get('energy_matrix', {}))
    apply_dict_overrides(energy_content, scenario.get('energy_content', {}))
    apply_dict_overrides(e_efs,          scenario.get('emission_factors', {}))

    # ---------- scenario parameter overrides (deep) ----------
    # Support BOTH keys for convenience: prefer 'param_overrides', fallback to 'parameters'
    _param_patch = scenario.get('param_overrides', None)
    if _param_patch is None:
        _param_patch = scenario.get('parameters', {})
    _recursive_ns_update(params, _param_patch)
    _renorm_blend(params)

    # Debug: show final blend after overrides
    try:
        print("DBG blend after overrides →",
              {'sinter': params.blend.sinter, 'pellet': params.blend.pellet, 'lump': params.blend.lump})
    except Exception as _e:
        print("DBG blend after overrides → (missing)", _e)
        
        
       
    # ---------- intensity adjustments (now dicts are final) ----------
    adjust_blast_furnace_intensity(energy_int, energy_shares, params)
    adjust_process_gas_intensity('Coke Production', 'process_gas_coke',
                                 energy_int, energy_shares, params)

    # ---------- recipes (load ONCE) ----------
    recipes = load_recipes_from_yaml(
        os.path.join(base, 'recipes.yml'),
        params, energy_int, energy_shares, energy_content
    )
    recipes = _apply_recipe_overrides(recipes, scenario.get('recipe_overrides', {}), {
        **vars(params), 'energy_int': energy_int, 'energy_shares': energy_shares, 'energy_content': energy_content
    })
    recipes_dict = {r.name: r for r in recipes}
    
    # ---------- NOW build routes (path or classic overrides) ----------
    path_mode = bool(scenario.get("path_mode", False))
    user_path = scenario.get("path", None)
    
    if path_mode:
        if not isinstance(user_path, dict):
            raise ValueError("path_mode=True but 'path' missing in scenario.")
        validate_path(user_path)
        route_cfg = build_onoff_from_path(user_path, recipes)   # <-- recipes now defined
    else:
        route_cfg = apply_route_overrides(route_cfg, scenario.get('route_overrides', {}))
        
    # Apply scenario recipe overrides (e.g., EAF 90% PI / 10% Scrap)
    eval_ctx = {**vars(params),
                'energy_int': energy_int,
                'energy_shares': energy_shares,
                'energy_content': energy_content}
    recipes = _apply_recipe_overrides(recipes, scenario.get('recipe_overrides', {}), eval_ctx)
    recipes_dict = {r.name: r for r in recipes}  # refresh

    # Debug: confirm recipes reflect blend and overrides
    bf = recipes_dict.get('Blast Furnace')
    dri = recipes_dict.get('Direct Reduction Iron')
    eaf = recipes_dict.get('Electric Arc Furnace')
    if bf:
        print(f"DBG BF recipe → Sinter={bf.inputs.get('Sinter'):.6f}, Pellet={bf.inputs.get('Pellet'):.6f}, Lump={bf.inputs.get('Iron Ore'):.6f}")
    if dri:
        print(f"DBG DRI recipe → Sinter={dri.inputs.get('Sinter'):.6f}, Pellet={dri.inputs.get('Pellet'):.6f}, Lump={dri.inputs.get('Iron Ore'):.6f}")
    if eaf:
        print("DBG EAF recipe inputs →", eaf.inputs)

    # ---------- other configs ----------
    mkt_cfg = load_market_config  (os.path.join(base, 'mkt_config.yml'))
    p_efs   = load_data_from_yaml(os.path.join(base, 'process_emissions.yml'))
    
    # ---------- material balance ----------
    # PATCH B: preflight debug — who will produce each stage's material?
    from collections import defaultdict
    producers = defaultdict(list)
    for r in recipes:
        for m in r.outputs:
            producers[m].append(r.name)
    
    print("[DBG] active processes:", [k for k,v in route_cfg.items() if v])
    
    # These should each show exactly one active producer in your chosen path:
    for mat in [
        "Liquid Steel R",          # produced by Regular Steel
        "Cast Steel (IP1)",        # produced by one CC variant
        "Raw Products (types)",    # produced by HR / Rod/bar / Ingot
        "Cold Raw Steel (IP2)",    # only if Cold Rolling is selected
        "Intermediate Process 3",  # produced by treatment or bypass
        "Manufactured Feed (IP4)", # produced by manufacturing step
        "Finished Products",       # produced by final coat
    ]:
        actives = [p for p in producers.get(mat, []) if route_cfg.get(p, 0)]
        print(f"[ACTIVE] {mat}: {actives}")
    
    # Sanity: what does Regular Steel output?
    rs = recipes_dict.get("Regular Steel") or recipes_dict.get("Regular Steel Production")
    if rs:
        print("[DBG] Regular Steel outputs →", list(rs.outputs.keys()))
    else:
        print("[DBG] Regular Steel recipe not found")
    
    final_demand = {'Finished Products': 1000.0}
    balance_matrix, prod_lvl = calculate_balance_matrix(recipes, final_demand, route_cfg)
    if balance_matrix is None:
        print("Material balance failed")
        raise SystemExit
       
    # Make sure variants have energy rows (e.g., 'Continuous Casting (R)' -> 'Continuous Casting')
    _active = [p for p, r in prod_lvl.items() if r > 1e-9]
    expand_energy_tables_for_active(_active, energy_shares, energy_int)

    for proc, runs in prod_lvl.items():
        if runs > 1e-9:
            print(f"  {proc:<24s} → {runs:.3f}")
            
            print("Runs → CC:", prod_lvl.get("Continuous Casting", 0))
            print("Runs → HR:", prod_lvl.get("Hot Rolling", 0))
            print("Runs → Rod/bar:", prod_lvl.get("Rod/bar/section Mill", 0))

    # ---------- internal electricity ----------
    internal_elec = calculate_internal_electricity(prod_lvl, recipes_dict, params)

    # ---------- energy balance ----------
    energy_balance = calculate_energy_balance(prod_lvl, energy_int, energy_shares)

    # repair BF to base intensity for reporting (thermal carriers only)
    if 'Blast Furnace' in energy_balance.index and hasattr(params, 'bf_base_intensity'):
        bf_runs = prod_lvl.get('Blast Furnace', 0.0)
        base_bf = params.bf_base_intensity
        bf_sh   = energy_shares.get('Blast Furnace', {})
        for carrier in energy_balance.columns:
            if carrier != 'Electricity':
                energy_balance.loc['Blast Furnace', carrier] = bf_runs * base_bf * bf_sh.get(carrier, 0.0)

    # repair Coke Production to base (thermal carriers only)
    cp_runs = prod_lvl.get('Coke Production', 0.0)
    base_cp = getattr(params, 'coke_production_base_intensity',
                      energy_int.get('Coke Production', 0.0))
    cp_sh   = energy_shares.get('Coke Production', {})
    if cp_runs and cp_sh:
        for carrier in energy_balance.columns:
            if carrier != 'Electricity':
                energy_balance.loc['Coke Production', carrier] = cp_runs * base_cp * cp_sh.get(carrier, 0.0)
                
    # after prod_lvl is built
    active = [p for p, r in prod_lvl.items() if r > 1e-9]
    expand_energy_tables_for_active(active, energy_shares, energy_int)            

    # utility-plant credit
    energy_balance = calculate_energy_balance(prod_lvl, energy_int, energy_shares)

    # ---------- recovered-gas EF (dynamic) ----------
    gas_coke_MJ = prod_lvl.get('Coke Production', 0) * \
                  recipes_dict.get('Coke Production', Process('',{},{})).outputs.get('Process Gas', 0)
    gas_bf_MJ   = (params.bf_adj_intensity - params.bf_base_intensity) * \
                  prod_lvl.get('Blast Furnace', 0)
    total_gas_MJ = gas_coke_MJ + gas_bf_MJ

    # dynamic EF for Coke-oven gas (exclude Electricity share)
    cp_shares = energy_shares.get('Coke Production', {})
    fuels_cp  = [c for c in cp_shares if c != 'Electricity' and cp_shares[c] > 0]
    EF_coke_gas = (sum(cp_shares[c] * e_efs.get(c, 0) for c in fuels_cp) /
                   sum(cp_shares[c] for c in fuels_cp)) if fuels_cp else 0.0
    avoided_coke_CO2 = gas_coke_MJ * EF_coke_gas

    # dynamic EF for BF top-gas (exclude Electricity share)
    bf_shares = energy_shares.get('Blast Furnace', {})
    fuels_bf  = [c for c in bf_shares if c != 'Electricity' and bf_shares[c] > 0]
    EF_bf_gas = (sum(bf_shares[c] * e_efs.get(c, 0) for c in fuels_bf) /
                 sum(bf_shares[c] for c in fuels_bf)) if fuels_bf else 0.0
    avoided_bf_CO2 = gas_bf_MJ * EF_bf_gas

    EF_process_gas = ((avoided_coke_CO2 + avoided_bf_CO2) / total_gas_MJ) if total_gas_MJ else 0.0

    util_eff      = recipes_dict['Utility Plant'].outputs.get('Electricity', 0)
    internal_elec = total_gas_MJ * util_eff

    # ---------- emissions ----------
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

    # guarantee TOTAL exists
    if emissions is not None and 'TOTAL' not in emissions.index:
        emissions.loc['TOTAL'] = emissions.sum()

    if emissions is not None:
        total = emissions.loc['TOTAL', 'TOTAL CO2e']
        print(f"\nTotal CO₂e for {final_demand['Finished Products']} units: {total:.2f} kg")

 # one folder per scenario under ./artifacts/<scenario_name>/
    base_dir = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path().cwd()
    sc_name  = pathlib.Path(args.scenario).stem  # e.g., "DRI_EAF"
    outdir   = base_dir / "artifacts" / sc_name
    outdir.mkdir(parents=True, exist_ok=True)

    # MASS SANKEY (materials ↔ processes)
    fig_mass = make_mass_sankey(
        prod_lvl=prod_lvl,
        recipes_dict=recipes_dict,
        min_flow=0.5,  # filter small flows
        title=f"Mass Flow Sankey — 1000 kg Finished Steel ({sc_name})"
    )
    fig_mass.write_html(outdir / "mass_sankey.html", include_plotlyjs="cdn")

    # ENERGY SANKEY (carriers → processes)
    fig_energy = make_energy_sankey(
        energy_balance_df=energy_balance,
        min_MJ=25.0,  # filter small links
        title=f"Energy Flow Sankey — Process Carriers ({sc_name})"
    )
    fig_energy.write_html(outdir / "energy_sankey.html", include_plotlyjs="cdn")
    
    fig_hybrid = make_hybrid_sankey(
    energy_balance_df=energy_balance,
    emissions_df=emissions,
    min_MJ=25.0,          # hide tiny energy links
    min_kg=1.0,           # hide tiny CO₂ links
    co2_scale=None,       # autoscale so totals match; or set a fixed number
    include_direct_and_energy_sinks=True
)
    fig_hybrid.write_html(outdir / "hybrid_sankey.html", include_plotlyjs="cdn")
    #print("Saved hybrid Sankey to:", (outdir / "hybrid_sankey.html").resolve())

    #print("Saved Sankey diagrams to:", outdir.resolve())
    
    fig_energy_ranked = make_energy_to_process_sankey(
    energy_balance_df=energy_balance,
    emissions_df=emissions,
    title="Energy → Processes (ranked by CO₂e)",
    min_MJ=25.0,
    sort_by="emissions"
    )
    fig_energy_ranked.write_html(outdir / "energy_to_process_sankey.html", include_plotlyjs="cdn")
    #print("Saved:", (outdir / "energy_to_process_sankey.html").resolve())
