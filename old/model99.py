# -*- coding: utf-8 -*-
"""
Final script to calculate a material balance, energy balance, and emissions,
including internal utility-plant electricity credit.

Route selection is interactive: whenever a material has multiple producers,
the user picks exactly one. No route_config.yml is used.
"""

import os
import argparse
import pathlib
import yaml
import json
import pandas as pd
from collections import defaultdict, deque
from types import SimpleNamespace
import plotly.graph_objects as go

# ===================================================================
#                           Data Models
# ===================================================================
class Process:
    """Represents a single recipe with its inputs and outputs."""
    __slots__ = ('name', 'inputs', 'outputs')
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = dict(inputs or {})
        self.outputs = dict(outputs or {})

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
        return SimpleNamespace()

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
        return data  # allow non-dict YAML (e.g., plain list/string)

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
    try:
        recipe_data = yaml.safe_load(open(filepath, 'r', encoding='utf-8')) or []
    except FileNotFoundError:
        print(f"FATAL ERROR: recipes file not found: {filepath}")
        return []

    # Context for evaluating expressions
    context = {**vars(params),
               'energy_int': energy_int,
               'energy_shares': energy_shares,
               'energy_content': energy_content}

    recipes = []
    for item in recipe_data:
        name = (item.get('process') or '').strip()
        if not name:
            continue

        # Evaluate inputs
        inputs, outputs = {}, {}
        for mat, formula in (item.get('inputs') or {}).items():
            if isinstance(formula, str):
                try:
                    inputs[mat] = float(eval(formula, context))
                except Exception as e:
                    print(f"Error in {name} input {mat}: {e}")
                    inputs[mat] = 0.0
            else:
                inputs[mat] = float(formula)

        # Evaluate outputs (can reference 'inputs')
        for mat, formula in (item.get('outputs') or {}).items():
            if isinstance(formula, str):
                out_ctx = {**context, 'inputs': inputs}
                try:
                    outputs[mat] = float(eval(formula, out_ctx))
                except Exception as e:
                    print(f"Error in {name} output {mat}: {e}")
                    outputs[mat] = 0.0
            else:
                outputs[mat] = float(formula)

        recipes.append(Process(name, inputs, outputs))

    return recipes

def load_market_config(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or []
        return {i['name'].strip(): i['value'] for i in cfg}
    except Exception as e:
        print(f"FATAL ERROR loading market config: {e}")
        return {}

def load_electricity_intensity(filepath):
    """Return dict {ISO3: gCO2_per_MJ_electricity} from electricity_intensity.yml."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: Electricity intensity file not found: {filepath}, using defaults.")
        return {}
    items = raw.get('electricity_intensity', [])
    out = {}
    for it in items:
        try:
            code = str(it['code']).upper()
            val  = float(it['intensity'])
            out[code] = val
        except Exception:
            pass
    return out

# ===================================================================
#                          Plot Builders
# ===================================================================
def make_mass_sankey(prod_lvl, recipes_dict, min_flow=0.5, title="Mass Flow Sankey"):
    """
    3-layer Sankey (Material_in → Process → Material_out)
    Units: recipe units × runs. Filters links < min_flow.
    """
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

    mat_in_labels  = sorted(f"[IN] {m}"  for m in mats_in)
    proc_labels    = sorted(f"[P] {p}"   for p in procs)
    mat_out_labels = sorted(f"[OUT] {m}" for m in mats_out)

    labels = mat_in_labels + proc_labels + mat_out_labels
    index  = {lab:i for i,lab in enumerate(labels)}

    sources, targets, values, link_labels = [], [], [], []
    for s, t, v in links:
        sources.append(index[s])
        targets.append(index[t])
        values.append(v)
        link_labels.append(f"{s} → {t}: {v:,.3f}")

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=15, thickness=18),
        link=dict(source=sources, target=targets, value=values, label=link_labels)
    )])
    fig.update_layout(title_text=title, font_size=12)
    return fig


def make_energy_sankey(energy_balance_df, min_MJ=10.0, title="Energy Flow Sankey"):
    df = energy_balance_df.copy().drop(index=[r for r in ["TOTAL"] if r in energy_balance_df.index],
                                       errors="ignore")
    carriers = [c for c in df.columns if (df[c].abs() > min_MJ).any()]
    procs = [p for p in df.index if (df.loc[p, carriers].abs() > min_MJ).any()]
    carrier_labels = [f"[E] {c}" for c in carriers]
    proc_labels    = [f"[P] {p}" for p in procs]
    labels = carrier_labels + proc_labels
    index  = {lab:i for i,lab in enumerate(labels)}
    sources, targets, values, link_labels = [], [], [], []
    for p in procs:
        for c in carriers:
            val = float(df.at[p, c])
            if val > min_MJ:
                sources.append(index[f"[E] {c}"])
                targets.append(index[f"[P] {p}"])
                values.append(val)
                link_labels.append(f"{c} → {p}: {val:,.1f} MJ")
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=15, thickness=18),
        link=dict(source=sources, target=targets, value=values, label=link_labels)
    )])
    fig.update_layout(title_text=title, font_size=12)
    return fig

def make_energy_to_process_sankey(energy_balance_df,
                                  emissions_df=None,
                                  title="Energy → Processes (ranked)",
                                  min_MJ=10.0,
                                  sort_by="emissions",
                                  exclude_carriers=("TOTAL", "Utility Plant")):
    """
    Simple Sankey: energy carriers (MJ) → processes, ranked by CO₂ or MJ.
    """
    eb = energy_balance_df.copy()
    carriers = [c for c in eb.columns if c not in exclude_carriers]
    procs = [r for r in eb.index if r not in ("TOTAL", "Utility Plant")]

    if sort_by == "emissions" and emissions_df is not None and "TOTAL CO2e" in emissions_df.columns:
        order = (emissions_df.loc[[p for p in procs if p in emissions_df.index], "TOTAL CO2e"]
                 .fillna(0).sort_values(ascending=False).index.tolist())
    else:
        order = (eb.loc[procs, carriers].sum(axis=1)
                 .sort_values(ascending=False).index.tolist())

    nodes = carriers + order
    idx = {n:i for i,n in enumerate(nodes)}

    src, tgt, val, hover = [], [], [], []
    for p in order:
        for c in carriers:
            mj = float(eb.at[p, c]) if (p in eb.index and c in eb.columns) else 0.0
            if mj > min_MJ:
                src.append(idx[c]); tgt.append(idx[p]); val.append(mj)
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
    Carriers (MJ) → processes → CO₂ sinks (kg). CO₂ link widths scaled to match MJ.
    """
    import numpy as np

    eb = energy_balance_df.copy()
    if emissions_df is None or emissions_df.empty:
        raise ValueError("emissions_df is empty, cannot build hybrid sankey.")
    em = emissions_df.copy()

    proc_rows = [r for r in eb.index if r not in ("TOTAL",)]
    proc_rows = [r for r in proc_rows if r in em.index and r != "Utility Plant"]
    carriers = [c for c in eb.columns if c != "TOTAL"]

    total_energy_MJ = float(eb.loc[proc_rows, carriers].clip(lower=0).sum().sum())
    total_co2_kg = float(em.loc[proc_rows, "TOTAL CO2e"].clip(lower=0).sum())
    if co2_scale is None:
        co2_scale = (total_energy_MJ / total_co2_kg) if total_co2_kg > 0 else 1.0

    carrier_nodes = carriers[:]
    process_nodes = proc_rows[:]
    sink_nodes = ["CO₂ (energy)", "CO₂ (direct)"] if include_direct_and_energy_sinks else ["CO₂"]
    nodes = carrier_nodes + process_nodes + sink_nodes
    node_index = {n: i for i, n in enumerate(nodes)}

    src, tgt, val, hover = [], [], [], []

    # carriers → processes
    for p in process_nodes:
        for c in carriers:
            mj = float(eb.at[p, c]) if (p in eb.index and c in eb.columns) else 0.0
            if mj > min_MJ:
                src.append(node_index[c]); tgt.append(node_index[p]); val.append(mj)
                hover.append(f"{c} → {p}<br>{mj:,.1f} MJ")

    # processes → CO₂ sinks
    for p in process_nodes:
        e_kg = float(em.at[p, "Energy Emissions"]) if "Energy Emissions" in em.columns and p in em.index else 0.0
        d_kg = float(em.at[p, "Direct Emissions"]) if "Direct Emissions" in em.columns and p in em.index else 0.0
        if include_direct_and_energy_sinks:
            if e_kg > min_kg:
                src.append(node_index[p]); tgt.append(node_index["CO₂ (energy)"]); val.append(e_kg * co2_scale)
                hover.append(f"{p} → CO₂ (energy)<br>{e_kg:,.2f} kg (×{co2_scale:,.3g})")
            if d_kg > min_kg:
                src.append(node_index[p]); tgt.append(node_index["CO₂ (direct)"]); val.append(d_kg * co2_scale)
                hover.append(f"{p} → CO₂ (direct)<br>{d_kg:,.2f} kg (×{co2_scale:,.3g})")
        else:
            co2_kg = e_kg + d_kg
            if co2_kg > min_kg:
                src.append(node_index[p]); tgt.append(node_index["CO₂"]); val.append(co2_kg * co2_scale)
                hover.append(f"{p} → CO₂<br>{co2_kg:,.2f} kg (×{co2_scale:,.3g})")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, pad=18, thickness=16),
        link=dict(source=src, target=tgt, value=val,
                  hovertemplate="%{customdata}<extra></extra>", customdata=hover)
    ))
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>Scale: 1 kg × {co2_scale:,.3g} = 1 width unit</sup>"),
        font=dict(size=12), height=700
    )
    return fig

# ===================================================================
#                   Scenario-helper utilities
# ===================================================================
def apply_fuel_substitutions(sub_map, energy_shares, energy_int,
                             energy_content, emission_factors, recipes=None):
    """Re-map energy shares (and optionally rename recipe inputs/outputs) for carriers."""
    for old, new in (sub_map or {}).items():
        if old == new:
            continue
        for proc, shares in energy_shares.items():
            if old in shares:
                old_val = shares.get(old) or 0.0
                shares[new] = (shares.get(new) or 0.0) + old_val
                shares[old] = 0.0
        if recipes is not None:
            for r in recipes:
                if old in r.inputs:
                    r.inputs[new] = r.inputs.pop(old)
                if r.name != 'Coke Production' and old in r.outputs:
                    r.outputs[new] = r.outputs.pop(old)

def apply_dict_overrides(target_dict, overrides):
    """Shallow update helper (energy_int, energy_content, emission_factors …)."""
    target_dict.update(overrides or {})

def apply_recipe_overrides(recipes, overrides, params, energy_int, energy_shares, energy_content):
    if not overrides:
        return recipes
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
    return list(by_name.values())

# ===================================================================
#                       Calculation Functions
# ===================================================================
def adjust_blast_furnace_intensity(energy_int, energy_shares, params):
    """
    Scales the BF intensity and saves both the original and adjusted values
    so top-gas = adjusted_intensity – base_intensity can be harvested.
    """
    pg = getattr(params, 'process_gas', 0.0)
    if 'Blast Furnace' not in energy_int:
        return

    base = energy_int['Blast Furnace']
    params.bf_base_intensity = base

    shares = energy_shares.get('Blast Furnace', {})
    carriers = ['Gas', 'Coal', 'Coke', 'Charcoal']
    S = sum(shares.get(c, 0.0) for c in carriers)
    denom = max(1e-9, 1 - pg * S)

    adj = base / denom
    energy_int['Blast Furnace'] = adj
    params.bf_adj_intensity = adj

    print(f"Adjusted BF intensity: {base:.2f} → {adj:.2f} MJ/t steel (recovering {pg*100:.1f}% of carriers)")

def adjust_process_gas_intensity(proc_name, param_key, energy_int, energy_shares, params):
    pg = getattr(params, param_key, 0.0)
    if proc_name not in energy_int or pg <= 0:
        return
    base = energy_int[proc_name]
    safe = proc_name.replace(' ','_').lower()
    setattr(params, f"{safe}_base_intensity", base)

    shares = energy_shares.get(proc_name, {})
    S = sum(shares.get(c,0.0) for c in ['Gas','Coal','Coke','Charcoal'])
    denom = max(1e-9, 1 - pg*S)
    adj = base/denom
    energy_int[proc_name] = adj
    setattr(params, f"{safe}_adj_intensity", adj)
    print(f"Adjusted {proc_name}: {base:.2f}→{adj:.2f} MJ/run")

def calculate_balance_matrix(recipes, final_demand, production_routes):
    """
    Solve production levels by walking upstream from 'final_demand' material.
    production_routes: dict {process_name: 0/1}; missing key means allowed (1).
    Exactly one producer per material must be enabled, or we treat as external.
    """
    all_m, producers = set(), defaultdict(list)
    recipes_dict = {r.name:r for r in recipes}
    for r in recipes:
        all_m |= set(r.inputs) | set(r.outputs)
        for m in r.outputs:
            producers[m].append(r)

    demand = list(final_demand.keys())[0]
    if demand not in producers:
        # demand may be a market-only material; still build external requirement row
        required = defaultdict(float, final_demand)
        mats = sorted(all_m | {demand})
        df = pd.DataFrame([[required.get(m,0) for m in mats], [-final_demand.get(m,0) for m in mats]],
                          index=['External Inputs','Final Demand'], columns=mats)
        return df.loc[:, (df.abs()>1e-9).any()], defaultdict(float)

    required = defaultdict(float, final_demand)
    prod_level = defaultdict(float)
    queue = deque([demand])
    seen = {demand}

    while queue:
        mat = queue.popleft(); seen.remove(mat)
        amt = required[mat]
        if amt <= 1e-9:
            continue

        cand_all = producers.get(mat, [])
        cand = [p for p in cand_all if (production_routes.get(p.name, 1.0) > 0.0)]

        if not cand:
            # No enabled internal producer → external purchase; leave as requirement
            # (Downstream will still balance.)
            continue

        if len(cand) > 1:
            names = [p.name for p in cand]
            raise ValueError(f"Ambiguous producers for '{mat}': {names}. Pick exactly one.")

        p = cand[0]
        out_amt = float(p.outputs.get(mat, 0.0))
        if out_amt <= 0:
            continue

        runs = amt / out_amt
        prod_level[p.name] += runs

        for im, ia in p.inputs.items():
            required[im] += runs * float(ia)
            if im in producers and im not in seen:
                queue.append(im); seen.add(im)

        required[mat] = 0.0

    # Prepare matrix (process net flows + external + final demand)
    ext = {m:amt for m,amt in required.items() if amt>1e-9 and m not in producers}
    mats = sorted(all_m | set(required.keys()))
    data, rows = [], []

    for nm, lvl in prod_level.items():
        if lvl > 1e-9:
            rec = recipes_dict[nm]
            row = [ (rec.outputs.get(m,0.0) - rec.inputs.get(m,0.0)) * lvl for m in mats ]
            data.append(row); rows.append(nm)

    data.append([ext.get(m,0.0) for m in mats]); rows.append('External Inputs')
    data.append([-final_demand.get(m,0.0) for m in mats]); rows.append('Final Demand')

    df = pd.DataFrame(data, index=rows, columns=mats)
    return df.loc[:, (df.abs()>1e-9).any()], prod_level

def calculate_energy_balance(prod_level, energy_int, energy_shares):
    """
    Build energy balance (MJ) from production levels, per-run intensity, and carrier shares.
    """
    es = pd.DataFrame.from_dict(energy_shares, orient='index').fillna(0.0)
    ei = pd.Series(energy_int).fillna(0.0)

    per_run = es.multiply(ei, axis='index')  # MJ per run by carrier
    runs = pd.Series(prod_level)
    common = per_run.index.intersection(runs.index)
    data = per_run.loc[common].multiply(runs, axis=0)

    all_carriers = sorted(es.columns.union(pd.Index(['Electricity'])))
    bal = pd.DataFrame(data, index=common, columns=all_carriers).fillna(0.0)
    bal.loc['TOTAL'] = bal.sum()
    return bal

def adjust_energy_balance(energy_df, internal_elec):
    """
    Apply internal electricity credit: subtract internal_elec from TOTAL's Electricity
    and add a 'Utility Plant' row producing negative Electricity (export).
    """
    df = energy_df.copy()
    if 'Electricity' not in df.columns:
        df['Electricity'] = 0.0
    df.loc['TOTAL', 'Electricity'] -= internal_elec
    df.loc['Utility Plant'] = 0.0
    df.loc['Utility Plant', 'Electricity'] = -internal_elec
    return df

def calculate_internal_electricity(prod_level, recipes_dict, params):
    """
    Internal electricity from recovered gases: BF top-gas delta + Coke-oven gas,
    converted with Utility Plant efficiency (recipe output Electricity per MJ gas).
    """
    util_eff = 0.0
    if 'Utility Plant' in recipes_dict:
        util_eff = recipes_dict['Utility Plant'].outputs.get('Electricity', 0.0)

    internal_elec = 0.0

    # BF top-gas (difference between adjusted and base intensities)
    bf_runs = float(prod_level.get('Blast Furnace', 0.0))
    if bf_runs > 0 and hasattr(params,'bf_base_intensity') and hasattr(params,'bf_adj_intensity'):
        bf_delta = params.bf_adj_intensity - params.bf_base_intensity
        gf = bf_runs * bf_delta
        print(f"DBG gas BF: runs={bf_runs:.3f}, delta={bf_delta:.2f} MJ/run → {gf:.1f} MJ")
        internal_elec += gf * util_eff

    # Coke-oven gas (recipe-defined)
    cp_runs = float(prod_level.get('Coke Production', 0.0))
    if 'Coke Production' in recipes_dict:
        gas_per_run_cp = recipes_dict['Coke Production'].outputs.get('Process Gas', 0.0)
        if cp_runs > 0 and gas_per_run_cp > 0:
            gf_cp = cp_runs * gas_per_run_cp
            print(f"DBG gas Coke: runs={cp_runs:.3f}, gas_per_run={gas_per_run_cp:.2f} MJ/run → {gf_cp:.1f} MJ")
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
    Calculates total emissions. Electricity from Utility Plant is only
    credited to in-mill processes (≤ IP3). IP4 + final coating are grid-only.
    """

    if energy_df is None or energy_df.empty:
        return None

    # ---------------------------
    # 1) Split: inside vs outside
    # ---------------------------
    idx_all = [r for r in energy_df.index if r not in ("TOTAL",)]
    idx_inside  = [p for p in idx_all if p not in ("Utility Plant",) and p not in OUTSIDE_MILL_PROCS]
    idx_outside = [p for p in idx_all if p in OUTSIDE_MILL_PROCS]

    # Sum ONLY inside-mill electricity for the internal share
    if "Electricity" in energy_df.columns and idx_inside:
        inside_elec_MJ = float(energy_df.loc[idx_inside, "Electricity"].clip(lower=0).sum())
    else:
        inside_elec_MJ = 0.0

    # Internal electricity EF (from recovered gas)
    if total_gas_MJ > 1e-9:
        util_eff      = internal_elec / total_gas_MJ if total_gas_MJ > 0 else 0.0
        ef_internal_e = (EF_process_gas / util_eff) if util_eff > 0 else 0.0
    else:
        ef_internal_e = 0.0

    ef_grid = energy_efs.get("Electricity", 0.0)

    # Internal share applies ONLY to inside-mill electricity use
    if inside_elec_MJ > 1e-9:
        internal_share_inside = min(internal_elec / inside_elec_MJ, 1.0)
    else:
        internal_share_inside = 0.0
    grid_share_inside = 1.0 - internal_share_inside

    # --- DEBUG ---
    print(f"[DBG] total_gas_MJ       = {total_gas_MJ:.2f} MJ")
    print(f"[DBG] EF_process_gas     = {EF_process_gas:.2f} gCO2/MJ_gas")
    if total_gas_MJ > 1e-9:
        util_eff = internal_elec / total_gas_MJ
        print(f"[DBG] util_efficiency    = {util_eff:.3f}")
        print(f"[DBG] ef_internal_e      = {ef_internal_e:.2f} gCO2/MJ_elec")
    else:
        print("[DBG] No gas recovered, ef_internal_e = 0")
    print(f"[DBG] ef_grid            = {ef_grid:.2f} gCO2/MJ_elec")
    print(f"[DBG] inside_elec_MJ     = {inside_elec_MJ:.2f} MJ")
    print(f"[DBG] internal_share_in  = {internal_share_inside:.1%}")
    print(f"[DBG] grid_share_in      = {grid_share_inside:.1%}")
    print("— end DBG —\n")
    # -----------

    # 3) Per-process emissions
    rows = []
    for proc_name, runs in prod_level.items():
        if runs <= 1e-9 or proc_name not in energy_df.index:
            continue

        row = {"Process": proc_name, "Energy Emissions": 0.0, "Direct Emissions": 0.0}
        is_outside = proc_name in OUTSIDE_MILL_PROCS

        for carrier, cons in energy_df.loc[proc_name].items():
            cons = float(cons)
            if cons <= 0:
                continue

            if carrier == "Electricity":
                if is_outside:
                    # Outside the mill: ALWAYS grid
                    row["Energy Emissions"] += cons * ef_grid
                else:
                    # Inside the mill: split internal vs grid by inside-only share
                    row["Energy Emissions"] += cons * (
                        internal_share_inside * ef_internal_e
                        + grid_share_inside     * ef_grid
                    )
            else:
                row["Energy Emissions"] += cons * energy_efs.get(carrier, 0.0)

        # Direct process emissions
        row["Direct Emissions"] += runs * process_efs.get(proc_name, 0.0)
        rows.append(row)

    if not rows:
        return None

    emissions_df = pd.DataFrame(rows).set_index("Process") / 1000.0
    emissions_df["TOTAL CO2e"] = emissions_df["Energy Emissions"] + emissions_df["Direct Emissions"]

    # Zero Coke Production totals (if you keep that rule)
    if "Coke Production" in emissions_df.index:
        emissions_df.loc["Coke Production", ["Energy Emissions", "Direct Emissions", "TOTAL CO2e"]] = 0.0

    emissions_df.loc["TOTAL"] = emissions_df.sum()
    return emissions_df


def derive_energy_shares(recipes, energy_content):
    """
    Optional helper: derive shares by scanning recipe inputs (electricity in MJ; fuels by LHV).
    """
    shares = {}
    for proc in recipes:
        MJ_by_carrier = {}
        for c, amt in proc.inputs.items():
            if c == 'Electricity':
                MJ_by_carrier[c] = float(amt)
            elif c in energy_content:
                MJ_by_carrier[c] = float(amt) * float(energy_content[c])
        total = sum(MJ_by_carrier.values())
        if total > 1e-12:
            shares[proc.name] = {c: mj/total for c, mj in MJ_by_carrier.items()}
        else:
            shares[proc.name] = {}
    return shares

# ===================================================================
#                   Interactive Route Selection
# ===================================================================
STAGE_MATS = {
    "Finished": "Finished Products",
    "IP4": "Manufactured Feed (IP4)",
    "IP3": "Intermediate Process 3",
    "Raw": "Raw Products (types)",
    "Cast": "Cast Steel (IP1)",
    "Liquid": "Liquid Steel",
    "PigIron": "Pig Iron",
    "GradeR": "Liquid Steel R",
    "GradeL": "Liquid Steel L",
    "GradeH": "Liquid Steel H",
}

OUTSIDE_MILL_PROCS = {
    "Direct use of Basic Steel Products (IP4)",
    "Casting/Extrusion/Conformation",
    "Stamping/calendering/lamination",
    "Machining",
    "No Coating",
    "Hot Dip Metal Coating FP",
    "Electrolytic Metal Coating FP",
    "Organic or Sintetic Coating (painting)",
}

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
            
def _override_eaf_feed(recipes, mode: str):
    """
    Mutate the EAF recipe *in memory* for this run.
    mode: 'scrap' | 'dri' | 'bf'
    - 'scrap' → Scrap = 1.0, Pig Iron = 0.0
    - 'dri' or 'bf' → Pig Iron = 1.0, Scrap = 0.0
    Keeps other consumables unchanged.
    """
    eaf = next((r for r in recipes if r.name == "Electric Arc Furnace"), None)
    if not eaf:
        return
    base_in = dict(eaf.inputs)
    keep = {k: v for k, v in base_in.items() if k not in ("Pig Iron", "Scrap")}
    if mode == "scrap":
        eaf.inputs = {**keep, "Scrap": 1.0}
    else:  # 'dri' or 'bf'
        eaf.inputs = {**keep, "Pig Iron": 1.0}            

def build_routes_interactive(recipes, demand_mat, pre_select=None, pre_mask=None, interactive=True):
    """
    Build on/off mask interactively.
    - recipes: list[Process]
    - demand_mat: str (final material like 'Finished Products')
    - pre_select: dict {process_name: 0/1} initial picks/bans
    - pre_mask:   dict {process_name: 0/1} hard bans/forces (overrides pre_select)
    - interactive: if False, raise on ambiguity instead of prompting.
    """
    from collections import defaultdict, deque

    # material -> [Process,...]
    producers = defaultdict(list)
    for r in recipes:
        for m in r.outputs:
            producers[m].append(r)

    chosen = dict(pre_select or {})
    if pre_mask:
        # hard locks override pre_select
        for k, v in pre_mask.items():
            chosen[k] = v

    visited_mats = set()
    q = deque([demand_mat])

    while q:
        mat = q.popleft()
        if mat in visited_mats:
            continue
        visited_mats.add(mat)

        cand = producers.get(mat, [])
        if not cand:
            # no internal producer -> external purchase
            continue

        active = [r for r in cand if chosen.get(r.name, 1) > 0]

        if len(active) == 1:
            pick = active[0]
        elif len(active) > 1:
            if not interactive:
                raise ValueError(
                    f"Ambiguous producers for '{mat}': {[r.name for r in active]}. "
                    f"Provide pre_select/pre_mask to make it unique."
                )
            print(f"\nChoose ONE producer for '{mat}':")
            for i, r in enumerate(active, 1):
                ins = ", ".join(r.inputs.keys()) or "(no inputs)"
                outs = ", ".join(r.outputs.keys())
                print(f"  [{i}] {r.name}    inputs: {ins}    outputs: {outs}")
            while True:
                sel = input("Enter number: ").strip()
                if sel.isdigit() and 1 <= int(sel) <= len(active):
                    pick = active[int(sel) - 1]
                    break
                print("Invalid selection, try again.")
        else:
            # all disabled -> external
            continue

        chosen[pick.name] = 1
        for r in cand:
            if r.name != pick.name:
                chosen[r.name] = 0

        for im in pick.inputs.keys():
            if im not in visited_mats:
                q.append(im)

    return chosen



def build_route_mask(route_name, recipes):
    """
    Returns a dict {process_name: 0/1} used as a pre-mask for the interactive
    builder. 0 = disabled (never offered), 1 = allowed.
    Keeps downstream shaping/treatment/coating available; only clamps upstream.
    """
    # Lists below should match your recipe names
    ban = set()
    if route_name == 'EAF-Scrap':
        ban = {
            'Blast Furnace', 'Basic Oxygen Furnace',
            'Direct Reduction Iron'
        }
    elif route_name == 'DRI-EAF':
        ban = {
            'Blast Furnace'
        }
    elif route_name == 'BF-BOF':
        ban = {
            'Direct Reduction Iron', 'Electric Arc Furnace'
        }
    elif route_name == 'External':
        # Disable all primary steelmaking so the chain starts at purchased basic steel
        ban = {
            'Blast Furnace', 'Basic Oxygen Furnace',
            'Direct Reduction Iron', 'Electric Arc Furnace',
            'Coke Production', 'Charcoal Production',
            'Sintering', 'Pelletizing'
        }
    # 'auto' → no ban

    mask = {r.name: (0 if r.name in ban else 1) for r in recipes}
    return mask

ROUTE_DEFAULT_FEEDS = {
    "EAF-Scrap": "scrap",
    "DRI-EAF":   "dri",
    "BF-BOF":    None,
    "External":  None,
}

def _first_existing(candidates, pool):
    for k in candidates:
        if k in pool:
            return k
    return None

def enforce_eaf_feed(recipes, mode: str | None):
    """
    Force 'Electric Arc Furnace' to use only one feed:
      scrap  → Scrap (or Scrap Steel)
      dri    → Direct Reduced Iron (or DRI/HBI)
      pigiron→ Pig Iron (or Hot Metal)
    Keeps all non-feed inputs (O2, N2, fluxes, energy) unchanged.
    Safe no-op if EAF not found or target feed material not found.
    """
    if not mode:
        return

    # find the EAF recipe
    eaf = next((r for r in recipes if r.name == "Electric Arc Furnace"), None)
    if not eaf:
        print("[WARN] enforce_eaf_feed: EAF recipe not found — skipping.")
        return

    # build a set of known materials across the model
    all_mats = set()
    for r in recipes:
        all_mats.update(r.outputs.keys())

    # which material name to feed?
    if mode == "scrap":
        want = _first_existing(("Scrap", "Scrap Steel"), all_mats)
    elif mode == "dri":
        want = _first_existing(("Direct Reduced Iron", "DRI", "HBI"), all_mats)
    elif mode == "pigiron":
        want = _first_existing(("Pig Iron", "Hot Metal"), all_mats)
    else:
        print(f"[WARN] enforce_eaf_feed: unknown mode '{mode}' — skipping.")
        return

    if not want:
        print(f"[WARN] enforce_eaf_feed: no matching material for mode '{mode}' — skipping.")
        return

    # keep non-feed inputs; drop any feed keys we might have
    feed_keys = {"Pig Iron", "Hot Metal", "Direct Reduced Iron", "DRI", "HBI", "Scrap", "Scrap Steel"}
    non_feed = {k: v for k, v in eaf.inputs.items() if k not in feed_keys}

    # set the forced feed to 1.0 (you can change the amount if your unit differs)
    eaf.inputs = {**non_feed, want: 1.0}
    print(f"[INFO] EAF feed forced to '{want}' ({mode})")
    
def build_pre_for_route(route_key):
    """Return (pre_select, pre_mask, recipe_overrides) for a clean, unique path."""
    route = route_key.upper()
    pre_select = {
        # lock one CC variant and a simple downstream
        "Continuous Casting (R)": 1,
        "Hot Rolling": 0,                    # prefer rods for this sweep; flip if you want
        "Rod/bar/section Mill": 1,
        "Ingot Casting": 0,
        "Direct use of Basic Steel Products (IP4)": 1,
        "No Coating": 1,
        # keep treatments OFF for the simple baseline
        "Cold Rolling": 0,
        "Steel Thermal Treatment": 0,
        "Hot Dip Metal Coating": 0,
        "Electrolytic Metal Coating": 0,
        "Casting/Extrusion/Conformation": 0,
        "Stamping/calendering/lamination": 0,
        "Machining": 0,
        "Bypass Raw→IP3": 0,
        "Bypass CR→IP3": 0,
    }
    pre_mask = {}
    recipe_overrides = {}

    if route == "BF-BOF":
        # Liquid Steel → BOF; Pig Iron → BF
        pre_mask.update({
            "Direct Reduction Iron": 0,
            "Electric Arc Furnace": 0,
        })
    elif route == "DRI-EAF":
        # Liquid Steel → EAF; Pig Iron → DRI; ensure EAF uses Pig Iron (not Scrap)
        pre_mask.update({
            "Blast Furnace": 0,
            "Basic Oxygen Furnace": 0,
        })
        recipe_overrides["Electric Arc Furnace"] = {"inputs": {"Pig Iron": 1.0, "Scrap": 0.0}}
    elif route == "EAF-SCRAP":
        # Liquid Steel → EAF; NO pig-iron path; force scrap mode
        pre_mask.update({
            "Blast Furnace": 0,
            "Basic Oxygen Furnace": 0,
            "Direct Reduction Iron": 0,
        })
        recipe_overrides["Electric Arc Furnace"] = {"inputs": {"Pig Iron": 0.0, "Scrap": 1.0}}
    else:
        raise ValueError(f"Unknown route '{route_key}'")

    return pre_select, pre_mask, recipe_overrides
    

# ===================================================================
#                           MAIN EXECUTION
# ===================================================================
if __name__ == '__main__':

    # ---------- scenario / args ----------
    p = argparse.ArgumentParser()
    
    p.add_argument('-s', '--scenario', default='DRI_EAF.yml',
                   help='file name inside data/scenarios (or a full path)')
    p.add_argument('--stage', choices=list(STAGE_MATS.keys()), default='Finished',
                   help='Where to stop the chain (default: Finished)')
    p.add_argument('--demand', type=float, default=1000.0,
                   help='Demand quantity at the selected stage (default: 1000)')
    p.add_argument(
    '--route',
    choices=['auto', 'BF-BOF', 'DRI-EAF', 'EAF-Scrap', 'External'],
    default='auto',
    help='Upstream route preset to constrain producer choices (default: auto)'
    )
    args = p.parse_args()

    base = os.path.join('data', '')
    # Allow either "path_demo.yml" or "data/scenarios/path_demo.yml"
    if os.path.sep in args.scenario or '/' in args.scenario:
        sc_path = pathlib.Path(args.scenario)
    else:
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
    params          = load_parameters      (os.path.join(base,'parameters.yml'))
    
    # ---- country → electricity EF selection (first prompt) ----
    elec_map = load_electricity_intensity(os.path.join(base, 'electricity_intensity.yml'))
    
    # optional pre-selection from scenario.yml (grid_country: BRA, etc.)
    pre_code = (scenario.get('grid_country') or scenario.get('country') or '').upper()
    
    def _pick_country_code(elec_map, pre_code=None, default_code='USA'):
        if pre_code and pre_code in elec_map:
            return pre_code
        if not elec_map:
            return None
        codes = sorted(elec_map.keys())
        print("\nSelect country for electricity EF (gCO₂/MJ):")
        for i, c in enumerate(codes, 1):
            print(f"  [{i}] {c}  → {elec_map[c]:.2f}")
        prompt = f"Enter number or code (default={default_code if default_code in elec_map else codes[0]}): "
        while True:
            sel = input(prompt).strip()
            if not sel:
                return default_code if default_code in elec_map else codes[0]
            if sel.isdigit():
                k = int(sel) - 1
                if 0 <= k < len(codes):
                    return codes[k]
            sel = sel.upper()
            if sel in elec_map:
                return sel
            print("Invalid selection, try again.")
    
    country_code = _pick_country_code(elec_map, pre_code=pre_code, default_code='USA')
    
    if country_code:
        e_efs['Electricity'] = elec_map[country_code]
        params.grid_country = country_code
        print(f"[INFO] Electricity EF set by country {country_code}: {e_efs['Electricity']:.2f} gCO₂/MJ")
    else:
        print("[WARN] Using default Electricity EF from emission_factors.yml (no country map found).")


    # ---------- scenario-level overrides ----------
    apply_fuel_substitutions(scenario.get('fuel_substitutions', {}),
                             energy_shares, energy_int, energy_content, e_efs)
    apply_dict_overrides(energy_int,     scenario.get('energy_int', {}))
    apply_dict_overrides(energy_shares,  scenario.get('energy_matrix', {}))
    apply_dict_overrides(energy_content, scenario.get('energy_content', {}))
    apply_dict_overrides(e_efs,          scenario.get('emission_factors', {}))

    # parameters (deep merge)
    def _recursive_ns_update(ns, patch):
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
        try:
            b = ns.blend
            s = float(getattr(b, 'sinter', 0.0))
            p_ = float(getattr(b, 'pellet', 0.0))
            l = float(getattr(b, 'lump',   0.0))
            tot = s + p_ + l
            if tot and abs(tot - 1.0) > 1e-9:
                b.sinter = s / tot
                b.pellet = p_ / tot
                b.lump   = l / tot
        except AttributeError:
            pass

    _param_patch = scenario.get('param_overrides', None)
    if _param_patch is None:
        _param_patch = scenario.get('parameters', {})
    _recursive_ns_update(params, _param_patch)
    _renorm_blend(params)

    try:
        print("DBG blend after overrides →",
              {'sinter': params.blend.sinter, 'pellet': params.blend.pellet, 'lump': params.blend.lump})
    except Exception as _e:
        print("DBG blend after overrides → (missing)", _e)

    # ---------- intensity adjustments (after overrides) ----------
    adjust_blast_furnace_intensity(energy_int, energy_shares, params)
    adjust_process_gas_intensity('Coke Production', 'process_gas_coke',
                                 energy_int, energy_shares, params)

    # ---------- recipes (load once, then apply scenario recipe overrides) ----------
    recipes = load_recipes_from_yaml(
        os.path.join(base, 'recipes.yml'),
        params, energy_int, energy_shares, energy_content
    )
    recipes = apply_recipe_overrides(recipes, scenario.get('recipe_overrides', {}),
                                     params, energy_int, energy_shares, energy_content)
    recipes_dict = {r.name: r for r in recipes}

    # Debug: confirm some key recipes (if they exist)
    for key in ('Blast Furnace', 'Direct Reduction Iron', 'Electric Arc Furnace'):
        r = recipes_dict.get(key)
        if r:
            print(f"DBG {key} inputs →", r.inputs)

    # ---------- other configs ----------
    mkt_cfg = load_market_config  (os.path.join(base, 'mkt_config.yml'))
    p_efs   = load_data_from_yaml(os.path.join(base, 'process_emissions.yml'))

    # ---------- interactive path selection ----------
    demand_mat = STAGE_MATS[args.stage]
    final_demand = {demand_mat: float(args.demand)}

    print(f"\n=== Interactive path selection for demand: {demand_mat} ({args.demand}) ===")
    
    # Build the route pre-mask once
    pre_mask = build_route_mask(args.route, recipes)
    
    print(f"\n=== Interactive path selection for demand: {demand_mat} ({args.demand}) ===")
    if args.route != 'auto':
        print(f"[INFO] Route preset: {args.route} (incompatible upstream units disabled)")
    
    # Enforce EAF feed from route (scrap vs pig iron/DRI)
    feed_mode = ROUTE_DEFAULT_FEEDS.get(args.route)
    enforce_eaf_feed(recipes, feed_mode)
    
    # Also pre-disable conflicting upstream cores (soft – user can still choose where allowed)
    UPSTREAM_CORE = {
        "Blast Furnace", "Basic Oxygen Furnace", "Direct Reduction Iron",
        "Electric Arc Furnace", "Scrap Purchase"
    }
    route_disable = {
        "EAF-Scrap": {"Blast Furnace", "Basic Oxygen Furnace", "Direct Reduction Iron"},
        "DRI-EAF":   {"Blast Furnace", "Basic Oxygen Furnace"},
        "BF-BOF":    {"Direct Reduction Iron", "Electric Arc Furnace"},
        "External":  set(),
    }.get(args.route, set())
    pre_select = {p: 0 for p in route_disable if p in UPSTREAM_CORE}
    
    print(f"[INFO] Route preset: {args.route}")
    print(f"[INFO] Demand: {args.demand} at stage {args.stage} → {STAGE_MATS[args.stage]}")
    
    # ✅ Correct call: pass both pre_select and pre_mask
    production_routes = build_routes_interactive(
        recipes,
        STAGE_MATS[args.stage],
        pre_select=pre_select,
        pre_mask=pre_mask,
        interactive=True
    )

    # Solve material balance
    balance_matrix, prod_lvl = calculate_balance_matrix(recipes, final_demand, production_routes)
    if balance_matrix is None:
        print("Material balance failed")
        raise SystemExit

    # Ensure variant processes (e.g., CC variants) have energy rows
    active_procs = [p for p, r in prod_lvl.items() if r > 1e-9]
    expand_energy_tables_for_active(active_procs, energy_shares, energy_int)

    # ---------- internal electricity (before energy credit) ----------
    internal_elec = calculate_internal_electricity(prod_lvl, recipes_dict, params)

    # ---------- energy balance ----------
    energy_balance = calculate_energy_balance(prod_lvl, energy_int, energy_shares)

    # Optional: repair BF/CP reporting to base thermal carriers (kept as in your flow)
    if 'Blast Furnace' in energy_balance.index and hasattr(params, 'bf_base_intensity'):
        bf_runs = float(prod_lvl.get('Blast Furnace', 0.0))
        base_bf = float(params.bf_base_intensity)
        bf_sh   = energy_shares.get('Blast Furnace', {})
        for carrier in energy_balance.columns:
            if carrier != 'Electricity':
                energy_balance.loc['Blast Furnace', carrier] = bf_runs * base_bf * float(bf_sh.get(carrier, 0.0))

    cp_runs = float(prod_lvl.get('Coke Production', 0.0))
    base_cp = float(getattr(params, 'coke_production_base_intensity',
                            energy_int.get('Coke Production', 0.0)))
    cp_sh   = energy_shares.get('Coke Production', {})
    if cp_runs and cp_sh:
        for carrier in energy_balance.columns:
            if carrier != 'Electricity':
                energy_balance.loc['Coke Production', carrier] = cp_runs * base_cp * float(cp_sh.get(carrier, 0.0))

    # Apply internal electricity credit
    energy_balance = adjust_energy_balance(energy_balance, internal_elec)

    # ---------- recovered-gas EF (dynamic) ----------
    gas_coke_MJ = prod_lvl.get('Coke Production', 0.0) * \
                  recipes_dict.get('Coke Production', Process('',{},{})).outputs.get('Process Gas', 0.0)
    gas_bf_MJ   = (getattr(params, 'bf_adj_intensity', 0.0) - getattr(params, 'bf_base_intensity', 0.0)) * \
                  prod_lvl.get('Blast Furnace', 0.0)
    total_gas_MJ = float(gas_coke_MJ + gas_bf_MJ)

    # dynamic EF for Coke-oven gas (exclude Electricity share)
    cp_shares = energy_shares.get('Coke Production', {})
    fuels_cp  = [c for c in cp_shares if c != 'Electricity' and cp_shares[c] > 0]
    EF_coke_gas = (sum(cp_shares[c] * e_efs.get(c, 0.0) for c in fuels_cp) /
                   max(1e-12, sum(cp_shares[c] for c in fuels_cp))) if fuels_cp else 0.0
    avoided_coke_CO2 = total_gas_MJ * 0.0  # accounted via internal electricity path; keep placeholder

    # dynamic EF for BF top-gas (exclude Electricity share)
    bf_shares = energy_shares.get('Blast Furnace', {})
    fuels_bf  = [c for c in bf_shares if c != 'Electricity' and bf_shares[c] > 0]
    EF_bf_gas = (sum(bf_shares[c] * e_efs.get(c, 0.0) for c in fuels_bf) /
                 max(1e-12, sum(bf_shares[c] for c in fuels_bf))) if fuels_bf else 0.0

    # Average EF for process gas burned at the utility
    EF_process_gas = EF_coke_gas if total_gas_MJ <= 1e-9 else (
        (EF_coke_gas * (gas_coke_MJ / max(1e-12, total_gas_MJ))) +
        (EF_bf_gas   * (gas_bf_MJ   / max(1e-12, total_gas_MJ)))
    )

    util_eff      = recipes_dict.get('Utility Plant', Process('',{},{})).outputs.get('Electricity', 0.0)
    internal_elec = total_gas_MJ * util_eff  # recompute from total gas and efficiency (defensive)

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

    if emissions is not None and 'TOTAL' not in emissions.index:
        emissions.loc['TOTAL'] = emissions.sum()

    if emissions is not None:
        total = float(emissions.loc['TOTAL', 'TOTAL CO2e'])
        print(f"\nTotal CO₂e for {args.demand:.1f} units of '{demand_mat}': {total:.2f} kg")

    # ---------- outputs (artifacts) ----------
    base_dir = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path().cwd()
    sc_name  = pathlib.Path(args.scenario).stem
    outdir   = base_dir / "artifacts" / sc_name
    outdir.mkdir(parents=True, exist_ok=True)

    # MASS SANKEY (materials ↔ processes)
    recipes_dict = {r.name: r for r in recipes}  # ensure fresh
    fig_mass = make_mass_sankey(
        prod_lvl=prod_lvl,
        recipes_dict=recipes_dict,
        min_flow=0.5,
        title=f"Mass Flow Sankey — 1000 kg Finished Steel ({sc_name})"
    )
    fig_mass.write_html(outdir / "mass_sankey.html", include_plotlyjs="cdn")

    # ENERGY SANKEY (carriers → processes)
    fig_energy = make_energy_sankey(
        energy_balance_df=energy_balance,
        min_MJ=25.0,
        title=f"Energy Flow Sankey — Process Carriers ({sc_name})"
    )
    fig_energy.write_html(outdir / "energy_sankey.html", include_plotlyjs="cdn")

    # HYBRID (only if emissions computed)
    if emissions is not None and not emissions.empty:
        fig_hybrid = make_hybrid_sankey(
            energy_balance_df=energy_balance,
            emissions_df=emissions,
            min_MJ=25.0,
            min_kg=1.0,
            co2_scale=None,
            include_direct_and_energy_sinks=True
        )
        fig_hybrid.write_html(outdir / "hybrid_sankey.html", include_plotlyjs="cdn")

        fig_energy_ranked = make_energy_to_process_sankey(
            energy_balance_df=energy_balance,
            emissions_df=emissions,
            title="Energy → Processes (ranked by CO₂e)",
            min_MJ=25.0,
            sort_by="emissions"
        )
        fig_energy_ranked.write_html(outdir / "energy_to_process_sankey.html", include_plotlyjs="cdn")

    print("Saved Sankey diagrams to:", outdir.resolve())
