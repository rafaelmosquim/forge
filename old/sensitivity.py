# -*- coding: utf-8 -*-
"""
Steel Emissions & Sensitivity Analysis

This script:
1) Loads process, energy, and emissions configurations from YAML.
2) Calculates material and energy balances.
3) Computes total CO₂e emissions per process and aggregate.
4) Runs a 1% one-at-a-time sensitivity analysis over all key parameters.
5) Prints the emissions table and plots the top 10 most influential parameters.
"""
import os
import yaml
import json
import copy
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from types import SimpleNamespace

# ------------------- Data Model -------------------
class Process:
    """Represents a single process with its inputs and outputs."""
    __slots__ = ('name', 'inputs', 'outputs')
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs

# ------------------- Loaders -------------------
def load_parameters(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return json.loads(json.dumps(data), object_hook=lambda d: SimpleNamespace(**d))


def load_data_from_yaml(path, default=0.0):
    try:
        raw = yaml.safe_load(open(path, 'r', encoding='utf-8')) or {}
    except FileNotFoundError:
        return {}
    if isinstance(raw, dict) and len(raw) == 1:
        raw = next(iter(raw.values())) or {}
    out = {}
    for k, v in raw.items():
        if v is None:
            out[str(k)] = default
        elif isinstance(v, str):
            try: out[str(k)] = float(v)
            except ValueError: out[str(k)] = v
        else:
            out[str(k)] = v
    return out


def load_recipes(path, params, energy_int, energy_shares, energy_content):
    raw = yaml.safe_load(open(path, 'r', encoding='utf-8')) or []
    recipes = []
    ctx = dict(vars(params))
    ctx.update({'energy_int': energy_int,
                'energy_shares': energy_shares,
                'energy_content': energy_content})
    for item in raw:
        name = item.get('process', '').strip()
        if not name:
            continue
        inputs = {}
        for m, expr in item.get('inputs', {}).items():
            inputs[m] = eval(expr, ctx) if isinstance(expr, str) else expr
        outputs = {}
        for m, expr in item.get('outputs', {}).items():
            local = dict(ctx); local['inputs'] = inputs
            outputs[m] = eval(expr, local) if isinstance(expr, str) else expr
        recipes.append(Process(name, inputs, outputs))
    return recipes


def load_market_config(path):
    try:
        cfg = yaml.safe_load(open(path, 'r', encoding='utf-8')) or []
        return {item['name']: item['value'] for item in cfg}
    except:
        return {}

# ------------------- Adjustments -------------------
def adjust_blast_furnace_intensity(ei, es, params):
    pg = getattr(params, 'process_gas', 0.0)
    if 'Blast Furnace' not in ei:
        return
    base = ei['Blast Furnace']; params.bf_base_intensity = base
    shares = es.get('Blast Furnace', {})
    S = sum(shares.get(c, 0) for c in ['Gas','Coal','Coke','Charcoal'])
    adj = base / (1 - pg * S)
    ei['Blast Furnace'] = adj; params.bf_adj_intensity = adj


def adjust_process_gas_intensity(proc, key, ei, es, params):
    pg = getattr(params, key, 0.0)
    if pg <= 0 or proc not in ei:
        return
    base = ei[proc]
    setattr(params, f"{proc.replace(' ','_')}_base_intensity", base)
    shares = es.get(proc, {})
    S = sum(shares.get(c, 0) for c in ['Gas','Coal','Coke','Charcoal'])
    adj = base / (1 - pg * S)
    ei[proc] = adj
    setattr(params, f"{proc.replace(' ','_')}_adj_intensity", adj)

# ------------------- Balances -------------------
def calculate_balance(recipes, demand, routes):
    prod_map = defaultdict(list)
    for r in recipes:
        for m in r.outputs:
            prod_map[m].append(r)
    required = defaultdict(float, demand)
    prod_level = defaultdict(float)
    queue = deque([next(iter(demand))]); seen = set(queue)
    while queue:
        m = queue.popleft(); seen.discard(m)
        amt = required[m]
        for r in prod_map.get(m, []):
            share = routes.get(r.name, 0)
            out = r.outputs.get(m, 0)
            if share<=0 or out<=0: continue
            runs = amt * share / out
            prod_level[r.name] += runs
            for im, ia in r.inputs.items():
                required[im] += runs * ia
                if im not in seen and im in prod_map:
                    queue.append(im); seen.add(im)
        required[m] = 0
    return prod_level


def calculate_energy_balance(prod_level, ei, es):
    es_df = pd.DataFrame.from_dict(es, 'index').fillna(0)
    ei_s = pd.Series(ei).fillna(0)
    per = es_df.multiply(ei_s, axis=0)
    runs = pd.Series(prod_level)
    df = per.loc[runs.index.intersection(per.index)].multiply(runs, axis=0)
    cols = sorted(es_df.columns.union(['Electricity']))
    bal = pd.DataFrame(df, columns=cols).fillna(0)
    bal.loc['TOTAL'] = bal.sum()
    return bal


def adjust_energy_balance(bal, internal):
    eb = bal.copy(); eb['Electricity'] = eb.get('Electricity',0)
    eb.loc['TOTAL','Electricity'] -= internal
    eb.loc['Utility Plant','Electricity'] = -internal
    return eb


def calculate_internal_electricity(prod_level, recipes_dict, params):
    eff = recipes_dict['Utility Plant'].outputs.get('Electricity',0)
    ie = 0.0
    bf = prod_level.get('Blast Furnace',0)
    if bf and hasattr(params,'bf_base_intensity'):
        ie += bf*(params.bf_adj_intensity-params.bf_base_intensity)*eff
    cp = prod_level.get('Coke Production',0)
    ie += cp*recipes_dict['Coke Production'].outputs.get('Process Gas',0)*eff
    return ie

# ------------------- Emissions -------------------
def calculate_emissions(prod_level, bal, e_efs, p_efs, internal, tot_gas, EF_pg):
    tp = abs(bal.loc['TOTAL','Electricity']+internal)
    if tp>0:
        isr = min(internal/tp,1); gsr=1-isr
    else:
        isr,gsr=0,1
    ef_grid = e_efs.get('Electricity',0)
    util_eff = internal/tot_gas if tot_gas>0 else 0
    ef_int = EF_pg/util_eff if util_eff>0 else 0
    rows=[]
    for p,runs in prod_level.items():
        if runs<=0: continue
        ee=0
        if p in bal.index:
            for c,cons in bal.loc[p].items():
                ee += cons*(isr*ef_int+gsr*ef_grid) if c=='Electricity' else cons*e_efs.get(c,0)
        de = runs*p_efs.get(p,0)
        rows.append({'Process':p,'Energy Em':ee,'Direct Em':de})
    df = pd.DataFrame(rows).set_index('Process')/1000
    df['TOTAL CO2e'] = df['Energy Em']+df['Direct Em']
    df.loc['TOTAL'] = df.sum()
    if 'Coke Production' in df.index:
        df.loc['Coke Production',['Energy Em','Direct Em','TOTAL CO2e']] = 0
    return df

# ------- Sensitivity Analysis -------
def compute_total_emissions(ei, es, ec, params, rc, mkt, e_efs, p_efs, recipes, rd, fd):
    ei2 = copy.deepcopy(ei)
    adjust_blast_furnace_intensity(ei2, es, params)
    adjust_process_gas_intensity('Coke Production','process_gas_coke',ei2,es,params)
    pl = calculate_balance(recipes, fd, rc)
    ie = calculate_internal_electricity(pl, rd, params)
    eb = calculate_energy_balance(pl, ei2, es)
    eb = adjust_energy_balance(eb, ie)
    gas_c = pl.get('Coke Production',0)*rd['Coke Production'].outputs.get('Process Gas',0)
    gas_b = (params.bf_adj_intensity-params.bf_base_intensity)*pl.get('Blast Furnace',0)
    tg = gas_c+gas_b
    EF_c = e_efs.get('Coal',0)*gas_c\   
    shares = es.get('Blast Furnace',{})
    S2=sum(shares.get(c,0)*e_efs.get(c,0) for c in ['Coal','Coke'])
    EF_b = S2/sum(shares.get(c,0) for c in ['Coal','Coke']) if tg>0 else 0
    EF_pg=(EF_c+gas_b*EF_b)/tg if tg>0 else 0
    df = calculate_emissions(pl, eb, e_efs, p_efs, ie, tg, EF_pg)
    return df.loc['TOTAL','TOTAL CO2e']

def run_sensitivity(ei, es, ec, params, rc, mkt, e_efs, p_efs, recipes, rd, fd):
    base = compute_total_emissions(ei, es, ec, params, rc, mkt, e_efs, p_efs, recipes, rd, fd)
    d=0.01; recs=[]
    for nm,mp in [('energy_int',ei),('energy_content',ec),('e_efs',e_efs),('p_efs',p_efs)]:
        for k,v in mp.items():
            mp2=copy.deepcopy(mp); mp2[k]=v*(1+d)
            args={
                'ei':ei,'es':es,'ec':ec,'params':params,'rc':rc,'mkt':mkt,
                'e_efs':e_efs,'p_efs':p_efs,'recipes':recipes,'rd':rd,'fd':fd
            }
            args[nm]=mp2
            t=compute_total_emissions(**args)
            recs.append({'parameter':f"{nm}['{k}']",'sensitivity':(t/base-1)/d})
    for attr in ['process_gas','process_gas_coke']:
        if hasattr(params,attr):
            val=getattr(params,attr)
            p2=copy.deepcopy(params); setattr(p2,attr,val*(1+d))
            args={'ei':ei,'es':es,'ec':ec,'params':p2,'rc':rc,'mkt':mkt,
                  'e_efs':e_efs,'p_efs':p_efs,'recipes':recipes,'rd':rd,'fd':fd}
            t=compute_total_emissions(**args)
            recs.append({'parameter':f"params.{attr}",'sensitivity':(t/base-1)/d})
    return pd.DataFrame(recs).set_index('parameter')

# ---------------- Main ----------------
if __name__=='__main__':
    base='data'
    ei=load_data_from_yaml(os.path.join(base,'energy_int.yml'))
    es=load_data_from_yaml(os.path.join(base,'energy_matrix.yml'))
    ec=load_data_from_yaml(os.path.join(base,'energy_content.yml'))
    params=load_parameters(os.path.join(base,'parameters.yml'))
    recipes=load_recipes(os.path.join(base,'recipes.yml'),params,ei,es,ec)
    rd={r.name:r for r in recipes}
    rc=load_data_from_yaml(os.path.join(base,'route_config.yml'))
    mkt=load_market_config(os.path.join(base,'mkt_config.yml'))
    e_efs=load_data_from_yaml(os.path.join(base,'emission_factors.yml'))
    p_efs=load_data_from_yaml(os.path.join(base,'process_emissions.yml'))
    fd={'Finished Steel':1000.0}

    adjust_blast_furnace_intensity(ei, es, params)
    adjust_process_gas_intensity('Coke Production','process_gas_coke',ei,es,params)

    pl=calculate_balance(recipes,fd,rc)
    ie=calculate_internal_electricity(pl,rd,params)
    eb=calculate_energy_balance(pl,ei,es)
    eb=adjust_energy_balance(eb,ie)

    gas_c=pl.get('Coke Production',0)*rd['Coke Production'].outputs.get('Process Gas',0)
    gas_b=(params.bf_adj_intensity-params.bf_base_intensity)*pl.get('Blast Furnace',0)
    tg=gas_c+gas_b
    EF_c=e_efs.get('Coal',0)*gas_c
    shares=es.get('Blast Furnace',{})
    Ssum=sum(shares.get(c,0)*e_efs.get(c,0) for c in ['Coal','Coke'])
    EF_b=Ssum/sum(shares.get(c,0) for c in ['Coal','Coke']) if tg>0 else 0
    EF_pg=(EF_c+gas_b*EF_b)/tg if tg>0 else 0

    emissions_df=calculate_emissions(pl,eb,e_efs,p_efs,ie,tg,EF_pg)
    print("\n--- Emissions (kg CO2e) ---")
    print(emissions_df)

    sens_df=run_sensitivity(ei,es,ec,params,rc,mkt,e_efs,p_efs,recipes,rd,fd)
    top10=sens_df['sensitivity'].abs().sort_values(ascending=False).head(10)
    print("\n--- Top 10 Parameter Sensitivities (Elasticity) ---")
    print(top10)

    plt.figure(figsize=(8,6))
    plt.barh(top10.index, top10.values)
    plt.xlabel('Elasticity')
    plt.title('Top 10 Parameter Sensitivities on Total CO2e')
    plt.tight_layout()
    plt.show()
