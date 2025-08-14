# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 16:29:30 2025

@author: rafae
"""

# -*- coding: utf-8 -*-
"""
global_sensitivity.py
Run Sobol or LHS sensitivity on curated levers for your steel model.

Usage examples:
  # Sobol, 10 levers, N=256 (≈ (2D+2)N = 5632 runs per (route,country))
  python global_sensitivity.py --levers data/levers.yml \
    --routes BF-BOF,EAF-Scrap,DRI-EAF --countries BRA --method sobol --N 256 --plot

  # Faster LHS screening (SRC/PRCC-style), 400 samples
  python global_sensitivity.py --levers data/levers.yml \
    --routes BF-BOF --countries BRA --method lhs --N 400 --plot
"""
import os, copy, argparse, yaml, time, math
import numpy as np
import pandas as pd

from SALib.sample import saltelli, latin
from SALib.analyze import sobol

import model99 as ms  # your model

# ---------- route helpers (deterministic) ----------
def _material_picks_for(route_key: str) -> dict:
    if route_key == 'BF-BOF':
        liquid_steel = 'Basic Oxygen Furnace'; pig_iron = 'Blast Furnace'
    elif route_key == 'DRI-EAF':
        liquid_steel = 'Electric Arc Furnace'; pig_iron = 'Direct Reduction Iron'
    elif route_key == 'EAF-Scrap':
        liquid_steel = 'Electric Arc Furnace'; pig_iron = None
    else:
        raise ValueError(f"Unknown route '{route_key}'")

    return {
        'Finished Products':            'No Coating',
        'Manufactured Feed (IP4)':      'Direct use of Basic Steel Products (IP4)',
        'Raw Products (types)':         'Hot Rolling',
        'Cast Steel (IP1)':             'Continuous Casting (R)',
        'Liquid Steel':                 liquid_steel,
        'Pig Iron':                     pig_iron,
        'Coke':                         'Coke Production',
        'Coal':                         'Coal from Market',
        'Oxygen':                       'Oxygen Production',
        'Nitrogen':                     'Nitrogen Production',
        'Burnt Lime':                   'Burnt Lime Production',
        'Dolomite':                     'Dolomite Production',
        'Gas':                          'Natural gas from Market'
    }

def _enforce_unique_picks_in_mask(mask: dict, recipes, picks: dict) -> dict:
    from collections import defaultdict
    producers = defaultdict(list)
    for r in recipes:
        for m in r.outputs:
            producers[m].append(r.name)
    out = dict(mask)
    for material, chosen in picks.items():
        if not chosen:
            continue
        for name in producers.get(material, []):
            allowed = 1 if out.get(name, 1) == 1 else 0
            out[name] = 1 if (name == chosen and allowed == 1) else 0
    return out

FEEDS = {"EAF-Scrap": "scrap", "DRI-EAF": "dri", "BF-BOF": None}

# ---------- levers ----------
def _load_levers(path):
    data = yaml.safe_load(open(path, 'r', encoding='utf-8')) or {}
    L = data.get('levers', [])
    cleaned = []
    for i, lv in enumerate(L):
        name = str(lv.get('name') or f"lever_{i+1}")
        fam  = str(lv.get('family'))
        key  = str(lv.get('key'))
        typ  = str(lv.get('type','scale')).lower()
        lo   = float(lv.get('lo'))
        hi   = float(lv.get('hi'))
        if fam not in {'energy_int','e_efs','p_efs','params'}:
            print(f"[WARN] lever '{name}' unknown family → skip")
            continue
        cleaned.append({'name':name,'family':fam,'key':key,'type':typ,'lo':lo,'hi':hi})
    if not cleaned:
        raise ValueError("No valid levers in levers.yml")
    return cleaned

def _get_param_value(params, path):
    # path like "params.process_gas"
    assert path.startswith('params.')
    cur = params
    for p in path.split('.')[1:]:
        cur = getattr(cur, p)
    return float(cur)

def _set_param_value(params, path, value):
    parts = path.split('.')[1:]
    cur = params
    for i, p in enumerate(parts):
        if i == len(parts)-1:
            setattr(cur, p, float(value))
        else:
            cur = getattr(cur, p)

def _resolve_bounds(levers, params, energy_int, e_efs, p_efs):
    """
    Returns problem dict for SALib and a converter from X->mutator
    For scale-type levers, bounds are [base*lo, base*hi]; abs-type are [lo, hi].
    """
    names, bounds, basevals = [], [], []
    for lv in levers:
        fam, key, typ = lv['family'], lv['key'], lv['type']
        lo, hi = float(lv['lo']), float(lv['hi'])
        if fam == 'energy_int':
            v0 = float(energy_int.get(key, np.nan))
        elif fam == 'e_efs':
            v0 = float(e_efs.get(key, np.nan))
        elif fam == 'p_efs':
            v0 = float(p_efs.get(key, np.nan))
        else:  # params
            v0 = _get_param_value(params, key)
        if not math.isfinite(v0):
            raise ValueError(f"Base value not found for lever {lv['name']} ({fam}:{key})")
        if typ == 'scale':
            lo_b, hi_b = v0*lo, v0*hi
        else:
            lo_b, hi_b = lo, hi
        names.append(lv['name'])
        bounds.append([lo_b, hi_b])
        basevals.append(v0)

    problem = {'num_vars': len(names), 'names': names, 'bounds': bounds}
    def mutator_from_vector(vec):
        # vec is a list/array of absolute values within bounds
        def mutate(params_r, energy_int_r, e_efs_r, p_efs_r, energy_shares_r, energy_content_r):
            for x, lv in zip(vec, levers):
                fam, key = lv['family'], lv['key']
                if fam == 'energy_int':
                    energy_int_r[key] = float(x)
                elif fam == 'e_efs':
                    e_efs_r[key] = float(x)
                elif fam == 'p_efs':
                    p_efs_r[key] = float(x)
                else:
                    _set_param_value(params_r, key, float(x))
        return mutate
    return problem, mutator_from_vector

# ---------- one deterministic solve ----------
def _solve_once(country_code, route_key, demand, stage_key, base_objs, elec_idx,
                mutator=None):
    (params, energy_int, energy_shares, energy_content, e_efs, p_efs, recipes_base) = base_objs

    # Deep copies
    params_r         = copy.deepcopy(params)
    energy_int_r     = copy.deepcopy(energy_int)
    energy_shares_r  = copy.deepcopy(energy_shares)
    energy_content_r = copy.deepcopy(energy_content)
    e_efs_r          = copy.deepcopy(e_efs)
    p_efs_r          = copy.deepcopy(p_efs)
    recipes_r        = copy.deepcopy(recipes_base)
    rec_dict         = {r.name: r for r in recipes_r}

    # Country electricity EF override
    cc_map = ms.load_electricity_intensity(os.path.join('data','electricity_intensity.yml'))
    code = (country_code or '').upper()
    if code in cc_map:
        e_efs_r['Electricity'] = float(cc_map[code])

    # External mutation (levers)
    if mutator:
        mutator(params_r, energy_int_r, e_efs_r, p_efs_r, energy_shares_r, energy_content_r)

    # Intensity adjustments
    if hasattr(ms, 'adjust_blast_furnace_intensity'):
        ms.adjust_blast_furnace_intensity(energy_int_r, energy_shares_r, params_r)
    if hasattr(ms, 'adjust_process_gas_intensity'):
        ms.adjust_process_gas_intensity('Coke Production', 'process_gas_coke',
                                        energy_int_r, energy_shares_r, params_r)

    # Enforce EAF feed
    mode = FEEDS.get(route_key)
    if mode and hasattr(ms, "enforce_eaf_feed"):
        ms.enforce_eaf_feed(recipes_r, mode)

    # Deterministic route (no prompts)
    demand_node = ms.STAGE_MATS[stage_key]
    base_mask   = ms.build_route_mask(route_key, recipes_r)
    pre_mask    = _enforce_unique_picks_in_mask(base_mask, recipes_r, _material_picks_for(route_key))
    prod_routes = ms.build_routes_interactive(recipes_r, demand_node,
                                              pre_select=None, pre_mask=pre_mask, interactive=False)

    # Solve
    final_demand = {demand_node: float(demand)}
    balance_matrix, prod_lvl = ms.calculate_balance_matrix(recipes_r, final_demand, prod_routes)
    if balance_matrix is None or not prod_lvl:
        return float('nan')

    # ensure variant energy rows
    active = [p for p, r in prod_lvl.items() if r > 1e-12]
    if hasattr(ms, 'expand_energy_tables_for_active'):
        ms.expand_energy_tables_for_active(active, energy_shares_r, energy_int_r)

    # energy & emissions
    internal_elec = 0.0
    if hasattr(ms, 'calculate_internal_electricity'):
        internal_elec = ms.calculate_internal_electricity(prod_lvl, rec_dict, params_r)
    energy_balance = ms.calculate_energy_balance(prod_lvl, energy_int_r, energy_shares_r)
    if hasattr(ms, 'adjust_energy_balance'):
        energy_balance = ms.adjust_energy_balance(energy_balance, internal_elec)

    # recovered gas EF
    bf_adj  = getattr(params_r, 'bf_adj_intensity', 0.0)
    bf_base = getattr(params_r, 'bf_base_intensity', 0.0)
    gas_bf_MJ = max(0.0, (bf_adj - bf_base)) * float(prod_lvl.get('Blast Furnace', 0.0))
    cp_runs = float(prod_lvl.get('Coke Production', 0.0))
    gas_coke_MJ = 0.0
    cp = rec_dict.get('Coke Production')
    if cp and isinstance(cp.outputs, dict):
        gas_coke_MJ = cp_runs * float(cp.outputs.get('Process Gas', 0.0))
    total_gas_MJ = gas_coke_MJ + gas_bf_MJ

    def mix_ef(proc):
        sh = energy_shares_r.get(proc, {}) or {}
        fuels = [f for f,v in sh.items() if f != 'Electricity' and v>0]
        if not fuels: return 0.0
        num = sum(sh[f]*float(e_efs_r.get(f,0.0)) for f in fuels)
        den = sum(sh[f] for f in fuels)
        return num/den if den else 0.0
    EF_coke = mix_ef('Coke Production'); EF_bf = mix_ef('Blast Furnace')
    EF_process_gas = ((EF_coke*gas_coke_MJ + EF_bf*gas_bf_MJ)/total_gas_MJ) if total_gas_MJ>0 else 0.0

    util_eff = rec_dict.get('Utility Plant').outputs.get('Electricity', 0.0) if 'Utility Plant' in rec_dict else 0.0
    internal_elec = total_gas_MJ * float(util_eff)

    emissions = ms.calculate_emissions(
        mkt_cfg=None,
        prod_level=prod_lvl,
        energy_df=energy_balance,
        energy_efs=e_efs_r,
        process_efs=p_efs_r,
        internal_elec=internal_elec,
        final_demand=final_demand,
        total_gas_MJ=total_gas_MJ,
        EF_process_gas=EF_process_gas
    )
    if emissions is None or 'TOTAL' not in emissions.index:
        return float('nan')
    return float(emissions.loc['TOTAL','TOTAL CO2e'])

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--levers', required=True, help='Path to levers.yml')
    ap.add_argument('--routes', default='BF-BOF,EAF-Scrap,DRI-EAF')
    ap.add_argument('--countries', default='BRA')
    ap.add_argument('--stage', default='Finished', choices=list(ms.STAGE_MATS.keys()))
    ap.add_argument('--demand', type=float, default=1000.0)

    ap.add_argument('--method', choices=['sobol','lhs'], default='sobol')
    ap.add_argument('--N', type=int, default=256, help='Base sample size (Sobol) or sample count (LHS)')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--save-run', action='store_true')
    args = ap.parse_args()

    BASE = os.path.join('data','')
    # Load base configs
    energy_int      = ms.load_data_from_yaml(os.path.join(BASE,'energy_int.yml'))
    energy_shares   = ms.load_data_from_yaml(os.path.join(BASE,'energy_matrix.yml'))
    energy_content  = ms.load_data_from_yaml(os.path.join(BASE,'energy_content.yml'))
    e_efs           = ms.load_data_from_yaml(os.path.join(BASE,'emission_factors.yml'))
    p_efs           = ms.load_data_from_yaml(os.path.join(BASE,'process_emissions.yml'))
    params          = ms.load_parameters     (os.path.join(BASE,'parameters.yml'))
    recipes         = ms.load_recipes_from_yaml(os.path.join(BASE,'recipes.yml'),
                                                params, energy_int, energy_shares, energy_content)
    base_objs = (params, energy_int, energy_shares, energy_content, e_efs, p_efs, recipes)

    levers = _load_levers(args.levers)
    problem, mutator_from_vector = _resolve_bounds(levers, params, energy_int, e_efs, p_efs)

    # Output dir
    if args.save_run:
        ts = time.strftime('%Y%m%d_%H%M%S')
        outdir = os.path.join('artifacts', f'global_sens_{ts}')
    else:
        outdir = os.path.join('artifacts','dev')
    os.makedirs(outdir, exist_ok=True)

    routes    = [r.strip() for r in args.routes.split(',') if r.strip()]
    countries = [c.strip().upper() for c in args.countries.split(',') if c.strip()]

    for rk in routes:
        for cc in countries:
            print(f"\n[GS] {rk} @ {cc} — method={args.method}, D={problem['num_vars']}, N={args.N}")
            if args.method == 'sobol':
                X = saltelli.sample(problem, args.N, calc_second_order=False)
            else:  # LHS
                X = latin.sample(problem, args.N)

            # Evaluate
            Y = np.zeros(X.shape[0], dtype=float)
            for i, row in enumerate(X):
                mut = mutator_from_vector(row)
                Y[i] = _solve_once(cc, rk, args.demand, args.stage, base_objs, None, mutator=mut)
                if (i+1) % max(1, X.shape[0]//10) == 0:
                    print(f"  progress {i+1}/{X.shape[0]}")

            if args.method == 'sobol':
                Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
                df = pd.DataFrame({
                    'lever': problem['names'],
                    'S1': Si['S1'], 'S1_conf': Si['S1_conf'],
                    'ST': Si['ST'], 'ST_conf': Si['ST_conf'],
                }).sort_values('ST', ascending=False)
                out_csv = os.path.join(outdir, f"sens_sobol_{rk.replace('/','-')}_{cc}.csv")
                df.to_csv(out_csv, index=False)
                print("[GS] Saved:", out_csv)

                if args.plot:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    top = df.head(20)
                    fig, ax = plt.subplots(figsize=(10, max(4, 0.4*len(top))))
                    ax.barh(top['lever'], top['ST'])
                    ax.invert_yaxis()
                    ax.set_title(f"Sobol Total-order (ST) — {rk} @ {cc}")
                    ax.set_xlabel("ST")
                    plt.tight_layout()
                    out_png = os.path.join(outdir, f"sens_sobol_ST_{rk.replace('/','-')}_{cc}.png")
                    plt.savefig(out_png, dpi=180); plt.close()
                    print("[GS] Saved:", out_png)

            else:
                # LHS screening: report standardized rank correlations (SRC) & PRCC
                dfX = pd.DataFrame(X, columns=problem['names'])
                df = pd.DataFrame({'Y': Y})
                # SRC: standardized regression coefficients (quick proxy)
                Xn = (dfX - dfX.mean())/dfX.std(ddof=0)
                Yn = (df['Y'] - df['Y'].mean())/df['Y'].std(ddof=0)
                beta = np.linalg.lstsq(Xn.values, Yn.values, rcond=None)[0]
                SRC = pd.Series(beta, index=dfX.columns)
                # PRCC: partial rank correlation (simple implementation)
                RX = dfX.rank()  # rank-transform
                RY = df.rank()
                # regress each variable out of others, then correlate residuals with Y residuals
                from numpy.linalg import lstsq
                prcc = {}
                for col in RX.columns:
                    others = [c for c in RX.columns if c != col]
                    # Residual of Xi | X_others
                    b_x = lstsq(RX[others].values, RX[col].values, rcond=None)[0]
                    res_x = RX[col].values - RX[others].values.dot(b_x)
                    # Residual of Y | X_others
                    b_y = lstsq(RX[others].values, RY['Y'].values, rcond=None)[0]
                    res_y = RY['Y'].values - RX[others].values.dot(b_y)
                    prcc[col] = np.corrcoef(res_x, res_y)[0,1]
                df_out = pd.DataFrame({
                    'lever': problem['names'],
                    'SRC': SRC[problem['names']].values,
                    'PRCC': [prcc[n] for n in problem['names']]
                }).sort_values('PRCC', ascending=False)
                out_csv = os.path.join(outdir, f"sens_lhs_{rk.replace('/','-')}_{cc}.csv")
                df_out.to_csv(out_csv, index=False)
                print("[GS] Saved:", out_csv)

                if args.plot:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    top = df_out.head(20)
                    fig, ax = plt.subplots(figsize=(10, max(4, 0.4*len(top))))
                    ax.barh(top['lever'], top['PRCC'])
                    ax.invert_yaxis()
                    ax.set_title(f"LHS PRCC — {rk} @ {cc}")
                    ax.set_xlabel("PRCC")
                    plt.tight_layout()
                    out_png = os.path.join(outdir, f"sens_lhs_PRCC_{rk.replace('/','-')}_{cc}.png")
                    plt.savefig(out_png, dpi=180); plt.close()
                    print("[GS] Saved:", out_png)

    print("[GS] done.")

if __name__ == '__main__':
    main()
