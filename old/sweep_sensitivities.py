# -*- coding: utf-8 -*-
"""
sweep_sensitivities.py
Sensitivity & ranking toolbox for the steel model.

What it does:
  1) Route baseline comparison (no parameter change)     → --route-first
  2) Parameter sweeps (elec_ef, bf_intensity, ...)       → --param one|a,b,c|ALL
  3) Route+country parameter impact ranking (tornado)    → --rank

Outputs are organized per run in artifacts/sweep_<timestamp>/...

Requires: model99.py (imported as ms) in the same folder.
"""

import os, time, math, copy, argparse, datetime, yaml
import pandas as pd
from collections import defaultdict

import model99 as ms  # your model entry point

# -------------------------
# Config
# -------------------------
PARAM_KEYS = ['elec_ef','bf_intensity','coke_intensity','yield']


# -------------------------
# Small utilities
# -------------------------
def is_number(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def load_elec_idx(base_dir):
    """
    Read data/electricity_intensity.yml → {ISO3: intensity}.
    """
    path = os.path.join(base_dir, 'electricity_intensity.yml')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        items = data.get('electricity_intensity', [])
        out = {}
        for it in items:
            try:
                code = str(it['code']).upper()
                val  = float(it['intensity'])
                out[code] = val
            except Exception:
                continue
        if not out:
            print(f"[WARN] electricity_intensity.yml parsed but empty at {path}")
        return out
    except FileNotFoundError:
        print(f"[WARN] electricity_intensity.yml not found at {path}")
        return {}
    except Exception as e:
        print(f"[WARN] cannot load electricity_intensity.yml: {e}")
        return {}


def expand_energy_variants(active, energy_shares, energy_int):
    """If your model supports variant expansion (e.g., CC R/L/H), call it."""
    fn = getattr(ms, 'expand_energy_tables_for_active', None)
    if callable(fn):
        fn(active, energy_shares, energy_int)


def _mk_run_dir(base='artifacts'):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base, f"sweep_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _mk_param_dir(run_dir, param_key):
    d = os.path.join(run_dir, param_key)
    os.makedirs(d, exist_ok=True)
    return d


# -------------------------
# Non-interactive route picks
# -------------------------
def _material_picks_for(route_key: str) -> dict:
    """Your desired producer per material (by route)."""
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

    out = dict(mask)  # copy
    for material, chosen_proc in picks.items():
        if not chosen_proc:
            continue
        for name in producers.get(material, []):
            # keep 0 if already 0; only set 1 if both (chosen) AND currently allowed
            allowed = 1 if out.get(name, 1) == 1 else 0
            out[name] = 1 if (name == chosen_proc and allowed == 1) else 0
    return out


# -------------------------
# Core single run (sweep)
# -------------------------
def run_one(country_code, route_key, demand, stage_key,
            param_key, multiplier, base_objs, elec_idx):
    """
    One non-interactive run.
    - country_code: ISO3 electricity EF selector (overrides e_efs['Electricity'] if present)
    - route_key: 'BF-BOF' | 'EAF-Scrap' | 'DRI-EAF'
    - param_key: 'elec_ef' | 'bf_intensity' | 'coke_intensity' | 'yield' | 'baseline'
    - multiplier: numeric sensitivity factor
    - base_objs: (params, energy_int, energy_shares, energy_content, e_efs, p_efs, recipes_list_msprocess)
    """
    (params, energy_int, energy_shares, energy_content,
     e_efs, p_efs, recipes_base) = base_objs

    # Deep copies to isolate this run
    params_r         = copy.deepcopy(params)
    energy_int_r     = copy.deepcopy(energy_int)
    energy_shares_r  = copy.deepcopy(energy_shares)
    energy_content_r = copy.deepcopy(energy_content)
    e_efs_r          = copy.deepcopy(e_efs)
    p_efs_r          = copy.deepcopy(p_efs)
    recipes_r        = copy.deepcopy(recipes_base)
    rec_dict         = {r.name: r for r in recipes_r}

    # Country electricity EF override
    cc = (country_code or '').upper()
    if cc in elec_idx:
        e_efs_r['Electricity'] = float(elec_idx[cc])

    # Apply sensitivity before intensity adjustments
    if param_key == 'elec_ef':
        if 'Electricity' in e_efs_r and is_number(multiplier):
            e_efs_r['Electricity'] *= float(multiplier)
    elif param_key == 'bf_intensity' and 'Blast Furnace' in energy_int_r:
        energy_int_r['Blast Furnace'] *= float(multiplier)
    elif param_key == 'coke_intensity' and 'Coke Production' in energy_int_r:
        energy_int_r['Coke Production'] *= float(multiplier)
    # 'yield' handled after emissions; 'baseline' → no changes

    # Intensity adjustments (your model’s mechanics)
    if hasattr(ms, 'adjust_blast_furnace_intensity'):
        ms.adjust_blast_furnace_intensity(energy_int_r, energy_shares_r, params_r)
    if hasattr(ms, 'adjust_process_gas_intensity'):
        ms.adjust_process_gas_intensity('Coke Production', 'process_gas_coke',
                                        energy_int_r, energy_shares_r, params_r)

    # Deterministic route resolution (NO prompts)
    demand_node = ms.STAGE_MATS[stage_key]
    base_mask   = ms.build_route_mask(route_key, recipes_r)
    pre_mask    = _enforce_unique_picks_in_mask(base_mask, recipes_r, _material_picks_for(route_key))
    prod_routes = ms.build_routes_interactive(
        recipes_r, demand_node, pre_select=None, pre_mask=pre_mask, interactive=False
    )

    # Solve material balance
    final_demand = {demand_node: float(demand)}
    balance_matrix, prod_lvl = ms.calculate_balance_matrix(recipes_r, final_demand, prod_routes)
    if balance_matrix is None or not prod_lvl:
        raise RuntimeError("No balance (missing upstream path)")

    # Ensure variant rows (e.g., CC R/L/H) exist in energy tables
    active = [p for p, r in prod_lvl.items() if r > 1e-12]
    expand_energy_variants(active, energy_shares_r, energy_int_r)

    # Internal electricity (use your model helper)
    internal_elec = 0.0
    if hasattr(ms, 'calculate_internal_electricity'):
        internal_elec = ms.calculate_internal_electricity(prod_lvl, rec_dict, params_r)

    energy_balance = ms.calculate_energy_balance(prod_lvl, energy_int_r, energy_shares_r)
    if hasattr(ms, 'adjust_energy_balance'):
        energy_balance = ms.adjust_energy_balance(energy_balance, internal_elec)

    # Recovered process gas EF for emissions calc
    bf_adj  = getattr(params_r, 'bf_adj_intensity', 0.0)
    bf_base = getattr(params_r, 'bf_base_intensity', 0.0)
    gas_bf_MJ = max(0.0, (bf_adj - bf_base)) * float(prod_lvl.get('Blast Furnace', 0.0))

    gas_coke_MJ = 0.0
    cp_runs = float(prod_lvl.get('Coke Production', 0.0))
    cp = rec_dict.get('Coke Production')
    if cp and isinstance(cp.outputs, dict):
        val = cp.outputs.get('Process Gas', 0.0)
        if isinstance(val, (int, float)) and val > 0:
            gas_coke_MJ = cp_runs * float(val)
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

    total = float('nan')
    if emissions is not None and 'TOTAL' in emissions.index:
        total = float(emissions.loc['TOTAL', 'TOTAL CO2e'])

    # Yield sensitivity: multiplier as kept fraction (e.g., 0.85) → emissions per final unit ~ 1 / yield
    if param_key == 'yield' and is_number(multiplier) and float(multiplier) != 0.0:
        total = total / float(multiplier)

    return dict(country=country_code, route=route_key, stage=stage_key,
                param=param_key, multiplier=float(multiplier), demand=float(demand),
                total_CO2e_kg=total)


# -------------------------
# Plotting (headless)
# -------------------------
def _humanize_kg(x):
    x = float(x)
    if x >= 1e12: return f"{x/1e12:.2f} Tt"
    if x >= 1e9:  return f"{x/1e9:.2f} Gt"
    if x >= 1e6:  return f"{x/1e6:.2f} Mt"
    if x >= 1e3:  return f"{x/1e3:.2f} kt"
    return f"{x:.0f} kg"


def plot_route_baseline(df, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np, matplotlib.pyplot as plt

    if df.empty:
        print("[BASELINE] no data to plot")
        return

    mults = sorted(df['multiplier'].dropna().unique().tolist())
    base_m = min(mults, key=lambda m: abs(m-1.0)) if mults else None
    if base_m is None:
        print("[BASELINE] no multiplier column found")
        return

    pv = (df[df['multiplier']==base_m]
            .pivot_table(index='country', columns='route', values='total_CO2e_kg', aggfunc='mean'))
    if pv.empty:
        print("[BASELINE] pivot empty")
        return
    pv = pv.sort_index()

    fig, ax = plt.subplots(figsize=(12, max(4, 0.5*len(pv))))
    pv.plot(kind='barh', ax=ax)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Total CO₂e (kg)")
    ax.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print("[BASELINE] Saved:", out_png)


def make_param_plots(df, args, outdir, param_key):
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np, matplotlib.pyplot as plt

    def annotate_bars(ax, bars, values):
        xmax = ax.get_xlim()[1] or 1.0
        for b, v in zip(bars, values):
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                continue
            x = float(b.get_width()); y = b.get_y() + b.get_height()/2
            txt = _humanize_kg(v); pad = xmax * 0.01
            if x >= xmax * 0.18:
                ax.text(x - pad, y, txt, va='center', ha='right', fontsize=9, color='white')
            else:
                ax.text(x + pad, y, txt, va='center', ha='left', fontsize=9)

    routes_u = df['route'].dropna().unique().tolist()
    mults = sorted(df['multiplier'].dropna().unique().tolist())
    base_m = min(mults, key=lambda m: abs(m-1.0)) if mults else None

    cmap = plt.get_cmap('viridis')
    colors = {m: cmap(i/max(1, len(mults)-1)) for i, m in enumerate(sorted(mults))}

    for rk in routes_u:
        sub = df[df['route'] == rk].copy()
        if sub.empty: 
            continue

        pv = sub.pivot_table(index='country', columns='multiplier',
                             values='total_CO2e_kg', aggfunc='mean')
        pv = pv.replace([np.inf, -np.inf], np.nan)
        pv = pv.reindex(columns=sorted(pv.columns))

        sort_col = base_m if (base_m in pv.columns) else (pv.columns[0] if len(pv.columns) else None)
        if sort_col is not None:
            pv = pv.sort_values(by=sort_col, ascending=True)

        # A) grouped bars
        fig, ax = plt.subplots(figsize=(12, max(4, 0.4*len(pv))))
        y = np.arange(len(pv.index)); k = max(1, len(pv.columns))
        bar_h = 0.8 / k; offset0 = -(k-1) * bar_h / 2.0
        xmax = float(np.nanmax(pv.values)) if np.isfinite(pv.values).any() else 1.0
        ax.set_xlim(0, xmax * 1.15)

        for i, m in enumerate(pv.columns):
            vals = pv[m].values.astype(float)
            bars = ax.barh(y + offset0 + i*bar_h, np.nan_to_num(vals, nan=0.0),
                           height=bar_h, label=f"{m:g}×", color=colors.get(m))
            annotate_bars(ax, bars, vals)

        ax.set_yticks(y); ax.set_yticklabels(pv.index); ax.invert_yaxis()
        ttl = f"{rk} — sensitivity: {param_key}"
        if base_m in pv.columns: ttl += f" (baseline = {base_m:g}×)"
        ax.set_title(ttl); ax.set_xlabel(f"Total CO₂e per {int(args.demand)} {args.stage}")
        ax.legend(title="Multiplier", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        ax.grid(axis='x', alpha=0.2)
        plt.tight_layout()
        out_bars = os.path.join(outdir, f"{rk.replace('/','-')}_{param_key}_bars.png")
        plt.savefig(out_bars, dpi=180); plt.close()
        print("Saved:", out_bars)

        # B) heatmap ratios
        if base_m in pv.columns:
            ratio = pv.divide(pv[base_m], axis=0); legend_label = f"CO₂ vs {base_m:g}× baseline (×)"
            title_hm = f"{rk} — Ratio to baseline ({base_m:g}× = 1.0)"
        else:
            ratio = pv.divide(pv.min(axis=1), axis=0); legend_label = "CO₂ vs row min (×)"
            title_hm = f"{rk} — Ratio to row minimum (≈1.0)"

        ratio = ratio.replace([np.inf, -np.inf], np.nan).clip(lower=0.5, upper=1.5)
        fig, ax = plt.subplots(figsize=(max(8, 0.7*len(ratio.columns)), max(5, 0.4*len(ratio.index))))
        im = ax.imshow(ratio.values, aspect='auto', cmap='coolwarm', vmin=0.5, vmax=1.5)
        ax.set_xticks(np.arange(len(ratio.columns))); ax.set_xticklabels([f"{m:g}×" for m in ratio.columns])
        ax.set_yticks(np.arange(len(ratio.index)));  ax.set_yticklabels(ratio.index)
        ax.set_title(title_hm)
        for i in range(ratio.shape[0]):
            for j in range(ratio.shape[1]):
                val = ratio.iat[i, j]
                if not (isinstance(val, float) and math.isnan(val)):
                    ax.text(j, i, f"{val:,.2f}×", ha='center', va='center', fontsize=8,
                            color='black' if 0.9 <= val <= 1.1 else 'white')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label(legend_label)
        plt.tight_layout()
        out_hm = os.path.join(outdir, f"{rk.replace('/','-')}_{param_key}_heatmap.png")
        plt.savefig(out_hm, dpi=180); plt.close()
        print("Saved:", out_hm)


# -------------------------
# Batch sweeps
# -------------------------
def run_param_sweep(args, param_key, base_objs, elec_idx, run_dir):
    (params, energy_int, energy_shares, energy_content, e_efs, p_efs, recipes) = base_objs

    routes = [r.strip() for r in args.routes.split(',') if r.strip()]
    if args.countries.upper() == 'ALL':
        countries = sorted(elec_idx.keys()) or ['BRA']
    else:
        countries = [c.strip().upper() for c in args.countries.split(',') if c.strip()]
    grid = [float(x) for x in args.grid.split(',') if x.strip()]
    if not grid: grid = [1.0]

    combos = [(cc, rk, m) for cc in countries for rk in routes for m in grid]
    print(f"[INFO] [{param_key}] {len(countries)} countries × {len(routes)} routes × {len(grid)} mults = {len(combos)} runs")
    if param_key == 'yield' and any(m > 1.0 for m in grid):
        print("[WARN] yield multipliers > 1.0 — yield is a fraction; >1 decreases emissions")

    results = []
    for (cc, rk, m) in combos:
        try:
            print(f"[RUN {param_key}] {cc} — {rk} × {m} …", end='', flush=True)
            rec = run_one(cc, rk, args.demand, args.stage, param_key, m, base_objs, elec_idx)
            results.append(rec)
            print(" ok")
        except Exception as e:
            results.append(dict(country=cc, route=rk, stage=args.stage,
                                param=param_key, multiplier=m, demand=args.demand,
                                total_CO2e_kg=float('nan'), note=f'ERROR: {e}'))
            print(f" ERR: {e}")

    df = pd.DataFrame(results).sort_values(['route','country','multiplier'])
    outdir_param = _mk_param_dir(run_dir, param_key)
    out_csv = os.path.join(outdir_param, f"sensitivity_{param_key}.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    if args.plot:
        make_param_plots(df, args, outdir_param, param_key)

    return df


# -------------------------
# Ranking (where action matters most)
# -------------------------
from types import SimpleNamespace

def _walk_ns(ns, prefix="params"):
    """Yield (path, value) numeric leaves from a SimpleNamespace tree."""
    for k, v in vars(ns).items():
        p = f"{prefix}.{k}"
        if isinstance(v, SimpleNamespace):
            yield from _walk_ns(v, p)
        else:
            if isinstance(v, (int, float)) and v is not None:
                yield (p, float(v))


def _set_ns(ns, path, new_val):
    """Set a dot-path in SimpleNamespace, e.g., params.blend.sinter."""
    parts = path.split('.')[1:]  # drop leading 'params'
    cur = ns
    for i, key in enumerate(parts):
        if i == len(parts)-1:
            setattr(cur, key, float(new_val))
        else:
            cur = getattr(cur, key)


def _solve_once(country_code, route_key, demand, stage_key, base_objs, elec_idx,
                mutator=None):
    """
    Run the model once, optionally mutating configs BEFORE adjustments and route resolution.
    Returns (total, prod_lvl, energy_balance, recipes_map).
    """
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
    cc = (country_code or '').upper()
    if cc in elec_idx:
        e_efs_r['Electricity'] = float(elec_idx[cc])

    # External mutation hook
    if mutator:
        mutator(params_r, energy_int_r, e_efs_r, p_efs_r, energy_shares_r, energy_content_r)

    # Intensity adjustments
    if hasattr(ms, 'adjust_blast_furnace_intensity'):
        ms.adjust_blast_furnace_intensity(energy_int_r, energy_shares_r, params_r)
    if hasattr(ms, 'adjust_process_gas_intensity'):
        ms.adjust_process_gas_intensity('Coke Production', 'process_gas_coke',
                                        energy_int_r, energy_shares_r, params_r)

    # Deterministic route (no prompts)
    demand_node = ms.STAGE_MATS[stage_key]
    base_mask   = ms.build_route_mask(route_key, recipes_r)
    pre_mask    = _enforce_unique_picks_in_mask(base_mask, recipes_r, _material_picks_for(route_key))
    prod_routes = ms.build_routes_interactive(
        recipes_r, demand_node, pre_select=None, pre_mask=pre_mask, interactive=False
    )

    # Solve
    final_demand = {demand_node: float(demand)}
    balance_matrix, prod_lvl = ms.calculate_balance_matrix(recipes_r, final_demand, prod_routes)
    if balance_matrix is None or not prod_lvl:
        return (float('nan'), {}, pd.DataFrame(), rec_dict)

    active = [p for p, r in prod_lvl.items() if r > 1e-12]
    expand_energy_variants(active, energy_shares_r, energy_int_r)

    internal_elec = 0.0
    if hasattr(ms, 'calculate_internal_electricity'):
        internal_elec = ms.calculate_internal_electricity(prod_lvl, rec_dict, params_r)

    energy_balance = ms.calculate_energy_balance(prod_lvl, energy_int_r, energy_shares_r)
    if hasattr(ms, 'adjust_energy_balance'):
        energy_balance = ms.adjust_energy_balance(energy_balance, internal_elec)

    # Recovered gas EF
    bf_adj  = getattr(params_r, 'bf_adj_intensity', 0.0)
    bf_base = getattr(params_r, 'bf_base_intensity', 0.0)
    gas_bf_MJ = max(0.0, (bf_adj - bf_base)) * float(prod_lvl.get('Blast Furnace', 0.0))

    gas_coke_MJ = 0.0
    cp_runs = float(prod_lvl.get('Coke Production', 0.0))
    cp = rec_dict.get('Coke Production')
    if cp and isinstance(cp.outputs, dict):
        val = cp.outputs.get('Process Gas', 0.0)
        if isinstance(val, (int, float)) and val > 0:
            gas_coke_MJ = cp_runs * float(val)
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
    total = float(emissions.loc['TOTAL','TOTAL CO2e']) if (emissions is not None and 'TOTAL' in emissions.index) else float('nan')
    return (total, prod_lvl, energy_balance, rec_dict)


def _enumerate_candidates(families, params, energy_int, e_efs, p_efs, active_procs, energy_balance):
    """
    Return list of candidates as dicts:
    {'family': 'energy_int'|'e_efs'|'p_efs'|'params', 'key': name/path, 'value': float}
    Focus only on things that can influence the *active* route.
    """
    fams = {f.strip() for f in families.split(',') if f.strip()}
    cands = []

    if 'energy_int' in fams:
        for proc in active_procs:
            v = energy_int.get(proc, None)
            if isinstance(v, (int,float)) and v != 0:
                cands.append({'family':'energy_int','key':proc,'value':float(v)})

    if 'e_efs' in fams:
        used = [c for c in energy_balance.columns if c != 'TOTAL']
        for carrier in used:
            v = e_efs.get(carrier, None)
            if isinstance(v, (int,float)) and v != 0:
                cands.append({'family':'e_efs','key':carrier,'value':float(v)})

    if 'p_efs' in fams:
        for proc in active_procs:
            v = p_efs.get(proc, None)
            if isinstance(v, (int,float)) and v != 0:
                cands.append({'family':'p_efs','key':proc,'value':float(v)})

    if 'params' in fams and isinstance(params, SimpleNamespace):
        for path, v in _walk_ns(params):
            if path.endswith('_base_intensity') or path.endswith('_adj_intensity'):
                continue
            if v != 0:
                cands.append({'family':'params','key':path,'value':float(v)})

    # de-duplicate
    uniq = {}
    for c in cands:
        uniq[(c['family'], c['key'])] = c
    return list(uniq.values())


def _mutator_for(candidate, new_val):
    """Return a function that applies the override to the deep-copied objects."""
    fam, key = candidate['family'], candidate['key']
    def fn(params_r, energy_int_r, e_efs_r, p_efs_r, energy_shares_r, energy_content_r):
        if fam == 'energy_int':
            energy_int_r[key] = float(new_val)
        elif fam == 'e_efs':
            e_efs_r[key] = float(new_val)
        elif fam == 'p_efs':
            p_efs_r[key] = float(new_val)
        elif fam == 'params':
            _set_ns(params_r, key, float(new_val))
    return fn


def _tornado_plot(df_rank, out_png, title, topk=20):
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np, matplotlib.pyplot as plt

    if df_rank is None or df_rank.empty:
        return

    top = df_rank.sort_values('abs_delta_kg', ascending=True).tail(topk)  # smallest at top
    if top.empty:
        return

    # Build labels even if the 'label' column doesn't exist
    labels = (top.get('label') if 'label' in top.columns
              else (top['family'].astype(str) + ' :: ' + top['key'].astype(str)))

    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(12, max(4, 0.45*len(top))))
    bars = ax.barh(y, top['delta_kg_at_eps'], height=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_title(title)
    ax.set_xlabel(f"Δ CO₂e (kg) for +{top['eps'].iloc[0]*100:.1f}% parameter change")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180); plt.close()



# -------------------------
# Main
# -------------------------
def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('--countries', default='ALL', help="CSV like 'BRA,USA' or ALL (from electricity_intensity.yml)")
    ap.add_argument('--routes', default='BF-BOF,EAF-Scrap,DRI-EAF')
    ap.add_argument('--stage', default='Finished', choices=list(ms.STAGE_MATS.keys()))
    ap.add_argument('--demand', type=float, default=1000.0)

    ap.add_argument('--param', default='elec_ef',
                    help="One key, comma list, 'ALL' for all, or 'NONE'")
    ap.add_argument('--grid', default='0.75,1.0,1.25', help='comma-separated numbers')
    ap.add_argument('--route-first', action='store_true', help='Run baseline-by-route (multiplier=1) before sweeps')
    ap.add_argument('--plot', action='store_true')

    # Ranking options
    ap.add_argument('--rank', action='store_true',
                    help='Rank parameters by impact for each (route,country)')
    ap.add_argument('--eps', type=float, default=0.05,
                    help='Relative step for sensitivity (e.g., 0.05 = +/−5%)')
    ap.add_argument('--topk', type=int, default=20,
                    help='How many top parameters to chart per (route,country)')
    ap.add_argument('--families', default='energy_int,e_efs,p_efs,params',
                    help='Comma list from {energy_int,e_efs,p_efs,params}')
    args = ap.parse_args()

    BASE = os.path.join('data', '')
    elec_idx  = load_elec_idx(BASE)

    # Base data (once)
    energy_int      = ms.load_data_from_yaml(os.path.join(BASE,'energy_int.yml'))
    energy_shares   = ms.load_data_from_yaml(os.path.join(BASE,'energy_matrix.yml'))
    energy_content  = ms.load_data_from_yaml(os.path.join(BASE,'energy_content.yml'))
    e_efs           = ms.load_data_from_yaml(os.path.join(BASE,'emission_factors.yml'))
    p_efs           = ms.load_data_from_yaml(os.path.join(BASE,'process_emissions.yml'))
    params          = ms.load_parameters     (os.path.join(BASE,'parameters.yml'))
    recipes         = ms.load_recipes_from_yaml(os.path.join(BASE,'recipes.yml'),
                                                params, energy_int, energy_shares, energy_content)

    base_objs = (params, energy_int, energy_shares, energy_content, e_efs, p_efs, recipes)
    run_dir = _mk_run_dir('artifacts')
    print("[RUN] outputs →", os.path.abspath(run_dir))

    # 0) Optional: baseline-by-route (multiplier=1 only)
    if args.route_first:
        saved_grid = args.grid
        args.grid = '1.0'
        df_base = run_param_sweep(args, param_key='baseline', base_objs=base_objs, elec_idx=elec_idx, run_dir=run_dir)
        args.grid = saved_grid
        if args.plot:
            out_png = os.path.join(run_dir, 'baseline_routes.png')
            plot_route_baseline(df_base, out_png, title=f"Baseline (×1.0) by Route — {args.stage}")

    # 1) Parameter sweeps?
    if args.param.upper() == 'ALL':
        param_list = PARAM_KEYS
    elif args.param.upper() == 'NONE':
        param_list = []
    else:
        param_list = [p.strip() for p in args.param.split(',') if p.strip()]

    for pkey in param_list:
        run_param_sweep(args, param_key=pkey, base_objs=base_objs, elec_idx=elec_idx, run_dir=run_dir)

    # 2) Ranking?
    if args.rank:
        fams = args.families
        routes = [r.strip() for r in args.routes.split(',') if r.strip()]
        countries = (sorted(elec_idx.keys()) if args.countries.upper()=='ALL'
                     else [c.strip().upper() for c in args.countries.split(',') if c.strip()])
        if not countries: countries = ['BRA']

        rank_dir = os.path.join(run_dir, 'rank'); os.makedirs(rank_dir, exist_ok=True)

        for rk in routes:
            for cc in countries:
                print(f"\n[RANK] {rk} @ {cc} — baseline solve…")
                total0, prod_lvl0, eb0, _ = _solve_once(cc, rk, args.demand, args.stage, base_objs, elec_idx, mutator=None)
                if not isinstance(eb0, pd.DataFrame) or eb0.empty or not prod_lvl0:
                    print(f"[RANK] skip {rk}@{cc}: empty energy balance or prod level")
                    continue
                active = [p for p, r in prod_lvl0.items() if r > 1e-12]

                cands = _enumerate_candidates(fams, base_objs[0], base_objs[1], base_objs[4], base_objs[5],
                                              active, eb0)
                print(f"[RANK] candidates: {len(cands)} (families={fams}) baseline={total0:.2f} kg")

                rows = []
                eps = float(args.eps)

                for c in cands:
                    v0 = c['value']
                    if not math.isfinite(v0):
                        continue
                    # Prefer central difference when positive
                    step = eps * abs(v0) if v0 != 0 else None
                    if step and v0 - step > 0:
                        v_minus = v0 - step
                        v_plus  = v0 + step
                        total_m, *_ = _solve_once(cc, rk, args.demand, args.stage, base_objs, elec_idx,
                                                  mutator=_mutator_for(c, v_minus))
                        total_p, *_ = _solve_once(cc, rk, args.demand, args.stage, base_objs, elec_idx,
                                                  mutator=_mutator_for(c, v_plus))
                        dY = (total_p - total_m) / 2.0
                        dparam = step
                    else:
                        # Forward difference
                        v_plus = v0 * (1.0 + eps) if v0 != 0 else v0 + 1e-6
                        total_p, *_ = _solve_once(cc, rk, args.demand, args.stage, base_objs, elec_idx,
                                                  mutator=_mutator_for(c, v_plus))
                        dY = (total_p - total0)
                        dparam = (v_plus - v0)

                    elasticity = (dY / max(1e-12, total0)) * (v0 / max(1e-12, dparam)) if dparam else float('nan')
                    delta_eps = (dY / max(1e-12, dparam)) * (eps * (abs(v0) if v0!=0 else 1.0))

                    label = f"{c['family']} :: {c['key']}"
                    rows.append({
                        'route': rk, 'country': cc,
                        'family': c['family'], 'key': c['key'],
                        'baseline_CO2e_kg': total0,
                        'param_value': v0,
                        'eps': eps,
                        'delta_kg_at_eps': float(delta_eps),
                        'abs_delta_kg': abs(float(delta_eps)),
                        'elasticity': float(elasticity),
                        'label': f"{c['family']} :: {c['key']}",   # <-- add this
                        'note': ''
                    })

                df_rank = pd.DataFrame(rows).sort_values('abs_delta_kg', ascending=False)
                out_csv = os.path.join(rank_dir, f"rank_{rk.replace('/','-')}_{cc}.csv")
                df_rank.to_csv(out_csv, index=False)
                print("[RANK] Saved:", out_csv)

                _tornado_plot(df_rank, os.path.join(rank_dir, f"rank_{rk.replace('/','-')}_{cc}_tornado.png"),
                              title=f"Top {args.topk} impacts — {rk} @ {cc}", topk=args.topk)

        print("[RANK] done.")

    print(f"[DONE] in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
