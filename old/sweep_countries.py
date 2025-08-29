# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:36:58 2025

@author: rafae
"""

# -*- coding: utf-8 -*-
"""
Sweep countries × routes baseline (non-interactive, fast).
Writes artifacts/country_route_baselines.csv
"""

import os, copy, yaml, time
import pandas as pd

# ---- import your model module (robust name fallback) ----
try:
    import model_scenarios as ms
except ImportError:
    import model99 as ms  # if you kept an alt filename

# ----------------------------
# Helpers
# ----------------------------
def tnow():
    return f"[{time.strftime('%H:%M:%S')}]"

def load_elec_idx(base_dir):
    path = os.path.join(base_dir, 'electricity_intensity.yml')
    try:
        data = yaml.safe_load(open(path, 'r', encoding='utf-8')) or {}
        items = data.get('electricity_intensity', [])
        return {str(it['code']).upper(): float(it['intensity']) for it in items}
    except Exception as e:
        print(tnow(), f"WARNING: cannot load electricity_intensity.yml: {e}")
        return {}

def route_to_bans(route_key):
    """Upstream bans to lock each route (no prompts)."""
    route_key = (route_key or '').strip()
    bans = {}
    if route_key == 'BF-BOF':
        bans.update({'Electric Arc Furnace': 0, 'Direct Reduction Iron': 0})
    elif route_key == 'DRI-EAF':
        bans.update({'Blast Furnace': 0, 'Basic Oxygen Furnace': 0})
    elif route_key == 'EAF-Scrap':
        bans.update({'Blast Furnace': 0, 'Basic Oxygen Furnace': 0, 'Direct Reduction Iron': 0})
    return bans

def postcasting_defaults():
    """Keep downstream comparable & simple."""
    return {
        # choose Rod/bar + Direct use (IP4) + No coating by disabling others
        'Hot Rolling': 1,
        'Ingot Casting': 0,
        'Casting/Extrusion/Conformation': 0,
        'Stamping/calendering/lamination': 0,
        'Machining': 0,
        'Hot Dip Metal Coating FP': 0,
        'Electrolytic Metal Coating FP': 0,
        'Organic or Sintetic Coating (painting)': 0,
        'Cold Rolling': 0,
        'Steel Thermal Treatment': 0,
        'Hot Dip Metal Coating': 0,
        'Electrolytic Metal Coating': 0,
        'Bypass Raw→IP3': 0,
        # left implicitly active (if present): Rod/bar/section Mill, Direct use of Basic Steel Products (IP4), No Coating
    }

def grade_mask(grade='R'):
    """Ban CC variants you don't want (avoid interactive grade choice)."""
    g = (grade or 'R').upper()
    if g == 'R':
        return {'Continuous Casting (L)': 0, 'Continuous Casting (H)': 0}
    if g == 'L':
        return {'Continuous Casting (R)': 0, 'Continuous Casting (H)': 0}
    if g == 'H':
        return {'Continuous Casting (R)': 0, 'Continuous Casting (L)': 0}
    return {}

def build_routes_noninteractive(recipes, demand_mat, bans=None):
    """
    Deterministic, no-prompt builder.
    - Applies bans as explicit zeros in the returned dict.
    - For any material with >1 allowed producers, picks by a preference map.
    - Writes 0 for *all* non-picked producers of that material.
    """
    bans = bans or {}
    producers = {}
    for r in recipes:
        for m in r.outputs:
            producers.setdefault(m, []).append(r)

    prefer = {
        'Cast Steel (IP1)': ['Continuous Casting (R)', 'Continuous Casting (L)', 'Continuous Casting (H)'],
        'Raw Products (types)': ['Rod/bar/section Mill', 'Hot Rolling', 'Ingot Casting'],
        'Manufactured Feed (IP4)': ['Direct use of Basic Steel Products (IP4)',
                                    'Casting/Extrusion/Conformation',
                                    'Stamping/calendering/lamination',
                                    'Machining'],
        'Finished Products': ['No Coating', 'Hot Dip Metal Coating FP',
                              'Electrolytic Metal Coating FP',
                              'Organic or Sintetic Coating (painting)'],
        'Liquid Steel': ['Electric Arc Furnace', 'Basic Oxygen Furnace'],
        'Liquid Steel R': ['Regular Steel'],
        'Liquid Steel L': ['Steel Refining (Low Alloy)'],
        'Liquid Steel H': ['Steel Refining (High Alloy)'],
    }

    # start with all bans explicitly OFF (critical because balance defaults to 1.0)
    chosen = {name: 0 for name, val in (bans or {}).items() if val == 0}

    q, seen = [demand_mat], set()
    while q:
        m = q.pop(0)
        if m in seen:
            continue
        seen.add(m)

        # candidates that are NOT banned
        all_cands = producers.get(m, [])
        cands = [r for r in all_cands if bans.get(r.name, 1) > 0]
        if not cands:
            # external purchase; nothing to enable/disable
            continue

        # choose deterministically
        if len(cands) == 1:
            pick = cands[0]
        else:
            pref_list = prefer.get(m, [])
            name2r = {r.name: r for r in cands}
            pick = next((name2r[pn] for pn in pref_list if pn in name2r), None)
            if pick is None:
                pick = sorted(cands, key=lambda r: r.name)[0]

        # enable the pick
        chosen[pick.name] = 1

        # EXPLICITLY disable *all* other producers of this material (even if banned)
        for r in all_cands:
            if r.name != pick.name:
                chosen[r.name] = 0

        # walk upstream
        for im in pick.inputs:
            if im not in seen:
                q.append(im)

    return chosen


def expand_energy_variants(active, energy_shares, energy_int):
    """CC(R/L/H) inherit base 'Continuous Casting' energy rows."""
    if hasattr(ms, 'expand_energy_tables_for_active'):
        ms.expand_energy_tables_for_active(active, energy_shares, energy_int)
    else:
        def base(n): return n.split(" (")[0]
        for n in active:
            b = base(n)
            if n not in energy_shares and b in energy_shares:
                energy_shares[n] = dict(energy_shares[b])
            if n not in energy_int and b in energy_int:
                energy_int[n] = energy_int[b]

# ----------------------------
# Load base data once
# ----------------------------
BASE = os.path.join('data', '')
elec_idx  = load_elec_idx(BASE)

energy_int      = ms.load_data_from_yaml(os.path.join(BASE,'energy_int.yml'))
energy_shares   = ms.load_data_from_yaml(os.path.join(BASE,'energy_matrix.yml'))
energy_content  = ms.load_data_from_yaml(os.path.join(BASE,'energy_content.yml'))
e_efs           = ms.load_data_from_yaml(os.path.join(BASE,'emission_factors.yml'))
p_efs           = ms.load_data_from_yaml(os.path.join(BASE,'process_emissions.yml'))
params          = ms.load_parameters     (os.path.join(BASE,'parameters.yml'))

scenario = ms.load_data_from_yaml(os.path.join(BASE, 'scenarios', 'DRI_EAF.yml'),
                                  default_value=None, unwrap_single_key=False) or {}
ms.apply_dict_overrides(energy_int,     scenario.get('energy_int', {}))
ms.apply_dict_overrides(energy_shares,  scenario.get('energy_matrix', {}))
ms.apply_dict_overrides(energy_content, scenario.get('energy_content', {}))
ms.apply_dict_overrides(e_efs,          scenario.get('emission_factors', {}))

recipes = ms.load_recipes_from_yaml(os.path.join(BASE,'recipes.yml'),
                                    params, energy_int, energy_shares, energy_content)

# ----------------------------
# Configure sweep
# ----------------------------
routes    = ['BF-BOF', 'EAF-Scrap', 'DRI-EAF']
countries = sorted(elec_idx.keys())
stage_key = 'Finished'
demand    = 1000.0
grade     = 'R'  # fixed grade to avoid prompts

results = []

# ----------------------------
# Run the grid
# ----------------------------
print(tnow(), f"Starting sweep for {len(countries)} countries × {len(routes)} routes")
total_jobs = len(countries)*len(routes)
job = 0

for cc in countries:
    for route_key in routes:
        job += 1
        print(tnow(), f"[{job}/{total_jobs}] {cc} – {route_key}")

        # Deep copies for isolation
        params_r         = copy.deepcopy(params)
        energy_int_r     = copy.deepcopy(energy_int)
        energy_shares_r  = copy.deepcopy(energy_shares)
        energy_content_r = copy.deepcopy(energy_content)
        e_efs_r          = copy.deepcopy(e_efs)
        p_efs_r          = copy.deepcopy(p_efs)
        recipes_r        = copy.deepcopy(recipes)
        rec_dict         = {r.name: r for r in recipes_r}

        # country electricity EF
        ef = elec_idx.get(cc)
        if ef is not None:
            e_efs_r['Electricity'] = ef

        # intensity adjustments
        ms.adjust_blast_furnace_intensity(energy_int_r, energy_shares_r, params_r)
        ms.adjust_process_gas_intensity('Coke Production', 'process_gas_coke',
                                        energy_int_r, energy_shares_r, params_r)

        # build non-interactive route
        demand_mat = ms.STAGE_MATS[stage_key]
        bans = {}
        bans.update(route_to_bans(route_key))
        bans.update(postcasting_defaults())
        bans.update(grade_mask(grade))

        prod_routes = build_routes_noninteractive(recipes_r, demand_mat, bans)

        # balances
        final_demand = {demand_mat: demand}
        balance_matrix, prod_lvl = ms.calculate_balance_matrix(recipes_r, final_demand, prod_routes)
        if balance_matrix is None:
            results.append(dict(country=cc, route=route_key, stage=stage_key,
                                demand=demand, total_CO2e_kg=float('nan'), note='no balance'))
            print(tnow(), "  -> balance failed (nan)")
            continue

        # ensure energy rows exist for chosen variants (e.g., CC (R/L/H))
        active = [p for p, r in prod_lvl.items() if r > 1e-9]
        expand_energy_variants(active, energy_shares_r, energy_int_r)

        # energy & emissions
        internal_elec = ms.calculate_internal_electricity(prod_lvl, rec_dict, params_r)
        energy_balance = ms.calculate_energy_balance(prod_lvl, energy_int_r, energy_shares_r)
        energy_balance = ms.adjust_energy_balance(energy_balance, internal_elec)

        # recovered-gas EF
        gas_coke_MJ = prod_lvl.get('Coke Production', 0) * rec_dict.get('Coke Production', ms.Process('',{},{})).outputs.get('Process Gas', 0)
        gas_bf_MJ   = (getattr(params_r,'bf_adj_intensity',0) - getattr(params_r,'bf_base_intensity',0)) * prod_lvl.get('Blast Furnace', 0)
        total_gas_MJ = gas_coke_MJ + gas_bf_MJ

        cp_sh = energy_shares_r.get('Coke Production', {})
        fuels_cp = [c for c in cp_sh if c != 'Electricity' and cp_sh[c] > 0]
        EF_coke_gas = (sum(cp_sh[c]*e_efs_r.get(c,0) for c in fuels_cp)/sum(cp_sh[c] for c in fuels_cp)) if fuels_cp else 0.0

        bf_sh = energy_shares_r.get('Blast Furnace', {})
        fuels_bf = [c for c in bf_sh if c != 'Electricity' and bf_sh[c] > 0]
        EF_bf_gas = (sum(bf_sh[c]*e_efs_r.get(c,0) for c in fuels_bf)/sum(bf_sh[c] for c in fuels_bf)) if fuels_bf else 0.0

        avoided_coke_CO2 = gas_coke_MJ * EF_coke_gas
        avoided_bf_CO2   = gas_bf_MJ   * EF_bf_gas
        EF_process_gas = ((avoided_coke_CO2 + avoided_bf_CO2) / total_gas_MJ) if total_gas_MJ else 0.0

        util_eff      = rec_dict['Utility Plant'].outputs.get('Electricity', 0) if 'Utility Plant' in rec_dict else 0.0
        internal_elec = total_gas_MJ * util_eff

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

        total = float(emissions.loc['TOTAL','TOTAL CO2e']) if emissions is not None and 'TOTAL' in emissions.index else float('nan')
        results.append(dict(country=cc, route=route_key, stage=stage_key,
                            demand=demand, total_CO2e_kg=total))
        print(tnow(), f"  -> TOTAL CO2e = {total:,.2f} kg")

# ----------------------------
# Save for the paper
# ----------------------------
df = pd.DataFrame(results).sort_values(['country','route'])
out = os.path.join('artifacts', 'country_route_baselines.csv')
os.makedirs('artifacts', exist_ok=True)
df.to_csv(out, index=False)
print(tnow(), "Saved:", out)
try:
    print(df.head(12).to_string(index=False))
except Exception:
    pass

# ----------------------------
# Charts: one bar chart per route, plus a grouped overview
# ----------------------------
import plotly.express as px

outdir = 'artifacts'
order_c = sorted(df['country'].unique(), key=lambda x: x)  # consistent x-order
order_r = ['BF-BOF', 'EAF-Scrap', 'DRI-EAF']               # optional: consistent route order

# 1) One chart per route
for route in df['route'].dropna().unique():
    sub = df[df['route'] == route].copy()
    sub = sub.dropna(subset=['total_CO2e_kg'])
    # keep country order stable left→right
    sub['country'] = pd.Categorical(sub['country'], categories=order_c, ordered=True)

    fig = px.bar(
        sub,
        x='country',
        y='total_CO2e_kg',
        title=f'CO₂e per {int(sub["demand"].iloc[0])} kg — {route}',
        labels={'country': 'Country', 'total_CO2e_kg': 'kg CO₂e'},
        text='total_CO2e_kg'
    )
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside', cliponaxis=False)
    fig.update_layout(xaxis_tickangle=-45, bargap=0.25, margin=dict(t=70, b=80))
    fig.write_html(os.path.join(outdir, f'bar_{route.replace("/","-").replace(" ","_")}.html'),
                   include_plotlyjs='cdn')

# 2) One grouped chart for all routes (nice comparison)
df_plot = df.dropna(subset=['total_CO2e_kg']).copy()
df_plot['country'] = pd.Categorical(df_plot['country'], categories=order_c, ordered=True)
if set(order_r).issuperset(df_plot['route'].unique()):
    df_plot['route'] = pd.Categorical(df_plot['route'], categories=order_r, ordered=True)

fig_all = px.bar(
    df_plot,
    x='country',
    y='total_CO2e_kg',
    color='route',
    barmode='group',
    title=f'CO₂e per {int(df_plot["demand"].iloc[0])} kg — by route and country',
    labels={'country': 'Country', 'total_CO2e_kg': 'kg CO₂e', 'route': 'Route'}
)
fig_all.update_layout(xaxis_tickangle=-45, bargap=0.25, margin=dict(t=70, b=80), legend_title_text='Route')
fig_all.write_html(os.path.join(outdir, 'bar_all_routes.html'), include_plotlyjs='cdn')

print("Saved charts to:", os.path.abspath(outdir))

