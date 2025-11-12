"""Core engine trio kept together here.

Exports the main compute functions while we gradually refactor internals.
Currently implements balance matrix and energy balance locally, and delegates
emissions to the legacy monolith to preserve behavior.
"""
from __future__ import annotations

import pandas as pd
from collections import defaultdict, deque
from typing import Dict

def calculate_balance_matrix(recipes, final_demand, production_routes):
    """
    Solve production levels by walking upstream from 'final_demand' material.
    production_routes: dict {process_name: 0/1}; missing key means allowed (1).
    Exactly one producer per material must be enabled, or we treat as external.
    """
    all_m, producers = set(), defaultdict(list)
    recipes_dict = {r.name: r for r in recipes}
    for r in recipes:
        all_m |= set(r.inputs) | set(r.outputs)
        for m in r.outputs:
            producers[m].append(r)

    demand = list(final_demand.keys())[0]
    if demand not in producers:
        # demand may be a market-only material; still build external requirement row
        required: Dict[str, float] = defaultdict(float, final_demand)
        mats = sorted(all_m | {demand})
        df = pd.DataFrame(
            [[required.get(m, 0) for m in mats], [-final_demand.get(m, 0) for m in mats]],
            index=['External Inputs', 'Final Demand'], columns=mats,
        )
        return df.loc[:, (df.abs() > 1e-9).any()], defaultdict(float)

    required: Dict[str, float] = defaultdict(float, final_demand)
    prod_level: Dict[str, float] = defaultdict(float)
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
    ext = {m: amt for m, amt in required.items() if amt > 1e-9 and m not in producers}
    mats = sorted(all_m | set(required.keys()))
    data, rows = [], []

    for nm, lvl in prod_level.items():
        if lvl > 1e-9:
            rec = recipes_dict[nm]
            row = [(rec.outputs.get(m, 0.0) - rec.inputs.get(m, 0.0)) * lvl for m in mats]
            data.append(row); rows.append(nm)

    data.append([ext.get(m, 0.0) for m in mats]); rows.append('External Inputs')
    data.append([-final_demand.get(m, 0.0) for m in mats]); rows.append('Final Demand')

    df = pd.DataFrame(data, index=rows, columns=mats)
    return df.loc[:, (df.abs() > 1e-9).any()], prod_level


def calculate_energy_balance(prod_level, energy_int, energy_shares):
    """Build energy balance (MJ) from production levels, per-run intensity, and carrier shares."""
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


def calculate_emissions(
    mkt_cfg,
    prod_level,
    energy_df,
    energy_efs,
    process_efs,
    internal_elec,  # kept for signature compatibility (not used here)
    final_demand,
    total_gas_MJ,
    EF_process_gas,
    internal_fraction_plant=None,
    ef_internal_electricity=None,
    outside_mill_procs: set | None = None,
    allow_direct_onsite: set | None = None,
):
    """
    Enforce mutual exclusivity per process row:
    - Onsite production  → Energy Emissions only (Direct=0), unless whitelisted chemistry.
    - Market/outside     → Direct Emissions only (Energy=0).

    Electricity EF for onsite rows uses the plant-wide blend:
        ef_elec_mix = f_internal * ef_int + (1 - f_internal) * EF_grid
    Electricity for outside-mill rows is grid-only.
    Returns a DataFrame indexed by process with columns:
        [Energy Emissions, Direct Emissions, TOTAL CO2e] in tonnes.
    """
    # Helpers
    def _is_market_process(name: str) -> bool:
        n = name.lower()
        return (" from market" in n) or (" purchase" in n)

    # Determine outside-mill processes
    if outside_mill_procs is not None:
        outside_set = set(outside_mill_procs)
    else:
        try:
            outside_set = set(mkt_cfg.get('outside_mill_procs', [])) if isinstance(mkt_cfg, dict) else set()
        except Exception:
            outside_set = set()
        for name in list(process_efs.keys()):
            n = name.lower()
            if ' from market' in n or ' purchase' in n:
                outside_set.add(name)

    ALLOW_DIRECT_ONSITE = set(allow_direct_onsite or [])

    f_internal = float(internal_fraction_plant or 0.0)
    ef_grid = float(energy_efs.get('Electricity', 0.0))
    ef_int_e = float(ef_internal_electricity or 0.0)
    ef_elec_mix = f_internal * ef_int_e + (1.0 - f_internal) * ef_grid

    rows = []
    # Iterate over all processes appearing anywhere
    proc_index = list({*energy_df.index.tolist(), *process_efs.keys(), *prod_level.keys()})

    for proc_name in proc_index:
        runs = float(prod_level.get(proc_name, 0.0))
        if runs <= 1e-12:
            continue

        is_purchase = _is_market_process(proc_name)
        is_outside = proc_name in outside_set

        row = {"Process": proc_name, "Energy Emissions": 0.0, "Direct Emissions": 0.0}

        if is_purchase:
            row["Direct Emissions"] = runs * 1000.0 * float(process_efs.get(proc_name, 0.0))
            row["Energy Emissions"] = 0.0
        else:
            elec_ef_for_proc = ef_grid if is_outside else ef_elec_mix
            if proc_name in energy_df.index:
                for carrier, cons in energy_df.loc[proc_name].items():
                    if carrier == 'Electricity':
                        row['Energy Emissions'] += float(cons) * elec_ef_for_proc
                    else:
                        row['Energy Emissions'] += float(cons) * float(energy_efs.get(carrier, 0.0))

            # Direct emissions only when whitelisted chemistry for onsite
            if proc_name in ALLOW_DIRECT_ONSITE:
                row['Direct Emissions'] = runs * 1000.0 * float(process_efs.get(proc_name, 0.0))
            else:
                row['Direct Emissions'] = 0.0

        rows.append(row)

    if not rows:
        return None

    emissions_df = pd.DataFrame(rows).set_index('Process') / 1000.0  # kg -> t
    emissions_df['TOTAL CO2e'] = emissions_df['Energy Emissions'] + emissions_df['Direct Emissions']

    return emissions_df

__all__ = [
    "calculate_balance_matrix",
    "calculate_energy_balance",
    "calculate_emissions",
]
