"""
This is the core computation engine for the Forge modeling framework. Main function is 'calculate_balance_matrix', which walks upstream from a specified final demand to determine production levels across a network of recipes. It is set for error if multiple producers exist for a material without clear selection, to avoid guessing. 
The function is complex because recipes are configured once, for all possible steel production processes, to avoid duplication and drift (ie. one recipe for BF-BOF, another for DRI-EAF). The route must be constructed elsewhere and fed into engine, which is kept a pure calculation machine by design. The specific production scenario is set by enabling/disabling certain routes, via 'production_routes' dict.
Additional functions compute energy balances and emissions based on production levels and various intensity and emission factors configured externally via YAML files.
"""
from __future__ import annotations

from matplotlib.pylab import mat
import pandas as pd
from collections import defaultdict, deque
from typing import Dict

def calculate_balance_matrix(recipes, final_demand, production_routes, material_credit_map=None):
    """
    This function is the heart of the system. If solves material balances to give production levelsf or all processes needed to satisfy a given final demand, based on available recipes and enabled production routes.
    """
    
    """
    First step is to scan recipes to build a mapping of materials and their possible producers 
    (ie Pig Iron may be produced by the Blast Furnace or Direct Reduction, per recipes).
    """

    all_materials, producers = set(), defaultdict(list)
    material_credit_map = material_credit_map or {}
    credit_stock = defaultdict(float)  # accumulated credits for target materials
    recipes_dict = {r.name: r for r in recipes}
    for r in recipes:
        all_materials |= set(r.inputs) | set(r.outputs) # gather all possible materials
        for m in r.outputs: 
            producers[m].append(r) 
            # map material to possible producers 
            # (still allow more than one at this point)

    """
    Some materials are not produced internally, by design they have no output in recipes (ie. iron ore). These will remain as external requirements in the final balance.
    """        

    demand = list(final_demand.keys())[0]
    if demand not in producers:
        # demand may be a market-only material; still build external requirement row
        required: Dict[str, float] = defaultdict(float, final_demand)
        mats = sorted(all_materials | {demand})
        df = pd.DataFrame(
            [[required.get(m, 0) for m in mats], [-final_demand.get(m, 0) for m in mats]],
            index=['External Inputs', 'Final Demand'], columns=mats,
        )
        return df.loc[:, (df.abs() > 1e-9).any()], defaultdict(float)
    
    """
    Now we need to walk upstream from final demand, which is set by the user (ie cast steel, finished product).
    To avoid infinite loops we set a queue and seen set. Required and prod_level dicts track material requirements and production levels per process.
    """

    required = defaultdict(float, final_demand) # the walk starts with final demand set by user (ie pig iron, 1000 kg) - generic so we can change final demand on the fly without duplication
    prod_level = defaultdict(float) # production levels per process (recipe name)
    queue = deque([demand]) # puts final demand at the start of the queue. the rest is built upstream from recipes. user must ensure recipes i/o are consistent, code can't check that.
    seen = {demand} # to avoid infinite loops

    # Now that we set the table we can start walking upstream


    while queue:
        material = queue.popleft(); seen.remove(material) # pop material from queue and mark as unseen
        amount_required = required[material] # amount required for this material
        if amount_required <= 1e-9:
            continue

        credit_available = credit_stock.get(material, 0.0)
        if credit_available > 0.0:
            used_credit = min(credit_available, amount_required)
            amount_required -= used_credit
            credit_stock[material] = credit_available - used_credit
            required[material] = amount_required

        if amount_required <= 1e-9:
            continue

        all_recipes = producers.get(material, []) # possible recipes for this material
        enabled_recipes = [recipe for recipe in all_recipes if (production_routes.get(recipe.name, 1.0) > 0.0)] # filter recipes by enabled routes

        if not enabled_recipes:
            # No enabled internal producer → external purchase; leave as requirement
            continue

        if len(enabled_recipes) > 1:
            names = [p.name for p in enabled_recipes]
            raise ValueError(f"Ambiguous producers for '{material}': {names}. Pick exactly one.")
            # multiple enabled producers leads to error so code can't guess. recipes can and have multiple producers per material, but user must pick one via production_routes.

        selected_recipe = enabled_recipes[0]
        output_per_run = float(selected_recipe.outputs.get(material, 0.0))
        if output_per_run <= 0:
            continue

        production_runs_needed = amount_required / output_per_run
        prod_level[selected_recipe.name] += production_runs_needed

        for input_material, input_amount in selected_recipe.inputs.items():
            required[input_material] += production_runs_needed * float(input_amount)
            if input_material in producers and input_material not in seen:
                queue.append(input_material); seen.add(input_material)

        for output_material, output_amount in selected_recipe.outputs.items():
            rule = material_credit_map.get(output_material)
            if not rule:
                continue
            target_mat, ratio = rule
            try:
                ratio_val = float(ratio)
            except Exception:
                ratio_val = 1.0
            credit_amount = production_runs_needed * float(output_amount) * ratio_val
            if credit_amount <= 0.0:
                continue
            remaining_req = required.get(target_mat, 0.0)
            if remaining_req > 0.0:
                applied = min(remaining_req, credit_amount)
                required[target_mat] = remaining_req - applied
                credit_amount -= applied
            if credit_amount > 1e-9:
                credit_stock[target_mat] += credit_amount

        required[material] = 0.0 # loop stops when no material is left on the queue

    # Prepare matrix (process net flows + external + final demand)
    external_purchases = {m: amt for m, amt in required.items() if amt > 1e-9 and m not in producers}
    mats = sorted(all_materials | set(required.keys()))
    data, rows = [], []

    for process_name, production_volume in prod_level.items():
        if production_volume > 1e-9:
            rec = recipes_dict[process_name]
            row = [(rec.outputs.get(m, 0.0) - rec.inputs.get(m, 0.0)) * production_volume for m in mats]
            data.append(row); rows.append(process_name)

    data.append([external_purchases.get(m, 0.0) for m in mats]); rows.append('External Inputs')
    data.append([-final_demand.get(m, 0.0) for m in mats]); rows.append('Final Demand')

    df = pd.DataFrame(data, index=rows, columns=mats)
    return df.loc[:, (df.abs() > 1e-9).any()], prod_level


def calculate_energy_balance(prod_level, energy_int, energy_shares):
    """Build energy balance (MJ) from production levels, per-run intensity, and carrier shares."""
    energy_shares = pd.DataFrame.from_dict(energy_shares, orient='index').fillna(0.0) 
    # reads per carrier shares from dict (from energy_shares YAML)
    energy_intensity = pd.Series(energy_int).fillna(0.0)
    # reads per process energy intensity from dict (from energy_int YAML)

    per_run = energy_shares.multiply(energy_intensity, axis='index') # MJ per run by carrier
    runs = pd.Series(prod_level) # as energy intensity is per process run, we need to multiply by runs, so we need runs first
    common = per_run.index.intersection(runs.index) # only processes with both energy intensity and production levels
    data = per_run.loc[common].multiply(runs, axis=0) # now we multiply MJ/run by runs to get total MJ per process

    all_carriers = sorted(energy_shares.columns.union(pd.Index(['Electricity']))) # ensure Electricity is included even if zero (but why?)
    bal = pd.DataFrame(data, index=common, columns=all_carriers).fillna(0.0) # this simply creates the energy balance DataFrame
    bal.loc['TOTAL'] = bal.sum() # and this adds a total row summing total per carrier enrgy consumption
    return bal # bal will be read anywhere as the energy balance; do not refactor here


def calculate_emissions(
    mkt_cfg,
    prod_level,
    energy_df,
    energy_efs,
    process_efs,
    internal_elec,  # kept for signature compatibility (not used here because we changed logic)
    final_demand,
    total_gas_MJ,
    EF_process_gas,
    internal_fraction_plant=None,
    ef_internal_electricity=None,
    outside_mill_procs: set | None = None,
    allow_direct_onsite: set | None = None,
    process_output_emissions: dict | None = None,
):
    """
    This should be a simple emission calculator but needed some tailoring for steel processes. Aluminum will probably need similar treatment.
    First we separate purchases from onsite production. This is set via 'mkt_cfg' (or helper?) because we need some flexibility (i.e. Nitrogen can be purchased or produced onsite). As we have some processes that occur downstream of a typical mill, we need to identify those as well. Electricity/gas there is from grid/mkt, not blended.
    Second aspect is that both electricity and process gas need special handling. Both emission factors are blends between outside purchase and internal production, based on internal_fraction_plant. We calculate this elsewhere and explain there. 
    Finally, some internal process can have direct emissions, so we need to whitelist those via 'allow_direct_onsite' set.
    When all these aspects are considered, we can build the emissions DataFrame, which is simply a matter of iterating over processes and calculating energy and direct emissions accordingly.
    This function may benefit from some refactoring later, but for now it works as intended.
    """
    # Helpers
    def _is_market_process(name: str) -> bool:
        n = name.lower()
        return (" from market" in n) or (" purchase" in n) #maybe mkt_cfg is not even used anymore? check this later

    # Determine outside-mill processes - self explanatory; we set a plant boundary constant
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
    output_emissions = {}
    try:
        output_emissions = {
            str(k): float(v)
            for k, v in (process_output_emissions or {}).items()
            if v is not None
        }
    except Exception:
        output_emissions = {}
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
                    if proc_name == "Coke Production" and carrier == "Coal":
                        continue # we treat coal as feedstock, not energy, for coke production
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

    # Add per-process direct emissions derived from recipe outputs (e.g., Electrolysis gas streams)
    if output_emissions:
        indexed = {row["Process"]: row for row in rows}
        for proc, extra in output_emissions.items():
            if abs(extra) <= 1e-12:
                continue
            row = indexed.get(proc)
            if row is None:
                row = {"Process": proc, "Energy Emissions": 0.0, "Direct Emissions": 0.0}
                rows.append(row)
                indexed[proc] = row
            row["Direct Emissions"] = float(row.get("Direct Emissions", 0.0)) + float(extra)
        total_extra = sum(v for v in output_emissions.values() if abs(v) > 1e-12)
        if abs(total_extra) > 1e-12:
            rows.append({
                "Process": "Process emissions (Electrolysis)",
                "Energy Emissions": 0.0,
                "Direct Emissions": float(total_extra),
            })

    if not rows:
        return None

    emissions_df = pd.DataFrame(rows).set_index('Process') / 1000.0  # kg -> t
    emissions_df['TOTAL CO2e'] = emissions_df['Energy Emissions'] + emissions_df['Direct Emissions'] # separate accounting for energy and direct emissions for clarity; but total emissions always sum both

    return emissions_df

__all__ = [
    "calculate_balance_matrix",
    "calculate_energy_balance",
    "calculate_emissions",
]
