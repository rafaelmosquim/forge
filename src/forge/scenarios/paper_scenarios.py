# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 10:05:01 2025

@author: rafae
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 10:34:03 2025

@author: rafae
"""

import pandas as pd
import numpy as np
import matplotlib.cm as cm
import seaborn as sns
from itertools import product
from tqdm import tqdm  # Progress bar
import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # Added missing import
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple

# Ensure repo root is on sys.path for imports when running as a script
import sys, os as _os
from pathlib import Path as _Path
_ROOT_DIR = _Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Core bridge: run live FORGE model instead of static CSVs
from forge.steel_core_api_v2 import RouteConfig, ScenarioInputs, run_scenario
from forge.scenarios.utils import label_from_spec_path as _label_from_spec_path, configure_output_roots as _auto_output_roots

# ---- Make plt.show() safe in headless runs ----
import matplotlib as _mpl
import matplotlib.pyplot as _plt
_FIG_INDEX = 0

def _install_safe_show():
    global _FIG_INDEX
    backend = (_mpl.get_backend() or '').lower()
    # Save figures instead of showing when non-interactive backend is active
    save_mode = ('agg' in backend) or (_os.getenv('FORGE_SAVE_FIGS', '') == '1')
    if not save_mode:
        return
    base_dir = _os.getenv('FORGE_FIG_DIR', 'results/figs')
    # Overwrite in-place within the configured figs folder
    out_path = _Path(base_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _orig_show = _plt.show

    def _safe_show(*args, **kwargs):
        nonlocal out_path
        global _FIG_INDEX
        _FIG_INDEX += 1
        prefix = _os.getenv('FORGE_FIG_PREFIX', 'paper')
        fname = f'{prefix}_fig_{_FIG_INDEX:02d}.png'
        path = out_path / fname
        _plt.gcf().savefig(path, dpi=200, bbox_inches='tight')
        _plt.close(_plt.gcf())
        print(f"[fig] saved {path}")

    _plt.show = _safe_show

# defer installing safe show until output dirs are configured





# ============================================
# 1. SETUP (Input Data & Helper Functions)
# ============================================

DATA_DIR = "datasets/steel/likely"
FINAL_PICKS = {
    "Manufactured Feed (IP4)": "Stamping/calendering/lamination",
    "Finished Products": "No Coating",
}

# Toggle between simple product picks vs. portfolio basket
# - simple: uses FINAL_PICKS (stamping/no coating)
# - portfolio: blends EF across a portfolio spec (see configs/*finished_steel_portfolio*.yml)
PRODUCT_CONFIG = _os.getenv('FORGE_PAPER_PRODUCT_CONFIG', 'simple').strip().lower()
PORTFOLIO_SPEC_OVERRIDE = _os.getenv('FORGE_PAPER_PORTFOLIO_SPEC', '').strip() or None
PORTFOLIO_BLEND_OVERRIDE = _os.getenv('FORGE_PAPER_PORTFOLIO_BLEND', '').strip() or None

# Configure output roots then install safe show
_auto_output_roots()
_install_safe_show()

def _scenario_for_bf_config(config: str) -> Dict[str, Any]:
    """Scenario for BF configs; 'Charcoal' uses dataset scenario file for consistency."""
    name = str(config).strip().lower()
    if name == "charcoal":
        try:
            path = _Path(DATA_DIR) / 'scenarios' / 'BF_BOF_charcoal.yml'
            import yaml as _yaml_local
            with path.open('r', encoding='utf-8') as fh:
                payload = _yaml_local.safe_load(fh) or {}
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        # Fallback to coarse substitution if file missing
        return {"fuel_substitutions": {"Coal": "Charcoal", "Coke": "Charcoal"}}
    return {}

def _route_defaults_scenario(route: str, config: str) -> Dict[str, Any]:
    """Load dataset scenario defaults per route/config to avoid duplication.

    These are the route-level defaults used in the app (e.g., BF coal/charcoal,
    DRI-EAF, EAF-Scrap). They act as a baseline that we later augment with
    energy_int_schedule and any extra toggles.
    """
    route_u = (route or '').strip().upper()
    cfg_u = (config or '').strip().upper()
    fname = None
    if route_u.startswith('BF'):
        # Choose charcoal or coal variant
        if cfg_u == 'CHARCOAL':
            fname = 'BF_BOF_charcoal.yml'
        else:
            fname = 'BF_BOF_coal.yml'
    elif route_u.startswith('DRI'):
        fname = 'DRI_EAF.yml'
    elif route_u.startswith('EAF'):
        fname = 'scrap_EAF.yml'
    if not fname:
        return {}
    try:
        path = _Path(DATA_DIR) / 'scenarios' / fname
        import yaml as _yaml_local2
        with path.open('r', encoding='utf-8') as fh:
            payload = _yaml_local2.safe_load(fh) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}

def _scenario_for_dri_config(config: str) -> Dict[str, Any]:
    name = str(config).strip().lower()
    def _mix(gas=0.0, bio=0.0, h2=0.0):
        return {
            "dri_mix": "Custom",
            "dri_mix_definitions": {
                "Custom": {
                    2030: {"Gas": gas, "Biomethane": bio, "Green hydrogen": h2},
                    2040: {"Gas": gas, "Biomethane": bio, "Green hydrogen": h2},
                }
            },
        }
    if name in {"natural gas", "gas", "ng"}:
        return _mix(gas=1.0, bio=0.0, h2=0.0)
    if name in {"biomethane-100", "biomethane"}:
        return _mix(gas=0.0, bio=1.0, h2=0.0)
    if name in {"green h2", "green hydrogen", "h2"}:
        return _mix(gas=0.0, bio=0.0, h2=1.0)
    return _mix(gas=1.0, bio=0.0, h2=0.0)

@lru_cache(maxsize=2048)
def _ef_core_cached(route: str, config: str, year: int, base_year: int, annual_improvement: float) -> float:
    route = str(route).strip()
    config = str(config).strip()
    # Build scenario payload starting from route defaults
    base_defaults = _route_defaults_scenario(route, config)
    if route.upper().startswith("BF"):
        scn = _scenario_for_bf_config(config)
    elif route.upper().startswith("DRI"):
        scn = _scenario_for_dri_config(config)
    else:
        scn = {}
    rate_pct = float(annual_improvement) * 100.0
    scn["energy_int_schedule"] = {
        "rate_pct_per_year": rate_pct,
        "baseline_year": int(base_year),
        "target_year": int(year),
    }
    # For DRI mixes, set snapshot_year so _apply_dri_mix picks nearest <= year
    scn.setdefault("snapshot_year", int(year))
    # Enforce BF minimum base intensity (post-schedule) as an explicit floor
    if route.upper().startswith("BF"):
        scn.setdefault("energy_int_floor", {})["Blast Furnace"] = 11.0
    # Merge baseline route defaults first
    if base_defaults:
        scn = _merge_scenarios(base_defaults, scn)
    rc = RouteConfig(
        route_preset=route,
        stage_key="Finished",
        stage_role=None,
        demand_qty=1000.0,
        picks_by_material=dict(FINAL_PICKS),
        pre_select_soft={},
    )
    si = ScenarioInputs(country_code=None, scenario=scn, route=rc)
    out = run_scenario(DATA_DIR, si)
    raw = float(getattr(out, "total_co2e_kg", 0.0) or 0.0)
    return raw / 1000.0

# -----------------
# Optional disk cache
# -----------------
import pickle as _pkl
import threading as _th

_EF_CACHE_LOCK = _th.Lock()
_EF_CACHE: Dict[tuple, float] | None = None

def _cache_dir_file() -> tuple[_Path, _Path]:
    base = _os.getenv('FORGE_PAPER_CACHE_DIR', 'results/cache')
    d = _Path(base)
    f = d / 'paper_ef_cache.pkl'
    return d, f

def _load_ef_cache() -> Dict[tuple, float]:
    global _EF_CACHE
    if _EF_CACHE is not None:
        return _EF_CACHE
    d, f = _cache_dir_file()
    try:
        if f.exists():
            with f.open('rb') as fh:
                _EF_CACHE = _pkl.load(fh) or {}
        else:
            _EF_CACHE = {}
    except Exception:
        _EF_CACHE = {}
    return _EF_CACHE

def _save_ef_cache():
    d, f = _cache_dir_file()
    try:
        d.mkdir(parents=True, exist_ok=True)
        with f.open('wb') as fh:
            _pkl.dump(_EF_CACHE or {}, fh)
    except Exception:
        pass

def _make_ef_key(route: str, config: str, year: int, base_year: int, annual_improvement: float, picks: Optional[Dict[str, str]] = None) -> tuple:
    # Round improvement to 1e-6 to avoid float chatter
    imp = round(float(annual_improvement), 6)
    picks_src = picks if isinstance(picks, dict) else FINAL_PICKS
    picks_sig = tuple(sorted(picks_src.items()))
    return (str(route), str(config), int(year), int(base_year), imp, str(DATA_DIR), picks_sig)

def _ef_core_cached_disk(route: str, config: str, year: int, base_year: int, annual_improvement: float) -> float:
    """Disk-backed cache wrapper around _ef_core_cached.

    Controlled by FORGE_PAPER_CACHE env (defaults on). Set FORGE_PAPER_CACHE=0 to disable.
    """
    use_cache = _os.getenv('FORGE_PAPER_CACHE', '1').strip() not in {'0', 'false', 'off'}
    if not use_cache:
        return _ef_core_cached(route, config, year, base_year, annual_improvement)
    key = _make_ef_key(route, config, year, base_year, annual_improvement, None)
    with _EF_CACHE_LOCK:
        cache = _load_ef_cache()
        if key in cache:
            return cache[key]
    # Miss → compute
    val = _ef_core_cached(route, config, year, base_year, annual_improvement)
    with _EF_CACHE_LOCK:
        cache = _load_ef_cache()
        cache[key] = val
        _save_ef_cache()
    return val

def _merge_scenarios(base: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not extra:
        return base
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_scenarios(out[k], v)
        else:
            out[k] = v
    return out

def _ef_core_run_with_picks(route: str, config: str, year: int, base_year: int, annual_improvement: float, picks: Dict[str, str], scenario_defaults: Optional[Dict[str, Any]] = None) -> float:
    # Build scenario payload with route defaults + config mapping + schedule
    base_defaults = _route_defaults_scenario(route, config)
    if route.upper().startswith("BF"):
        scn = _scenario_for_bf_config(config)
    elif route.upper().startswith("DRI"):
        scn = _scenario_for_dri_config(config)
    else:
        scn = {}
    rate_pct = float(annual_improvement) * 100.0
    scn["energy_int_schedule"] = {
        "rate_pct_per_year": rate_pct,
        "baseline_year": int(base_year),
        "target_year": int(year),
    }
    scn.setdefault("snapshot_year", int(year))
    if route.upper().startswith("BF"):
        scn.setdefault("energy_int_floor", {})["Blast Furnace"] = 11.0
    # Merge route defaults, then portfolio/default scenario overrides
    if base_defaults:
        scn = _merge_scenarios(base_defaults, scn)
    if scenario_defaults:
        scn = _merge_scenarios(scn, scenario_defaults)
    rc = RouteConfig(
        route_preset=route,
        stage_key="Finished",
        stage_role=None,
        demand_qty=1000.0,
        picks_by_material=dict(picks or {}),
        pre_select_soft={},
    )
    si = ScenarioInputs(country_code=None, scenario=scn, route=rc)
    out = run_scenario(DATA_DIR, si)
    raw = float(getattr(out, "total_co2e_kg", 0.0) or 0.0)
    return raw / 1000.0

def _ef_core_cached_disk_with_picks(route: str, config: str, year: int, base_year: int, annual_improvement: float, picks: Dict[str, str], scenario_defaults: Optional[Dict[str, Any]] = None) -> float:
    use_cache = _os.getenv('FORGE_PAPER_CACHE', '1').strip() not in {'0', 'false', 'off'}
    if not use_cache:
        return _ef_core_run_with_picks(route, config, year, base_year, annual_improvement, picks, scenario_defaults)
    key = _make_ef_key(route, config, year, base_year, annual_improvement, picks)
    with _EF_CACHE_LOCK:
        cache = _load_ef_cache()
        if key in cache:
            return cache[key]
    val = _ef_core_run_with_picks(route, config, year, base_year, annual_improvement, picks, scenario_defaults)
    with _EF_CACHE_LOCK:
        cache = _load_ef_cache()
        cache[key] = val
        _save_ef_cache()
    return val

def _portfolio_spec_for_route(route: str) -> Optional[str]:
    """Return the processing basket spec applied across all routes.

    Priority:
      1) FORGE_PAPER_PORTFOLIO_SPEC if set
      2) configs/finished_steel_portfolio.yml if present
      3) fallback to configs/finished_steel_portfolio_eaf.yml
    """
    if PORTFOLIO_SPEC_OVERRIDE:
        return PORTFOLIO_SPEC_OVERRIDE
    # Prefer a unified portfolio file if it exists
    unified = _Path('configs/finished_steel_portfolio.yml')
    if unified.exists():
        return str(unified)
    # Fallback to previous EAF-based basket
    return 'configs/finished_steel_portfolio_eaf.yml'

def _compute_portfolio_ef(route: str, config: str, year: int, base_year: int, annual_improvement: float) -> float:
    import yaml as _yaml
    spec_path = _portfolio_spec_for_route(route)
    if not spec_path:
        # Fallback to simple picks
        return _ef_core_cached_disk(route, config, year, base_year, annual_improvement)
    try:
        with open(spec_path, 'r', encoding='utf-8') as fh:
            spec = _yaml.safe_load(fh) or {}
    except Exception:
        return _ef_core_cached_disk(route, config, year, base_year, annual_improvement)
    defaults = spec.get('defaults') or {}
    defaults_picks = (defaults.get('picks_by_material') or {})
    scenario_defaults = defaults.get('scenario') or {}
    runs = spec.get('runs') or []
    run_map = {r.get('name'): (r.get('picks_by_material') or {}) for r in runs if isinstance(r, dict)}
    blends = spec.get('blends') or []
    # Choose blend by override name or default to first
    blend = None
    if PORTFOLIO_BLEND_OVERRIDE:
        for b in blends:
            if isinstance(b, dict) and str(b.get('name','')).strip() == PORTFOLIO_BLEND_OVERRIDE:
                blend = b
                break
    if blend is None:
        for b in blends:
            if isinstance(b, dict):
                blend = b
                break
    if not blend:
        return _ef_core_cached_disk(route, config, year, base_year, annual_improvement)
    comps = blend.get('components') or []
    if not comps:
        return _ef_core_cached_disk(route, config, year, base_year, annual_improvement)
    total_share = 0.0
    parts: list[Tuple[float, Dict[str, str]]] = []
    for comp in comps:
        try:
            run_name = comp.get('run')
            share = float(comp.get('share', 0.0) or 0.0)
        except Exception:
            continue
        picks_run = run_map.get(run_name, {})
        picks_merged = dict(defaults_picks)
        picks_merged.update(picks_run)
        parts.append((share, picks_merged))
        total_share += max(0.0, share)
    if total_share <= 0.0:
        return _ef_core_cached_disk(route, config, year, base_year, annual_improvement)
    # Blend EF over portfolio parts
    ef = 0.0
    for share, picks in parts:
        w = max(0.0, share) / total_share
        ef_part = _ef_core_cached_disk_with_picks(route, config, year, base_year, annual_improvement, picks, scenario_defaults)
        ef += w * ef_part
    return ef


# Load BF fleet data (example); provide a CI-safe fallback
try:
    bf_fleet = pd.read_csv(
        "datasets/steel/plants.csv", sep=";", encoding="utf-8"
    )
    bf_fleet["Capacity"] = pd.to_numeric(bf_fleet["Capacity"], errors="coerce")
except FileNotFoundError:
    bf_fleet = pd.DataFrame(
        {
            "Fuel": ["coal", "charcoal"],
            "Capacity": [100.0, 20.0],
            "EOL": [2035, 2035],
            "Relining": [2030, 2030],
        }
    )

def get_emission_factor(route, config, year, base_year=2025, annual_improvement=0.0):
    """EF lookup (tCO2/t) computed entirely in core, including annual improvements.

    - energy_int_schedule is passed to the core with given baseline and year.
    - No external scaling is applied here; returned EF is ready to use.
    """
    route_s = str(route)
    config_s = str(config)
    y = int(year)
    by = int(base_year)
    imp = float(annual_improvement)
    if PRODUCT_CONFIG == 'portfolio':
        return _compute_portfolio_ef(route_s, config_s, y, by, imp)
    return _ef_core_cached_disk(route_s, config_s, y, by, imp)

# ============================================
# 2. PARAMETER SWEEP CONFIGURATION
# ============================================

param_grid = {
    'scenario': ['business_as_usual', 'conservative', 'aggressive'],
    'utilization_rate': [0.65, 0.80, 1],
    'annual_improvement': [0.0, 0.01, 0.02],
    'charcoal_expansion': [None, {2030: 3000, 2035: 3000, 2040: 3000, 2045: 3000}],
    'dri_mix': [
        {2030: {"Natural Gas": 1.0}, 
         2040: {"Natural Gas": 0.7, "Biomethane-100": 0.1, "Green H2": 0.2}},
        {2030: {"Natural Gas": 0.7, "Biomethane-100": 0.1, "Green H2": 0.2}, 
         2040: {"Natural Gas": 0.4, "Biomethane-100": 0.2, "Green H2": 0.4}}
    ]
}

def generate_combinations(grid):
    """Generate all parameter combinations."""
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in product(*values)]

scenarios = generate_combinations(param_grid)
print(f"Total scenarios to run: {len(scenarios)}")


def simulate_steel_transition(bf_fleet, **params):
    results = []
    coal_bfs = bf_fleet[bf_fleet["Fuel"] == "coal"].copy()
    charcoal_bfs = bf_fleet[bf_fleet["Fuel"] == "charcoal"].copy()
    
    # Note: enforce BF minimum intensity via energy_int_floor (11 MJ) in core,
    # not an EF floor here.
    
    # Baseline capacities
    original_coal_cap = coal_bfs["Capacity"].sum()
    original_charcoal_cap = charcoal_bfs["Capacity"].sum()
    total_capacity = original_coal_cap + original_charcoal_cap
    
    # Initialize tracking
    current_charcoal_cap = original_charcoal_cap
    dri_cap = 0
    coal_cap = original_coal_cap
    
    # Business-as-usual requires tracking original EOL schedule
    if params['scenario'] == "business_as_usual":
        original_eol_schedule = coal_bfs["EOL"].value_counts().sort_index()
    
    for year in range(2025, 2051):
        # Handle charcoal expansion FIRST (replaces coal-BF directly)
        if params.get('charcoal_expansion') and year in params['charcoal_expansion']:
            added_charcoal = params['charcoal_expansion'][year]
            # Charcoal can only replace available coal capacity
            added_charcoal = min(added_charcoal, coal_cap)
            current_charcoal_cap += added_charcoal
            coal_cap -= added_charcoal
        
        # Scenario-specific capacity logic
        if params['scenario'] == "business_as_usual":
            # Replace retiring BFs with identical new ones
            retiring_cap = original_eol_schedule.get(year, 0)
            coal_cap = coal_cap - retiring_cap + retiring_cap  # 1:1 replacement
            dri_cap = 0
        
        elif params['scenario'] == "conservative" and year >= 2030:
            # Retire coal plants reaching EOL and convert to DRI
            retiring_coal = coal_bfs[coal_bfs["EOL"] == year]["Capacity"].sum()
            # Only convert remaining coal capacity (after charcoal replacement)
            retiring_coal = min(retiring_coal, coal_cap)
            coal_cap -= retiring_coal
            dri_cap += retiring_coal
        
        elif params['scenario'] == "aggressive" and year >= 2030:
            # Convert coal plants reaching relining year to DRI
            relining_coal = coal_bfs[coal_bfs["Relining"] == year]["Capacity"].sum()
            # Only convert remaining coal capacity (after charcoal replacement)
            relining_coal = min(relining_coal, coal_cap)
            coal_cap -= relining_coal
            dri_cap += relining_coal
        
        # Final validation
        current_total = coal_cap + current_charcoal_cap + dri_cap
        if not np.isclose(current_total, total_capacity, rtol=0.01):
            print(f"Warning: Capacity imbalance in {year} - Total: {current_total:.1f} vs Original: {total_capacity:.1f}")
            # Force balance by adjusting coal capacity (since charcoal and DRI are fixed conversions)
            coal_cap = max(0, total_capacity - (current_charcoal_cap + dri_cap))
        
        # Emissions calculation
        utilization = params['utilization_rate']
        annual_improvement = params['annual_improvement']
        
        coal_ef = get_emission_factor("BF-BOF", "Coke", year, base_year=2025, annual_improvement=annual_improvement)
        coal_emis = coal_cap * utilization * coal_ef
        
        charcoal_ef = get_emission_factor("BF-BOF", "Charcoal", year, base_year=2025, annual_improvement=annual_improvement)
        charcoal_emis = current_charcoal_cap * utilization * charcoal_ef
        
        dri_emis = 0
        if dri_cap > 0 and params.get('dri_mix'):
            mix_year = max(y for y in params['dri_mix'].keys() if y <= year)
            for config, share in params['dri_mix'][mix_year].items():
                dri_ef = get_emission_factor("DRI-EAF", config, year, base_year=2025, annual_improvement=annual_improvement)
                dri_emis += dri_cap * share * utilization * dri_ef
        
        results.append({
            'Year': year,
            'Scenario': params['scenario'],
            'BF_Coal_Capacity': coal_cap,
            'BF_Charcoal_Capacity': current_charcoal_cap,
            'DRI_Capacity': dri_cap,
            'Total_Emissions': coal_emis + charcoal_emis + dri_emis,
            'Emissions_Intensity': (coal_emis + charcoal_emis + dri_emis) / 
                                  ((coal_cap + current_charcoal_cap + dri_cap) * utilization),
            'Total_Capacity': coal_cap + current_charcoal_cap + dri_cap
        })
    
    return pd.DataFrame(results)

# ============================================
# 4. RUN ALL SCENARIOS (WITH PROGRESS BAR)
# ============================================

all_results = []
for scenario in tqdm(scenarios, desc="Running scenarios"):
    df = simulate_steel_transition(bf_fleet, **scenario)
    summary = {
        **scenario,
        '2050_Emissions': df[df['Year'] == 2050]['Total_Emissions'].values[0],
        'Cumulative_Emissions': df['Total_Emissions'].sum(),
        'Final_Intensity': df[df['Year'] == 2050]['Emissions_Intensity'].values[0]
    }
    all_results.append(summary)

results_df = pd.DataFrame(all_results)

# ============================================
# 5. VISUALIZE RESULTS
# ============================================


# Identify best and worst scenarios
best_idx = results_df['Cumulative_Emissions'].idxmin()
worst_idx = results_df['Cumulative_Emissions'].idxmax()

best_params = results_df.loc[best_idx].to_dict()
worst_params = results_df.loc[worst_idx].to_dict()

# Run simulations to get annual data
best_df = simulate_steel_transition(bf_fleet, **best_params)
worst_df = simulate_steel_transition(bf_fleet, **worst_params)

# Calculate cumulative emissions
best_df['Cumulative'] = best_df['Total_Emissions'].cumsum()/1000000 
worst_df['Cumulative'] = worst_df['Total_Emissions'].cumsum()/1000000

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot with enhanced styling

ax.plot(worst_df['Year'], worst_df['Cumulative'], 
        color='r', linewidth=2.5, 
        label='Worst Case')

ax.plot(best_df['Year'], best_df['Cumulative'], 
        color='b', linewidth=2.5, 
        label='Best Case')



# Fill between for visual impact
ax.fill_between(best_df['Year'], best_df['Cumulative'], worst_df['Cumulative'],
               color='#3498db', alpha=0.1, label='Emission Reduction Potential')

# Annotations (no scaling needed since data is already in Gtons)
ax.annotate(f'{best_df["Cumulative"].iloc[-1]:.1f} Gt CO$_2$', 
            xy=(2043, best_df["Cumulative"].iloc[-1]), 
            xytext=(0, 10), textcoords='offset points',
            ha='left', va='bottom', color='b', fontsize = 12)

ax.annotate(f'{worst_df["Cumulative"].iloc[-1]:.1f} Gt CO$_2$', 
            xy=(2043, worst_df["Cumulative"].iloc[-1]), 
            xytext=(0, -20), textcoords='offset points',
            ha='left', va='top', color='r', fontsize = 12)

# Formatting
ax.set_xlabel('Year', labelpad=10)
ax.set_ylabel('Cumulative Emissions (Gt CO$_2$)', labelpad=10)  # Changed to Gt
ax.set_title('Brazil Steel Sector: Best vs Worst Emission Scenarios (2025-2050)', pad=20)
ax.legend(frameon=True, framealpha=1, loc='upper left')

# Remove scientific notation (since data is in Gtons)
ax.ticklabel_format(axis='y', useOffset=False, style='plain')

plt.tight_layout()
plt.savefig('best_worst_scenarios.png', dpi=300, bbox_inches='tight')
plt.show()



# ============================================
# B. Marginal Impact Plots
# ============================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
results_df['Cumulative_Emissions_Mtons'] = results_df['Cumulative_Emissions'] / 1e6

# Define tailored palettes
efficiency_palette = ['#a1d99b', '#74c476', '#238b45']
utilization_palette = ['#9ECAE1', '#6BAED6', '#2171B5']
scenario_palette = ['#d62728', '#ff7f0e', '#1f77b4']

# Efficiency
sns.boxplot(x='annual_improvement', y='Cumulative_Emissions_Mtons', 
            data=results_df, ax=axes[0], palette=efficiency_palette)
axes[0].set_title("Impact of Efficiency Gains", pad=12)
axes[0].set_ylabel("Cumulative Emissions (Gt CO₂)", labelpad=10)
axes[0].set_xlabel("Annual Efficiency Improvement (%)", labelpad=10)

# Utilization
sns.boxplot(x='utilization_rate', y='Cumulative_Emissions_Mtons', 
            data=results_df, ax=axes[1], palette=utilization_palette)
axes[1].set_title("Impact of Utilization Rate", pad=12)
axes[1].set_ylabel("")
axes[1].set_xlabel("Utilization Rate (%)", labelpad=10)

# Scenario
sns.boxplot(x='scenario', y='Cumulative_Emissions_Mtons', 
            data=results_df, ax=axes[2],
            order=['business_as_usual', 'conservative', 'aggressive'],
            palette=scenario_palette)
axes[2].set_xticklabels(['Business as Usual', 'Conservative', 'Aggressive'])
axes[2].set_title("Impact of Transition Strategy", pad=12)
axes[2].set_ylabel("")
axes[2].set_xlabel("Scenario Type", labelpad=10)

# Formatting
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# ============================================
# Emission evolution
# ============================================

# 1. First, let's properly create combined_df by collecting all simulation results
all_simulations = []

# Loop through each scenario and run the simulation
for _, scenario_params in results_df.iterrows():
    # Run the simulation for this scenario
    sim_results = simulate_steel_transition(bf_fleet, **scenario_params)
    
    # Add scenario parameters to the results
    for param, value in scenario_params.items():
        sim_results[param] = value
    
    # Store the results
    all_simulations.append(sim_results)

# Combine all results into one DataFrame
combined_df = pd.concat(all_simulations)

# ---- Aggregate EF table (median and quartiles) for selected years ----
def _write_aggregate_ef_table(df: pd.DataFrame) -> None:
    target_years = {2030, 2035, 2040, 2045, 2050}
    try:
        sub = df[df['Year'].isin(target_years)].copy()
        if sub.empty:
            return
        stats = (
            sub.groupby(['scenario', 'Year'])['Emissions_Intensity']
               .agg(median='median', q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
               .reset_index()
               .sort_values(['scenario', 'Year'])
        )
        base_dir = _os.getenv('FORGE_TABLE_DIR', 'results/tables')
        out_dir = _Path(base_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'aggregate_ef.csv'
        stats.to_csv(out_path, index=False)
        print(f"[table] aggregate EF written to {out_path}")
    except Exception as e:
        print(f"[table] failed to write aggregate EF: {e}")

_write_aggregate_ef_table(combined_df)

plt.figure(figsize=(12, 7))

# Define consistent styling with full lines
SCENARIO_STYLE = {
    'business_as_usual': {
        'color': '#d62728',  # Red
        'band_alpha': 0.2,
        'line_style': '-',  # Full line
        'line_width': 2.5
    },
    'conservative': {
        'color': '#ff7f0e',  # Orange
        'band_alpha': 0.15,
        'line_style': '-',  # Full line
        'line_width': 2.5
    },
    'aggressive': {
        'color': '#1f77b4',  # Blue
        'band_alpha': 0.1,
        'line_style': '-',  # Full line
        'line_width': 2.5
    }
}

# Create a dictionary to store annotation positions
annotation_positions = {}

for scenario, style in SCENARIO_STYLE.items():
    sc_data = combined_df[combined_df['scenario'] == scenario]
    
    # Calculate stats from THE SAME grouped object
    grouped = sc_data.groupby('Year')['Emissions_Intensity']
    stats = grouped.agg(['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    stats.columns = ['median', 'q25', 'q75']
    
    # Plot bands and line WITHOUT offset
    plt.fill_between(
        stats.index,
        stats['q25'],
        stats['q75'],
        color=style['color'],
        alpha=style['band_alpha'],
        label='_nolegend_'
    )
    
    plt.plot(
        stats.index,
        stats['median'],
        color=style['color'],
        linestyle='-',
        linewidth=style['line_width'],
        label=scenario.replace('_', ' ').title()
    )
    # Store annotation positions for this scenario
    annotation_positions[scenario] = {}
    for year in [2030, 2040, 2050]:
        if year in stats.index:
            annotation_positions[scenario][year] = stats.loc[year, 'median']

    # Add annotations with smart placement
for year in [2030, 2040, 2050]:
    # Collect all values at this year
    year_values = [pos[year] for pos in annotation_positions.values() if year in pos]
    
    # Calculate vertical offset step (3% of y-range)
    y_min, y_max = plt.ylim()
    y_range = y_max - y_min
    offset_step = 0.0 * y_range
    
    # Annotate each scenario
    for i, (scenario, style) in enumerate(SCENARIO_STYLE.items()):
        if year in annotation_positions[scenario]:
            y_val = annotation_positions[scenario][year]
            # Calculate vertical position with offset
            y_offset = y_val + (i - 1) * offset_step  # Center conservative, offset others
            
            plt.annotate(
                f"{y_val:.2f}",
                (year, y_val),
                xytext=(year, y_offset),
                color=style['color'],
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    fc='white',
                    ec=style['color'],
                    alpha=1,
                    lw=1
                )
            )
            # Add marker on the line
            #plt.scatter(year, y_val, color=style['color'], s=40, zorder=5)

plt.title('Emission intensity evolution for all scenarios with uncertainty band', pad=15)
plt.xlabel('Year')
plt.ylabel('Emission Intensity (tCO₂/t steel)')
plt.legend(frameon=True)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ============================================
# Transition year impact
# ============================================

def simulate_aggressive_transition(bf_fleet, transition_year, **params):
    """Run aggressive scenario where plants convert to DRI at relining after transition year"""
    results = []
    coal_bfs = bf_fleet[bf_fleet["Fuel"] == "coal"].copy()
    original_coal_cap = coal_bfs["Capacity"].sum()
    
    active_coal_bfs = coal_bfs.copy()
    dri_cap = 0  # Track DRI capacity directly
    
    for year in range(2025, 2051):
        # Get plants reaching relining year
        relining_mask = active_coal_bfs["Relining"] == year
        relining_bfs = active_coal_bfs[relining_mask]
        
        # Before transition: keep all coal plants
        if year < transition_year:
            coal_cap = active_coal_bfs["Capacity"].sum()
        
        # After transition: convert relining plants
        else:
            # Calculate conversion
            converted_cap = relining_bfs["Capacity"].sum()
            dri_cap += converted_cap
            active_coal_bfs = active_coal_bfs[~relining_mask]
            coal_cap = active_coal_bfs["Capacity"].sum()
            
            # Validate immediately
            if not np.isclose(coal_cap + dri_cap, original_coal_cap, rtol=0.01):
                print(f"Capacity error during {year} conversion: Coal={coal_cap:.1f}, DRI={dri_cap:.1f}")
                dri_cap = original_coal_cap - coal_cap  # Force balance
        
        # Emissions calculations
        utilization = params.get('utilization_rate', 0.8)
        coal_ef = get_emission_factor(
            "BF-BOF", "Coke", year,
            base_year=2025,
            annual_improvement=params.get('annual_improvement', 0.0)
        )
        coal_emis = coal_cap * utilization * coal_ef
        
        # DRI emissions (only if capacity exists)
        dri_emis = 0
        if dri_cap > 0:
            current_mix = next(
                (params['dri_mix'][y] for y in sorted(params['dri_mix'].keys(), reverse=True) 
                if y <= year), None
            )
            if current_mix:
                for config, share in current_mix.items():
                    dri_ef = get_emission_factor(
                        "DRI-EAF", config, year,
                        base_year=2025,
                        annual_improvement=params.get('annual_improvement', 0.0)
                    )
                    dri_emis += dri_cap * share * utilization * dri_ef
        
        results.append({
            'Year': year,
            'Transition_Year': transition_year,
            'BF_Coal_Capacity': coal_cap,
            'DRI_Capacity': dri_cap,
            'Total_Emissions': coal_emis + dri_emis
        })
    
    return pd.DataFrame(results)

# Run simulations with validation
transition_years = range(2026, 2050)
all_results = []

base_params = {
    'scenario': 'aggressive',
    'utilization_rate': 0.8,
    'annual_improvement': 0.01,
    'dri_mix': {
        2025: {"Natural Gas": 1.0},  # Baseline
        2030: {"Natural Gas": 0.7, "Biomethane-100": 0.1, "Green H2": 0.2},
        2040: {"Natural Gas": 0.4, "Biomethane-100": 0.2, "Green H2": 0.4}
    }
}

for tyear in tqdm(transition_years, desc="Testing transition years"):
    df = simulate_aggressive_transition(bf_fleet, tyear, **base_params)
    df['Cumulative_Emissions'] = df['Total_Emissions'].cumsum()
    all_results.append(df)

transition_results = pd.concat(all_results)

# Data validation
print("\nData validation:")
print(f"Total emissions range: {transition_results['Total_Emissions'].min()/1e9:.1f} to {transition_results['Total_Emissions'].max()/1e9:.1f} Gt")
print(f"Number of simulations: {len(transition_results['Transition_Year'].unique())}")

# Correct plotting
final_emissions = (transition_results
                  .groupby('Transition_Year')['Cumulative_Emissions']
                  .last() / 1000000)  # Now correctly in Gt

# Filter to relevant years
years_to_plot = [2030, 2035, 2040, 2045]
subset = final_emissions.loc[years_to_plot].sort_index()

# Color map
norm = plt.Normalize(min(subset.index), max(subset.index))
cmap = cm.get_cmap('Greens')
colors = [cmap(norm(year)) for year in subset.index]

# Plot
plt.figure(figsize=(10, 5))
bars = plt.barh(subset.index.astype(str), subset.values, color=colors, edgecolor='black')

# Label positioning with custom colors
for bar, val, label in zip(bars, subset.values, subset.index):
    bar_width = bar.get_width()
    text = f"{val:.2f} Gt"

    if bar_width < 1.8:
        x_pos = bar_width + 0.05
        ha = 'left'
    else:
        x_pos = bar_width - 0.05
        ha = 'right'

    # Custom color: white for 2030 and 2035, black otherwise
    if str(label) in ['2030', '2035']:
        color = 'black'
    else:
        color = 'white'

    plt.text(x_pos, bar.get_y() + bar.get_height() / 2,
             text, va='center', ha=ha, fontsize=14,
             fontweight='bold', color=color)

# Axis and layout
plt.xlabel('Cumulative CO₂ Emissions (Gt)')
plt.ylabel('Transition Start Year')
plt.title('Cumulative Emissions Increase with Delayed Transition\n(2025–2050, Aggressive Scenario)', fontsize=14, pad=15)
#plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# ============================================
# Capacity plot
# ============================================

def plot_capacity_comparison(results_df):
    """Visualize capacity by technology using pre-calculated scenarios"""
    # Filter scenarios
    scenarios = {
        'business_as_usual': {'None': None, '+Charcoal': None},
        'conservative': {'None': None, '+Charcoal': None}, 
        'aggressive': {'None': None, '+Charcoal': None}
    }
    
    # Find matching scenarios in results
    for idx, row in results_df.iterrows():
        scenario = row['scenario']
        charcoal = '+Charcoal' if row['charcoal_expansion'] is not None else 'None'
        scenarios[scenario][charcoal] = row
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plt.suptitle('Steel Production Capacity by Technology With/Without Charcoal Expansion', y=1.00, fontsize=14)
    
    # Plot each combination
    for i, scenario in enumerate(['business_as_usual', 'conservative', 'aggressive']):
        for j, charcoal in enumerate(['None', '+Charcoal']):
            ax = axes[i,j]
            data = scenarios[scenario][charcoal]
            
            if data is None:
                ax.axis('off')
                continue
                
            # Get simulation results (assuming you stored them)
            sim_results = simulate_steel_transition(bf_fleet, **data)
            
            # Stacked area plot
            ax.stackplot(sim_results['Year'],
                        sim_results['BF_Coal_Capacity']/1000,
                        sim_results['BF_Charcoal_Capacity']/1000, 
                        sim_results['DRI_Capacity']/1000,
                        colors = ['#636EFA', '#F5C518', '#00CC96'] ,  # muted reddish, orange, blue,
                        labels=['Coal-BF', 'Charcoal-BF', 'DRI'],
                        alpha=0.8)
            
            # Formatting
            if charcoal == 'None':
                ax.set_title(f"{scenario.replace('_', ' ').title()}")
            else:
                ax.set_title(f"{scenario.replace('_', ' ').title()} ({charcoal})")
            ax.set_xlabel('Year')
            ax.set_ylabel('Capacity (Mtpa)')
            ax.grid(alpha=0.3)
            for spine in ax.spines.values():
                spine.set_visible(False)
        # Add legend only once — above the first graph
        
            fig.legend(['Coal-BF', 'Charcoal-BF', 'DRI'],
                       loc='upper center',
                       bbox_to_anchor=(0.5, 0.99),  # Adjust X (horizontal) and Y (vertical) position
                       ncol=3,
                       frameon=False,
                       fontsize=12)
               
             
    plt.tight_layout()
    plt.show()

# Generate the plot
plot_capacity_comparison(results_df)

# ============================================
# Tornado Chart of Emission Drivers
# ============================================

# ============================================
# Tornado Chart of Emission Drivers (Cumulative)
# ============================================

def prepare_tornado_data(results_df):
    """Calculate the impact of each parameter on cumulative emissions"""
    
    # Calculate baseline emissions (median of all scenarios) in Gt CO₂
    baseline = results_df['Cumulative_Emissions'].median() / 1000000
    impact_data = []
    
    # Analyze each parameter's impact
    for param in ['utilization_rate', 'annual_improvement', 'dri_mix', 'charcoal_expansion']:
        
        # For efficiency parameters
        if param == 'annual_improvement':
            low_val = results_df[param].min()
            high_val = results_df[param].max()
            
            # Get emissions at low and high values
            low_emis = results_df[results_df[param] == low_val]['Cumulative_Emissions'].median() / 1000000
            high_emis = results_df[results_df[param] == high_val]['Cumulative_Emissions'].median() / 1000000
            
            impact_data.append({
                'Parameter': 'Efficiency: Low',
                'Impact': (low_emis - baseline),
                'Type': 'Reduction' if low_emis < baseline else 'Increase'
            })
            
            impact_data.append({
                'Parameter': 'Efficiency: High',
                'Impact': (high_emis - baseline),
                'Type': 'Reduction' if high_emis < baseline else 'Increase'
            })
        
        # For utilization rate
        elif param == 'utilization_rate':
            low_val = results_df[param].min()
            high_val = results_df[param].max()
            
            low_emis = results_df[results_df[param] == low_val]['Cumulative_Emissions'].median() / 1000000
            high_emis = results_df[results_df[param] == high_val]['Cumulative_Emissions'].median() / 1000000
            
            impact_data.append({
                'Parameter': 'Utilization: Low',
                'Impact': (low_emis - baseline),
                'Type': 'Reduction' if low_emis < baseline else 'Increase'
            })
            
            impact_data.append({
                'Parameter': 'Utilization: High',
                'Impact': (high_emis - baseline),
                'Type': 'Reduction' if high_emis < baseline else 'Increase'
            })
        
        # For charcoal expansion
        elif param == 'charcoal_expansion':
            # Get unique configurations
            unique_configs = results_df[param].astype(str).unique()
            
            for config in unique_configs:
                if config == 'None':
                    label = 'No Expansion'
                else:
                    label = 'Expansion'
                
                config_emis = results_df[results_df[param].astype(str) == config]['Cumulative_Emissions'].median() / 1000000
                
                impact_data.append({
                    'Parameter': f'Charcoal: {label}',
                    'Impact': (config_emis - baseline),
                    'Type': 'Reduction' if config_emis < baseline else 'Increase'
                })
        
        # For DRI mix
        elif param == 'dri_mix':
            # Get unique configurations
            unique_configs = results_df[param].astype(str).unique()
            
            for config in unique_configs:
                # Classify as blue (natural gas) or green (hydrogen) mix
                if 'Natural Gas' in config and 'Green H2' in config:
                    # Determine dominant fuel
                    if config.count('Natural Gas') > config.count('Green H2'):
                        label = 'Blue Mix'
                    else:
                        label = 'Green Mix'
                elif 'Natural Gas' in config:
                    label = 'Blue Mix'
                elif 'Green H2' in config:
                    label = 'Green Mix'
                else:
                    label = 'Other Mix'
                
                config_emis = results_df[results_df[param].astype(str) == config]['Cumulative_Emissions'].median() / 1000000
                
                impact_data.append({
                    'Parameter': f'DRI Mix: {label}',
                    'Impact': (config_emis - baseline),
                    'Type': 'Reduction' if config_emis < baseline else 'Increase'
                })
    
    return pd.DataFrame(impact_data)

# Prepare the data
tornado_df = prepare_tornado_data(results_df)

# Sort by absolute impact magnitude
tornado_df['Abs_Impact'] = tornado_df['Impact'].abs()
tornado_df = tornado_df.sort_values('Abs_Impact', ascending=True).drop('Abs_Impact', axis=1)

# Create the tornado chart with vibrant colors
plt.figure(figsize=(10, 8))

# Vibrant color scheme
colors = {
    'Increase': '#FF6B6B',  # Bright coral red
    'Reduction': '#4ECDC4'  # Bright turquoise
}

# Plot horizontal bars with more vibrant colors
for i, row in tornado_df.iterrows():
    plt.barh(
        row['Parameter'], 
        row['Impact'], 
        color=colors[row['Type']], 
        alpha=0.85,  # Slightly more opaque
        edgecolor='white',
        linewidth=0.8  # Slightly thicker border
    )

# Add zero line and labels
plt.axvline(0, color='#333333', linestyle='-', linewidth=1.5)  # Darker zero line
plt.xlabel('Impact on Cumulative Emissions (Gt CO₂)', fontsize=12, fontweight='bold')
plt.title('Key Drivers of Steel Sector Cumulative Emissions (2025-2050)', 
         fontsize=14, pad=20, fontweight='bold')
plt.grid(axis='x', alpha=0.2)

# Remove spines for cleaner look
for spine in ['top', 'right', 'left']:
    plt.gca().spines[spine].set_visible(False)
    
# Add subtle background color
plt.gca().set_facecolor('#F8F9FA')  # Very light gray background

# Add professional legend
legend_elements = [
    Patch(facecolor='#4ECDC4', label='Lowers Emissions', alpha=0.85),
    Patch(facecolor='#FF6B6B', label='Raises Emissions', alpha=0.85)
]
plt.legend(
    handles=legend_elements, 
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=2,
    frameon=False,
    fontsize=11
)

plt.tight_layout()
plt.show()

# ============================================
# Tornado Chart for Emission Intensity Drivers
# ============================================

def prepare_intensity_tornado_data(results_df):
    """Calculate the impact of each parameter on final emission intensity"""
    
    # Calculate baseline intensity (median of all scenarios)
    baseline = results_df['Final_Intensity'].median()
    
    impact_data = []
    
    # Analyze each parameter's impact
    for param in ['utilization_rate', 'annual_improvement', 'dri_mix', 'charcoal_expansion']:
        
        # For efficiency parameters
        if param == 'annual_improvement':
            low_val = results_df[param].min()
            high_val = results_df[param].max()
            
            # Get intensity at low and high values
            low_intensity = results_df[results_df[param] == low_val]['Final_Intensity'].median()
            high_intensity = results_df[results_df[param] == high_val]['Final_Intensity'].median()
            
            impact_data.append({
                'Parameter': 'Efficiency: Low',
                'Impact': (low_intensity - baseline),
                'Type': 'Reduction' if low_intensity < baseline else 'Increase'
            })
            
            impact_data.append({
                'Parameter': 'Efficiency: High',
                'Impact': (high_intensity - baseline),
                'Type': 'Reduction' if high_intensity < baseline else 'Increase'
            })
        
        # For utilization rate
        elif param == 'utilization_rate':
            low_val = results_df[param].min()
            high_val = results_df[param].max()
            
            low_intensity = results_df[results_df[param] == low_val]['Final_Intensity'].median()
            high_intensity = results_df[results_df[param] == high_val]['Final_Intensity'].median()
            
            impact_data.append({
                'Parameter': 'Utilization: Low',
                'Impact': (low_intensity - baseline),
                'Type': 'Reduction' if low_intensity < baseline else 'Increase'
            })
            
            impact_data.append({
                'Parameter': 'Utilization: High',
                'Impact': (high_intensity - baseline),
                'Type': 'Reduction' if high_intensity < baseline else 'Increase'
            })
        
        # For charcoal expansion
        elif param == 'charcoal_expansion':
            # Get unique configurations
            unique_configs = results_df[param].astype(str).unique()
            
            for config in unique_configs:
                if config == 'None':
                    label = 'No Expansion'
                else:
                    label = 'Expansion'
                
                config_intensity = results_df[results_df[param].astype(str) == config]['Final_Intensity'].median()
                
                impact_data.append({
                    'Parameter': f'Charcoal: {label}',
                    'Impact': (config_intensity - baseline),
                    'Type': 'Reduction' if config_intensity < baseline else 'Increase'
                })
        
        # For DRI mix
        elif param == 'dri_mix':
            # Get unique configurations
            unique_configs = results_df[param].astype(str).unique()
            
            for config in unique_configs:
                # Classify as blue (natural gas) or green (hydrogen) mix
                if 'Natural Gas' in config and 'Green H2' in config:
                    # Determine dominant fuel
                    if config.count('Natural Gas') > config.count('Green H2'):
                        label = 'Blue Mix'
                    else:
                        label = 'Green Mix'
                elif 'Natural Gas' in config:
                    label = 'Blue Mix'
                elif 'Green H2' in config:
                    label = 'Green Mix'
                else:
                    label = 'Other Mix'
                
                config_intensity = results_df[results_df[param].astype(str) == config]['Final_Intensity'].median()
                
                impact_data.append({
                    'Parameter': f'DRI Mix: {label}',
                    'Impact': (config_intensity - baseline),
                    'Type': 'Reduction' if config_intensity < baseline else 'Increase'
                })
    
    return pd.DataFrame(impact_data)

# Prepare the data
tornado_df = prepare_intensity_tornado_data(results_df)

# Sort by absolute impact magnitude
tornado_df['Abs_Impact'] = tornado_df['Impact'].abs()
tornado_df = tornado_df.sort_values('Abs_Impact', ascending=True).drop('Abs_Impact', axis=1)

# Create the tornado chart with vibrant colors
plt.figure(figsize=(10, 8))

# Vibrant color scheme
colors = {
    'Increase': '#FF6B6B',  # Bright coral red
    'Reduction': '#4ECDC4'  # Bright turquoise
}

# Plot horizontal bars with more vibrant colors
for i, row in tornado_df.iterrows():
    plt.barh(
        row['Parameter'], 
        row['Impact'], 
        color=colors[row['Type']], 
        alpha=0.85,  # Slightly more opaque
        edgecolor='white',
        linewidth=0.8  # Slightly thicker border
    )

# Add zero line and labels
plt.axvline(0, color='#333333', linestyle='-', linewidth=1.5)  # Darker zero line
plt.xlabel('Impact on Emission Intensity (tCO₂/t steel)', fontsize=12, fontweight='bold')
plt.title('Key Drivers of Steel Sector Emissions Intensity (2025-2050)', 
         fontsize=14, pad=20, fontweight='bold')
plt.grid(axis='x', alpha=0.2)

# Remove spines for cleaner look
for spine in ['top', 'right', 'left']:
    plt.gca().spines[spine].set_visible(False)
    
# Add subtle background color
plt.gca().set_facecolor('#F8F9FA')  # Very light gray background

# Add professional legend
legend_elements = [
    Patch(facecolor='#4ECDC4', label='Lowers Emissions', alpha=0.85),
    Patch(facecolor='#FF6B6B', label='Raises Emissions', alpha=0.85)
]
plt.legend(
    handles=legend_elements, 
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=2,
    frameon=False,
    fontsize=11
)

plt.tight_layout()
plt.show()
