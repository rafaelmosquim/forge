#!/usr/bin/env python3
"""
Validation runner for assumption‑explicit source cards.

Usage
-----
python scripts/validate_cards.py \
  --cards validation/source_cards/eaf_scrap_card.yml \
  --data data \
  --out artifacts/validation

- Reads one or more YAML "source cards" that define a literature boundary.
- Projects the rich model to that boundary (route, stage, EF overrides, etc.).
- Runs the engine (no interactivity), computes EF (tCO2/t), and compares to target/band.
- Writes a CSV summary and per‑card packet (inputs + results) under --out.

Notes
-----
This script assumes you renamed your original CLI engine to `steel_model_core.py` and kept
all YAML data under the `data/` folder.
"""
from __future__ import annotations

# Ensure we can import steel_model_core from the repo root (parent of scripts/)
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("DEBUG sys.path[0]=", sys.path[0])
print("DEBUG repo root  =", ROOT)

import argparse
import json
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import pandas as pd
import yaml

# --- Import your engine pieces ------------------------------------------------
from steel_model_core import (
    # Data models & loaders
    Process,
    load_data_from_yaml,
    load_parameters,
    load_recipes_from_yaml,
    load_market_config,
    load_electricity_intensity,
    apply_fuel_substitutions,
    apply_dict_overrides,
    apply_recipe_overrides,
    # Calculations
    adjust_blast_furnace_intensity,
    adjust_process_gas_intensity,
    calculate_balance_matrix,
    calculate_energy_balance,
    calculate_internal_electricity,
    adjust_energy_balance,
    calculate_emissions,
    # Routing helpers & constants
    STAGE_MATS,
    build_route_mask,
    enforce_eaf_feed,
    expand_energy_tables_for_active,
    build_routes_interactive,
)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

CARD_FIELDS_REQUIRED = ["id", "boundary", "model_projection"]


def _read_card(path: pathlib.Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        card = yaml.safe_load(f) or {}
    for key in CARD_FIELDS_REQUIRED:
        if key not in card:
            raise ValueError(f"Card {path} missing required field: {key}")
    return card


def _resolve_electricity_ef(card: Dict[str, Any], elec_map: Dict[str, float]) -> float | None:
    el = card.get("electricity", {}) or {}
    # If value provided directly, use it.
    direct = el.get("value_gCO2_per_MJ")
    if isinstance(direct, (int, float)):
        return float(direct)
    # Else resolve by country_code using electricity_intensity.yml
    code = (el.get("country_code") or "").upper()
    if code and code in elec_map:
        return float(elec_map[code])
    return None


def _apply_emission_factor_overrides(e_efs: Dict[str, float], overrides: Dict[str, Any], elec_map: Dict[str, float]):
    if not overrides:
        return
    pat = re.compile(r"^@electricity_intensity\[([A-Za-z]{3})\]$")
    for k, v in overrides.items():
        if isinstance(v, (int, float)):
            e_efs[k] = float(v)
        elif isinstance(v, str):
            m = pat.match(v.strip())
            if m:
                code = m.group(1).upper()
                if code in elec_map:
                    e_efs[k] = float(elec_map[code])
        # ignore others

def _run_once_with_bf(base_energy_int: dict, bf_base: float, runner):
    """Copy energy_int, set BF base intensity, then run the model via `runner`."""
    ei = dict(base_energy_int)
    ei["Blast Furnace"] = float(bf_base)
    return runner(ei)


@dataclass
class RunResult:
    card_id: str
    reported_ef: float | None
    model_ef: float
    diff: float | None
    pct_diff: float | None
    pass_rule: str
    pass_flag: bool | None
    total_kg_co2e: float
    demand_kg: float


# ----------------------------------------------------------------------------
# Core run (rebuilt from scratch: deterministic bans & selections)
# ----------------------------------------------------------------------------

def run_card(card_path: pathlib.Path, data_dir: pathlib.Path, out_dir: pathlib.Path) -> RunResult:
    # --- Load card ------------------------------------------------------------
    card = _read_card(card_path)
    card_id = card.get("id") or card_path.stem

    # --- Load base configs ----------------------------------------------------
    base = os.path.join(str(data_dir), "")
    energy_int      = load_data_from_yaml(os.path.join(base, "energy_int.yml"))
    energy_shares   = load_data_from_yaml(os.path.join(base, "energy_matrix.yml"))
    energy_content  = load_data_from_yaml(os.path.join(base, "energy_content.yml"))
    e_efs           = load_data_from_yaml(os.path.join(base, "emission_factors.yml"))
    params          = load_parameters(      os.path.join(base, "parameters.yml"))
    mkt_cfg         = load_market_config(   os.path.join(base, "mkt_config.yml"))
    elec_map        = load_electricity_intensity(os.path.join(base, "electricity_intensity.yml"))
    process_efs     = load_data_from_yaml(os.path.join(base, "process_emissions.yml"))

    # Electricity EF resolution (card > map) + optional EF overrides
    ef_elec = _resolve_electricity_ef(card, elec_map)
    if ef_elec is not None:
        e_efs["Electricity"] = float(ef_elec)
    _apply_emission_factor_overrides(
        e_efs,
        (card.get("model_projection") or {}).get("emission_factors_overrides", {}),
        elec_map,
    )

    # --- Read model projection & demand ---------------------------------------
    boundary = card.get("boundary", {}) or {}
    proj     = card.get("model_projection", {}) or {}

    stage_key = proj.get("demand", {}).get("stage_key") or boundary.get("stop_stage") or "Finished"
    if stage_key not in STAGE_MATS:
        raise ValueError(f"Unknown stage_key '{stage_key}'. Must be one of: {list(STAGE_MATS)}")
    demand_mat = STAGE_MATS[stage_key]
    demand_kg  = float(proj.get("demand", {}).get("quantity_kg", 1000.0))
    final_demand = {demand_mat: demand_kg}

    route = str(proj.get("route_preset", "auto"))

    # --- One-shot engine run ---------------------------------------------------
    def _engine_run(energy_int_override: dict | None = None):
        # Load recipes with the effective energy_int
        ei = dict(energy_int if energy_int_override is None else energy_int_override)
        recipes = load_recipes_from_yaml(os.path.join(base, "recipes.yml"), params, ei, energy_shares, energy_content)

        # Route-specific feed constraints (EAF feed only)
        if route == "EAF-Scrap":
            enforce_eaf_feed(recipes, "scrap")
        elif route == "DRI-EAF":
            enforce_eaf_feed(recipes, "dri")

        # Apply recipe overrides once
        recipes = apply_recipe_overrides(recipes, proj.get("recipe_overrides", {}), params, ei, energy_shares, energy_content)

        # Intensity adjustments that may affect expressions → reload once
        adjust_blast_furnace_intensity(ei, energy_shares, params)
        adjust_process_gas_intensity("Coke Production", "process_gas_coke", ei, energy_shares, params)
        recipes = load_recipes_from_yaml(os.path.join(base, "recipes.yml"), params, ei, energy_shares, energy_content)
        recipes = apply_recipe_overrides(recipes, proj.get("recipe_overrides", {}), params, ei, energy_shares, energy_content)
        recipes_dict = {r.name: r for r in recipes}

        # ---- Helper: merge masks (0 dominates) --------------------------------
        def merge_masks(upstream_mask: dict[str,int], downstream_mask: dict[str,int]) -> dict[str,int]:
            out = dict(upstream_mask)
            for k, v in downstream_mask.items():
                if v == 0:
                    out[k] = 0
                elif v == 1:
                    if out.get(k, 0) != 0:  # do not resurrect bans
                        out[k] = 1
            return out

        # ---- Helper: assert exactly one producer for a given product ----------
        def assert_exactly_one_selected(mask: dict[str,int], product: str):
            producers = [r.name for r in recipes if product in (r.outputs or {})]
            selected  = [p for p in producers if mask.get(p, 0) == 1]
            if len(selected) != 1:
                raise ValueError(
                    f"Need exactly one producer for '{product}', got {selected or 'none'} from {producers}"
                )

        # ---- Downstream deterministic clamps ----------------------------------
        downstream: dict[str,int] = {}

        def _on(name: str):  # enable if recipe exists
            if name in recipes_dict: downstream[name] = 1
        def _off(name: str): # disable if recipe exists
            if name in recipes_dict: downstream[name] = 0

        # CC variant
        cc_map = {"R": "Continuous Casting (R)", "L": "Continuous Casting (L)", "H": "Continuous Casting (H)"}
        cc_choice = cc_map.get(str(proj.get("cc_variant", "R")).upper(), "Continuous Casting (R)")
        for n in cc_map.values(): _off(n)
        _on(cc_choice)

        # Rolling family
        rb = str(proj.get("rolling_block", "hot_strip")).lower()  # hot_strip | plate | long
        if rb in ("hot_strip", "plate"):
            _on("Hot Rolling"); _off("Rod/bar/section Mill")
        else:
            _off("Hot Rolling"); _on("Rod/bar/section Mill")

        # Cold rolling
        use_cr = bool(proj.get("include_cold_rolling", False))
        if "Cold Rolling" in recipes_dict:
            downstream["Cold Rolling"] = 1 if use_cr else 0

        # IP3: choose one on RAW or CR side
        ip3_map  = proj.get("ip3_map") or {}
        side     = "CR" if use_cr else "RAW"
        ip3_key  = str(proj.get("ip3_treatment", "BYPASS")).upper()
        ip3_all  = set((ip3_map.get("RAW") or {}).values()) | set((ip3_map.get("CR") or {}).values())
        chosen_ip3 = (ip3_map.get(side) or {}).get(ip3_key)
        for n in ip3_all: _off(n)
        if chosen_ip3: _on(chosen_ip3)

        # IP4: auto-detect by output ("Manufactured Feed (IP4)") and clamp to one
        def clamp_unique_output(mask: dict[str,int], product: str, preferred: list[str] | None = None) -> tuple[dict[str,int], str]:
            candidates = [r.name for r in recipes if product in (r.outputs or {})]
            if not candidates:
                return mask, ""
            for n in candidates: mask[n] = 0
            chosen = None
            for p in (preferred or []):
                if p in candidates:
                    chosen = p; break
            if chosen is None:
                if "Direct use of Basic Steel Products (IP4)" in candidates:
                    chosen = "Direct use of Basic Steel Products (IP4)"
                else:
                    chosen = sorted(candidates)[0]
            mask[chosen] = 1
            return mask, chosen

        ip4_pref = []
        ip4_map  = proj.get("ip4_map") or {}
        ip4_key  = str(proj.get("ip4_process", "DIRECT_USE")).upper()
        if ip4_map:
            maybe = ip4_map.get(ip4_key) or ip4_map.get("DIRECT_USE")
            if maybe: ip4_pref.append(maybe)
        ip4_pref.append("Direct use of Basic Steel Products (IP4)")
        downstream, chosen_ip4 = clamp_unique_output(downstream, "Manufactured Feed (IP4)", ip4_pref)

        # Finish (auto-detect by output "Finished Products")
        coat_map = proj.get("coating_map") or {}
        coat_choice = str(proj.get("coating_type", "NONE")).upper()
        preferred_finish = []
        if coat_choice in ("", "NONE", "NO_COATING", "NO-COATING"):
            preferred_finish.append("No Coating")
        else:
            maybe = coat_map.get(coat_choice)
            if maybe: preferred_finish.append(maybe)
        # Always consider "No Coating" as a safe fallback
        preferred_finish.append("No Coating")
        downstream, chosen_finish = clamp_unique_output(downstream, "Finished Products", preferred_finish)
        # ---- AUTO-CLAMP: Raw Products (types) ---------------------------------------
        # Reuse the same clamp_unique_output helper you already have above.
        
        # Preference: if rolling_block is hot_strip/plate → Hot Rolling;
        # if long → Rod/bar/section Mill (if present); otherwise fall back to Hot Rolling;
        # keep Ingot Casting only if explicitly chosen.
        preferred_raw = []
        if rb in ("hot_strip", "plate"):
            preferred_raw = ["Hot Rolling", "Rod/bar/section Mill", "Ingot Casting"]
        elif rb == "long":
            preferred_raw = ["Rod/bar/section Mill", "Hot Rolling", "Ingot Casting"]
        else:
            preferred_raw = ["Hot Rolling", "Rod/bar/section Mill", "Ingot Casting"]
        
        downstream, chosen_raw = clamp_unique_output(
            downstream, "Raw Products (types)", preferred=preferred_raw
        )
        print("[DBG] RAW clamp:", {"preferred_order": preferred_raw, "chosen": chosen_raw})
        # ---- AUTO-CLAMP: utilities, fluxes, and fuels --------------------------------
        # Goes after IP4/Finish clamps and before route_mask merge.
        
        import re
        
        def _producer_names_for(product: str) -> list[str]:
            return [r.name for r in recipes if product in (r.outputs or {})]
        
        def _choose_by_patterns(candidates: list[str], patterns: list[str]) -> str | None:
            for pat in patterns:
                rx = re.compile(pat, re.IGNORECASE)
                for c in candidates:
                    if rx.search(c):
                        return c
            return None
        
        def _clamp_product(mask: dict[str,int], product: str, mode: str | None, boundary_cfg: dict) -> tuple[dict,str]:
            """
            Ensure exactly one producer for `product` by name heuristics.
            mode: "market" or "onsite" (default = sensible per item).
            Returns (mask, chosen_name). No-op if <=1 candidate exists.
            """
            candidates = _producer_names_for(product)
            if len(candidates) <= 1:
                return mask, candidates[0] if candidates else ""
        
            # Build preference list (market first unless specified otherwise)
            m = (mode or "market").lower()
        
            def _patterns_for(prod: str) -> list[str]:
                if prod in ("Nitrogen", "Oxygen"):
                    return (["from market|purchase|purchased|market",
                             "air separation|asu|plant|production"]
                            if m == "market" else
                            ["air separation|asu|plant|production",
                             "from market|purchase|purchased|market"])
                if prod in ("Dolomite", "Burnt Lime", "Lime"):
                    return (["from market|purchase|purchased|market",
                             "kiln|calcination|production"]
                            if m == "market" else
                            ["kiln|calcination|production",
                             "from market|purchase|purchased|market"])
                if prod == "Natural Gas":
                    return (["from market|purchase|purchased|market",
                             "production|reformer|generator"]
                            if m == "market" else
                            ["production|reformer|generator",
                             "from market|purchase|purchased|market"])
                if prod == "Coal":
                    return (["from market|purchase|purchased|market",
                             "mining|production"]
                            if m == "market" else
                            ["mining|production",
                             "from market|purchase|purchased|market"])
                if prod == "Coke":
                    prefer_onsite = bool(boundary_cfg.get("include_coke", False))
                    return (["production|coking", "from market|purchase|purchased|market"]
                            if prefer_onsite or m == "onsite" else
                            ["from market|purchase|purchased|market", "production|coking"])
                # generic fallback
                return ["from market|purchase|purchased|market",
                        "production|plant|kiln|calcination|asu|air separation"]
        
            pats = _patterns_for(product)
        
            # Ban all, then enable the first that matches preference patterns
            for n in candidates:
                mask[n] = 0
            chosen = _choose_by_patterns(candidates, pats)
        
            if chosen is None:
                # sensible hard fallback
                canonical = f"{product} from market"
                chosen = canonical if canonical in candidates else sorted(candidates)[0]
        
            mask[chosen] = 1
            print(f"[DBG] clamp {product}: candidates={candidates}, chosen={chosen}, mode={mode or 'default'}")
            return mask, chosen
        
        sourcing = (proj.get("sourcing") or {})
        
        downstream, _ = _clamp_product(downstream, "Nitrogen",    sourcing.get("nitrogen"),    boundary)
        downstream, _ = _clamp_product(downstream, "Oxygen",      sourcing.get("oxygen"),      boundary)
        downstream, _ = _clamp_product(downstream, "Dolomite",    sourcing.get("dolomite"),    boundary)
        downstream, _ = _clamp_product(downstream, "Burnt Lime",  sourcing.get("burnt_lime") or sourcing.get("lime"), boundary)
        downstream, _ = _clamp_product(downstream, "Natural Gas", sourcing.get("natural_gas"), boundary)
        downstream, _ = _clamp_product(downstream, "Coal",        sourcing.get("coal"),        boundary)
        downstream, _ = _clamp_product(downstream, "Coke",        sourcing.get("coke"),        boundary)

        

        # ---- Upstream route mask & final merge --------------------------------
        route_mask = build_route_mask(route, recipes)
        final_mask = merge_masks(route_mask, downstream)

        # Honor explicit force-offs (once, at the end)
        for name in (proj.get("production_routes_forced_off") or []):
            if name in recipes_dict:
                final_mask[name] = 0

        # **Optional boundary bans** (if you later want them, they go here)
        # e.g., if not boundary.get("include_processing", True): ban all coater/finishers by output scan

        # Sanity: assert uniqueness for the key ambiguous products we clamped
        for prod in ("Manufactured Feed (IP4)", "Finished Products"):
            assert_exactly_one_selected(final_mask, prod)

        # ---- Build routes & solve ---------------------------------------------
        production_routes = build_routes_interactive(
            recipes=recipes,
            demand_mat=demand_mat,
            pre_mask=final_mask,
            interactive=False
        )

        balance_matrix, prod_lvl = calculate_balance_matrix(recipes, final_demand, production_routes)

        # Expand energy tables for the active processes
        active_procs = [p for p, r in prod_lvl.items() if r > 1e-9]
        expand_energy_tables_for_active(active_procs, energy_shares, ei)

        # Internal electricity from recovered gas BEFORE credit
        internal_elec = calculate_internal_electricity(prod_lvl, recipes_dict, params)

        # Energy balance (+ credit)
        energy_balance = calculate_energy_balance(prod_lvl, ei, energy_shares)
        energy_balance = adjust_energy_balance(energy_balance, internal_elec)

        # Recovered-gas EF for internal electricity (Coke+BF mix)
        coke_runs = float(prod_lvl.get("Coke Production", 0.0))
        bf_runs   = float(prod_lvl.get("Blast Furnace", 0.0))
        gas_coke_MJ = coke_runs * recipes_dict.get("Coke Production", Process("", {}, {})).outputs.get("Process Gas", 0.0)
        gas_bf_MJ   = (float(getattr(params, "bf_adj_intensity", 0.0)) - float(getattr(params, "bf_base_intensity", 0.0))) * bf_runs
        total_gas_MJ = float(gas_coke_MJ + gas_bf_MJ)

        def _mix_ef(shares: dict[str, float]) -> float:
            fuels = [c for c, s in shares.items() if c != "Electricity" and s > 0]
            if not fuels: return 0.0
            num = sum(shares[c] * float(e_efs.get(c, 0.0)) for c in fuels)
            den = sum(shares[c] for c in fuels)
            return num / max(1e-12, den)

        EF_coke_gas = _mix_ef(energy_shares.get("Coke Production", {}))
        EF_bf_gas   = _mix_ef(energy_shares.get("Blast Furnace", {}))
        EF_process_gas = 0.0 if total_gas_MJ <= 1e-9 else (
            (EF_coke_gas * (gas_coke_MJ / total_gas_MJ)) + (EF_bf_gas * (gas_bf_MJ / total_gas_MJ))
        )

        # Emissions
        emissions = calculate_emissions(
            mkt_cfg,
            prod_lvl,
            energy_balance,
            e_efs,
            process_efs,
            internal_elec,
            final_demand,
            total_gas_MJ,
            EF_process_gas,
        )
        if emissions is not None and "TOTAL" not in emissions.index:
            emissions.loc["TOTAL"] = emissions.sum()

        total_kg = float(emissions.loc["TOTAL", "TOTAL CO2e"]) if emissions is not None else 0.0
        ef_t_per_t = total_kg / demand_kg if demand_kg else 0.0

        # Persist per-run packet
        return {
            "ef_t_per_t": ef_t_per_t,
            "emissions": emissions,
            "energy_balance": energy_balance,
            "downstream_choices": {
                "cc": cc_choice, "rolling": rb, "cold_rolling": use_cr,
                "ip3": chosen_ip3, "ip4": chosen_ip4, "finish": chosen_finish,
            },
            "final_mask": final_mask,
        }

    # --- Sweep (optional) ------------------------------------------------------
    sweep_vals = (proj.get("sweep", {}) or {}).get("bf_base_mj_per_t", [])
    results_for_sweep = []

    if sweep_vals:
        base_ei = dict(energy_int)  # snapshot
        for val in sweep_vals:
            ei = dict(base_ei)
            ei["Blast Furnace"] = float(val)
            res = _engine_run(ei)
            results_for_sweep.append({"bf": float(val), **res})

        # Summarize band
        efs = [r["ef_t_per_t"] for r in results_for_sweep if r.get("ef_t_per_t") is not None]
        band_min = min(efs) if efs else None
        band_med = sorted(efs)[len(efs)//2] if efs else None
        band_max = max(efs) if efs else None

        # Write sweep CSV
        out_dir_card = (out_dir / card_id)
        out_dir_card.mkdir(parents=True, exist_ok=True)
        import csv
        with open(out_dir_card / "bf_sweep_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["BF_base_MJ_per_t", "EF_tCO2_per_t"])
            for r in results_for_sweep:
                w.writerow([r["bf"], r["ef_t_per_t"]])
            w.writerow([])
            w.writerow(["EF_min", band_min])
            w.writerow(["EF_med", band_med])
            w.writerow(["EF_max", band_max])

        # Save inputs.json snapshot from median run (if available)
        snap = results_for_sweep[len(results_for_sweep)//2] if results_for_sweep else {}
        with open(out_dir_card / "inputs.json", "w", encoding="utf-8") as f:
            json.dump({
                "card": card,
                "resolved": {
                    "stage_key": stage_key,
                    "demand_kg": demand_kg,
                    "electricity_gCO2_per_MJ": e_efs.get("Electricity"),
                    "route_preset": route,
                    "downstream_choices": snap.get("downstream_choices", {}),
                }
            }, f, indent=2)

        # For validator summary, report the band
        expected = card.get("expected_result", {}) or {}
        pass_rule = expected.get("pass_rule", "report min/median/max over BF sweep")
        # Band check is deferred to caller; we just return the central value as ef_model
        return RunResult(
            card_id=card_id,
            reported_ef=None,
            model_ef=band_med if band_med is not None else 0.0,
            diff=None,
            pct_diff=None,
            pass_rule=pass_rule,
            pass_flag=None,  # band reporting only
            total_kg_co2e=0.0,  # not meaningful for band
            demand_kg=demand_kg,
        )

    # --- Single run ------------------------------------------------------------
    single = _engine_run()

    # Save packet
    card_out = out_dir / card_id
    card_out.mkdir(parents=True, exist_ok=True)

    with open(card_out / "inputs.json", "w", encoding="utf-8") as f:
        json.dump({
            "card": card,
            "resolved": {
                "stage_key": stage_key,
                "demand_kg": demand_kg,
                "electricity_gCO2_per_MJ": e_efs.get("Electricity"),
                "route_preset": route,
                "downstream_choices": single.get("downstream_choices", {}),
            }
        }, f, indent=2)

    if single.get("emissions") is not None:
        single["emissions"].to_csv(card_out / "emissions.csv")
    single["energy_balance"].to_csv(card_out / "energy_balance.csv")

    # Compare to expected/band
    expected = card.get("expected_result", {}) or {}
    reported = card.get("reported_ef", None)
    pass_rule = expected.get("pass_rule", "within ±15% of reported_ef OR inside band")
    band      = expected.get("band_tCO2_per_t", {}) or {}
    low, high = band.get("low", None), band.get("high", None)

    model_ef = single["ef_t_per_t"]
    diff = pct = passed = None
    if isinstance(reported, (int, float)):
        diff = model_ef - float(reported)
        pct  = (diff / float(reported)) if float(reported) else None
        tol_ok  = (pct is not None and abs(pct) <= 0.15)
        band_ok = (isinstance(low, (int, float)) and isinstance(high, (int, float)) and low <= model_ef <= high)
        passed  = bool(tol_ok or band_ok)
    elif isinstance(low, (int, float)) and isinstance(high, (int, float)):
        passed = bool(low <= model_ef <= high)

    return RunResult(
        card_id=card_id,
        reported_ef=float(reported) if isinstance(reported, (int, float)) else None,
        model_ef=model_ef,
        diff=diff if isinstance(diff, (int, float)) else None,
        pct_diff=pct if isinstance(pct, (int, float)) else None,
        pass_rule=pass_rule,
        pass_flag=passed,
        total_kg_co2e=(single.get("emissions").loc["TOTAL", "TOTAL CO2e"] if single.get("emissions") is not None else 0.0),
        demand_kg=demand_kg,
    )

    
# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", nargs="+", default=["validation/source_cards/eaf_scrap_card.yml"], help="List of card YAMLs to run")
    ap.add_argument("--data", default="data", help="Data directory (where YAMLs live)")
    ap.add_argument("--out", default="artifacts/validation", help="Output directory for results")
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data).resolve()
    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[RunResult] = []
    for card_path in args.cards:
        p = pathlib.Path(card_path)
        print(f"[RUN] Card: {p}")
        res = run_card(p, data_dir, out_dir)
        results.append(res)
        status = "PASS" if res.pass_flag else ("FAIL" if res.pass_flag is not None else "N/A")
        rep = f", reported={res.reported_ef:.3f}" if res.reported_ef is not None else ""
        delta = f", Δ={res.diff:+.3f} ({res.pct_diff:+.1%})" if res.diff is not None else ""
        print(f" → EF(model)={res.model_ef:.3f}{rep}  [{status}] {delta}")

    # Aggregate CSV
    df = pd.DataFrame([
        {
            "card_id": r.card_id,
            "reported_tCO2_per_t": r.reported_ef,
            "model_tCO2_per_t": r.model_ef,
            "diff_tCO2_per_t": r.diff,
            "pct_diff": r.pct_diff,
            "pass_rule": r.pass_rule,
            "pass": r.pass_flag,
            "total_kg_CO2e": r.total_kg_co2e,
            "demand_kg": r.demand_kg,
        }
        for r in results
    ])
    df.to_csv(out_dir / "summary.csv", index=False)
    print(f"Saved summary: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
