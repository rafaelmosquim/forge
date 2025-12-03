"""Scenario-level energy tweaks shared by CLI and UI."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


# Split baseline DRI gas share across carriers when mix is requested
DEFAULT_DRI_MIX_DEFINITIONS: Dict[str, Dict[int, Dict[str, float]]] = {
    "Blue": {
        2030: {"Gas": 1.0},
        2040: {"Gas": 0.70, "Biomethane": 0.10, "Green hydrogen": 0.20},
    },
    "Green": {
        2030: {"Gas": 0.70, "Biomethane": 0.10, "Green hydrogen": 0.20},
        2040: {"Gas": 0.40, "Biomethane": 0.20, "Green hydrogen": 0.40},
    },
}

# Replace a slice of BF solid fuels with charcoal on an expansion path
DEFAULT_CHARCOAL_SCHEDULE: Dict[int, float] = {
    2030: 0.0,
    2035: 0.2,
    2040: 0.4,
    2045: 0.6,
}


def _as_float(val: Any) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


def _pick_year_value(table: Mapping[Any, Any], year: Optional[int]) -> Optional[Any]:
    """Choose nearest <= year (or max) from a year→value mapping."""
    if not isinstance(table, Mapping) or not table:
        return None
    parsed = []
    for k, v in table.items():
        try:
            parsed.append((int(k), v))
        except Exception:
            continue
    if not parsed:
        return None
    parsed.sort(key=lambda x: x[0])
    if year is None:
        return parsed[-1][1]
    chosen = None
    for y, v in parsed:
        if y <= year:
            chosen = v
    if chosen is not None:
        return chosen
    # fallback to smallest key when all keys are > year
    return parsed[0][1]


def _scenario_year(scn: Mapping[str, Any]) -> Optional[int]:
    for key in ("snapshot_year", "year", "target_year", "analysis_year"):
        val = scn.get(key)
        try:
            return int(val)
        except Exception:
            continue
    return None


def apply_dri_mix(energy_shares: Dict[str, Dict[str, float]], scenario: Dict[str, Any]) -> None:
    """Redistribute DRI gas share into configured carrier mix."""
    if not isinstance(energy_shares, dict) or not isinstance(scenario, dict):
        return

    mix_cfg = scenario.get("dri_mix")
    if not mix_cfg:
        return

    # Accept inline dict (year→carrier mix) or named mix
    if isinstance(mix_cfg, Mapping):
        mix_defs = {"_inline": mix_cfg}
        mix_name = "_inline"
    else:
        mix_name = str(mix_cfg)
        mix_defs = scenario.get("dri_mix_definitions")
        if not isinstance(mix_defs, Mapping):
            mix_defs = DEFAULT_DRI_MIX_DEFINITIONS

    mix_table = mix_defs.get(mix_name)
    if not isinstance(mix_table, Mapping):
        return

    year = _scenario_year(scenario)
    mix_spec = _pick_year_value(mix_table, year)
    if not isinstance(mix_spec, Mapping):
        return

    # Find DRI process rows (keep lenient matching)
    dri_keys = [k for k in energy_shares if str(k).lower() == "direct reduction iron"]
    if not dri_keys:
        return

    shares = mix_spec
    weights = {str(c): _as_float(v) for c, v in shares.items() if _as_float(v) > 0}
    if not weights:
        return
    total_w = sum(weights.values())
    if total_w <= 0:
        return

    for proc in dri_keys:
        row = energy_shares.get(proc) or {}
        gas_share = _as_float(row.get("Gas", 0.0))
        if gas_share <= 0:
            continue
        row["Gas"] = 0.0
        for carrier, w in weights.items():
            row[carrier] = _as_float(row.get(carrier, 0.0)) + gas_share * (w / total_w)
        energy_shares[proc] = row


def apply_charcoal_expansion(energy_shares: Dict[str, Dict[str, float]], scenario: Dict[str, Any]) -> None:
    """Shift a fraction of BF solid fuels (Coal+Coke) to Charcoal."""
    if not isinstance(energy_shares, dict) or not isinstance(scenario, dict):
        return

    mode = scenario.get("charcoal_expansion")
    if not mode or str(mode).strip().lower() in {"none", "false", "0"}:
        return

    schedule = scenario.get("charcoal_share_schedule")
    if not isinstance(schedule, Mapping):
        schedule = DEFAULT_CHARCOAL_SCHEDULE

    year = _scenario_year(scenario)
    fraction = _pick_year_value(schedule, year)
    frac = _as_float(fraction)
    if frac > 1.0 and frac <= 100.0:
        frac = frac / 100.0
    frac = min(max(frac, 0.0), 1.0)
    if frac <= 0:
        return

    bf_keys = [k for k in energy_shares if str(k).lower() == "blast furnace"]
    if not bf_keys:
        return

    for proc in bf_keys:
        row = energy_shares.get(proc) or {}
        coal = _as_float(row.get("Coal", 0.0))
        coke = _as_float(row.get("Coke", 0.0))
        solids = coal + coke
        if solids <= 0:
            continue
        shift = solids * frac
        if solids > 0:
            if coal > 0:
                row["Coal"] = max(0.0, coal - shift * (coal / solids))
            if coke > 0:
                row["Coke"] = max(0.0, coke - shift * (coke / solids))
        row["Charcoal"] = _as_float(row.get("Charcoal", 0.0)) + shift
        energy_shares[proc] = row
