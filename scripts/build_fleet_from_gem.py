"""Build a national blast-furnace fleet (datasets/steel/plants*.csv) from Global Energy Monitor.

Brazil is the default and what the paper uses; --country selects any other
country in the tracker (the charcoal-plant list below is Brazilian).

Usage (needs pandas + openpyxl; the GEM workbooks are not shipped with FORGE):

    python scripts/build_fleet_from_gem.py \
        --bf Global-Blast-Furnace-Tracker-April-2024-Standard-Copy-V1.xlsx \
        --plants Global-Steel-Plant-Tracker-April-2024-Standard-Copy-V1.xlsx

Writes
  datasets/steel/plants_units.csv  one row per furnace: GEM unit ID, plant, status,
                                   capacity, start-up, last and planned relining,
                                   reducing agent, and the window/EOL with the rule
                                   that set it
  datasets/steel/plants.csv        the columns the transition model reads

Inclusion: operating and mothballed furnaces. Mothballed capacity stays because
the scenarios' utilization rates are production over installed capacity, and
installed capacity counts idled units (2024: 25.7 Mt pig iron over 38.8 Mt, 66%).
Announced, under-construction, cancelled and retired units are excluded: the
scenarios add no capacity. The relining rule is in forge.scenarios.fleet.

GEM data: CC BY 4.0, Global Energy Monitor, Global Blast Furnace Tracker and
Global Steel Plant Tracker, April 2024.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "forge" / "scenarios"))
from fleet import to_plants_frame, windows_from_units  # noqa: E402  (pure pandas, no FORGE import)

# Plants whose furnaces run on charcoal (GEM plant name); the rest are coke-based.
CHARCOAL_PLANTS = ("Viena", "Divinópolis", "Aperam", "Juiz de Fora", "AVB", "Vallourec", "Sinobras")


def _year(v):
    m = re.search(r"(19|20)\d{2}", str(v))
    return int(m.group(0)) if m else None


def load_units(bf_xlsx: Path, plants_xlsx: Path, country: str = "Brazil") -> pd.DataFrame:
    plants = pd.read_excel(plants_xlsx, sheet_name="Steel Plants")
    sel = plants.loc[plants["Country/Area"] == country, ["Plant ID", "Plant name (English)"]].drop_duplicates("Plant ID")
    bf = pd.read_excel(bf_xlsx, sheet_name="Blast Furnaces")
    units = bf.merge(sel, left_on="GEM Plant ID", right_on="Plant ID").drop_duplicates("GEM Unit ID")
    rel = pd.read_excel(bf_xlsx, sheet_name="Furnace Relinings")
    rel = rel[rel["GEM Unit ID"].isin(units["GEM Unit ID"])]
    done, planned = {}, {}
    for _, r in rel.iterrows():
        y = _year(r["Furnace Relining Stop Date"]) or _year(r["Furnace Relining Start Date"])
        if y is None:
            continue
        target = planned if "plan" in str(r["Relining Status"]).lower() else done
        target[r["GEM Unit ID"]] = max(target.get(r["GEM Unit ID"], 0), y)
    rows = []
    for _, u in units.iterrows():
        uid = u["GEM Unit ID"]
        last = max(filter(None, [_year(u["Most Recent Relining"]), done.get(uid)]), default=None)
        name = str(u["Plant name (English)"])
        rows.append({
            "unit_id": uid,
            "plant": name,
            "unit": str(u["Unit Name"]),
            "status": str(u["Unit Status"]).strip().lower(),
            "capacity": float(pd.to_numeric(u["Current Capacity (ttpa)"], errors="coerce") or 0.0),
            "fuel": "charcoal" if any(k in name for k in CHARCOAL_PLANTS) else "coal",
            "start": _year(u["Start Date"]),
            "last_relining": last,
            "planned_relining": planned.get(uid),
        })
    out = pd.DataFrame(rows).sort_values(["fuel", "plant", "unit"]).reset_index(drop=True)
    for c in ("start", "last_relining", "planned_relining"):
        out[c] = out[c].astype("Int64")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bf", required=True, type=Path, help="GEM Global Blast Furnace Tracker (xlsx)")
    ap.add_argument("--plants", required=True, type=Path, help="GEM Global Steel Plant Tracker (xlsx)")
    ap.add_argument("--country", default="Brazil", help="GEM 'Country/Area' name (default: Brazil)")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "steel")
    args = ap.parse_args()

    units = load_units(args.bf, args.plants, country=args.country)
    fleet = windows_from_units(units)
    units_out = units.merge(fleet[["unit_id", "Relining", "EOL", "rule"]], on="unit_id", how="left")
    units_out["in_model"] = units_out["Relining"].notna()
    units_out.to_csv(args.out / "plants_units.csv", index=False)
    to_plants_frame(fleet).to_csv(args.out / "plants.csv", sep=";", index=False)

    kept = fleet["capacity"].sum()
    coal = fleet.loc[fleet.fuel == "coal"]
    print(f"{len(fleet)} furnaces kept ({kept:,.0f} kt/yr; coal {coal['capacity'].sum():,.0f}); "
          f"excluded: {units_out.loc[~units_out['in_model'], ['plant', 'unit', 'status']].values.tolist()}")
    print(coal.groupby("Relining")["capacity"].sum().round(0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
