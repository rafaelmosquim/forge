"""The blast-furnace fleet: one relining rule for every unit, recurring windows.

plants.csv was once built with a 2010 placeholder wherever the last relining was
unknown, which put 11.9 Mt into a single, fictitious 2030 window and included a
cancelled and an announced furnace. These pin the rule and the file to it.
"""
from pathlib import Path

import pandas as pd
import pytest

from forge.scenarios.fleet import relining_due, to_plants_frame, windows_from_units

STEEL = Path(__file__).resolve().parents[1] / "datasets" / "steel"


def _unit(**kw):
    base = dict(unit_id="u", plant="p", unit="1", status="operating", capacity=1000.0, fuel="coal",
                start=2000, last_relining=None, planned_relining=None)
    base.update(kw)
    return base


def test_rule_planned_then_last_then_startup():
    units = pd.DataFrame([
        _unit(unit_id="planned", last_relining=2001, planned_relining=2025),
        _unit(unit_id="relined", last_relining=2019),
        _unit(unit_id="never", start=2016),
        _unit(unit_id="overdue", start=1965, last_relining=2001),
        _unit(unit_id="idle", status="mothballed", last_relining=2018),
        _unit(unit_id="gone", status="cancelled"),
        _unit(unit_id="future", status="announced", start=2026),
        _unit(unit_id="charcoal", fuel="charcoal"),
    ])
    f = windows_from_units(units, campaign=20, lifespan=50).set_index("unit_id")
    assert f.loc["planned", "Relining"] == 2025
    assert f.loc["relined", "Relining"] == 2039
    assert f.loc["never", "Relining"] == 2036 and f.loc["never", "EOL"] == 2066
    assert f.loc["overdue", "Relining"] == 2025          # floored at the base year
    assert f.loc["idle", "Relining"] == 2038              # mothballed capacity is kept
    assert "gone" not in f.index and "future" not in f.index
    assert f.loc["charcoal", "Relining"] == 2101


def test_windows_recur_every_campaign():
    fleet = pd.DataFrame({"Relining": [2025, 2030, 2101]})
    assert relining_due(fleet, 2025, 20).tolist() == [True, False, False]
    assert relining_due(fleet, 2045, 20).tolist() == [True, False, False]
    assert relining_due(fleet, 2050, 20).tolist() == [False, True, False]
    assert relining_due(fleet, 2044, 20).tolist() == [False, False, False]


def test_plants_csv_is_the_rule_applied_to_plants_units():
    units = pd.read_csv(STEEL / "plants_units.csv")
    expected = to_plants_frame(windows_from_units(units))
    actual = pd.read_csv(STEEL / "plants.csv", sep=";")
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                  check_dtype=False)
    assert set(units.loc[~units["in_model"], "status"]) <= {"cancelled", "announced", "construction", "retired"}
