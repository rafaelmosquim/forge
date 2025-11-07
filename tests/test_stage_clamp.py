from forge.scenarios.monte_carlo_tri import _clamp_downstream_processes


def test_clamp_cast_disables_downstream():
    scn = {}
    out = _clamp_downstream_processes("Cast", scn)
    ro = out.get("route_overrides", {})
    # IP3
    assert ro.get("Hot Rolling") == 0
    assert ro.get("Rod/bar/section Mill") == 0
    # IP4
    assert ro.get("Stamping/calendering/lamination") == 0
    assert ro.get("Machining") == 0
    # Finishing
    assert ro.get("No Coating") == 0
    assert ro.get("Organic or Sintetic Coating (painting)") == 0


def test_clamp_finished_no_changes():
    scn = {"route_overrides": {"Hot Rolling": 1}}
    out = _clamp_downstream_processes("Finished", scn)
    assert out["route_overrides"]["Hot Rolling"] == 1
