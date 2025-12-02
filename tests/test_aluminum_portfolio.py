import pytest

from forge.cli import steel_batch_cli as batch_cli


# Canonical outputs for `make aluminum_fgv` as of this branch.
AL_FGV_EXPECTED = {
    "aluminum_rolled_direct_no_coat": {
        "raw_co2e_kg": 8664.446343132184,
        "total_co2e_kg": 9120.469834875983,
    },
    "aluminum_extruded_direct_no_coat": {
        "raw_co2e_kg": 8816.369127132184,
        "total_co2e_kg": 9280.388554875984,
    },
    "aluminum_cast_direct_no_coat": {
        "raw_co2e_kg": 8191.184493532184,
        "total_co2e_kg": 8622.299466875984,
    },
    "aluminum_fgv_portfolio": {
        "raw_co2e_kg": 8400.998809212184,
        "total_co2e_kg": 8843.156641275984,
    },
}


@pytest.fixture(scope="session")
def aluminum_fgv_spec(repo_root):
    path = repo_root / "configs" / "aluminum_fgv_portfolio.yml"
    if not path.exists():
        pytest.skip("aluminum_fgv portfolio spec not found")
    return path


@pytest.fixture(scope="session")
def aluminum_data_dir(repo_root):
    path = repo_root / "datasets" / "aluminum" / "baseline"
    if not path.exists():
        pytest.skip("datasets/aluminum/baseline directory not found; skipping aluminum tests.")
    return path


@pytest.fixture(scope="session")
def _load_yaml():
    import yaml
    def _load(path):
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return _load


@pytest.mark.integration
def test_aluminum_fgv_portfolio_regression(aluminum_fgv_spec, tmp_path):
    plans, blends = batch_cli._plans_from_spec(aluminum_fgv_spec, cli_defaults={})
    if not plans:
        pytest.skip("no runs produced from aluminum_fgv spec")

    # Keep logs under the test tempdir
    for plan in plans:
        plan.log_dir = tmp_path
    for blend in blends:
        blend.log_dir = tmp_path

    summaries, failures, records = batch_cli.run_batch(
        plans,
        log_dir_default=tmp_path,
    )
    assert failures == 0

    blend_summaries = batch_cli._run_blends(
        blends,
        records,
        log_dir_default=tmp_path,
    )

    all_summaries = {s["name"]: s for s in [*summaries, *blend_summaries]}
    for name, expected in AL_FGV_EXPECTED.items():
        assert name in all_summaries, f"missing summary for {name}"
        got = all_summaries[name]
        assert got["raw_co2e_kg"] == pytest.approx(expected["raw_co2e_kg"], rel=1e-9, abs=1e-6)
        assert got["total_co2e_kg"] == pytest.approx(expected["total_co2e_kg"], rel=1e-9, abs=1e-6)
        if not got.get("blend"):
            assert got.get("route_preset") == "Primary"
            assert str(got.get("stage_key", "")).lower().startswith("finish")

    # Blend totals should match the weighted components
    blend = all_summaries["aluminum_fgv_portfolio"]
    rolled = AL_FGV_EXPECTED["aluminum_rolled_direct_no_coat"]["total_co2e_kg"]
    extruded = AL_FGV_EXPECTED["aluminum_extruded_direct_no_coat"]["total_co2e_kg"]
    cast = AL_FGV_EXPECTED["aluminum_cast_direct_no_coat"]["total_co2e_kg"]
    expected_blend_total = 0.265 * rolled + 0.135 * extruded + 0.6 * cast
    assert blend["total_co2e_kg"] == pytest.approx(expected_blend_total, rel=1e-12, abs=1e-6)


@pytest.mark.integration
def test_aluminum_primary_baseline(aluminum_data_dir, _load_yaml):
    from forge.steel_core_api_v2 import ScenarioInputs, RouteConfig, run_scenario

    scenario = _load_yaml(aluminum_data_dir / "scenarios" / "primary-al.yml")
    stage_key = scenario.get("stage_key", "primary")
    scn = ScenarioInputs(
        country_code="BRA",
        scenario=scenario,
        route=RouteConfig(
            route_preset="Primary",
            stage_key=stage_key,
            demand_qty=1000.0,
        ),
    )
    out = run_scenario(str(aluminum_data_dir), scn)
    assert out is not None
    assert out.emissions is not None and not out.emissions.empty
    assert out.total_co2e_kg == pytest.approx(7801.666253532184, rel=0.0, abs=1e-6)
    assert out.meta.get("sector_key") == "aluminum"
    assert out.meta.get("stage_key") == stage_key


@pytest.mark.integration
def test_aluminum_secondary_remelt(aluminum_data_dir, _load_yaml):
    from forge.steel_core_api_v2 import ScenarioInputs, RouteConfig, run_scenario

    scenario = _load_yaml(aluminum_data_dir / "scenarios" / "secondary-al.yml")
    stage_key = scenario.get("stage_key", "remelt")
    scn = ScenarioInputs(
        country_code="BRA",
        scenario=scenario,
        route=RouteConfig(
            route_preset="Secondary",
            stage_key=stage_key,
            demand_qty=1000.0,
        ),
    )
    out = run_scenario(str(aluminum_data_dir), scn)
    assert out is not None
    assert out.emissions is not None and not out.emissions.empty
    assert out.total_co2e_kg == pytest.approx(820.78, rel=0.0, abs=1e-2)
    assert out.meta.get("sector_key") == "aluminum"
    assert out.meta.get("stage_key") == stage_key


@pytest.mark.integration
def test_aluminum_secondary_is_lower_than_primary(aluminum_data_dir, _load_yaml):
    from forge.steel_core_api_v2 import ScenarioInputs, RouteConfig, run_scenario

    prim = _load_yaml(aluminum_data_dir / "scenarios" / "primary-al.yml")
    sec = _load_yaml(aluminum_data_dir / "scenarios" / "secondary-al.yml")

    out_primary = run_scenario(
        str(aluminum_data_dir),
        ScenarioInputs(
            country_code="BRA",
            scenario=prim,
            route=RouteConfig(
                route_preset="Primary",
                stage_key=prim.get("stage_key", "primary"),
                demand_qty=1000.0,
            ),
        ),
    )
    out_secondary = run_scenario(
        str(aluminum_data_dir),
        ScenarioInputs(
            country_code="BRA",
            scenario=sec,
            route=RouteConfig(
                route_preset="Secondary",
                stage_key=sec.get("stage_key", "remelt"),
                demand_qty=1000.0,
            ),
        ),
    )
    assert out_secondary.total_co2e_kg < out_primary.total_co2e_kg
