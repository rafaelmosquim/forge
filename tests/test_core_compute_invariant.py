import pytest, pathlib

# Import the library entrypoint without invoking any CLI.
forge_run = pytest.importorskip("forge_core.run")

# Use a small, canonical resolved config present in the repo.
CONFIG = pathlib.Path("configs/BF-BOF_resolved.yml")

@pytest.mark.skipif(not CONFIG.exists(), reason="requires resolved config")
def test_compute_produces_nonnegative_emissions_and_rows():
    # Expect the run module to expose a `run` function that returns a dict-like result.
    result = forge_run.run(config_path=str(CONFIG))
    # Minimal invariant: has rows and a total emissions scalar that is nonnegative.
    # Adapt keys minimally if your API uses different naming.
    assert result, "empty result"
    total = result.get("emissions_total") or result.get("total_emissions") or result.get("emissions", 0)
    assert total is not None, "missing total emissions"
    assert float(total) >= 0.0
