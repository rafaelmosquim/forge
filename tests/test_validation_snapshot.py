import importlib
import os
import math
import pytest

# Published likely BRA crude-steel “Validation (as cast)” targets (tCO2e/tcs)
EXPECTED = {
    "BF-BOF": 2.216,
    "DRI-EAF": 0.971,
    "EAF-Scrap": 0.208,
}
TOL = 0.08  # wide-ish to avoid false red; adjust if your core is tighter

def _import_core():
    for mod in ("steel_core_api_v2", "steel_model_core", "steel_core"):
        try:
            return importlib.import_module(mod)
        except Exception:
            continue
    return None

@pytest.mark.skipif(
    os.getenv("FORGE_ENABLE_NUMERIC_TESTS") not in ("1", "true", "True"),
    reason="Numeric snapshot disabled; set FORGE_ENABLE_NUMERIC_TESTS=1 to enable."
)
def test_crude_steel_validation_likely_bra(repo_root):
    core = _import_core()
    if core is None:
        pytest.skip("Core API not importable")

    # We try a couple of expected entry points; skip if missing.
    # Please wire one of these in your core to return route intensities for the
    # ‘Validation (as cast)’ boundary with Likely dataset and Brazil grid.
    candidates = [
        # preferred: single call
        getattr(core, "run_validation_snapshot", None),
        # fallback: a class-based engine with a convenience method
        getattr(core, "ForgeModel", None),
    ]
    if candidates[0]:
        res = candidates[0](dataset="Likely", grid="BRA", boundary="validation_as_cast")
        # Expect: dict like {"BF-BOF": x, "DRI-EAF": y, "EAF-Scrap": z}
        for k, v_exp in EXPECTED.items():
            v = float(res[k])
            assert math.isclose(v, v_exp, rel_tol=0, abs_tol=TOL), f"{k}: {v} vs {v_exp}"
        return

    if candidates[1]:
        # illustrative wiring; adjust to your real constructor/params:
        model = candidates[1](
            data_path=str(repo_root / "data"),
            dataset="Likely",
            grid="BRA",
            boundary="validation_as_cast",
        )
        res = model.compute_route_intensities(["BF-BOF", "DRI-EAF", "EAF-Scrap"])
        for k, v_exp in EXPECTED.items():
            v = float(res[k])
            assert math.isclose(v, v_exp, rel_tol=0, abs_tol=TOL), f"{k}: {v} vs {v_exp}"
        return

    pytest.skip("No validation API found; expose run_validation_snapshot(...) or ForgeModel(...).")
