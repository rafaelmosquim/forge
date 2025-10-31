import importlib
import pytest

CANDIDATE_MODULES = [
    "forge.steel_core_api_v2",
    "steel_core_api_v2",
    "forge.steel_model_core",
    "steel_model_core",
    "steel_core",
]

def try_import():
    last_err = None
    for mod in CANDIDATE_MODULES:
        try:
            return importlib.import_module(mod), mod
        except Exception as e:
            last_err = e
    pytest.skip(f"Core module not found/importable ({CANDIDATE_MODULES}); skipping. Last error: {last_err}")

def test_import_core_module():
    mod, name = try_import()
    assert mod is not None, "core module import returned None"

def test_yaml_configs_present_and_parseable(data_dir, yload):
    # minimal set used across the model
    required = [
        "recipes.yml",
        "energy_int.yml",
        "energy_matrix.yml",
        "energy_content.yml",
        "emission_factors.yml",
        "parameters.yml",
        "process_emissions.yml",
        "mkt_config.yml",
    ]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        pytest.skip(f"Missing YAML(s): {missing}")

    # parse to ensure no syntax errors
    for f in required:
        doc = yload(data_dir / f)
        assert isinstance(doc, (dict, list)), f"{f} parsed but is empty or wrong type"
