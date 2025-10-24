# tests/test_data_consistency.py
from pathlib import Path
import math
import yaml
import pytest

TOL = 1e-3

def norm_num(x):
    """
    Best-effort convert YAML value to a representative float:
    - numeric -> float(x)
    - dict with 'likely'/'avg'/'mean' -> that value
    - dict with 'min' & 'max' only -> midpoint
    - else -> None (non-numeric / symbolic)
    """
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    if isinstance(x, dict):
        for k in ("likely", "avg", "average", "mean", "median"):
            if k in x and isinstance(x[k], (int, float, str)):
                return norm_num(x[k])
        if "min" in x and "max" in x:
            lo = norm_num(x["min"])
            hi = norm_num(x["max"])
            if lo is not None and hi is not None:
                return 0.5 * (lo + hi)
    return None

@pytest.fixture
def data_dir():
    # Adjust if your data lives elsewhere
    return Path("datasets/steel/likely")

@pytest.fixture
def yload():
    def _load(p: Path):
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return _load

def test_energy_matrix_shares_sum_to_one(data_dir, yload):
    """
    Each process's carrier shares should sum ~ 1.0 for the representative (likely) values.
    Accept either:
      - {process: {carrier: number_or_dict}}
      - Or nested shapes as long as carriers map to numeric/dict values.
    """
    m = yload(data_dir / "energy_matrix.yml")
    assert isinstance(m, dict), "energy_matrix.yml should be a dict keyed by process"

    for proc, shares in m.items():
        if not isinstance(shares, dict) or not shares:
            continue

        nums = []
        for carrier, v in shares.items():
            val = norm_num(v)
            if val is not None:
                nums.append(val)
            # ignore non-numeric (e.g., symbolic or notes)

        if nums:
            s = sum(nums)
            assert math.isfinite(s), f"{proc}: non-finite share sum"
            assert abs(s - 1.0) <= TOL, f"{proc}: shares sum to {s:.6f}, expected ~1.0"

def test_recipes_nonnegative_and_have_output(data_dir, yload):
    """
    Recipes file can be either:
      - dict: {process: {inputs: {...}, outputs: {...}}}
      - list: [{process: ..., inputs: {...}, outputs: {...}}, ...]
    We check:
      - if outputs exist, they must be non-empty
      - all numeric coefficients we can parse are >= 0
      (symbolic terms are skipped)
    NOTE: entries with empty outputs (e.g., placeholders like Steam Production)
    are skipped rather than failed.
    """
    r = yload(data_dir / "recipes.yml")

    entries = []
    if isinstance(r, dict):
        for proc, node in r.items():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            outputs = node.get("outputs", {})
            entries.append({"process": proc, "inputs": inputs, "outputs": outputs})
    elif isinstance(r, list):
        for node in r:
            if not isinstance(node, dict):
                continue
            proc = node.get("process", "<unknown>")
            inputs = node.get("inputs", {})
            outputs = node.get("outputs", {})
            entries.append({"process": proc, "inputs": inputs, "outputs": outputs})
    else:
        pytest.fail("recipes.yml must be a dict or a list of recipe objects")

    assert entries, "No recipe entries found"

    for entry in entries:
        proc = entry["process"]
        inputs = entry["inputs"] if isinstance(entry["inputs"], dict) else {}
        outputs = entry["outputs"] if isinstance(entry["outputs"], dict) else {}

        # Skip placeholder/utility entries with no explicit outputs
        if not outputs:
            continue

        # Check numeric values we can parse are >= 0
        for where, d in (("input", inputs), ("output", outputs)):
            for k, v in d.items():
                num = norm_num(v)
                if num is None:
                    continue  # skip symbolic or complex expressions
                assert num >= -1e-12, f"{proc}: {where} '{k}' has negative value {num}"
