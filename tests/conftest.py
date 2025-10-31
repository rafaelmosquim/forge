from pathlib import Path
import os
import sys
import pytest
import yaml

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def data_dir(repo_root: Path) -> Path:
    # default steel dataset lives under datasets/steel/likely
    d = repo_root / "datasets" / "steel" / "likely"
    if not d.exists():
        pytest.skip("datasets/steel/likely directory not found; skipping data-dependent tests.")
    return d

@pytest.fixture(scope="session")
def yload():
    def _load(p: Path):
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return _load
