from pathlib import Path
import os
import pytest
import yaml

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def data_dir(repo_root: Path) -> Path:
    # assumes data/ alongside code, as in your README and methods note
    d = repo_root / "data"
    if not d.exists():
        pytest.skip("data/ directory not found; skipping data-dependent tests.")
    return d

@pytest.fixture(scope="session")
def yload():
    def _load(p: Path):
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return _load
