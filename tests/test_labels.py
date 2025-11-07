import os


def test_label_from_spec_path_finished():
    os.environ["FORGE_PAPER_PRODUCT_CONFIG"] = "portfolio"
    from forge.scenarios.utils import label_from_spec_path

    assert label_from_spec_path("configs/finished_steel_portfolio.yml") == "finished"
    assert label_from_spec_path("configs/paper_portfolio.yml") == "paper"
    assert label_from_spec_path("configs/as_cast_portfolio.yml") == "as_cast"
    assert label_from_spec_path("configs/custom_portfolio.yml") == "custom"


def test_output_roots_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_OUTPUT_LABEL", "foo")
    from forge.scenarios.utils import configure_output_roots
    configure_output_roots()
    assert os.environ.get("FORGE_FIG_DIR", "").endswith("results/foo/figs")
    assert os.environ.get("FORGE_TABLE_DIR", "").endswith("results/foo/tables")
