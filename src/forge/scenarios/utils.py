from __future__ import annotations

import os
from pathlib import Path


def label_from_spec_path(spec_path: str | None) -> str:
    if not spec_path:
        return "simple"
    name = Path(spec_path).stem
    mapping = {
        "finished_steel_portfolio": "finished",
        "paper_portfolio": "paper",
        "as_cast_portfolio": "as_cast",
    }
    if name in mapping:
        return mapping[name]
    if name.endswith("_portfolio"):
        return name[:-10] or name
    return name


def configure_output_roots() -> None:
    """Set FORGE_FIG_DIR and FORGE_TABLE_DIR deterministically based on env.

    Priority:
      - If FORGE_OUTPUT_LABEL is set → results/<label>/{figs,tables}
      - Else if FORGE_PAPER_PORTFOLIO_SPEC is set → derive label from spec filename
      - Else if both FORGE_FIG_DIR and FORGE_TABLE_DIR already set → leave as-is
      - Else if FORGE_PAPER_PRODUCT_CONFIG=portfolio → results/portfolio/{figs,tables}
      - Else → results/simple/{figs,tables}
    """
    fig_dir_env = os.getenv("FORGE_FIG_DIR")
    table_dir_env = os.getenv("FORGE_TABLE_DIR")

    label = os.getenv("FORGE_OUTPUT_LABEL", "").strip()
    if not label:
        spec = os.getenv("FORGE_PAPER_PORTFOLIO_SPEC", "").strip() or None
        if spec:
            label = label_from_spec_path(spec)
        elif fig_dir_env and table_dir_env:
            return  # keep preexisting config
        elif os.getenv("FORGE_PAPER_PRODUCT_CONFIG", "").strip().lower() == "portfolio":
            label = "portfolio"
        else:
            label = "simple"

    # FGV runs are grouped under results/fgv/<label>; others stay at results/<label>
    if label.lower().startswith("fgv"):
        base = Path("results") / "fgv" / label
    else:
        base = Path("results") / label
    os.environ["FORGE_FIG_DIR"] = str(base / "figs")
    os.environ["FORGE_TABLE_DIR"] = str(base / "tables")
