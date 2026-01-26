from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import pandas as pd


def _move_rows_last(df: pd.DataFrame, rows_last: Sequence[str]) -> pd.DataFrame:
    if df is None or df.empty or not rows_last:
        return df
    existing_last = [r for r in rows_last if r in df.index]
    if not existing_last:
        return df
    keep = [r for r in df.index if r not in existing_last]
    return df.loc[keep + existing_last]


def _clean_numeric(
    df: pd.DataFrame,
    *,
    zero_tol: float,
    decimals: int,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        return out
    numeric = out[numeric_cols].copy()
    try:
        numeric = numeric.astype(float)
    except Exception:
        pass
    numeric = numeric.fillna(0.0)
    if zero_tol is not None and zero_tol > 0:
        numeric = numeric.mask(numeric.abs() < float(zero_tol), 0.0)
    if decimals is not None:
        numeric = numeric.round(int(decimals))
    out.loc[:, numeric_cols] = numeric
    return out


def _drop_all_zero_columns(df: pd.DataFrame, *, zero_tol: float) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return df
    mask = (numeric.abs() >= float(zero_tol)).any(axis=0)
    keep_cols = [c for c in df.columns if c not in numeric.columns] + mask.index[mask].tolist()
    return df.loc[:, keep_cols]


def _drop_all_zero_rows(df: pd.DataFrame, *, zero_tol: float) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return df
    mask = (numeric.abs() >= float(zero_tol)).any(axis=1)
    return df.loc[mask.index[mask].tolist()]


def render_df_to_latex(
    df: pd.DataFrame,
    *,
    index_name: Optional[str] = None,
    add_total_row: Optional[str] = None,
    move_rows_last: Sequence[str] = (),
    drop_zero_cols: bool = False,
    drop_zero_rows: bool = False,
    decimals: int = 3,
    zero_tol: float = 1e-9,
    scale: float = 1.0,
    escape: bool = True,
    resizebox: bool = False,
    resizebox_width: str = "\\linewidth",
    latex_kwargs: Optional[dict] = None,
) -> str:
    """
    Render a DataFrame as LaTeX (tabular only by default).

    Notes:
    - Uses pandas ``DataFrame.to_latex`` (booktabs style).
    - When ``resizebox=True``, wraps the entire tabular inside
      ``\\resizebox{<width>}{!}{...}`` to fit wide tables.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "% (empty table)\n"

    out = df.copy()
    if index_name is not None:
        out.index.name = index_name

    if add_total_row and str(add_total_row) in out.index:
        out = out.drop(index=str(add_total_row))

    numeric_cols = out.select_dtypes(include="number").columns
    if len(numeric_cols) > 0 and scale is not None and float(scale) != 1.0:
        try:
            out.loc[:, numeric_cols] = out.loc[:, numeric_cols].astype(float).mul(float(scale))
        except Exception:
            pass

    if add_total_row:
        numeric = out.select_dtypes(include="number")
        if not numeric.empty:
            out.loc[str(add_total_row)] = numeric.sum(axis=0)

    out = _move_rows_last(out, move_rows_last)

    out = _clean_numeric(out, zero_tol=zero_tol, decimals=decimals)

    if drop_zero_cols:
        out = _drop_all_zero_columns(out, zero_tol=zero_tol)
    if drop_zero_rows:
        out = _drop_all_zero_rows(out, zero_tol=zero_tol)

    def _float_format(x: float) -> str:
        try:
            return f"{float(x):.{int(decimals)}f}"
        except Exception:
            return str(x)

    kwargs = dict(
        index=True,
        escape=bool(escape),
        na_rep="",
        float_format=_float_format,
        bold_rows=False,
    )
    if isinstance(latex_kwargs, dict):
        kwargs.update({k: v for k, v in latex_kwargs.items() if v is not None})

    tabular = out.to_latex(**kwargs).strip() + "\n"

    if not resizebox:
        return tabular

    width = str(resizebox_width or "\\linewidth")
    return f"\\resizebox{{{width}}}{{!}}{{%\n{tabular}}}\n"


def render_run_tables_to_latex(
    *,
    balance_matrix: Optional[pd.DataFrame],
    energy_balance: Optional[pd.DataFrame],
    emissions: Optional[pd.DataFrame],
    decimals: int = 3,
    zero_tol: float = 1e-9,
    emissions_scale: float = 1.0,
) -> dict[str, str]:
    """Convenience wrapper to render the three core output tables."""
    tables: dict[str, str] = {}
    if isinstance(balance_matrix, pd.DataFrame) and not balance_matrix.empty:
        tables["balance_matrix"] = render_df_to_latex(
            balance_matrix,
            index_name="Process",
            move_rows_last=("External Inputs", "Final Demand"),
            decimals=decimals,
            zero_tol=zero_tol,
            resizebox=True,
        )
    if isinstance(energy_balance, pd.DataFrame) and not energy_balance.empty:
        tables["energy_balance"] = render_df_to_latex(
            energy_balance,
            index_name="Process",
            move_rows_last=("TOTAL",),
            drop_zero_cols=True,
            decimals=decimals,
            zero_tol=zero_tol,
            resizebox=True,
        )
    if isinstance(emissions, pd.DataFrame) and not emissions.empty:
        tables["emissions"] = render_df_to_latex(
            emissions,
            index_name="Process",
            add_total_row="TOTAL",
            drop_zero_rows=True,
            drop_zero_cols=True,
            decimals=decimals,
            zero_tol=zero_tol,
            scale=emissions_scale,
            resizebox=True,
        )
    return tables
