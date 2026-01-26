import pandas as pd

from forge.core.latex import render_df_to_latex, render_run_tables_to_latex


def test_render_df_to_latex_moves_rows_last_and_resizebox():
    df = pd.DataFrame(
        {"Mat1": [1.0, 2.0, 3.0], "Mat2": [0.0, 4.0, 5.0]},
        index=["Proc A", "Final Demand", "External Inputs"],
    )
    tex = render_df_to_latex(
        df,
        index_name="Process",
        move_rows_last=("External Inputs", "Final Demand"),
        decimals=0,
        zero_tol=1e-12,
        resizebox=True,
    )
    assert tex.lstrip().startswith("\\resizebox")
    assert "\\begin{tabular}" in tex
    assert tex.find("Proc A") < tex.find("External Inputs") < tex.find("Final Demand")


def test_render_df_to_latex_drop_zero_columns_and_add_total():
    df = pd.DataFrame(
        {"A": [0.0, 0.0], "B": [1.0, 0.0]},
        index=["p1", "p2"],
    )
    tex = render_df_to_latex(df, drop_zero_cols=True, decimals=3)
    assert "A" not in tex
    assert "B" in tex

    emissions = pd.DataFrame(
        {"Energy Emissions": [1.0], "Direct Emissions": [2.0], "TOTAL CO2e": [3.0]},
        index=["Proc"],
    )
    tex2 = render_df_to_latex(emissions, add_total_row="TOTAL", decimals=3)
    assert "TOTAL" in tex2


def test_render_run_tables_to_latex_smoke():
    balance = pd.DataFrame({"X": [1.0]}, index=["External Inputs"])
    energy = pd.DataFrame({"Electricity": [10.0]}, index=["Proc"])
    emissions = pd.DataFrame({"TOTAL CO2e": [0.5]}, index=["Proc"])
    tables = render_run_tables_to_latex(balance_matrix=balance, energy_balance=energy, emissions=emissions)
    assert set(tables.keys()) == {"balance_matrix", "energy_balance", "emissions"}

