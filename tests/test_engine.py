import pytest

# will test here the balance matrix logic only

from forge.core.engine import calculate_balance_matrix
from forge.core.models import Process

# we will test here if balance matrix walks upstream from final demand
# we will ask for steel and see if basic oxygen furnace asks for inputs from blast furnace
def test_balance_matrix_walks_upstream_and_tracks_runs():
    recipes = [
        Process(
            "Basic Oxygen Furnace",
            inputs={"Pig Iron": 1.0, "Electricity": 1.0},
            outputs={"Steel": 1.0},
        ),
        Process(
            "Blast Furnace",
            inputs={"Iron Ore": 1.1, "Coke": 10.0, "Sinter": 1.0},
            outputs={"Pig Iron": 1.0},
        ),
        Process(
            "Iron Ore From Market",
            inputs={},
            outputs={"Iron Ore": 1.0},
        ),
    ]

    df, prod_level = calculate_balance_matrix(
        recipes=recipes,
        final_demand={"Steel": 100.0},
        production_routes={"Basic Oxygen Furnace": 1.0, "Blast Furnace": 1.0},
    )

    # Check production levels; we need 100 units of steel and each BOF produces 1 unit of steel, so we need 100 runs of BOF; each BOF needs 1 unit of Pig Iron, so we need 100 units of Pig Iron; each Blast Furnace produces 1 unit of Pig Iron, so we need 100 runs of Blast Furnace. All model is based on production levels, so this is the most important check.
    assert prod_level["Basic Oxygen Furnace"] == pytest.approx(100.0)
    assert prod_level["Blast Furnace"] == pytest.approx(100.0)

    # Check balance matrix values. Process outputs should be positive, inputs negative. Note a product needs to be an output of a process to appear in the balance matrix. And also an input must be produced by an upstream process, or it will be treated as a pruchase.

    assert df.loc["Basic Oxygen Furnace", "Steel"] == pytest.approx(100.0)
    assert df.loc["Blast Furnace", "Iron Ore"] == pytest.approx(-110.0)
    assert df.loc["Blast Furnace", "Coke"] == pytest.approx(-1000.0)
    assert df.loc["Blast Furnace", "Sinter"] == pytest.approx(-100.0)
    assert df.loc["Basic Oxygen Furnace", "Pig Iron"] == pytest.approx(-100.0)
    assert df.loc["Basic Oxygen Furnace", "Electricity"] == pytest.approx(-100.0)
    
    # Intermediate product is fully balanced internally
    assert df["Pig Iron"].sum() == pytest.approx(0.0)

# test with multiple producers should fail by design; core model accepts only one production route per process which should be resolved elsewhere.
def test_balance_matrix_with_multiple_routes():
    recipes = [
        Process(
            "Basic Oxygen Furnace",
            inputs={"Pig Iron": 1.0, "Electricity": 1.0},
            outputs={"Steel": 1.0},
        ),
        Process(
            "Blast Furnace",
            inputs={"Iron Ore": 1.1, "Coke": 10.0, "Sinter": 1.0},
            outputs={"Pig Iron": 1.0},
        ),
        Process(
            "Direct Reduction Iron",
            inputs={"Iron Ore": 1.1, "Natural Gas": 5.0, "Sinter": 1.0},
            outputs={"Pig Iron": 1.0},
        ),
        Process(
            "Iron Ore From Market",
            inputs={},
            outputs={"Iron Ore": 1.0},
        ),
    ]
    with pytest.raises(ValueError, match="Ambiguous producers for 'Pig Iron'"):
        calculate_balance_matrix(
            recipes=recipes,
            final_demand={"Steel": 100},
            production_routes={"Basic Oxygen Furnace": 1.0, "Blast Furnace": 1.0,
                               "Direct Reduction Iron": 1.0},
        )

# def test_external_producers():
#     recipes = [
#         Process(
#             "Blast Furnace",
#             inputs={"Iron Ore": 1.1, "Coke": 10.0, "Sinter": 1.0},
#             outputs={"Pig Iron": 1.0},
#         ),
#         Process(
#             "Iron Ore From Market",
#             inputs={},
#             outputs={"Iron Ore": 1.0},
#         ),
#     ]
#     df, prod_level = calculate_balance_matrix(
#         recipes=recipes,
#         final_demand={"Pig Iron": 100.0},
#         production_routes={"Blast Furnace":1.0, "Iron Ore From Market":1.0}    
#     )        

#     assert prod_level["Blast Furnace"] == pytest.approx(100.0)
#     assert df.loc["External Inputs", "Iron Ore"] == pytest.approx(110.0)

