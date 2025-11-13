import types

from forge.core.transforms import (
    apply_dict_overrides,
    apply_recipe_overrides,
    apply_fuel_substitutions,
)


def test_apply_dict_and_recipe_overrides():
    class R:
        def __init__(self, name, inputs, outputs):
            self.name = name; self.inputs = dict(inputs); self.outputs = dict(outputs)

    recipes = [R('X', {'A': 1}, {'B': 2})]
    overrides = {
        'X': {
            'inputs': {'A': '2 * 1'},
            'outputs': {'B': 'inputs["A"] + 1'},
        }
    }
    params = types.SimpleNamespace()
    energy_int, energy_shares, energy_content = {}, {}, {}

    new_recipes = apply_recipe_overrides(recipes, overrides, params, energy_int, energy_shares, energy_content)
    r = next(r for r in new_recipes if r.name == 'X')
    assert r.inputs['A'] == 2
    assert r.outputs['B'] == 3

    d = {'k': 1}
    apply_dict_overrides(d, {'k': 2, 'z': 9})
    assert d['k'] == 2 and d['z'] == 9


def test_apply_fuel_substitutions_moves_shares_and_recipe_io():
    energy_shares = {'P': {'Coal': 0.3, 'Gas': 0.7}}
    energy_int = {}
    energy_content = {}
    efs = {}

    class R:
        def __init__(self):
            self.name = 'Proc'
            self.inputs = {'Coal': 1.0}
            self.outputs = {'X': 1.0}

    recipes = [R()]
    apply_fuel_substitutions({'Coal': 'Charcoal'}, energy_shares, energy_int, energy_content, efs, recipes)

    assert energy_shares['P']['Coal'] == 0.0
    assert energy_shares['P']['Charcoal'] == 0.3
    r = recipes[0]
    assert 'Coal' not in r.inputs and r.inputs['Charcoal'] == 1.0
