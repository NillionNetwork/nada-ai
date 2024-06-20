import nada_numpy as na
from nada_dsl import *

from nada_ai.nn import Flatten


def nada_main():
    party = Party("party")

    x = na.array([2, 2, 2, 1], party, "input_x", SecretInteger)

    flatten = Flatten(start_dim=0)

    x_flat = flatten(x)

    assert x_flat.shape == (8,), x_flat.shape

    assert Flatten()(x).shape == (2, 4), Flatten(start_dim=0)(x).shape
    assert Flatten(start_dim=0, end_dim=-2)(x).shape == (8, 1), Flatten(
        start_dim=0, end_dim=-2
    )(x).shape
    assert Flatten(start_dim=2, end_dim=3)(x).shape == (2, 2, 2), Flatten(
        start_dim=2, end_dim=3
    )(x).shape

    x_flat_out = x_flat.output(party, "x_flat")

    return x_flat_out
