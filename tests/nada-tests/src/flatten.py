from nada_dsl import *
from nada_ai.nn import Flatten
import nada_algebra as na


def nada_main():
    party = Party("party")

    x = na.array([2, 2, 2, 1], party, "input_x")

    flatten = Flatten()

    x_flat = flatten(x)

    assert x_flat.shape == (2, 4), x_flat.shape

    assert Flatten(start_dim=0)(x).shape == (8,), Flatten(start_dim=0)(x).shape
    assert Flatten(start_dim=0, end_dim=-2)(x).shape == (8, 1), Flatten(
        start_dim=0, end_dim=-2
    )(x).shape
    assert Flatten(start_dim=2, end_dim=3)(x).shape == (2, 2, 2), Flatten(
        start_dim=2, end_dim=3
    )(x).shape

    x_flat_out = x_flat.output(party, "x_flat")

    return x_flat_out
