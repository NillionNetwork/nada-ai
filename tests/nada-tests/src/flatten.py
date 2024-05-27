from nada_dsl import *
from nada_ai.nn import Flatten
import nada_algebra as na


def nada_main():
    party = Party("party")

    x = na.array([2, 2, 2, 1], party, "input_x")
    y = na.array([4], party, "input_y")

    flatten = Flatten()

    x_flat = flatten(x)
    y_flat = flatten(y)

    assert x_flat.shape == (8,), x_flat.shape
    assert y_flat.shape == (4,), y_flat.shape

    x_flat_out = x_flat.output(party, "x_flat")
    y_flat_out = y_flat.output(party, "y_flat")

    return x_flat_out + y_flat_out
