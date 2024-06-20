import nada_numpy as na
from nada_dsl import *

from nada_ai.nn import ReLU


def nada_main():
    party = Party("party")

    x = na.array([4], party, "input_x", SecretInteger)
    y = na.array([4], party, "input_y", na.SecretRational)

    relu = ReLU()

    relu_x = relu(x)
    relu_y = relu(y)

    assert relu_x.shape == x.shape, relu_x.shape
    assert relu_y.shape == y.shape, relu_y.shape

    relu_x_out = relu_x.output(party, "relu_x")
    relu_y_out = relu_y.output(party, "relu_y")

    return relu_x_out + relu_y_out
