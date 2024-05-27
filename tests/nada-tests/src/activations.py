from nada_dsl import *
from nada_ai.nn import ReLU
import nada_algebra as na


def nada_main():
    party = Party("party")

    x = na.array([4], party, "input")

    relu = ReLU()

    relu_out = relu(x)

    assert relu_out.shape == x.shape, relu_out.shape

    return relu_out.output(party, "relu")
