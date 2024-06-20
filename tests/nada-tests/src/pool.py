import nada_numpy as na
from nada_dsl import *

from nada_ai.nn import AvgPool2d


def nada_main():
    party = Party("party")

    x = na.array((1, 4, 4, 2), party, "input_x", SecretInteger)
    y = na.array((4, 4, 2), party, "input_y", SecretInteger)

    pool1 = AvgPool2d(
        kernel_size=2,
        padding=0,
        stride=1,
    )
    pool2 = AvgPool2d(
        kernel_size=2,
        padding=1,
        stride=2,
    )

    x_pool1 = pool1(x)
    x_pool2 = pool2(x)

    y_pool1 = pool1(y)
    y_pool2 = pool2(y)

    assert x_pool1.shape == (1, 4, 3, 1), x_pool1.shape
    assert x_pool2.shape == (1, 4, 3, 2), x_pool2.shape

    assert y_pool1.shape == (4, 3, 1), y_pool1.shape
    assert y_pool2.shape == (4, 3, 2), y_pool2.shape

    x_pool1_out = x_pool1.output(party, "x_pool1")
    x_pool2_out = x_pool2.output(party, "x_pool2")

    y_pool1_out = y_pool1.output(party, "y_pool1")
    y_pool2_out = y_pool2.output(party, "y_pool2")

    return x_pool1_out + x_pool2_out + y_pool1_out + y_pool2_out
