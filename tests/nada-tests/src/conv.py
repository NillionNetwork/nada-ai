import nada_algebra as na
from nada_dsl import *

from nada_ai.nn import Conv2d


def nada_main():
    party = Party("party")

    x = na.array((1, 3, 4, 2), party, "input_x", na.Rational)
    y = na.array((3, 4, 2), party, "input_y", na.Rational)

    conv1 = Conv2d(
        kernel_size=2,
        in_channels=3,
        out_channels=1,
        padding=0,
        stride=1,
    )

    conv1.load_state_from_network("conv1", party, na.Rational)

    conv2 = Conv2d(
        kernel_size=2,
        in_channels=3,
        out_channels=2,
        padding=1,
        stride=2,
    )

    conv2.load_state_from_network("conv2", party, na.Rational)

    x_conv1 = conv1(x)
    x_conv2 = conv2(x)

    y_conv1 = conv1(y)
    y_conv2 = conv2(y)

    assert x_conv1.shape == (1, 1, 3, 1), x_conv1.shape
    assert x_conv2.shape == (1, 2, 3, 2), x_conv2.shape

    assert y_conv1.shape == (1, 3, 1), y_conv1.shape
    assert y_conv2.shape == (2, 3, 2), y_conv2.shape

    x_conv1_out = x_conv1.output(party, "x_conv1")
    x_conv2_out = x_conv2.output(party, "x_conv2")

    y_conv1_out = y_conv1.output(party, "y_conv1")
    y_conv2_out = y_conv2.output(party, "y_conv2")

    return x_conv1_out + x_conv2_out + y_conv1_out + y_conv2_out
