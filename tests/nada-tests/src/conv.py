import nada_numpy as na
import pytest
from nada_dsl import Party, SecretInteger

from nada_ai.nn import Conv2d


def nada_main():
    party = Party("party")

    x = na.array((1, 3, 4, 2), party, "input_x", na.SecretRational)
    y = na.array((3, 4, 2), party, "input_y", SecretInteger)

    conv1 = Conv2d(
        kernel_size=2,
        in_channels=3,
        out_channels=1,
        padding=0,
        stride=1,
        nada_type=na.SecretRational,
    )
    conv1.load_state_from_network("conv1", party, na.SecretRational)

    conv2 = Conv2d(
        kernel_size=2,
        in_channels=3,
        out_channels=2,
        padding=1,
        stride=2,
        nada_type=SecretInteger,
    )
    conv2.load_state_from_network("conv2", party, SecretInteger)

    x_conv = conv1(x)
    y_conv = conv2(y)

    with pytest.raises(TypeError):
        conv1(y)
    with pytest.raises(TypeError):
        conv2(x)

    assert x_conv.shape == (1, 1, 3, 1), x_conv.shape
    assert y_conv.shape == (2, 3, 2), y_conv.shape

    x_conv_out = x_conv.output(party, "x_conv")
    y_conv_out = y_conv.output(party, "y_conv")

    return x_conv_out + y_conv_out
