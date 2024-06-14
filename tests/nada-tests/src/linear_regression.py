import nada_algebra as na
from nada_dsl import *

from nada_ai.linear_model import LinearRegression


def nada_main():
    party = Party("party")

    x = na.array([4], party, "input", na.SecretRational)

    model = LinearRegression(4)

    model.load_state_from_network("testmod", party, na.SecretRational)

    result = model(x)

    assert result.shape == (1,), result.shape

    return result.output(party, "output")
