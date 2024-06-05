from nada_dsl import *
import nada_algebra as na
from nada_ai.nn import LinearRegression


def nada_main():
    party = Party("party")

    x = na.array([4], party, "input", SecretInteger)

    model = LinearRegression(4)

    model.load_state_from_network("testmod", party, SecretInteger)

    result = model(x)

    assert result.shape == (1,), result.shape

    return result.output(party, "output")
