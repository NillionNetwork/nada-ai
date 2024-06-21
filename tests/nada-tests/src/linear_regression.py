import nada_numpy as na
from nada_dsl import *

from nada_ai.linear_model import LinearRegression


def nada_main():
    party = Party("party")

    x = na.array([4], party, "input", SecretInteger)

    model = LinearRegression(4, nada_type=SecretInteger)

    model.load_state_from_network("testmod", party, SecretInteger)

    result = model(x)

    return na.output(result, party, "my_output")
