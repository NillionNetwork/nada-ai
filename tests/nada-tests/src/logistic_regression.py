import nada_numpy as na
from nada_dsl import *

from nada_ai.linear_model import LogisticRegression


def nada_main():
    party = Party("party")

    x = na.array([4], party, "input", SecretInteger)

    model = LogisticRegression(4, 3, nada_type=SecretInteger)

    model.load_state_from_network("testmod", party, SecretInteger)

    result = model(x)

    return result.output(party, "my_output")
