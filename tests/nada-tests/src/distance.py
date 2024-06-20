import nada_numpy as na
from nada_dsl import *

from nada_ai.nn import DotProductSimilarity


def nada_main():
    party = Party("party")

    # 2 queries, each of size 3
    queries = na.array((2, 3), party, "input_x", SecretInteger)
    # 5 values, each also of size 3
    values = na.array((5, 3), party, "input_y", SecretInteger)

    dps = DotProductSimilarity()

    similarities = dps(queries, values)
    assert similarities.shape == (2, 5)  # for each query, the similarity to each value

    return similarities.output(party, "my_output")
