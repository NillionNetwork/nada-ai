import nada_numpy as na
from nada_dsl import *

from nada_ai.nn import Linear, Module


def nada_main():
    party = Party("party")

    x = na.array([3], party, "input", SecretInteger)

    class TestModule(Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_0 = Linear(3, 2, nada_type=SecretInteger)
            self.linear_1 = Linear(2, 1, nada_type=SecretInteger)

        def forward(self, x: na.NadaArray) -> na.NadaArray:
            x = self.linear_0(x)
            x = self.linear_1(x)
            return x

    mod = TestModule()

    mod.load_state_from_network("testmod", party, SecretInteger)

    result = mod(x)

    assert result.shape == (1,), result.shape

    return result.output(party, "output")
