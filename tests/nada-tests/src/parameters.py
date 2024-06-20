import nada_numpy as na
from nada_dsl import *

from nada_ai.nn import Module, Parameter


def nada_main():
    party = Party("party")

    x = na.array([2], party, "input", na.SecretRational)

    class TestModule(Module):
        def __init__(self) -> None:
            self.param1 = Parameter(na.zeros((3, 2), na.Rational))
            self.param2 = Parameter(na.zeros((3,), na.Rational))

        def forward(self, x: na.NadaArray) -> na.NadaArray:
            return (self.param1 @ x) + self.param2

    mod = TestModule()

    mod.load_state_from_network("testmod", party, na.SecretRational)

    result = mod(x)

    assert result.shape == (3,), result.shape

    return result.output(party, "output")
