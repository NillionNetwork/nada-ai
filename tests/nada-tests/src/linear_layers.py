from nada_dsl import *
from nada_ai.nn import Module
import nada_algebra as na
from nada_ai.nn import Linear


def nada_main():
    party = Party("party")

    class TestModule(Module):
        def __init__(self) -> None:
            self.linear_0 = Linear(3, 2)
            self.linear_1 = Linear(2, 1)

        def forward(self, x: na.NadaArray) -> na.NadaArray:
            x = self.linear_0(x)
            x = self.linear_1(x)
            return x

    x = na.array([3], party, "input")

    mod = TestModule()

    mod.load_state_from_network("testmod", party, as_rational=False)

    result = mod(x)

    assert result.shape == (1,), result.shape

    return result.output(party, "output")
