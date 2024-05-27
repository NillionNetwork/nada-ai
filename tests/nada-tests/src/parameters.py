from nada_dsl import *
from nada_ai.nn import Module, Parameter
import nada_algebra as na


def nada_main():
    party = Party("party")

    class TestModule(Module):
        def __init__(self) -> None:
            self.param1 = Parameter((3, 2))
            self.param2 = Parameter(3)

        def forward(self, x: na.NadaArray) -> na.NadaArray:
            return (self.param1 @ x) + self.param2

    x = na.array([2], party, "input")

    mod = TestModule()

    mod.load_state_from_network("testmod", party)

    result = mod(x)

    assert result.shape == (3,), result.shape

    return result.output(party, "output")
