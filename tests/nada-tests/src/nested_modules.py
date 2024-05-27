from nada_dsl import *
from nada_ai.nn import Module, Parameter
import nada_algebra as na

def nada_main():
    party = Party("party")

    class TestModule1(Module):
        def __init__(self) -> None:
            self.param1 = Parameter((3,2))
            self.param2 = Parameter(3)
        def forward(self, x: na.NadaArray) -> na.NadaArray:
            return (self.param1 @ x) + self.param2

    class TestModule2(Module):
        def __init__(self, module1: TestModule1) -> None:
            self.mod = module1
            self.param1 = Parameter(3)
        def forward(self, x: na.NadaArray) -> na.NadaArray:
            x = self.mod(x)
            return x + self.param1

    x = na.array([2], party, "input")

    mod = TestModule2(module1=TestModule1())

    mod.load_state_from_network("testmod", party, as_rational=False)

    result = mod(x)

    return result.output(party, "output")
