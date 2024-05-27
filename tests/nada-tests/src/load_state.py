from nada_dsl import *
from nada_ai.nn import Module, Parameter
import nada_algebra as na

def nada_main():
    party = Party("party")

    class TestModule(Module):
        def __init__(self) -> None:
            self.param1 = Parameter((3,2))
            self.param2 = Parameter(2)
        def forward(self, x: na.NadaArray) -> na.NadaArray:
            ...

    mod1 = TestModule()
    mod2 = TestModule()

    mod1.load_state_from_network("module1", party, as_rational=True, scale=4)
    mod2.load_state_from_network("module2", party, as_rational=False)

    m1_p1_out = mod1.param1.output(party, "module1_param1")
    m1_p2_out = mod1.param2.output(party, "module1_param2")

    m2_p1_out = mod2.param1.output(party, "module2_param1")
    m2_p2_out = mod2.param2.output(party, "module2_param2")

    return m1_p1_out + m1_p2_out + m2_p1_out + m2_p2_out
