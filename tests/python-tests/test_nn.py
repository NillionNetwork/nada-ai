"""NN components unit tests"""

import nada_numpy as na
import numpy as np
import pytest
from nada_dsl import Integer

from nada_ai.exceptions import MismatchedShapesException
from nada_ai.nn.module import Module, Parameter


class TestModule:

    def test_parameters_1(self):
        class TestModule(Module):
            def __init__(self) -> None:
                self.param1 = Parameter(na.zeros((1, 2), na.Rational))
                self.param2 = Parameter(na.zeros((2, 2), na.Rational))

            def forward(x: na.NadaArray) -> na.NadaArray: ...

        mod = TestModule()
        parameters = [param[0] for param in mod.named_parameters()]

        assert len(parameters) == 2

        assert "param1" in parameters
        assert "param2" in parameters

    def test_parameters_2(self):
        class TestModule(Module):
            def __init__(self) -> None:
                self.param1 = Parameter(na.zeros((1, 2)))
                self.param2 = Parameter(na.zeros((2, 2)))

            def forward(x: na.NadaArray) -> na.NadaArray: ...

        class TestModule2(Module):
            def __init__(self) -> None:
                self.mod = TestModule()
                self.param3 = Parameter(na.zeros((3, 2)))

            def forward(x: na.NadaArray) -> na.NadaArray: ...

        mod = TestModule2()
        parameters = [param[0] for param in mod.named_parameters()]

        assert len(parameters) == 3

        assert "mod.param1" in parameters
        assert "mod.param2" in parameters
        assert "param3" in parameters

    def test_parameters_3(self):
        param = Parameter(na.zeros((2, 3, 1, 3)))
        numel = param.numel()

        assert numel == 2 * 3 * 1 * 3

    def test_parameters_4(self):
        class TestModule(Module):
            def __init__(self) -> None:
                self.param1 = Parameter(na.zeros((3, 3)))
                self.param2 = Parameter(na.zeros((2, 2)))

            def forward(x: na.NadaArray) -> na.NadaArray: ...

        class TestModule2(Module):
            def __init__(self) -> None:
                self.mod = TestModule()
                self.param3 = Parameter(na.zeros((3, 2)))

            def forward(x: na.NadaArray) -> na.NadaArray: ...

        mod = TestModule2()
        numel = mod.numel()

        assert numel == (3 * 3) + (2 * 2) + (3 * 2)

    def test_parameters_5(self):
        param = Parameter(na.zeros((2, 3)))

        alphas = na.alphas((2, 3), alpha=Integer(42))
        param.load(alphas, Integer)

        assert param[0][0] == Integer(42)

    def test_parameters_6(self):
        class TestModule(Module):
            def __init__(self) -> None:
                self.param1 = Parameter(na.zeros((2, 3)))
                self.param2 = Parameter(na.zeros((2, 3)))

            def forward(x: na.NadaArray) -> na.NadaArray: ...

        mod = TestModule()

        alphas = na.alphas((2, 3), alpha=Integer(42))

        for _, param in mod.named_parameters():
            param.load(alphas, Integer)

        for _, param in mod.named_parameters():
            assert param[0][0] == Integer(42)

    def test_parameters_7(self):
        class TestModule(Module):
            def __init__(self) -> None:
                self.param1 = Parameter(na.zeros((2, 3)))
                self.param2 = Parameter(na.zeros((2, 3)))

            def forward(x: na.NadaArray) -> na.NadaArray: ...

        class TestModule2(Module):
            def __init__(self) -> None:
                self.mod = TestModule()
                self.param3 = Parameter(na.zeros((2, 3)))

            def forward(x: na.NadaArray) -> na.NadaArray: ...

        mod = TestModule2()

        alphas = na.alphas((2, 3), alpha=Integer(42))

        for _, param in mod.named_parameters():
            param.load(alphas, Integer)

        for _, param in mod.named_parameters():
            assert param[0][0] == Integer(42)

    def test_parameters_8(self):
        param = Parameter(na.zeros((2, 3)))

        alphas = na.alphas((3, 3), alpha=Integer(42))

        with pytest.raises(MismatchedShapesException):
            param.load(alphas, Integer)

    def test_parameters_9(self):
        class TestModule(Module):
            def __init__(self) -> None:
                self.param = Parameter(na.zeros((2, 3)))
                self.param = Parameter(na.zeros((2, 2)))
                self.something = na.NadaArray(inner=na.zeros((3, 2)))
                self.something_else = 42

            def forward(x: na.NadaArray) -> na.NadaArray: ...

        mod = TestModule()

        assert len(list(mod.named_parameters())) == 1

    def test_parameters_10(self):
        param = Parameter(na.zeros((2,)))

        assert len(param.shape) == 1
        assert param.numel() == 2

    def test_parameters_11(self):
        param1 = Parameter(na.zeros((2, 3)))
        param2 = Parameter(na.zeros((3, 2)))

        assert isinstance(param1.inner, np.ndarray)
        assert isinstance(param2.inner, np.ndarray)

        result = param1 @ param2

        assert isinstance(result, na.NadaArray)
