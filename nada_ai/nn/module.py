"""Neural Network Module logic"""

from abc import ABC, abstractmethod
import inspect
from typing import Iterator, Tuple, Union
from nada_ai.nn.parameter import Parameter
import nada_algebra as na
from nada_dsl import (
    Party,
    SecretInteger,
    SecretUnsignedInteger,
    PublicInteger,
    PublicUnsignedInteger,
)

_NadaInteger = Union[
    SecretInteger,
    SecretUnsignedInteger,
    PublicInteger,
    PublicUnsignedInteger,
    na.SecretRational,
]


class Module(ABC):
    """Generic neural network module"""

    @abstractmethod
    def forward(self, x: na.NadaArray, *args, **kwargs) -> na.NadaArray:
        """
        Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Output array.
        """
        ...

    def __call__(self, x: na.NadaArray, *args, **kwargs) -> na.NadaArray:
        """
        Proxy for forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Output array.
        """
        return self.forward(x, *args, **kwargs)

    def __named_parameters(self, prefix: str) -> Iterator[Tuple[str, Parameter]]:
        """
        Recursively generates all parameters in Module, its submodules, their submodules, etc.

        Args:
            prefix (str): Named parameter prefix. Parameter names have a "."-delimited trace of
                all modules that encapsulate them.

        Yields:
            Iterator[Tuple[str, Parameter]]: Iterator over named parameters.
        """
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                name = ".".join([prefix, name]) if prefix else name
                yield name, value
            elif isinstance(value, Module):
                name = ".".join([prefix, name]) if prefix else name
                yield from value.__named_parameters(prefix=name)

    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        """
        Generates all parameters in Module, its submodules, their submodules, etc.

        Yields:
            Iterator[Tuple[str, Parameter]]: Iterator over named parameters.
        """
        yield from self.__named_parameters(prefix="")

    def __numel(self) -> Iterator[int]:
        """
        Recursively generates number of elements in each Parameter in the module.

        Yields:
            Iterator[int]: Number of elements in each Parameter.
        """
        for _, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value.numel()
            elif isinstance(value, Module):
                yield from value.__numel()

    def numel(self) -> int:
        """
        Returns total number of elements in the module.

        Returns:
            int: Total number of elements.
        """
        return sum(self.__numel())

    def load_state_from_network(
        self,
        name: str,
        party: Party,
        nada_type: _NadaInteger = na.SecretRational,
    ) -> None:
        """
        Loads the model state from the Nillion network.

        Args:
            name (str): Name to be used to find state secrets in the network.
            party (Party): Party that provided the model state in the network.
            nada_type (_NadaInteger, optional): NadaType to interpret the state values as. Defaults to na.SecretRational.
        """
        for param_name, param in self.named_parameters():
            state_name = f"{name}_{param_name}"
            param_state = na.array(param.shape, party, state_name, nada_type)
            param.load_state(param_state)
