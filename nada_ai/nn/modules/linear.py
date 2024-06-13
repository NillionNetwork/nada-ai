"""Linear layer implementation"""

import nada_algebra as na

from nada_ai.nn.module import Module
from nada_ai.nn.parameter import Parameter

__all__ = ["Linear"]


class Linear(Module):
    """Linear layer implementation"""

    def __init__(
        self, in_features: int, out_features: int, include_bias: bool = True
    ) -> None:
        """
        Linear (or fully-connected) layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            include_bias (bool, optional): Whether or not to include a bias term. Defaults to True.
        """
        self.weight = Parameter((out_features, in_features))
        self.bias = Parameter(out_features) if include_bias else None

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """
        Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Module output.
        """
        if self.bias is None:
            return self.weight @ x
        return self.weight @ x + self.bias
