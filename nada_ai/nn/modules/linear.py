"""Linear layer implementation"""

import nada_numpy as na

from nada_ai.nada_typing import NadaInteger
from nada_ai.nn.module import Module
from nada_ai.nn.parameter import Parameter
from nada_ai.utils import check_nada_type, ensure_cleartext

__all__ = ["Linear"]


class Linear(Module):
    """Linear layer implementation"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        include_bias: bool = True,
        *,
        nada_type: NadaInteger = na.SecretRational,
    ) -> None:
        """
        Linear (or fully-connected) layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            include_bias (bool, optional): Whether or not to include a bias term. Defaults to True.
            nada_type (NadaInteger, optional): Nada data type to use. Defaults to na.SecretRational.
        """
        super().__init__()

        self.weight = Parameter(
            na.zeros((out_features, in_features), ensure_cleartext(nada_type))
        )
        self.bias = (
            Parameter(na.zeros((out_features,), ensure_cleartext(nada_type)))
            if include_bias
            else None
        )

    @check_nada_type(level="error")
    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """
        Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Module output.
        """
        if self.bias is None:
            return x @ self.weight.T
        return x @ self.weight.T + self.bias
