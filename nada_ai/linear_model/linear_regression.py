"""Linear regression implementation"""

import nada_numpy as na

from nada_ai.nada_typing import NadaInteger
from nada_ai.nn.module import Module
from nada_ai.nn.parameter import Parameter
from nada_ai.utils import check_nada_type, ensure_cleartext

__all__ = ["LinearRegression"]


class LinearRegression(Module):
    """Linear regression implementation"""

    def __init__(
        self,
        in_features: int,
        include_bias: bool = True,
        *,
        nada_type: NadaInteger = na.SecretRational,
    ) -> None:
        """
        Initialization.

        Args:
            in_features (int): Number of input features to regression.
            include_bias (bool, optional): Whether or not to include a bias term. Defaults to True.
            nada_type (NadaInteger, optional): Nada data type to use. Defaults to na.SecretRational.
        """
        super().__init__()

        self.coef = Parameter(na.zeros((in_features,), ensure_cleartext(nada_type)))
        self.intercept = (
            Parameter(na.zeros((1,), ensure_cleartext(nada_type)))
            if include_bias
            else None
        )

    @check_nada_type(level="error")
    def forward(self, x: na.NadaArray) -> NadaInteger:
        """
        Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            NadaInteger: Module output.
        """
        if self.intercept is None:
            return x @ self.coef.T
        return x @ self.coef.T + self.intercept[0]
