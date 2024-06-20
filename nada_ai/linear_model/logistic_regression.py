"""Logistic regression implementation"""

import nada_numpy as na

from nada_ai.nada_typing import NadaInteger
from nada_ai.nn.module import Module
from nada_ai.nn.parameter import Parameter
from nada_ai.utils import check_nada_type, ensure_cleartext

__all__ = ["LogisticRegression"]


class LogisticRegression(Module):
    """Logistic regression implementation"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        include_bias: bool = True,
        nada_type: NadaInteger = na.SecretRational,
    ) -> None:
        """
        Initialization.
        NOTE: this model will produce logistic regression logits instead of
        the actual output probabilities!
        To convert logits to probabilities, they need to be passed through a
        sigmoid activation function.

        Args:
            in_features (int): Number of input features to regression.
            num_classes (int): Number of classes to predict.
            include_bias (bool, optional): Whether or not to include a bias term. Defaults to True.
            nada_type (NadaInteger, optional): Nada data type to use. Defaults to na.SecretRational.
        """
        super().__init__()

        self.coef = Parameter(
            na.zeros((out_features, in_features), ensure_cleartext(nada_type))
        )
        self.intercept = (
            Parameter(na.zeros((out_features,), ensure_cleartext(nada_type)))
            if include_bias
            else None
        )

    @check_nada_type(level="error")
    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """
        Forward pass.
        NOTE: this forward pass will return the logistic regression logits instead
        of the actual output probabilities!
        To convert logits to probabilities, they need to be passed through a
        sigmoid activation function.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Module output.
        """
        if self.intercept is None:
            return self.coef @ x
        return self.coef @ x + self.intercept
