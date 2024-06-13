"""Linear regression implementation"""

import nada_algebra as na

from nada_ai.nn.module import Module
from nada_ai.nn.parameter import Parameter

__all__ = ["LinearRegression", "LogisticRegression"]


class LinearRegression(Module):
    """Linear regression implementation"""

    def __init__(self, in_features: int, include_bias: bool = True) -> None:
        """
        Initialization.

        Args:
            in_features (int): Number of input features to regression.
            include_bias (bool, optional): Whether or not to include a bias term. Defaults to True.
        """
        self.coef = Parameter(in_features)
        self.intercept = Parameter() if include_bias else None

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """
        Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Module output.
        """
        if self.intercept is None:
            return self.coef @ x
        return self.coef @ x + self.intercept


class LogisticRegression(Module):
    """Logistic regression implementation"""

    def __init__(
        self, in_features: int, out_features: int, include_bias: bool = True
    ) -> None:
        """
        Initialization.

        Args:
            in_features (int): Number of input features to regression.
            num_classes (int): Number of classes to predict.
            include_bias (bool, optional): Whether or not to include a bias term. Defaults to True.
        """
        self.coef = Parameter((out_features, in_features))
        self.intercept = Parameter(out_features) if include_bias else None

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """
        Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Module output.
        """
        if self.intercept is None:
            return self.coef @ x
        return self.coef @ x + self.intercept
