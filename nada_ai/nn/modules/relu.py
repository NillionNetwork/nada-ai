"""NN activations logic"""

from typing import Union

import nada_numpy as na
from nada_dsl import (Integer, NadaType, PublicBoolean, PublicInteger,
                      SecretBoolean, SecretInteger)

from nada_ai.nn.module import Module

__all__ = ["ReLU"]


class ReLU(Module):
    """ReLU layer implementation"""

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """
        Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Module output.
        """
        relu = self._rational_relu if x.is_rational else self._relu
        mask = x.apply(relu)
        return x * mask

    @staticmethod
    def _rational_relu(
        value: Union[na.Rational, na.SecretRational]
    ) -> Union[na.Rational, na.SecretRational]:
        """
        Element-wise ReLU logic for rational values.

        Args:
            value (Union[na.Rational, na.SecretRational]): Input rational.

        Returns:
            Union[na.Rational, na.SecretRational]: ReLU output rational.
        """
        above_zero: Union[na.PublicBoolean, na.SecretBoolean] = value > na.rational(0)
        return above_zero.if_else(na.rational(1), na.rational(0))

    @staticmethod
    def _relu(value: NadaType) -> Union[PublicInteger, SecretInteger]:
        """
        Element-wise ReLU logic for NadaType values.

        Args:
            value (NadaType): Input nada value.

        Returns:
            Union[PublicInteger, SecretInteger]: Output nada value.
        """
        above_zero: Union[PublicBoolean, SecretBoolean] = value > Integer(0)
        return above_zero.if_else(Integer(1), Integer(0))
