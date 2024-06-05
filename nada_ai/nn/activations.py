"""NN activations logic"""

from typing import Union
import nada_algebra as na
from nada_ai.nn.module import Module
from nada_dsl import Integer, NadaType, SecretInteger, SecretBoolean, PublicBoolean


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
        if x.dtype in (na.Rational, na.SecretRational):
            mask = x.apply(self._rational_relu)
        else:
            mask = x.apply(self._relu)

        return x * mask

    @staticmethod
    def _rational_relu(value: Union[na.Rational, na.SecretRational]) -> na.SecretRational:
        """
        Element-wise ReLU logic for rational values.

        Args:
            value (Union[na.Rational, na.SecretRational]): Input rational.

        Returns:
            na.SecretRational: ReLU output rational.
        """
        above_zero: Union[PublicBoolean, SecretBoolean] = value > na.rational(0)
        return above_zero.if_else(na.rational(1), na.rational(0))

    @staticmethod
    def _relu(value: NadaType) -> SecretInteger:
        """
        Element-wise ReLU logic for NadaType values.

        Args:
            value (NadaType): Input nada value.

        Returns:
            SecretInteger: Output nada value.
        """
        above_zero: Union[PublicBoolean, SecretBoolean] = value > Integer(0)
        return above_zero.if_else(Integer(1), Integer(0))
