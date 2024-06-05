"""NN activations logic"""

from typing import Union
import nada_algebra as na
from nada_ai.nn.module import Module
from nada_dsl import Integer, NadaType, SecretBoolean, PublicBoolean


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
        if x.is_rational:
            mask = x.apply(self._rational_relu)
        else:
            mask = x.apply(self._relu)

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
    def _relu(value: NadaType) -> NadaType:
        """
        Element-wise ReLU logic for NadaType values.

        Args:
            value (NadaType): Input nada value.

        Returns:
            NadaType: Output nada value.
        """
        above_zero: Union[PublicBoolean, SecretBoolean] = value > Integer(0)
        return above_zero.if_else(Integer(1), Integer(0))
