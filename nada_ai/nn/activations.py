"""NN activations logic"""

import nada_algebra as na
from nada_ai.nn.module import Module
from nada_dsl import Integer


class ReLU(Module):
    """ReLU layer implementation"""

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Module output.
        """
        x = x.applypyfunc(
            lambda a: a.value
            if isinstance(a, (na.Rational, na.SecretRational))
            else a
        )
        return x.applypyfunc(
            lambda a: (
                a > Integer(0)
            ).if_else(a, Integer(0))
        )
