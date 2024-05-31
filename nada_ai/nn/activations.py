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
        binary_mask = x.applypyfunc(
            lambda a: (a > Integer(0)).if_else(Integer(1), Integer(0))
        )
        return x * binary_mask
