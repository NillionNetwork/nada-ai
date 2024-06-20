"""Distance implementation"""

import nada_numpy as na
from typing_extensions import override

from nada_ai.nn.module import Module

__all__ = ["DotProductSimilarity"]


class DotProductSimilarity(Module):
    """Dot product similarity module"""

    # pylint:disable=arguments-differ
    @override  # type: ignore
    def forward(self, x_1: na.NadaArray, x_2: na.NadaArray) -> na.NadaArray:
        """
        Forward pass logic.

        Args:
            x_1 (na.NadaArray): First input array.
            x_2 (na.NadaArray): Second input array.

        Returns:
            na.NadaArray: Dot product between input arrays.
        """
        if x_1.dtype != x_2.dtype:
            raise TypeError(
                f"Incompatible nada types detected: {x_1.dtype} and {x_2.dtype}"
            )
        return x_1 @ x_2.T
