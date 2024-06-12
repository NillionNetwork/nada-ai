"""Distance implementation"""

from typing import override
import nada_algebra as na
from nada_ai.nn.module import Module

__all__ = ["DotProductSimilarity"]


class DotProductSimilarity(Module):
    """Dot product similarity module"""

    @override
    def forward(self, x_1: na.NadaArray, x_2: na.NadaArray) -> na.NadaArray:
        """
        Forward pass logic.

        Args:
            x_1 (na.NadaArray): First input array.
            x_2 (na.NadaArray): Second input array.

        Returns:
            na.NadaArray: Dot product between input arrays.
        """
        return x_1 @ x_2.T
