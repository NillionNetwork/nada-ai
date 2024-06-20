"""Flatten layer implementation"""

import nada_numpy as na
import numpy as np

from nada_ai.nn.module import Module

__all__ = ["Flatten"]


class Flatten(Module):
    """Flatten layer implementation"""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        """
        Flatten operator.

        Args:
            start_dim (int, optional): Flatten start dimension. Defaults to 1.
            end_dim (int, optional): Flatten end dimenion. Defaults to -1.
        """
        super().__init__()

        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """
        Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Module output.
        """
        shape = x.shape

        end_dim = self.end_dim
        if end_dim < 0:
            end_dim += len(shape)

        flattened_dim_size = int(np.prod(shape[self.start_dim : end_dim + 1]))
        flattened_shape = (
            shape[: self.start_dim] + (flattened_dim_size,) + shape[end_dim + 1 :]
        )

        return x.reshape(flattened_shape)
