"""Parameter logic"""

import numpy as np
import nada_algebra as na
from nada_ai.utils import ShapeLike, MismatchedShapesException


class Parameter(na.NadaArray):
    """Parameter class"""

    def __init__(self, shape: ShapeLike) -> None:
        """
        Initializes light NadaArray wrapper.

        Args:
            shape (ShapeLike, optional): Parameter array shape.
        """
        zeros = na.zeros(shape)
        super().__init__(inner=zeros.inner)

    def numel(self) -> int:
        """
        Returns number of elements in inner array.

        Returns:
            int: Number of elements in inner array.
        """
        return np.prod(self.shape)

    def load_state(self, state: na.NadaArray) -> None:
        """
        Loads a provided NadaArray as new Parameter state.

        Args:
            state (na.NadaArray): New state.

        Raises:
            MismatchedShapesException: Raised when state of incompatible shape is provided.
        """
        if state.shape != self.shape:
            msg = f"State shape `{state.shape}` does not match expected `{self.shape}`."
            raise MismatchedShapesException(msg)

        self.inner = state.inner
