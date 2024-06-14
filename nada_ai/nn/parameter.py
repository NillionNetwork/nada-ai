"""Parameter logic"""

import nada_algebra as na
import numpy as np

from nada_ai.exceptions import MismatchedShapesException
from nada_ai.nada_typing import ShapeLike


class Parameter(na.NadaArray):
    """Parameter class"""

    def __init__(self, shape: ShapeLike = 1) -> None:
        """
        Initializes light NadaArray wrapper.

        Args:
            shape (ShapeLike, optional): Parameter array shape. Defaults to 1.
        """
        super().__init__(inner=np.empty(shape))

    def numel(self) -> int:
        """
        Returns number of elements in inner array.

        Returns:
            int: Number of elements in inner array.
        """
        return int(np.prod(self.shape))

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
