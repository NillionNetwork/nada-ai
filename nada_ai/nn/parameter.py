"""Parameter logic"""

from functools import reduce
from typing import Iterable, Union

import numpy as np
import nada_algebra as na
from nada_ai.exceptions import MismatchedShapesException
from nada_dsl import Integer

_ShapeLike = Union[int, Iterable[int]]

class Parameter(na.NadaArray):
    """Parameter class"""

    def __init__(self, shape: _ShapeLike) -> None:
        """Initializes light NadaArray wrapper.

        Args:
            shape (_ShapeLike, optional): Parameter array shape.
        """
        zeros = np.zeros(shape, dtype=int)
        zeros = np.frompyfunc(Integer, 1, 1)(zeros)
        super().__init__(inner=zeros)

    def numel(self) -> int:
        """Returns number of elements in inner array.

        Returns:
            int: Number of elements in inner array.
        """
        return reduce(lambda x, y: x * y, self.shape, 1)

    def load_state(self, state: na.NadaArray) -> None:
        """Loads a provided NadaArray as new Parameter state.

        Args:
            state (na.NadaArray): New state.

        Raises:
            MismatchedShapesException: Raised when state of incompatible shape is provided.
        """
        if state.shape != self.shape:
            raise MismatchedShapesException("Could not load state of shape `%s` for parameter of shape `%s`" % (state.shape, self.shape))

        self.inner = state.inner
