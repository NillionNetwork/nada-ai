"""Parameter logic"""

from typing import Union, get_args

import nada_numpy as na
import numpy as np

from nada_ai.exceptions import MismatchedShapesException
from nada_ai.nada_typing import NadaInteger


class Parameter(na.NadaArray):
    """Parameter class"""

    def __init__(self, data: Union[na.NadaArray, np.ndarray]) -> None:
        """
        Initializes light NadaArray wrapper.

        Args:
            data (Union[na.NadaArray, np.ndarray]): Parameter data.
        """
        if isinstance(data, na.NadaArray):
            data = data.inner
        super().__init__(inner=data)

    def numel(self) -> int:
        """
        Returns number of elements in inner array.

        Returns:
            int: Number of elements in inner array.
        """
        return int(np.prod(self.shape))

    def load(
        self, data: Union[na.NadaArray, np.ndarray], nada_type: NadaInteger
    ) -> None:
        """
        Loads a provided array as new Parameter data.

        Args:
            data (Union[na.NadaArray, np.ndarray]): New parameter data.
            nada_type (NadaInteger): NadaType to interpret the state values as.

        Raises:
            MismatchedShapesException: Raised when state of incompatible shape is provided.
            TypeError: Raised when incompatible data type is passed.
        """
        if isinstance(data, np.ndarray):
            data = na.NadaArray(data)

        if data.shape != self.shape:
            msg = f"Data shape `{data.shape}` does not match expected `{self.shape}`."
            raise MismatchedShapesException(msg)

        if nada_type not in get_args(self.dtype):
            msg = f"Nada type `{self.dtype}` is not compatible with expected `{nada_type}`."
            raise TypeError(msg)

        self.inner = data.inner
