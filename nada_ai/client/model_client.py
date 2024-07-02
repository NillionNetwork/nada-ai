"""Model client implementation"""

from abc import ABC, ABCMeta
from typing import Any, Dict, Sequence

import nada_numpy.client as na_client
import numpy as np
import torch

from nada_ai.nada_typing import NillionType


class ModelClientMeta(ABCMeta):
    """ML model client metaclass"""

    def __call__(cls, *args, **kwargs) -> object:
        """
        Ensures __init__ defines a value for `self.state_dict`.

        Raises:
            AttributeError: Raised when no value for `self.state_dict` is defined.

        Returns:
            object: Result object.
        """
        obj = super().__call__(*args, **kwargs)
        if not getattr(obj, "state_dict") or getattr(obj, "state_dict") is None:
            raise AttributeError("Required attribute `state_dict` not set")
        return obj


class ModelClient(ABC, metaclass=ModelClientMeta):
    """ML model client"""

    state_dict: Dict[str, Any]

    def export_state_as_secrets(
        self,
        name: str,
        nada_type: NillionType,
    ) -> Dict[str, NillionType]:
        """
        Exports model state as a Dict of Nillion secret types.

        Args:
            name (str): Name to be used to store state secrets in the network.
            nada_type (NillionType): Data type to convert weights to.

        Raises:
            NotImplementedError: Raised when unsupported model state type is passed.
            TypeError: Raised when model state has incompatible values.

        Returns:
            Dict[str, NillionType]: Dict of Nillion secret types that represents model state.
        """
        state_secrets = {}
        for state_layer_name, state_layer_weight in self.state_dict.items():
            layer_name = f"{name}_{state_layer_name}"
            layer_state = self.__ensure_numpy(state_layer_weight)
            state_secret = na_client.array(layer_state, layer_name, nada_type)
            state_secrets.update(state_secret)

        return state_secrets

    def __ensure_numpy(self, array_like: Any) -> np.ndarray:
        """
        Ensures an array-like input is converted to a NumPy array.

        Args:
            array_like (Any): Some array-like input.

        Raises:
            TypeError: Raised when an input is passed of an incompatible type.

        Returns:
            np.ndarray: NumPy-converted result.
        """
        if isinstance(array_like, torch.Tensor):
            return array_like.detach().numpy()
        if isinstance(array_like, (Sequence, np.ndarray)):
            return np.array(array_like)
        if isinstance(array_like, (float, int, np.floating)):
            return np.array([array_like])
        error_msg = f"Could not convert type `{type(array_like)}` to NumPy array"
        raise TypeError(error_msg)
