"""Model client implementation"""

from abc import ABC, ABCMeta
import nada_algebra as na
import nada_algebra.client as na_client
from typing import Any, Dict, Sequence
from nada_ai.utils import NillionType

import torch
import numpy as np


class ModelClientMeta(ABCMeta):
    """ML model client metaclass"""

    def __call__(self, *args, **kwargs) -> object:
        """
        Ensures __init__ defines a value for `self.state_dict`.

        Raises:
            AttributeError: Raised when no value for `self.state_dict` is defined.

        Returns:
            object: Result object.
        """
        obj = super(ModelClientMeta, self).__call__(*args, **kwargs)
        if not getattr(obj, "state_dict"):
            raise AttributeError("required attribute `state_dict` not set")
        return obj


class ModelClient(ABC, metaclass=ModelClientMeta):
    """ML model client"""

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
        if nada_type not in (na.Rational, na.SecretRational):
            raise NotImplementedError("Exporting non-rational state is not supported")

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
        raise TypeError(
            "Could not convert type `%s` to NumPy array" % type(array_like).__name__
        )
