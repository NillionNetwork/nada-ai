"""
This module provides functions to work with the Python Nillion Client
"""

from abc import ABC, ABCMeta
import nada_algebra as na
import nada_algebra.client as na_client
from typing import Any, Dict, Sequence, Union

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
)
import prophet
import torch
from torch import nn
import sklearn
import numpy as np
import py_nillion_client as nillion

_NillionType = Union[
    na.Rational,
    na.SecretRational,
    nillion.SecretInteger,
    nillion.SecretUnsignedInteger,
    nillion.PublicVariableInteger,
    nillion.PublicVariableUnsignedInteger,
]
_LinearModel = Union[LinearRegression, LogisticRegression, LogisticRegressionCV]


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
        nada_type: _NillionType,
    ) -> Dict[str, _NillionType]:
        """
        Exports model state as a Dict of Nillion secret types.

        Args:
            name (str): Name to be used to store state secrets in the network.
            nada_type (_NillionType): Data type to convert weights to.

        Raises:
            NotImplementedError: Raised when unsupported model state type is passed.
            TypeError: Raised when model state has incompatible values.

        Returns:
            Dict[str, _NillionType]: Dict of Nillion secret types that represents model state.
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

        error_msg = f"Could not convert `{type(array_like).__name__}` to NumPy array"
        raise TypeError(error_msg)


class StateClient(ModelClient):
    """ModelClient for generic model states"""

    def __init__(self, state_dict: Dict[str, Any]) -> None:
        """
        Client initialization.
        This client accepts an arbitrary model state as input.

        Args:
            state_dict (Dict[str, Any]): State dict.
        """
        self.state_dict = state_dict


class TorchClient(ModelClient):
    """ModelClient for PyTorch models"""

    def __init__(self, model: nn.Module) -> None:
        """
        Client initialization.

        Args:
            model (nn.Module): PyTorch model object to wrap around.
        """
        self.state_dict = model.state_dict()


class SklearnClient(ModelClient):
    """ModelClient for Scikit-learn models"""

    def __init__(self, model: sklearn.base.BaseEstimator) -> None:
        """
        Client initialization.

        Args:
            model (sklearn.base.BaseEstimator): Sklearn model object to wrap around.
        """
        if isinstance(model, _LinearModel):
            state_dict = {"coef": model.coef_}
            if model.fit_intercept is True:
                state_dict.update({"intercept": model.intercept_})
        else:
            raise NotImplementedError(
                f"Instantiating ModelClient from Sklearn model type `{type(model).__name__}` is not yet implemented."
            )

        self.state_dict = state_dict


class ProphetClient(ModelClient):
    """ModelClient for Prophet models"""

    def __init__(self, model: "prophet.forecaster.Prophet") -> None:
        """
        Client initialization.

        Args:
            model (prophet.forecaster.Prophet): Prophet model.
        """
        self.state_dict = {
            "k": model.params["k"],
            "m": model.params["m"],
            "delta": model.params["delta"],
            "beta": model.params["beta"],
            "changepoints_t": model.changepoints_t,
            "y_scale": model.y_scale,
        }
