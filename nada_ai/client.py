"""
This module provides functions to work with the Python Nillion Client
"""

import nada_algebra as na
import nada_algebra.client as na_client
from collections import OrderedDict
from typing import Any, Dict, Iterable, Union

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
)
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


class ModelClient:
    """ML model client"""

    def __init__(self, model: Any, state_dict: OrderedDict[str, np.ndarray]) -> None:
        """Initialization.

        Args:
            model (Any): Model object to wrap around.
            state_dict (OrderedDict[str, np.ndarray]): Model state.
        """
        self.model = model
        self.state_dict = state_dict

    @classmethod
    def from_torch(cls, model: nn.Module) -> "ModelClient":
        """Instantiates a model client from a PyTorch model.

        Args:
            model (nn.Module): PyTorch nn.Module object.

        Returns:
            ModelClient: Instantiated model client.
        """
        state_dict = model.state_dict()
        return cls(model=model, state_dict=state_dict)

    @classmethod
    def from_sklearn(cls, model: sklearn.base.BaseEstimator) -> "ModelClient":
        """Instantiates a model client from a Sklearn estimator.

        Args:
            model (sklearn.base.BaseEstimator): Sklearn estimator object.

        Returns:
            ModelClient: Instantiated model client.
        """
        if not isinstance(model, sklearn.base.BaseEstimator):
            raise TypeError(
                "Cannot interpret type `%s` as Sklearn model. Expected (sub)type of `sklearn.base.BaseEstimator`"
                % type(model).__name__
            )

        if isinstance(model, LinearRegression):
            state_dict = OrderedDict(
                {
                    "coef": model.coef_,
                    "intercept": (
                        model.intercept_
                        if isinstance(model.intercept_, Iterable)
                        else np.array([model.intercept_])
                    ),
                }
            )
        elif isinstance(model, (LogisticRegression, LogisticRegressionCV)):
            state_dict = OrderedDict(
                {
                    "coef": model.coef_,
                    "intercept": model.intercept_,
                }
            )
        else:
            raise NotImplementedError(
                "Instantiating ModelClient from Sklearn model type `%s` is not yet implemented."
            )

        return cls(model=model, state_dict=state_dict)

    def export_state_as_secrets(
        self,
        name: str,
        nada_type: _NillionType = na.SecretRational,
    ) -> Dict[str, _NillionType]:
        """Exports model state as a Dict of Nillion secret types.

        Args:
            name (str): Name to be used to store state secrets in the network.
            nada_type (_NillionType, optional): Data type to convert weights to. Defaults to SecretRational.

        Raises:
            TypeError: Raised when model state has incompatible values.

        Returns:
            Dict[str, _NillionType]: Dict of Nillion secret types that represents model state.
        """
        state_secrets = {}
        for state_layer_name, state_layer_weight in self.state_dict.items():
            layer_name = f"{name}_{state_layer_name}"
            state_secret = na_client.array(
                self.__ensure_numpy(state_layer_weight), layer_name, nada_type
            )
            state_secrets.update(state_secret)

        return state_secrets

    def __ensure_numpy(self, array_like: Any) -> np.ndarray:
        """Ensures an array-like input is converted to a NumPy array.

        Args:
            array_like (Any): Some array-like input.

        Raises:
            TypeError: Raised when an input is passed of an incompatible type.

        Returns:
            np.ndarray: NumPy-converted result.
        """
        if isinstance(array_like, torch.Tensor):
            return array_like.detach().numpy()
        if isinstance(array_like, (float, int, np.ndarray)):
            return np.array(array_like)
        raise TypeError(
            "Could not convert type `%s` to NumPy array" % type(array_like).__name__
        )
