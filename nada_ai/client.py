"""
This module provides functions to work with the Python Nillion Client
"""

import warnings

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
        as_rational: bool = True,
        scale: int = 16,
        nada_type: _NillionType = nillion.SecretInteger,
    ) -> Dict[str, _NillionType]:
        """Exports model state as a Dict of Nillion secret types.

        Args:
            name (str): Name to be used to store state secrets in the network.
            as_rational (bool, optional): Whether or not to export the state as rational values. Defaults to True.
            scale (int, optional): Scaling factor to be used as base-2 exponent during quantization. Only used if state is
                exported as rational values. Defaults to 16.
            nada_type (_NillionType, optional): Data type to convert weights to. Defaults to nillion.SecretInteger.

        Raises:
            TypeError: Raised when model state has incompatible values.

        Returns:
            Dict[str, _NillionType]: Dict of Nillion secret types that represents model state.
        """
        if scale <= 0:
            warnings.warn(
                "Provided scaling factor `%d` is very low. This scale will be used as a base-2 exponent during quantization. Expected a value above 0."
                % scale
            )
        if scale >= 64:
            warnings.warn(
                "Provided scaling factor `%d` is very high. This scale will be used as a base-2 exponent during quantization. Expected a value below 64."
                % scale
            )

        state_secrets = {}
        for layer_name, layer_weight in self.state_dict.items():

            if isinstance(layer_weight, torch.Tensor):
                layer_weight = layer_weight.detach().numpy()
            elif isinstance(layer_weight, (float, int)):
                layer_weight = np.array(layer_weight)
            elif not isinstance(layer_weight, np.ndarray):
                raise TypeError(
                    "Could not parse layer weight of type `%s` in state_dict"
                    % type(layer_weight).__name__
                )

            if as_rational:
                layer_weight = layer_weight * (2**scale)

            layer_weight = layer_weight.astype(int)
            layer_weight[layer_weight == 0] = (
                1  # TODO: remove line when pushing zero as Secret is implemented
            )

            state_secret = na_client.array(
                layer_weight, prefix=f"{name}_{layer_name}", nada_type=nada_type
            )

            state_secrets.update(state_secret)

        return state_secrets
