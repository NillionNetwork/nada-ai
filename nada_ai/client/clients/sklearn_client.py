"""Scikit-learn client implementation"""

from nada_ai.client.model_client import ModelClient
from typing import Union

import sklearn
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
)

_LinearModel = Union[LinearRegression, LogisticRegression, LogisticRegressionCV]

__all__ = ["SklearnClient"]


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
