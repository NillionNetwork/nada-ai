"""Scikit-learn client implementation"""

import sklearn

from nada_ai.client.model_client import ModelClient
from nada_ai.nada_typing import LinearModel

__all__ = ["SklearnClient"]


class SklearnClient(ModelClient):
    """ModelClient for Scikit-learn models"""

    def __init__(self, model: sklearn.base.BaseEstimator) -> None:
        """
        Client initialization.

        Args:
            model (sklearn.base.BaseEstimator): Sklearn model object to wrap around.
        """
        if isinstance(model, LinearModel):  # type: ignore
            state_dict = {"coef": model.coef_}
            if model.fit_intercept is True:
                state_dict.update({"intercept": model.intercept_})
        else:
            error_msg = (
                f"Instantiating SklearnClient from `{type(model)}` is not implemented."
            )
            raise NotImplementedError(error_msg)

        self.state_dict = state_dict
