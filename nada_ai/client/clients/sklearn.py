"""Scikit-learn client implementation"""

import sklearn
from nada_ai.client.model_client import ModelClient
from nada_ai.typing import LinearModel

__all__ = ["SklearnClient"]


class SklearnClient(ModelClient):
    """ModelClient for Scikit-learn models"""

    def __init__(self, model: sklearn.base.BaseEstimator) -> None:
        """
        Client initialization.

        Args:
            model (sklearn.base.BaseEstimator): Sklearn model object to wrap around.
        """
        if isinstance(model, LinearModel):
            state_dict = {"coef": model.coef_}
            if model.fit_intercept is True:
                state_dict.update({"intercept": model.intercept_})
        else:
            raise NotImplementedError(
                f"Instantiating ModelClient from Sklearn model type `{type(model).__name__}` is not yet implemented."
            )

        self.state_dict = state_dict
