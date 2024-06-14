"""Scikit-learn client implementation"""

import prophet

from nada_ai.client.model_client import ModelClient

__all__ = ["ProphetClient"]


class ProphetClient(ModelClient):
    """ModelClient for Prophet models"""

    def __init__(self, model: prophet.forecaster.Prophet) -> None:
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
