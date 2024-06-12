"""Generic state client implementation"""

from typing import Any, Dict
from nada_ai.client.model_client import ModelClient

__all__ = ["StateClient"]


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
