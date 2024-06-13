"""PyTorch client implementation"""

from torch import nn

from nada_ai.client.model_client import ModelClient

__all__ = ["TorchClient"]


class TorchClient(ModelClient):
    """ModelClient for PyTorch models"""

    def __init__(self, model: nn.Module) -> None:
        """
        Client initialization.

        Args:
            model (nn.Module): PyTorch model object to wrap around.
        """
        self.state_dict = model.state_dict()
