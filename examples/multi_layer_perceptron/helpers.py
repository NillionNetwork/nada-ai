"""Helper functions and classes"""

from typing import Dict
import torch
import py_nillion_client as nillion

from torch import nn


class MyModel(nn.Module):
    """Fully customizable model"""

    def __init__(self, model_name: str) -> None:
        """Initialization

        Args:
            model_name (str): Model name to be used as prefix for secret names.
        """
        super().__init__()
        self.model_name = model_name
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1, stride=4)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input array.

        Returns:
            torch.Tensor: Output array.
        """
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

    def export_weights_as_secrets(self, precision: int) -> Dict:
        """Exports all current model weights as quantized Nillion secrets.

        Args:
            precision (int): Desired precision for quantization.

        Raises:
            ValueError: Raised when unexpected layer is encountered

        Returns:
            Dict: Model weight secrets.
        """
        weight_secrets = {}
        def return_data(result: dict, name: str, array: torch.Tensor):
            if len(array.shape) == 1:
                result.update(
                    {
                        f"{name}_{i}": nillion.SecretInteger(round(value.item() * precision))
                        if round(value.item() * precision) != 0
                        else nillion.SecretInteger(round(value.item() * precision)+1)
                        for i, value in enumerate(array)
                    }
                )
                return
            [return_data(result, f"{name}_{i}", array[i]) for i in range(array.shape[0])]
            return result

        for weight_name, weight_tensor in self.state_dict().items():
            result = {}
            return_data(result, f"{self.model_name}_{weight_name}", weight_tensor)
            weight_secrets.update(result)

        return weight_secrets
