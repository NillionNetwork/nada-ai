import nada_numpy as na

from nada_ai import nn


class MyNN(nn.Module):
    """My simple neural net"""

    def __init__(self) -> None:
        """Model is a two layers and an activations"""
        super().__init__()
        self.linear_0 = nn.Linear(8, 4)
        self.linear_1 = nn.Linear(4, 2)
        self.relu = nn.ReLU()

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """My forward pass logic"""
        x = self.linear_0(x)
        x = self.relu(x)
        x = self.linear_1(x)
        return x
