import nada_algebra as na
from nada_ai import nn


class MyModel(nn.Module):
    """My custom model architecture"""

    def __init__(self) -> None:
        """Model is a simple feed-forward NN with 2 layers and a ReLU activation"""
        self.linear_0 = nn.Linear(8, 4)
        self.linear_1 = nn.Linear(4, 2)
        self.relu = nn.ReLU()

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """My custom forward pass logic"""
        x = self.linear_0(x)
        x = self.relu(x)
        return self.linear_1(x)
