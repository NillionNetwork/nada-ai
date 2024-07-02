import nada_numpy as na

from nada_ai import nn


class MyNN(nn.Module):
    """My brand new model"""

    def __init__(self) -> None:
        """Model is a two layers and an activations"""
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=2, kernel_size=3, padding=1, stride=3
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=18, out_features=2)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """My forward pass logic"""
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x
