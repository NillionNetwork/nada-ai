"""Inference runners for standard AI use cases"""

import torch
import nada_ai
import nada_numpy as na
from abc import ABC, abstractmethod


class InferenceRunner(ABC):
    @abstractmethod
    def train(dataset) -> None:
        ...

    @abstractmethod
    def run(input_data) -> dict:
        ...

class ConvNetRunner(InferenceRunner):
    def __init__(self) -> None:

        class ConvNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=1, out_channels=2, kernel_size=3, padding=1, stride=3
                )
                self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
                self.fc1 = torch.nn.Linear(in_features=18, out_features=2)

                self.relu = torch.nn.ReLU()
                self.flatten = torch.nn.Flatten()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.relu(self.conv1(x))
                x = self.pool(x)
                x = self.flatten(x)
                x = self.fc1(x)
                return x

        class NadaConvNet(nada_ai.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nada_ai.nn.Conv2d(
                    in_channels=1, out_channels=2, kernel_size=3, padding=1, stride=3
                )
                self.pool = nada_ai.nn.AvgPool2d(kernel_size=2, stride=2)
                self.fc1 = nada_ai.nn.Linear(in_features=18, out_features=2)

                self.relu = nada_ai.nn.ReLU()
                self.flatten = nada_ai.nn.Flatten()

            def forward(self, x: na.NadaArray) -> na.NadaArray:
                x = self.relu(self.conv1(x))
                x = self.pool(x)
                x = self.flatten(x)
                x = self.fc1(x)
                return x

        self.model = ConvNet()
        self.nada_model = NadaConvNet()

class SingleClassificationRunner(InferenceRunner):
    ...

class MultiClassificationRunner(InferenceRunner):
    ...

class RegressionRunner(InferenceRunner):
    ...
