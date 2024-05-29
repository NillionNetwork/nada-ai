import nada_algebra as na
from nada_ai import nn
from nada_dsl import Integer


class MyConvModule(nn.Module):
    """My Convolutional module"""

    def __init__(self) -> None:
        """Contains some ConvNet components"""
        self.conv = nn.Conv2d(kernel_size=2, in_channels=3, out_channels=2)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """Takes convolution & pools"""
        return self.pool(self.conv(x))

class PlusOne(nn.Module):
    """My +1 module"""

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """Simply does +1"""
        return x + Integer(1)

class MyModel(nn.Module):
    """My aribitrarily specific model architecture"""

    def __init__(self) -> None:
        """Model is a collection of arbitrary custom components"""
        self.conv_module = MyConvModule()
        self.plus_one = PlusOne()
        self.linear = nn.Linear(4, 2)
        self.flatten = nn.Flatten()

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """My custom forward pass logic"""
        x = self.conv_module(x)
        x = self.flatten(x)
        x = self.plus_one(x)
        x = self.linear(x)
        return x
