"""NN layers logic"""

from typing import Iterable, Union
import numpy as np
import nada_algebra as na
from nada_ai.nn.module import Module
from nada_ai.nn.parameter import Parameter
from nada_dsl import Integer

_ShapeLike = Union[int, Iterable[int]]


class Linear(Module):
    """Linear layer implementation"""

    def __init__(self, in_features: int, out_features: int, include_bias: bool = True) -> None:
        """Linear (or fully-connected) layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            include_bias (bool, optional): Whether or not to include a bias term. Defaults to True.
        """
        self.weight = Parameter((out_features, in_features))
        self.bias = Parameter(out_features) if include_bias else None

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Module output.
        """
        if self.bias is None:
            return self.weight @ x
        return self.weight @ x + self.bias


class Conv2d(Module):
    """Conv2D layer implementation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _ShapeLike,
        padding: int = 0,
        stride: int = 1,
        include_bias: bool = True,
    ) -> None:
        """2D-convolutional operator.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (_ShapeLike): Size of convolution kernel.
            padding (int, optional): Padding length. Defaults to 0.
            stride (int, optional): Stride length. Defaults to 1.
            include_bias (bool, optional): Whether or not to include a bias term. Defaults to True.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        kernel_height, kernel_width = kernel_size

        self.weight = Parameter((out_channels, in_channels, kernel_height, kernel_width))
        self.bias = Parameter((1, out_channels, 1, 1)) if include_bias else None

        self.padding = padding
        self.stride = stride

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Module output.
        """
        unbatched=False
        if len(x.shape) == 3:
            # Assume unbatched --> assign batch_size of 1
            x = x.reshape(1, *x.shape)
            unbatched=True

        batch_size, _, input_rows, input_cols = x.shape
        out_channels, _, kernel_rows, kernel_cols = self.weight.shape

        if self.padding > 0:
            padded_input = np.pad(
                x.inner,
                [
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ],
                mode="constant",
            )
            padded_input = np.frompyfunc(
                lambda x: Integer(x.item()) if isinstance(x, np.int64) else x, 1, 1
            )(padded_input)
        else:
            padded_input = x.inner

        output_rows = (input_rows + 2 * self.padding - kernel_rows) // self.stride + 1
        output_cols = (input_cols + 2 * self.padding - kernel_cols) // self.stride + 1

        output_tensor = np.zeros(
            (batch_size, out_channels, output_rows, output_cols)
        ).astype(Integer)
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(output_rows):
                    for j in range(output_cols):
                        start_i = i * self.stride
                        start_j = j * self.stride

                        receptive_field = padded_input[
                            b,
                            :,
                            start_i : start_i + kernel_rows,
                            start_j : start_j + kernel_cols,
                        ]
                        output_tensor[b, oc, i, j] = np.sum(
                            self.weight.inner[oc] * receptive_field
                        )

        if self.bias is not None:
            output_tensor = output_tensor + self.bias.inner

        if unbatched:
            output_tensor = output_tensor[0]

        return na.NadaArray(output_tensor)


class AvgPool2d(Module):
    """2d-Average pooling layer implementation"""

    def __init__(self, kernel_size: _ShapeLike, padding: int = 0, stride: int = 1) -> None:
        """2D-average pooling layer.

        Args:
            kernel_size (_ShapeLike): Size of pooling kernel.
            padding (int, optional): Padding length. Defaults to 0.
            stride (int, optional): Stride length. Defaults to 1.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size

        self.padding = padding
        self.stride = stride

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Module output.
        """
        unbatched=False
        if len(x.shape) == 3:
            # Assume unbatched --> assign batch_size of 1
            x = x.reshape(1, *x.shape)
            unbatched=True

        batch_size, channels, input_height, input_width = x.shape

        if self.padding > 0:
            padded_input = np.pad(
                x.inner,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
            )
            padded_input = np.frompyfunc(
                lambda x: Integer(x.item()) if isinstance(x, np.int64) else x, 1, 1
            )(padded_input)
        else:
            padded_input = x.inner

        output_height = (
            input_height + 2 * self.padding - self.kernel_size[0]
        ) // self.stride + 1
        output_width = (
            input_width + 2 * self.padding - self.kernel_size[1]
        ) // self.stride + 1

        output_array = np.zeros(
            (batch_size, channels, output_height, output_width)
        ).astype(Integer)
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_h = i * self.stride
                        start_w = j * self.stride
                        end_h = start_h + self.kernel_size[0]
                        end_w = start_w + self.kernel_size[1]

                        pool_region = padded_input[b, c, start_h:end_h, start_w:end_w]
                        output_array[b, c, i, j] = np.sum(pool_region) / Integer(
                            pool_region.size
                        )
        if unbatched:
            output_array = output_array[0]

        return na.NadaArray(output_array)


class Flatten(Module):
    """Flatten layer implementation"""

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Module output.
        """
        return x.flatten()
