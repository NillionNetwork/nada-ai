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

    def __init__(
        self, in_features: int, out_features: int, include_bias: bool = True
    ) -> None:
        """
        Linear (or fully-connected) layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            include_bias (bool, optional): Whether or not to include a bias term. Defaults to True.
        """
        self.weight = Parameter((out_features, in_features))
        self.bias = Parameter(out_features) if include_bias else None

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """
        Forward pass.

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
        padding: _ShapeLike = 0,
        stride: _ShapeLike = 1,
        include_bias: bool = True,
    ) -> None:
        """
        2D-convolutional operator.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (_ShapeLike): Size of convolution kernel.
            padding (_ShapeLike, optional): Padding length. Defaults to 0.
            stride (_ShapeLike, optional): Stride length. Defaults to 1.
            include_bias (bool, optional): Whether or not to include a bias term. Defaults to True.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        self.weight = Parameter((out_channels, in_channels, *kernel_size))
        self.bias = Parameter(out_channels) if include_bias else None

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """
        Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Module output.
        """
        unbatched = False
        if x.ndim == 3:
            # Assume unbatched --> assign batch_size of 1
            x = x.reshape(1, *x.shape)
            unbatched = True

        batch_size, _, input_height, input_width = x.shape
        out_channels, _, kernel_rows, kernel_cols = self.weight.shape

        if any(pad > 0 for pad in self.padding):
            x = na.pad(
                x,
                [
                    (0, 0),
                    (0, 0),
                    self.padding,
                    self.padding,
                ],
                mode="constant",
            )

        out_height = (
            input_height + 2 * self.padding[0] - self.kernel_size[0]
        ) // self.stride[0] + 1
        out_width = (
            input_width + 2 * self.padding[1] - self.kernel_size[1]
        ) // self.stride[1] + 1

        output_tensor = na.zeros((batch_size, out_channels, out_height, out_width))
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        start_i = i * self.stride[0]
                        start_j = j * self.stride[1]

                        receptive_field = x[
                            b,
                            :,
                            start_i : start_i + kernel_rows,
                            start_j : start_j + kernel_cols,
                        ]
                        output_tensor[b, oc, i, j] = na.sum(
                            self.weight[oc] * receptive_field
                        )

        if self.bias is not None:
            output_tensor += self.bias.reshape(1, out_channels, 1, 1)

        if unbatched:
            output_tensor = output_tensor[0]

        return output_tensor


class AvgPool2d(Module):
    """2d-Average pooling layer implementation"""

    def __init__(
        self,
        kernel_size: _ShapeLike,
        stride: _ShapeLike = None,
        padding: _ShapeLike = 0,
    ) -> None:
        """
        2D-average pooling layer.

        Args:
            kernel_size (_ShapeLike): Size of pooling kernel.
            stride (_ShapeLike, optional): Stride length. Defaults to the size of the pooling kernel.
            padding (_ShapeLike, optional): Padding length. Defaults to 0.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

        if stride is None:
            stride = self.kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """
        Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Module output.
        """
        unbatched = False
        if x.ndim == 3:
            # Assume unbatched --> assign batch_size of 1
            x = x.reshape(1, *x.shape)
            unbatched = True

        batch_size, channels, input_height, input_width = x.shape
        is_rational = x.is_rational

        if any(pad > 0 for pad in self.padding):
            x = na.pad(
                x,
                (
                    (0, 0),
                    (0, 0),
                    self.padding,
                    self.padding,
                ),
                mode="constant",
            )

        out_height = (
            input_height + 2 * self.padding[0] - self.kernel_size[0]
        ) // self.stride[0] + 1
        out_width = (
            input_width + 2 * self.padding[1] - self.kernel_size[1]
        ) // self.stride[1] + 1

        output_tensor = na.zeros((batch_size, channels, out_height, out_width))
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        start_h = i * self.stride[0]
                        start_w = j * self.stride[1]
                        end_h = start_h + self.kernel_size[0]
                        end_w = start_w + self.kernel_size[1]

                        pool_region = x[b, c, start_h:end_h, start_w:end_w]

                        if is_rational:
                            pool_size = na.rational(pool_region.size)
                        else:
                            pool_size = Integer(pool_region.size)

                        output_tensor[b, c, i, j] = na.sum(pool_region) / pool_size

        if unbatched:
            output_tensor = output_tensor[0]

        return na.NadaArray(output_tensor)


class Flatten(Module):
    """Flatten layer implementation"""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        """
        Flatten operator.

        Args:
            start_dim (int, optional): Flatten start dimension. Defaults to 1.
            end_dim (int, optional): Flatten end dimenion. Defaults to -1.
        """
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: na.NadaArray) -> na.NadaArray:
        """
        Forward pass.

        Args:
            x (na.NadaArray): Input array.

        Returns:
            na.NadaArray: Module output.
        """
        shape = x.shape

        end_dim = self.end_dim
        if end_dim < 0:
            end_dim += len(shape)

        flattened_dim_size = int(np.prod(shape[self.start_dim : end_dim + 1]))
        flattened_shape = (
            shape[: self.start_dim] + (flattened_dim_size,) + shape[end_dim + 1 :]
        )

        return x.reshape(flattened_shape)
