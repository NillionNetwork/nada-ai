"""Convolutional operator implementation"""

import nada_algebra as na
from nada_ai.nn.module import Module
from nada_ai.nn.parameter import Parameter
from nada_ai.utils import ShapeLike

__all__ = ["Conv2d"]


class Conv2d(Module):
    """Conv2D layer implementation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: ShapeLike,
        padding: ShapeLike = 0,
        stride: ShapeLike = 1,
        include_bias: bool = True,
    ) -> None:
        """
        2D-convolutional operator.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (ShapeLike): Size of convolution kernel.
            padding (ShapeLike, optional): Padding length. Defaults to 0.
            stride (ShapeLike, optional): Stride length. Defaults to 1.
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
