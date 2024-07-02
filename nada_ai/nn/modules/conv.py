"""Convolutional operator implementation"""

import nada_numpy as na

from nada_ai.nada_typing import NadaInteger, ShapeLike2d
from nada_ai.nn.module import Module
from nada_ai.nn.parameter import Parameter
from nada_ai.utils import (check_nada_type, ensure_cleartext, ensure_tuple,
                           kernel_output_shape)

__all__ = ["Conv2d"]


class Conv2d(Module):
    """Conv2D layer implementation"""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: ShapeLike2d,
        padding: ShapeLike2d = 0,
        stride: ShapeLike2d = 1,
        include_bias: bool = True,
        *,
        nada_type: NadaInteger = na.SecretRational,
    ) -> None:
        """
        2D-convolutional operator.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (ShapeLike2d): Size of convolution kernel.
            padding (ShapeLike2d, optional): Padding length. Defaults to 0.
            stride (ShapeLike2d, optional): Stride length. Defaults to 1.
            include_bias (bool, optional): Whether or not to include a bias term. Defaults to True.
            nada_type (NadaInteger, optional): Nada data type to use. Defaults to na.SecretRational.
        """
        super().__init__()

        self.kernel_size = ensure_tuple(kernel_size)
        self.padding = ensure_tuple(padding)
        self.stride = ensure_tuple(stride)

        self.weight = Parameter(
            na.zeros(
                (out_channels, in_channels, *self.kernel_size),
                ensure_cleartext(nada_type),
            )
        )
        self.bias = (
            Parameter(na.zeros((out_channels,), ensure_cleartext(nada_type)))
            if include_bias
            else None
        )

    # pylint:disable=too-many-locals
    @check_nada_type(level="error")
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

        out_height, out_width = kernel_output_shape(
            (input_height, input_width), self.padding, self.kernel_size, self.stride
        )

        output_tensor = na.zeros(
            (batch_size, out_channels, out_height, out_width),
            x.cleartext_nada_type,
        )
        with na.context.UnsafeArithmeticSession():
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

        if x.is_rational:
            output_tensor = output_tensor.apply(lambda value: value.rescale_down())

        if self.bias is not None:
            output_tensor += self.bias.reshape(1, out_channels, 1, 1)

        if unbatched:
            output_tensor = output_tensor[0]

        return output_tensor
