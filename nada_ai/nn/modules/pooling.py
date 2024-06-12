"""Pooling layer implementation"""

from typing import Iterable, Union
import nada_algebra as na
from nada_ai.nn.module import Module
from nada_dsl import Integer

_ShapeLike = Union[int, Iterable[int]]

__all__ = ["AvgPool2d"]


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
