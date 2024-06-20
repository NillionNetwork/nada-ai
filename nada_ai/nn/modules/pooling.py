"""Pooling layer implementation"""

from typing import Optional

import nada_numpy as na

from nada_ai.nada_typing import ShapeLike2d
from nada_ai.nn.module import Module
from nada_ai.utils import ensure_tuple, kernel_output_shape, to_nada_type

__all__ = ["AvgPool2d"]


class AvgPool2d(Module):
    """2d-Average pooling layer implementation"""

    def __init__(
        self,
        kernel_size: ShapeLike2d,
        stride: Optional[ShapeLike2d] = None,
        padding: ShapeLike2d = 0,
    ) -> None:
        """
        2D-average pooling layer.

        Args:
            kernel_size (ShapeLike2d): Size of pooling kernel.
            stride (ShapeLike2d, optional): Stride length. Defaults to the
                size of the pooling kernel.
            padding (ShapeLike2d, optional): Padding length. Defaults to 0.
        """
        super().__init__()

        self.kernel_size = ensure_tuple(kernel_size)
        self.padding = ensure_tuple(padding)

        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = ensure_tuple(stride)

    # pylint:disable=too-many-locals
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

        out_height, out_width = kernel_output_shape(
            (input_height, input_width), self.padding, self.kernel_size, self.stride
        )

        output_tensor = na.zeros(
            (batch_size, channels, out_height, out_width), x.cleartext_nada_type
        )
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        start_h = i * self.stride[0]
                        start_w = j * self.stride[1]
                        end_h = start_h + self.kernel_size[0]
                        end_w = start_w + self.kernel_size[1]

                        pool_region = x[b, c, start_h:end_h, start_w:end_w]
                        pool_size = to_nada_type(
                            pool_region.size, x.cleartext_nada_type
                        )

                        output_tensor[b, c, i, j] = na.sum(pool_region) / pool_size

        if unbatched:
            output_tensor = output_tensor[0]

        return na.NadaArray(output_tensor)
