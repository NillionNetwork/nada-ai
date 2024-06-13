"""Stores various util functions"""

from typing import Tuple

import numpy as np

from nada_ai.nada_typing import ShapeLike2d


def fourier_series(
    dates: np.ndarray,
    period: int | float,
    series_order: int,
) -> np.ndarray:
    """
    Generates (plain-text) Fourier series.

    Args:
        dates (np.ndarray): Array of timestamp values.
        period (int | float): Fourier period.
        series_order (int): Order of Fourier series.

    Returns:
        np.ndarray: Generates Fourier series.
    """
    x_transpose = dates * 2 * np.pi
    fourier_components = np.empty((dates.shape[0], 2 * series_order))
    for i in range(series_order):
        c = x_transpose * (i + 1) / period
        fourier_components[:, 2 * i] = np.sin(c)
        fourier_components[:, (2 * i) + 1] = np.cos(c)
    return fourier_components


def kernel_output_shape(
    input_dims: Tuple[int, int],
    padding: Tuple[int, int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Determines the output shape after a kernel operation.

    Args:
        input_dims (Tuple[int, int]): Input dimensions.
        padding (Tuple[int, int]): Padding.
        kernel_size (Tuple[int, int]): Size of kernel.
        stride (Tuple[int, int]): Stride.

    Returns:
        Tuple[int, int]: Output shape.
    """
    out_height = (input_dims[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    out_width = (input_dims[1] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    return out_height, out_width


def ensure_tuple(tuple_like: ShapeLike2d) -> Tuple[int, int]:
    """
    Ensures input gets converted to a shape tuple.

    Args:
        tuple_like (ShapeLike2d): Input.

    Returns:
        Tuple[int, int]: Converted output.
    """
    if isinstance(tuple_like, int):
        return (tuple_like, tuple_like)
    return tuple_like
