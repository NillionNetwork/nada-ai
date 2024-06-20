"""Stores various util functions"""

import warnings
from typing import Any, Callable, Tuple

import nada_numpy as na
import numpy as np
from nada_dsl import (Boolean, Integer, PublicBoolean, PublicInteger,
                      PublicUnsignedInteger, SecretBoolean, SecretInteger,
                      SecretUnsignedInteger, UnsignedInteger)

from nada_ai.nada_typing import AnyNadaType, NadaCleartextType, ShapeLike2d


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


def ensure_cleartext(nada_type: AnyNadaType) -> NadaCleartextType:
    """
    Converts Nada type to clear text equivalent if necessary.

    Args:
        nada_type (AnyNadaType): Arbitrary Nada type.

    Returns:
        NadaCleartextType: Cleartext Nada type.
    """
    return {
        SecretInteger: Integer,
        SecretUnsignedInteger: UnsignedInteger,
        PublicInteger: Integer,
        PublicUnsignedInteger: UnsignedInteger,
        SecretBoolean: Boolean,
        PublicBoolean: Boolean,
        na.SecretRational: na.Rational,
        na.SecretBoolean: na.PublicBoolean,
    }.get(nada_type, nada_type)


def check_nada_type(level: str = "error") -> Callable:
    """
    Decorator that checks Nada type compatibility.

    Args:
        level (str, optional): Type check severity level. Defaults to "error".

    Returns:
        Callable: Decorated callable.
    """

    def decorator(func: Callable) -> Callable:
        """
        Decorator function.

        Args:
            func (Callable): Function to be decorated.

        Returns:
            Callable: Decorated function.
        """

        def wrapper(self, *args, **kwargs) -> na.NadaArray:
            """
            Forward pass wrapper.

            Args:
                self (Module): Module self object.

            Raises:
                TypeError: Raised when incompatibly typed input array is detected.
                ValueError: Raised when an invalid level is passed.

            Returns:
                na.NadaArray: Callable output.
            """
            if not any(True for _ in self.named_parameters()):
                return func(self, *args, **kwargs)

            array_dtypes = [
                x.dtype
                for x in list(args) + list(kwargs.values())
                if isinstance(x, na.NadaArray)
            ]
            param_dtypes = [param.dtype for _, param in self.named_parameters()]

            if any(
                param_dtype != array_dtype
                for param_dtype in param_dtypes
                for array_dtype in array_dtypes
            ):
                message = (
                    f"Input dtypes `{array_dtypes}` are not compatible with"
                    f" parameter dtypes `{param_dtypes}`"
                )
                if level == "warn":
                    warnings.warn(message, UserWarning)
                elif level == "error":
                    raise TypeError(message)
                else:
                    raise ValueError(f"Invalid level passed: {level}")
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def to_nada_type(value: Any, nada_type: NadaCleartextType) -> NadaCleartextType:
    """
    Converts a python-native value to a specified clear-text nada type.

    Args:
        value (Any): Python-native value.
        nada_type (NadaCleartextType): Desired nada type.

    Returns:
        NadaCleartextType: Converted value.
    """
    if nada_type == na.Rational:
        nada_type = na.rational
    return nada_type(value)
