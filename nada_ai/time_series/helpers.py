"""Time series helper functions"""

import numpy as np


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
    x_T = dates * 2 * np.pi
    fourier_components = np.empty((dates.shape[0], 2 * series_order))
    for i in range(series_order):
        c = x_T * (i + 1) / period
        fourier_components[:, 2 * i] = np.sin(c)
        fourier_components[:, (2 * i) + 1] = np.cos(c)
    return fourier_components
