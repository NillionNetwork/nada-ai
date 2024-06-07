"""Time series helper functions"""

from collections import OrderedDict
import pandas as pd
import numpy as np
import nada_algebra as na


def fourier_series(
    dates: pd.Series,
    period: int | float,
    series_order: int,
) -> np.ndarray:
    x_T = dates * np.pi * 2
    fourier_components = np.empty((dates.shape[0], 2 * series_order))
    for i in range(series_order):
        c = x_T * (i + 1) / period
        fourier_components[:, 2 * i] = np.sin(c)
        fourier_components[:, (2 * i) + 1] = np.cos(c)
    return fourier_components


def decode_component_matrix(component_matrix: na.NadaArray):
    additive_terns, multiplicative_terms, yearly, weekly, daily = component_matrix.inner
    return {
        "additive_terms": additive_terns,
        "multiplicative_terms": multiplicative_terms,
        "yearly": yearly,
        "weekly": weekly,
        "daily": daily,
    }


def decode_seasonality_matrix(seasonality_matrix: na.NadaArray):
    period_names = ["yearly", "yearly", "weekly", "weekly", "daily", "daily"]
    modes = ["additive", "multiplicative"] * 3

    result = {}
    for row, period_name, mode in zip(seasonality_matrix.inner, period_names, modes):
        period, fourier_order, prior_scale = row
        result[f"{period_name}_{mode}"] = {
            "period": period,
            "fourier_order": fourier_order,
            "prior_scale": prior_scale,
            "mode": mode,
        }

    return OrderedDict(result)
