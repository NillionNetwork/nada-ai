"""Facebook Prophet implementation"""

import numpy as np
from typing import Dict, Tuple, override

import nada_algebra as na
from nada_ai.nn.module import Module
from nada_ai.nn.parameter import Parameter
from nada_ai.time_series.helpers import fourier_series


class Prophet(Module):
    """Prophet forecasting implementation"""

    def __init__(
        self,
        n_changepoints: int,
        growth: str = "linear",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = "additive",
    ) -> None:
        """
        Prophet model initialization.

        Args:
            n_changepoints (int): Number of changepoints.
            growth (str, optional): Forecasting growth mode. Defaults to "linear".
            yearly_seasonality (bool, optional): Whether or not to include a yearly
                seasonality term. Defaults to True.
            weekly_seasonality (bool, optional): Whether or not to include a weekly
                seasonality term. Defaults to True.
            daily_seasonality (bool, optional): Whether or not to include a daily
                seasonality term. Defaults to False.
            seasonality_mode (str, optional): Seasonality mode. Defaults to 'additive'.
        """
        self.growth = growth

        self.seasonalities = {
            "additive": {},
            "multiplicative": {},
        }
        if yearly_seasonality:
            self.seasonalities[seasonality_mode]["yearly"] = {
                "period": 365.25,
                "fourier_order": 10,
            }
        if weekly_seasonality:
            self.seasonalities[seasonality_mode]["weekly"] = {
                "period": 7,
                "fourier_order": 3,
            }
        if daily_seasonality:
            self.seasonalities[seasonality_mode]["daily"] = {
                "period": 1,
                "fourier_order": 4,
            }

        num_fourier = 0
        for _, mode_seasonality in self.seasonalities.items():
            for _, period_seasonality in mode_seasonality.items():
                # NOTE: times two because there is always a term for both sin and cos
                num_fourier += period_seasonality["fourier_order"] * 2

        M = 1  # NOTE: MAP estimation is assumed, so M=1 guaranteed

        self.k = Parameter((M, 1))
        self.m = Parameter((M, 1))
        self.beta = (
            Parameter((M, num_fourier))
            if num_fourier != 0
            else na.NadaArray(np.array([None]))
        )
        self.delta = Parameter((M, n_changepoints))
        self.changepoints_t = Parameter(n_changepoints)
        self.y_scale = Parameter(1)

    def predict_seasonal_comps(
        self, dates: np.ndarray
    ) -> Tuple[na.NadaArray, na.NadaArray]:
        """
        Predicts seasonal components.

        Args:
            dates (np.ndarray): Array of timestamp values.

        Returns:
            Tuple[na.NadaArray, na.NadaArray]: Additive and multiplicative
                seasonal components.
        """
        [beta] = self.beta

        seasonal_components = {}
        for mode in ["additive", "multiplicative"]:
            seasonal_features = self.make_seasonality_features(
                dates, self.seasonalities[mode]
            )

            components = []
            for _, features in seasonal_features.items():
                if features is None:
                    continue

                comp = features @ beta.T
                components.append(comp)

            if len(components) == 0:
                seasonal_components[mode] = na.zeros(dates.shape, na.Rational)
            else:
                seasonal_components[mode] = na.NadaArray(np.array(components))

        additive_component = seasonal_components["additive"] * self.y_scale
        additive_component = additive_component.sum(axis=0)

        multiplicative_component = -seasonal_components["multiplicative"] + na.rational(
            1
        )
        multiplicative_component = multiplicative_component.prod(axis=0)

        return additive_component, multiplicative_component

    def make_seasonality_features(
        self, dates: np.ndarray, seasonalities: Dict[str, Dict[str, int | float]]
    ) -> Dict[str, na.NadaArray]:
        """
        Generates seasonality features per seasonal component.

        Args:
            dates (np.ndarray): Array of timestamp values.
            seasonalities (Dict[str, Dict[str, int  |  float]]): Seasonality config.

        Returns:
            Dict[str, na.NadaArray]: Generated seasonality features.
        """
        features = {}
        for name, props in seasonalities.items():
            period, fourier_order = props["period"], props["fourier_order"]
            if fourier_order == 0:
                feats = None
            else:
                feats = fourier_series(dates, period, fourier_order)
            features[name] = na.frompyfunc(na.rational, 1, 1)(feats)
        return features

    def predict_trend(self, floor: na.NadaArray, t: na.NadaArray) -> na.NadaArray:
        """
        Predicts trend values.

        Args:
            floor (na.NadaArray): Array of floor values.
            t (na.NadaArray): Array of t values.

        Raises:
            NotImplementedError: Raised when unsupported growth mode is provided.

        Returns:
            na.NadaArray: Predicted trend.
        """
        # NOTE: this indexing is possible because M=1 is guaranteed
        # If this were not the case, we should take the arithmetic mean
        [[k]] = self.k
        [[m]] = self.m
        [delta] = self.delta

        if self.growth == "linear":
            mask = na.frompyfunc(
                lambda a, b: (a <= b).if_else(na.rational(1), na.rational(0)), 2, 1
            )(self.changepoints_t[None, :], t[..., None])
            deltas_t = delta * mask
            k_t = deltas_t.sum(axis=1) + k
            m_t = (-self.changepoints_t * deltas_t).sum(axis=1) + m
            trend = k_t * t + m_t
        elif self.growth == "flat":
            trend = na.ones_like(t, na.rational) * m
        else:
            raise NotImplementedError(self.growth + " is not supported")

        return trend * self.y_scale + floor

    def predict(
        self,
        dates: np.ndarray,
        # TODO: often all zero - opportunity to compress
        floor: na.NadaArray,
        # TODO: can be deterministically generated from len(horizon)
        t: na.NadaArray,
    ) -> na.NadaArray:
        """
        Generates time series forecasts.

        Args:
            dates (np.ndarray): Array of timestamp values.
            floor (na.NadaArray): Array of floor values.
            t (na.NadaArray): Array of t values.

        Returns:
            na.NadaArray: Forecasted values.
        """
        assert len(dates) == len(
            floor
        ), "Provided Prophet inputs must be equally sized."
        assert len(floor) == len(t), "Provided Prophet inputs must be equally sized."

        dates = self.ensure_numeric_dates(dates)
        trend = self.predict_trend(floor, t)
        additive_comps, multiplicative_comps = self.predict_seasonal_comps(dates)
        yhat = trend * multiplicative_comps + additive_comps
        return yhat

    def ensure_numeric_dates(self, dates: np.ndarray) -> np.ndarray:
        """
        Ensures an array of dates is of the correct data format.

        Args:
            dates (np.ndarray): Data array.

        Raises:
            TypeError: Raised when dates array of incompatible type is passed.

        Returns:
            np.ndarray: Standardized dates.
        """
        if isinstance(dates.dtype, np.floating):
            return dates
        if np.issubdtype(dates.dtype, np.datetime64):
            return dates.astype(np.float64)
        raise TypeError(
            f"Could not convert dates array of type `{dates}` to a NumPy array of numerics."
        )

    @override
    def __call__(
        self,
        dates: np.ndarray,
        floor: na.NadaArray,
        t: na.NadaArray,
    ) -> na.NadaArray:
        """
        Forward pass.
        Note: requires multiple input arrays due to special nature of forecasting.

        Args:
            dates (np.ndarray): Array of timestamp values.
            floor (na.NadaArray): Array of floor values.
            t (na.NadaArray): Array of t values.

        Returns:
            na.NadaArray: Forecasted values.
        """
        return self.predict(dates=dates, floor=floor, t=t)

    @override
    def forward(
        self,
        dates: np.ndarray,
        floor: na.NadaArray,
        t: na.NadaArray,
    ) -> na.NadaArray:
        """
        Forward pass.
        Note: requires multiple input arrays due to special nature of forecasting.

        Args:
            dates (np.ndarray): Array of timestamp values.
            floor (na.NadaArray): Array of floor values.
            t (na.NadaArray): Array of t values.

        Returns:
            na.NadaArray: Forecasted values.
        """
        return self.predict(dates=dates, floor=floor, t=t)
