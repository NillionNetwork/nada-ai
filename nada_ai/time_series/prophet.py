"""Facebook Prophet implementation"""

from typing import Any, Dict, Tuple

import nada_numpy as na
import numpy as np
from typing_extensions import override

from nada_ai.nada_typing import NadaInteger
from nada_ai.nn.module import Module
from nada_ai.nn.parameter import Parameter
from nada_ai.utils import ensure_cleartext, fourier_series


class Prophet(Module):  # pylint:disable=too-many-instance-attributes
    """Prophet forecasting implementation"""

    # pylint:disable=too-many-arguments
    def __init__(
        self,
        n_changepoints: int,
        growth: str = "linear",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = "additive",
        *,
        nada_type: NadaInteger = na.SecretRational,
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
            nada_type (NadaInteger, optional): Nada data type to use. Defaults to na.SecretRational.
        """
        super().__init__()
        self.growth = growth

        self.seasonalities: Dict[str, Any] = {
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

        num_fourier = self._num_fourier_terms()

        # NOTE: MAP estimation is assumed, so M=1 guaranteed
        M = 1  # pylint:disable=invalid-name

        self.k = Parameter(na.zeros((M, 1), ensure_cleartext(nada_type)))
        self.m = Parameter(na.zeros((M, 1), ensure_cleartext(nada_type)))
        self.beta = (
            Parameter(na.zeros((M, num_fourier), ensure_cleartext(nada_type)))
            if num_fourier != 0
            else na.NadaArray(np.array([None]))
        )
        self.delta = Parameter(
            na.zeros((M, n_changepoints), ensure_cleartext(nada_type))
        )
        self.changepoints_t = Parameter(
            na.zeros((n_changepoints,), ensure_cleartext(nada_type))
        )
        self.y_scale = Parameter(na.zeros((1,), ensure_cleartext(nada_type)))

    def _num_fourier_terms(self) -> int:
        """
        Calculates the number of Fourier terms.

        Returns:
            int: Number of Fourier terms.
        """
        num_fourier = 0
        for _, mode_seasonality in self.seasonalities.items():
            for _, period_seasonality in mode_seasonality.items():
                # NOTE: times two because there is always a term for both sin and cos
                num_fourier += period_seasonality["fourier_order"] * 2
        return num_fourier

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

            seasonal_components[mode] = (
                na.NadaArray(np.array(components))
                if len(components) != 0
                else na.zeros(dates.shape, na.Rational)
            )

        add_component = seasonal_components["additive"] * self.y_scale
        add_component = add_component.sum(axis=0)

        mult_component = seasonal_components["multiplicative"] + na.rational(1)
        mult_component = mult_component.prod(axis=0)

        return add_component, mult_component

    def make_seasonality_features(
        self, dates: np.ndarray, seasonalities: Dict[str, Any]
    ) -> Dict[str, na.NadaArray]:
        """
        Generates seasonality features per seasonal component.

        Args:
            dates (np.ndarray): Array of timestamp values.
            seasonalities (Dict[str, Any]): Seasonality config.

        Returns:
            Dict[str, na.NadaArray]: Generated seasonality features.
        """
        features = {}
        for name, props in seasonalities.items():
            period, fourier_order = props["period"], props["fourier_order"]
            feats = None
            if fourier_order != 0:
                feats = fourier_series(dates, period, fourier_order)
                feats = na.frompyfunc(na.rational, 1, 1)(feats)
            features[name] = feats
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
            # pylint:disable=unnecessary-lambda-assignment
            less_than_fn = lambda a, b: (a <= b).if_else(na.rational(1), na.rational(0))
            less_than_vectorized = na.frompyfunc(less_than_fn, 2, 1)
            mask = less_than_vectorized(self.changepoints_t[None, :], t[..., None])
            deltas_t = delta * mask
            k_t = deltas_t.sum(axis=1) + k
            m_t = (-self.changepoints_t * deltas_t).sum(axis=1) + m
            trend = k_t * t + m_t
        elif self.growth == "flat":
            trend = na.ones_like(t, na.Rational) * m
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
        assert dates.shape == floor.shape, "Prophet inputs must be equally sized."
        assert floor.shape == t.shape, "Prophet inputs must be equally sized."

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
        dtype = dates.dtype
        if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating):
            return dates
        if np.issubdtype(dtype, np.datetime64):
            return dates.astype(np.float64)
        error_msg = f"Could not convert dates of type `{dates}` to a numeric array."
        raise TypeError(error_msg)

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

    # pylint:disable=arguments-differ
    @override  # type: ignore
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
