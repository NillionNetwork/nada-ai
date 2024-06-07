"""Facebook Prophet implementation"""

from typing import override
import pandas as pd
import nada_algebra as na
from nada_ai.nn.module import Module
from nada_ai.nn.parameter import Parameter
from nada_ai.time_series.helpers import (
    fourier_series,
    decode_seasonality_matrix,
    decode_component_matrix,
)


class Prophet(Module):
    """Prophet forecasting implementation"""

    def __init__(
        self,
        growth: str = "linear",
        n_changepoints: int = 25,  # TODO: somehow enforce a fixed number of changepoints in case n_changepoints < len(sequence)
        num_seasonality_features: int = 1,  # TODO: somehow enforce a fixed number of seasonalities
    ) -> None:
        if growth not in {"linear"}:
            raise NotImplementedError(f"Growth mode `{growth}` is not supported")

        self.growth = growth

        M = 1  # NOTE: MAP estimation is assumed, so M=1 guaranteed

        self.k = Parameter((M, 1))
        self.m = Parameter((M, 1))
        self.delta = Parameter((M, n_changepoints))
        self.beta = Parameter((M, num_seasonality_features))
        self.changepoints_t = Parameter(n_changepoints)

        self.y_scale = Parameter(1)

        self.seasonality_matrix = Parameter((6, 3))
        self.component_matrix = Parameter(5)

        self.seasonalities = decode_seasonality_matrix(self.seasonality_matrix)
        self.component_modes = decode_component_matrix(self.component_matrix)

    def predict_seasonal_comps(self, df):
        seasonal_features, _, component_cols, _ = self.make_all_seasonality_features(df)

        X = seasonal_features.values
        data = {}
        for component in component_cols.columns:
            component_values = component_cols[component].values
            beta_c = self.beta * component_values

            comp = X @ beta_c.T

            comp = (
                comp
                * self.component_modes[
                    component.replace("_additive", "").replace("_multiplicative", "")
                ]
            )
            data[component] = comp.mean(axis=1)

        return pd.DataFrame(data)

    def make_all_seasonality_features(self, df):
        seasonal_features = []
        prior_scales = []
        modes = {"additive": [], "multiplicative": []}

        for name, props in self.seasonalities.items():
            features = fourier_series(df["ds"], props["period"], props["fourier_order"])
            columns = [
                "{}_delim_{}".format(name, i + 1) for i in range(features.shape[1])
            ]
            features = pd.DataFrame(features, columns=columns)

            seasonal_features.append(features)
            prior_scales.extend([props["prior_scale"]] * features.shape[1])
            modes[props["mode"]].append(name)

        if len(seasonal_features) == 0:
            seasonal_features.append(pd.DataFrame({"zeros": na.zeros(df.shape[0])}))
            prior_scales.append(1.0)

        seasonal_features = pd.concat(seasonal_features, axis=1)
        component_cols, modes = self.regressor_column_matrix(seasonal_features, modes)
        return seasonal_features, prior_scales, component_cols, modes

    def regressor_column_matrix(self, seasonal_features, modes):
        components = pd.DataFrame(
            {
                "col": na.arange(seasonal_features.shape[1]),
                "component": [x.split("_delim_")[0] for x in seasonal_features.columns],
            }
        )

        for mode in ["additive", "multiplicative"]:
            components = self.add_group_component(
                components, mode + "_terms", modes[mode]
            )
            modes[mode].append(mode + "_terms")

        component_cols = pd.crosstab(
            components["col"],
            components["component"],
        ).sort_index(level="col")

        for name in ["additive_terms", "multiplicative_terms"]:
            if name not in component_cols:
                component_cols[name] = 0

        component_cols.drop("zeros", axis=1, inplace=True, errors="ignore")

        return component_cols, modes

    def add_group_component(self, components, name, group):
        new_comp = components[components["component"].isin(set(group))].copy()
        group_cols = new_comp["col"].unique()
        if len(group_cols) > 0:
            new_comp = pd.DataFrame({"col": group_cols, "component": name})
            components = pd.concat([components, new_comp])
        return components

    def predict_trend(self, df: pd.DataFrame) -> pd.Series:
        # NOTE: this indexing is possible because M=1 guaranteed
        # If this were not the case, we should take the arithmetic mean
        [[k]] = self.k
        [[m]] = self.m
        [delta] = self.delta

        t = df["t"].to_numpy()
        if self.growth == "linear":
            mask = self.changepoints_t[None, :] <= t[..., None]
            deltas_t = delta * mask
            k_t = deltas_t.sum(axis=1) + k
            m_t = (-self.changepoints_t * deltas_t).sum(axis=1) + m
            trend = k_t * t + m_t
        else:
            raise NotImplementedError(self.growth + " is not supported")

        return trend * self.y_scale + df["floor"]

    def predict(self, df: pd.DataFrame) -> na.NadaArray:
        df["trend"] = self.predict_trend(df)
        seasonal_comps = self.predict_seasonal_comps(df)

        result = pd.concat((df[["ds", "trend"]], seasonal_comps), axis=1)

        return (result["trend"] * (result["multiplicative_terms"] + 1)) + result[
            "additive_terms"
        ]

    @override
    def __call__(
        self,
        dates: na.NadaArray,
        floor: na.NadaArray,
        t: na.NadaArray,
        trend: na.NadaArray,
    ) -> na.NadaArray:
        """
        Forward pass.
        Note: requires multiple input arrays due special nature of forecasting.

        Args:
            dates (na.NadaArray): Array of timestamp values.
            floor (na.NadaArray): Array of floor values.
            t (na.NadaArray): Array of t values.
            trend (na.NadaArray): Array of trend values.

        Returns:
            na.NadaArray: Forecasted values.
        """
        return self.forward(dates=dates, floor=floor, t=t, trend=trend)

    @override
    def forward(
        self,
        dates: na.NadaArray,
        floor: na.NadaArray,
        t: na.NadaArray,
        trend: na.NadaArray,
    ) -> na.NadaArray:
        """
        Forward pass.
        Note: requires multiple input arrays due special nature of forecasting.

        Args:
            dates (na.NadaArray): Array of timestamp values.
            floor (na.NadaArray): Array of floor values.
            t (na.NadaArray): Array of t values.
            trend (na.NadaArray): Array of trend values.

        Returns:
            na.NadaArray: Forecasted values.
        """
        return self.predict(
            pd.DataFrame(
                {
                    "ds": dates.inner,
                    "floor": floor.inner,
                    "t": t.inner,
                    "trend": trend.inner,
                }
            )
        )
