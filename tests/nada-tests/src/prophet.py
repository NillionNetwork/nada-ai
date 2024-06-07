import pytest
from nada_dsl import *
import nada_algebra as na
from nada_ai.time_series import Prophet


def nada_main():
    party = Party("party")

    dates = na.array([4], party, "dates", SecretInteger)
    floor = na.array([4], party, "floor", SecretInteger)
    t = na.array([4], party, "t", na.SecretRational)
    trend = na.array([4], party, "floor", na.SecretRational)

    prophet = Prophet(
        growth="linear",
        n_changepoints=2,
        num_seasonality_features=1,
    )

    with pytest.raises(NotImplementedError):
        Prophet(growth="to_the_moon")

    result = prophet(dates, floor, t, trend)

    return result.output(party, "forecast")
