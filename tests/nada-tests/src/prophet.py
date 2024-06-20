import nada_numpy as na
import numpy as np
from nada_dsl import *

from nada_ai.time_series import Prophet


def nada_main():
    party = Party("party")

    dates = np.linspace(1, 10, 4)
    floor = na.array([4], party, "floor", na.SecretRational)
    t = na.array([4], party, "t", na.SecretRational)

    prophet = Prophet(
        n_changepoints=2,
        growth="linear",
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        nada_type=na.SecretRational,
    )

    prophet.load_state_from_network("my_prophet", party, na.SecretRational)

    result = prophet(dates, floor, t)

    return result.output(party, "forecast")
