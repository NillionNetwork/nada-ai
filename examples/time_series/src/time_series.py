from typing import List

import nada_numpy as na
import numpy as np
from config import TIME_HORIZON
from nada_ai.time_series import Prophet
from nada_dsl import Output


def nada_main() -> List[Output]:
    """
    Main Nada program.

    Returns:
        List[Output]: Program outputs.
    """
    # Step 1: We use Nada NumPy wrapper to create "Party0" and "Party1"
    parties = na.parties(2)

    # Step 2: Instantiate model object
    my_prophet = Prophet(
        n_changepoints=12,  # NOTE: this is a learned hyperparameter
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
    )

    # Step 3: Load model weights from Nillion network by passing model name (acts as ID)
    # In this examples Party0 provides the model and Party1 runs inference
    my_prophet.load_state_from_network("my_prophet", parties[0], na.SecretRational)

    # Step 4: Load input data to be used for inference (provided by Party1)
    start_date = np.datetime64("2024-05-01")
    end_date = start_date + TIME_HORIZON
    dates = np.arange(start_date, end_date)

    floor = na.array((TIME_HORIZON,), parties[1], "floor", na.SecretRational)
    t = na.array((TIME_HORIZON,), parties[1], "t", na.SecretRational)

    # Step 5: Compute inference
    # Note: completely equivalent to `my_model.forward(...)` or `model.predict(...)`
    result = my_prophet(dates, floor, t)

    # Step 6: We can use result.output() to produce the output for Party1 and variable name "my_output"
    return result.output(parties[1], "my_output")
