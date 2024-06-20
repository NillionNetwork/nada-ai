# Time Series

This example shows how you can run time series forecasting using Nada AI. It highlights that, although there exists a major parallel between Nada AI's design and that of PyTorch, it also integrates with other frameworks such as in this case `prophet`

You will find the nada program in `src/time_series.py`

What this script does is simply:
- Load the model provided by Party0 via `my_prophet = Prophet(n_changepoints=12, yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)` and `my_model.load_state_from_network("my_prophet", parties[0], na.SecretRational)`. Note that we explicitly provide some Prophet configurations (i.e. seasonalities and number of changepoints) that it picked up during training.
- Establish the desired forecasting horizon (20 days: 2024-05-01 to 2024-05-21) via `dates = np.arange(np.datetime64("2024-05-01"), np.datetime64("2024-05-21"))`
- Load in input secrets via `floor = na.array((20,), parties[1], "floor", na.SecretRational)` and `t = na.array((20,), parties[1], "t", na.SecretRational)`.
- Run inference via `result = my_prophet(dates, floor, t)` 
- Return the inference result to Party1 via `return result.output(parties[1], "my_output")`
