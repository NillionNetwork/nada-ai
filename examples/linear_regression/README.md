# Linear regression

This example shows how you can run a linear regression model using Nada AI. It highlights that, although there exists a major parallel between Nada AI's design and that of PyTorch, it also integrates with other frameworks such as in this case `sci-kitlearn`

You will find the nada program in `src/linear_regression.py`

What this script does is simply:
- Load the model provided by Party0 via `my_model = LinearRegression(in_features=10)` and `my_model.load_state_from_network("my_model", parties[0], na.SecretRational)`
- Load in the 10 input features as a 1-d array called "my_input" provided by Party1 via `my_input = na.array((10,), parties[1], "my_input", na.SecretRational)`
- Run inference via `result = my_model.forward(my_input)` 
- Return the inference result to Party1 via `return [Output(result.value, "my_output", parties[1])]`
