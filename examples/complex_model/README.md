# Complex model

This example shows how you can build and run an arbitrarily complex AI model - much like you can in PyTorch!

The model architecture is defined in `src/my_model.py`. You will notice that it is syntactically nearly identical to the equivalent PyTorch model.

This model is then used in the main Nada program - defined in `src/complex_model.py`. What this script does is simply:
- Load the model provided by Party0 via `my_model = MyModel()` and `my_model.load_state_from_network("my_model", parties[0], na.SecretRational)`
- Load in the (3, 4, 3) input data matrix called "my_input" provided by Party1 via `na.array((3, 4, 3), parties[1], "my_input", na.SecretRational)`
- Run inference via `result = my_model(my_input)` 
- Return the inference result to Party1 via `return result.output(parties[1], "my_output")`
