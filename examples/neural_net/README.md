# Neural net

This example shows how you can build and run a feed-forward neural network using Nada AI.

The model architecture is defined in `src/my_nn.py`. You will notice that it is syntactically exactly equal to the equivalent PyTorch model.

This model is then used in the main Nada program - defined in `src/neural_net.py`. What this script does is simply:
- Load the model provided by Party0 via `my_model = MyNN()` and `my_model.load_state_from_network("my_nn", parties[0], na.SecretRational)`
- Load in the (3, 4, 3) input data matrix called "my_input" provided by Party1 via `na.array((3, 4, 3), parties[1], "my_input", na.SecretRational)`
- Run inference via `result = my_model(my_input)` 
- Return the inference result to Party1 via `return result.output(parties[1], "my_output")`
