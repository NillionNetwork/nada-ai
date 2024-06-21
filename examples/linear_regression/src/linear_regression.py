import nada_numpy as na

from nada_ai.linear_model import LinearRegression


def nada_main():
    # Step 1: We use Nada NumPy wrapper to create "Party0" and "Party1"
    parties = na.parties(2)

    # Step 2: Instantiate linear regression object
    my_model = LinearRegression(in_features=10)

    # Step 3: Load model weights from Nillion network by passing model name (acts as ID)
    # In this examples Party0 provides the model and Party1 runs inference
    my_model.load_state_from_network("my_model", parties[0], na.SecretRational)

    # Step 4: Load input data to be used for inference (provided by Party1)
    my_input = na.array((10,), parties[1], "my_input", na.SecretRational)

    # Step 5: Compute inference
    # Note: completely equivalent to `my_model(...)`
    result = my_model.forward(my_input)

    # Step 6: We can use result.output() to produce the output for Party1 and variable name "my_output"
    return na.output(result, parties[1], "my_output")
