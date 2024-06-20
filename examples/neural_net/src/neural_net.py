import nada_numpy as na
from my_nn import MyNN


def nada_main():
    # Step 1: We use Nada NumPy wrapper to create "Party0" and "Party1"
    parties = na.parties(2)

    # Step 2: Instantiate model object
    my_model = MyNN()

    # Step 3: Load model weights from Nillion network by passing model name (acts as ID)
    # In this examples Party0 provides the model and Party1 runs inference
    my_model.load_state_from_network("my_nn", parties[0], na.SecretRational)

    # Step 4: Load input data to be used for inference (provided by Party1)
    my_input = na.array((8,), parties[1], "my_input", na.SecretRational)

    # Step 5: Compute inference
    # Note: completely equivalent to `my_model.forward(...)`
    result = my_model(my_input)

    # Step 6: We can use result.output() to produce the output for Party1 and variable name "my_output"
    return result.output(parties[1], "my_output")
