from nada_dsl import *
import nada_algebra as na
from nada_ai.nn import LinearRegression


def nada_main():
    # Step 1: We use Nada Algebra wrapper to create "Party0" and "Party1"
    parties = na.parties(2)

    # Step 2: Instantiate linear regression object
    my_model = LinearRegression(10)

    # Step 3: Load model weights from Nillion network by passing model name (acts as ID)
    # In this examples Party0 provides the model and Party1 runs inference
    my_model.load_state_from_network(
        "my_model",
        parties[0],
        as_rational=True,  # Set to True b/c model weights are native floats
        scale=16,
    )

    # Step 4: Load input data to be used for inference (provided by Party1)
    my_input = na.array((10,), parties[1], "my_input")

    # Step 5: Compute inference
    # Note: completely equivalent to `my_model(...)`
    result = my_model.forward(my_input)

    # Step 6: We can use result.output() to produce the output for Party1 and variable name "my_output"
    return result.output(parties[1], "my_output")
