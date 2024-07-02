import nada_numpy as na
from config import NUM_FEATS
from nada_dsl import Party

from nada_ai.linear_model import LogisticRegression


def nada_main():
    # Step 1: We use Nada NumPy wrapper to create "Party" and "Party1"
    # Define parties
    user = Party(name="User")
    provider = Party(name="Provider")

    # Step 2: Instantiate logistic regression object
    my_model = LogisticRegression(NUM_FEATS, 1)

    # Step 3: Load model weights from Nillion network by passing model name (acts as ID)
    # In this examples Party0 provides the model and Party1 runs inference
    my_model.load_state_from_network("my_model", user, na.SecretRational)

    # Step 4: Load input data to be used for inference (provided by Party1)
    my_input = na.array((NUM_FEATS,), provider, "my_input", na.SecretRational)

    # Step 5: Compute inference
    # Note: completely equivalent to `my_model(...)`
    result = my_model.forward(my_input)

    # Step 6: We can use result.output() to produce the output for Party1 and variable name "my_output"
    return result.output(user, "logit")
