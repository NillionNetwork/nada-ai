"""Linear regression example"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import asyncio

import nada_numpy as na
import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
from dotenv import load_dotenv
# Import helper functions for creating nillion client and getting keys
from nillion_python_helpers import (create_nillion_client, getNodeKeyFromFile,
                                    getUserKeyFromFile)
from sklearn.linear_model import LinearRegression

from examples.common.utils import compute, store_program, store_secrets
from nada_ai.client import SklearnClient

# Load environment variables from a .env file
load_dotenv()

NUM_FEATS = 10


# Main asynchronous function to coordinate the process
async def main():
    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    userkey = getUserKeyFromFile(os.getenv("NILLION_USERKEY_PATH_PARTY_1"))
    nodekey = getNodeKeyFromFile(os.getenv("NILLION_NODEKEY_PATH_PARTY_1"))
    client = create_nillion_client(userkey, nodekey)
    party_id = client.party_id
    user_id = client.user_id
    party_names = na_client.parties(2)
    program_name = "linear_regression"
    program_mir_path = f"./target/{program_name}.nada.bin"

    if not os.path.exists("bench"):
        os.mkdir("bench")

    # Store the program
    program_id = await store_program(
        client, user_id, cluster_id, program_name, program_mir_path
    )

    # Train a linear regression
    X = np.random.randn(1_000, NUM_FEATS)
    # We generate the data from a specific linear model
    coeffs_gt = np.ones(
        NUM_FEATS,
    )
    bias_gt = 4.2

    y = X @ coeffs_gt + bias_gt

    model = LinearRegression()
    # The learned params will likely be close to the coefficients & bias we used to generate the data
    fit_model = model.fit(X, y)

    print("Learned model coeffs are:", model.coef_)
    print("Learned model intercept is:", model.intercept_)

    # Create and store model secrets via ModelClient
    model_client = SklearnClient(fit_model)
    model_secrets = nillion.Secrets(
        model_client.export_state_as_secrets("my_model", na.SecretRational)
    )

    model_store_id = await store_secrets(
        client, cluster_id, program_id, party_id, party_names[0], model_secrets
    )

    # Store inputs to perform inference for
    my_input = na_client.array(np.ones((NUM_FEATS,)), "my_input", na.SecretRational)
    input_secrets = nillion.Secrets(my_input)

    data_store_id = await store_secrets(
        client, cluster_id, program_id, party_id, party_names[1], input_secrets
    )

    # Set up the compute bindings for the parties
    compute_bindings = nillion.ProgramBindings(program_id)
    [
        compute_bindings.add_input_party(party_name, party_id)
        for party_name in party_names
    ]
    compute_bindings.add_output_party(party_names[1], party_id)

    print(f"Computing using program {program_id}")
    print(f"Use secret store_id: {model_store_id} {data_store_id}")

    # Perform the computation and return the result
    result = await compute(
        client,
        cluster_id,
        compute_bindings,
        [model_store_id, data_store_id],
        nillion.Secrets({}),
    )
    # Rescale the obtained result by the quantization scale
    outputs = [na_client.float_from_rational(result["my_output"])]
    print(f"🖥️  The result is {outputs}")

    expected = fit_model.predict(np.ones((NUM_FEATS,)).reshape(1, -1))
    print(f"🖥️  VS expected plain-text result {expected}")
    return result


# Run the main function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
