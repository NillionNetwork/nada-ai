import asyncio
import py_nillion_client as nillion
import os
import sys
import numpy as np
import time
import nada_algebra as na
from nada_ai import SklearnClient
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv

# Add the parent directory to the system path to import modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import helper functions for creating nillion client and getting keys
from neural_net.network.helpers.nillion_client_helper import create_nillion_client
from neural_net.network.helpers.nillion_keypath_helper import (
    getUserKeyFromFile,
    getNodeKeyFromFile,
)
import nada_algebra.client as na_client

# Load environment variables from a .env file
load_dotenv()

NUM_FEATS = 10


# Decorator function to measure and log the execution time of asynchronous functions
def async_timer(file_path):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Log the execution time to a file
            with open(file_path, "a") as file:
                file.write(f"{NUM_FEATS} feats: {elapsed_time:.6f},\n")
            return result

        return wrapper

    return decorator


# Asynchronous function to store a program on the nillion client
@async_timer("bench/store_program.txt")
async def store_program(client, user_id, cluster_id, program_name, program_mir_path):
    action_id = await client.store_program(cluster_id, program_name, program_mir_path)
    program_id = f"{user_id}/{program_name}"
    print("Stored program. action_id:", action_id)
    print("Stored program_id:", program_id)
    return program_id


# Asynchronous function to store secrets on the nillion client
@async_timer("bench/store_secrets.txt")
async def store_secrets(client, cluster_id, program_id, party_id, party_name, secrets):
    secret_bindings = nillion.ProgramBindings(program_id)
    secret_bindings.add_input_party(party_name, party_id)

    # Store the secret for the specified party
    store_id = await client.store_secrets(cluster_id, secret_bindings, secrets, None)
    return store_id


# Asynchronous function to perform computation on the nillion client
@async_timer("bench/compute.txt")
async def compute(
    client, cluster_id, compute_bindings, store_ids, computation_time_secrets
):
    compute_id = await client.compute(
        cluster_id,
        compute_bindings,
        store_ids,
        computation_time_secrets,
        nillion.PublicVariables({}),
    )

    # Monitor and print the computation result
    print(f"The computation was sent to the network. compute_id: {compute_id}")
    while True:
        compute_event = await client.next_compute_event()
        if isinstance(compute_event, nillion.ComputeFinishedEvent):
            print(f"✅  Compute complete for compute_id {compute_event.uuid}")
            return compute_event.result.value


# Main asynchronous function to coordinate the process
async def main():
    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    userkey = getUserKeyFromFile(os.getenv("NILLION_USERKEY_PATH_PARTY_1"))
    nodekey = getNodeKeyFromFile(os.getenv("NILLION_NODEKEY_PATH_PARTY_1"))
    client = create_nillion_client(userkey, nodekey)
    party_id = client.party_id
    user_id = client.user_id
    party_names = na_client.parties(2)
    program_name = "main"
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
    # Rescale the obtained result by the quantization scale (here: 16)
    outputs = [result["my_output_0"] / 2**16]
    print(f"🖥️  The result is {outputs}")

    expected = fit_model.predict(np.ones((NUM_FEATS,)).reshape(1, -1))
    print(f"🖥️  VS expected plain-text result {expected}")
    return result


# Run the main function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
