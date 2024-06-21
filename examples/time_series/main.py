"""Time series forecasting example"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import asyncio

import nada_numpy as na
import nada_numpy.client as na_client
import numpy as np
import pandas as pd
import py_nillion_client as nillion
from dotenv import load_dotenv
# Import helper functions for creating nillion client and getting keys
from nillion_python_helpers import (create_nillion_client, getNodeKeyFromFile,
                                    getUserKeyFromFile)
from prophet import Prophet

from examples.common.utils import compute, store_program, store_secrets
from nada_ai.client import ProphetClient

# Load environment variables from a .env file
load_dotenv()


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

    na.set_log_scale(50)

    # Store the program
    program_id = await store_program(
        client, user_id, cluster_id, program_name, program_mir_path
    )

    # Train prophet model
    model = Prophet()

    ds = pd.date_range("2024-05-01", "2024-05-17").tolist()
    y = np.arange(1, 18).tolist()

    fit_model = model.fit(df=pd.DataFrame({"ds": ds, "y": y}))

    print("Model params are:", fit_model.params)
    print("Number of detected changepoints:", fit_model.n_changepoints)

    # Create and store model secrets via ModelClient
    model_client = ProphetClient(fit_model)
    model_secrets = nillion.Secrets(
        model_client.export_state_as_secrets("my_prophet", na.SecretRational)
    )

    model_store_id = await store_secrets(
        client, cluster_id, program_id, party_id, party_names[0], model_secrets
    )

    # Store inputs to perform inference for
    future_df = fit_model.make_future_dataframe(periods=3)
    inference_ds = fit_model.setup_dataframe(future_df.copy())

    my_input = {}
    my_input.update(
        na_client.array(inference_ds["floor"].to_numpy(), "floor", na.SecretRational)
    )
    my_input.update(
        na_client.array(inference_ds["t"].to_numpy(), "t", na.SecretRational)
    )

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

    # Sort & rescale the obtained results by the quantization scale
    outputs = [
        na_client.float_from_rational(result[1])
        for result in sorted(
            result.items(),
            key=lambda x: int(x[0].replace("my_output", "").replace("_", "")),
        )
    ]

    print(f"🖥️  The result is {outputs}")

    expected = fit_model.predict(inference_ds)["yhat"].to_numpy()
    print(f"🖥️  VS expected plain-text result {expected}")
    return result


# Run the main function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
