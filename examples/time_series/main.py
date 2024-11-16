"""Dot Product example script"""

import asyncio
import os

import nada_numpy as na
import nada_numpy.client as na_client
import numpy as np
import pandas as pd
from config import DIM, FORECAST_HORIZON
from dotenv import load_dotenv
from nillion_client import (InputPartyBinding, Network, NilChainPayer,
                            NilChainPrivateKey, OutputPartyBinding,
                            Permissions, PrivateKey, SecretInteger, VmClient)
from prophet import Prophet

from nada_ai.client import ProphetClient

home = os.getenv("HOME")
load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")


async def new_client(network, id: int, private_key: str = None):
    # Create payments config and set up Nillion wallet with a private key to pay for operations
    nilchain_key: str = os.getenv(f"NILLION_NILCHAIN_PRIVATE_KEY_{id}")  # type: ignore
    payer = NilChainPayer(
        network,
        wallet_private_key=NilChainPrivateKey(bytes.fromhex(nilchain_key)),
        gas_limit=10000000,
    )

    # Use a random key to identify ourselves
    signing_key = PrivateKey(private_key)
    print(signing_key.private_key)
    client = await VmClient.create(signing_key, network, payer)
    return client


# 1 Party running simple addition on 1 stored secret and 1 compute time secret
async def main() -> None:
    """Main nada program"""
    network = Network.from_config("devnet")

    # WARNING: In a real use case, the Provider and User would never have access to the Private Key
    # This is just for demonstration purposes
    # Provider and User should only exchange their IDs
    model_provider_name = "Party0"
    model_provider = await new_client(
        network,
        0,
        b'\xbf\xdf7\xa9\x1eL\x10i"\xd8\x1f\xbb\xe8\r;\x1b`\x1a\xd1\xa1;\xef\xd8\xbbf|\xf9\x12\xe9\xef\x03\xc7',
    )
    model_user_name = "Party1"
    model_user = await new_client(
        network,
        1,
        b"\x15\xa0\xc1\xcc\x12\xb5r\xf9\xcb\x89\x95\x8d\x94\xfb\xfe)\xdf\xfe\xbd3\x00\x18\x80\xc1\xd9W\x8b\xf7\xc0\x92S\xe9",
    )

    program_name = "time_series"
    program_mir_path = f"./target/{program_name}.nada.bin"

    ##### STORE PROGRAM
    print("-----STORE PROGRAM")

    # Store program
    program_mir = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), program_mir_path), "rb"
    ).read()
    program_id = await model_provider.store_program(program_name, program_mir).invoke()

    # Print details about stored program
    print(f"Stored program_id: {program_id}")

    ##### STORE SECRETS
    print("-----STORE SECRETS Party 0")

    # Train prophet model
    model = Prophet()
    start_date = np.datetime64("2024-05-01")
    end_date = start_date + DIM - 1
    ds = pd.date_range(start_date, end_date).tolist()

    y = np.random.randint(-100, 100, (DIM,)).tolist()

    fit_model = model.fit(df=pd.DataFrame({"ds": ds, "y": y}))
    print("Model params are:", fit_model.params)
    print("Number of detected changepoints:", fit_model.n_changepoints)

    # Create and store model secrets via ModelClient
    model_client = ProphetClient(fit_model)
    model_secrets = model_client.export_state_as_secrets(
        "my_prophet", na_client.SecretRational
    )

    # Create a permissions object to attach to the stored secret
    permissions = Permissions.defaults_for_user(model_provider.user_id).allow_compute(
        model_user.user_id, program_id
    )

    # Store a secret, passing in the receipt that shows proof of payment
    my_nn_store_id = await model_provider.store_values(
        model_secrets, ttl_days=5, permissions=permissions
    ).invoke()

    print("-----STORE SECRETS Party 1")

    # Create a secret
    future_df = fit_model.make_future_dataframe(periods=FORECAST_HORIZON)
    inference_ds = fit_model.setup_dataframe(future_df.copy())
    my_input = {}
    my_input.update(
        na_client.array(inference_ds["floor"].to_numpy(), "floor", na.SecretRational)
    )
    my_input.update(
        na_client.array(inference_ds["t"].to_numpy(), "t", na.SecretRational)
    )  # Create a permissions object to attach to the stored secret
    permissions = Permissions.defaults_for_user(model_user.user_id).allow_compute(
        model_user.user_id, program_id
    )

    # Store a secret, passing in the receipt that shows proof of payment
    my_inputs_store_id = await model_user.store_values(
        my_input, ttl_days=5, permissions=permissions
    ).invoke()

    ##### COMPUTE
    print("-----COMPUTE")

    # Bind the parties in the computation to the client to set input and output parties
    input_bindings = [
        InputPartyBinding(model_provider_name, model_provider.user_id),
        InputPartyBinding(model_user_name, model_user.user_id),
    ]
    output_bindings = [OutputPartyBinding(model_user_name, [model_user.user_id])]

    # Create a computation time secret to use
    compute_time_values = {
        # "my_int2": SecretInteger(10)
    }

    # Compute, passing in the compute time values as well as the previously uploaded value.
    print(
        f"Invoking computation using program {program_id} and values id {my_nn_store_id}, {my_inputs_store_id}"
    )
    compute_id = await model_user.compute(
        program_id,
        input_bindings,
        output_bindings,
        values=compute_time_values,
        value_ids=[my_nn_store_id, my_inputs_store_id],
    ).invoke()

    # Print compute result
    print(f"The computation was sent to the network. compute_id: {compute_id}")
    result = await model_user.retrieve_compute_results(compute_id).invoke()
    print(f"✅  Compute complete for compute_id {compute_id}")
    print(f"🖥️  The result is {result}")
    return result


if __name__ == "__main__":
    asyncio.run(main())
