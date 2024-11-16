"""Run model inference"""

import asyncio
import os
import pytest
import uuid

import nada_numpy as na
import nada_numpy.client as na_client
import numpy as np
import json
import torch
from torch import nn
from nada_ai.client import TorchClient

from nillion_client.ids import UserId
from nillion_client import (
 
    InputPartyBinding,
    Network,
    NilChainPayer,
    NilChainPrivateKey,
    OutputPartyBinding,
    Permissions,
    SecretInteger,
    VmClient,
    PrivateKey,
)
from dotenv import load_dotenv
import argparse
home = os.getenv("HOME")
load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")

PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    "--features-path",
    dest="features_path",
    type=str,
    required=True,
)
PARSER.add_argument(
    "--in-path",
    dest="in_path",
    type=str,
    required=True,
)
ARGS = PARSER.parse_args()

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
async def main(features_path: str, in_path: str) -> None:
    """Main nada program"""
    network = Network.from_config("devnet")

    # WARNING: In a real use case, the Provider and User would never have access to the Private Key
    # This is just for demonstration purposes
    # Provider and User should only exchange their IDs
    model_provider_name = "Provider"
    model_provider = await new_client(network, 0, b'\xbf\xdf7\xa9\x1eL\x10i"\xd8\x1f\xbb\xe8\r;\x1b`\x1a\xd1\xa1;\xef\xd8\xbbf|\xf9\x12\xe9\xef\x03\xc7')
    model_user_name = "User"
    model_user = await new_client(network, 1, b'\x15\xa0\xc1\xcc\x12\xb5r\xf9\xcb\x89\x95\x8d\x94\xfb\xfe)\xdf\xfe\xbd3\x00\x18\x80\xc1\xd9W\x8b\xf7\xc0\x92S\xe9')

    # This information was provided by the model provider
    with open(in_path, "r") as provider_variables_file:
        provider_variables = json.load(provider_variables_file)

    program_id = provider_variables["program_id"]
    model_store_id = provider_variables["model_store_id"]
    model_store_id = uuid.UUID(hex=model_store_id)
    model_provider_user_id = UserId.parse(provider_variables["model_provider_user_id"])
    

    features = np.load(features_path)

    # Print details about stored program
    print(f"Stored program_id: {program_id}")

    ##### STORE SECRETS
    print("-----STORE SECRETS Party 0")

    # Create a secret
    features= na_client.array(features, "my_input", na_client.SecretRational)

    # Create a permissions object to attach to the stored secret
    permissions = Permissions.defaults_for_user(model_user.user_id).allow_compute(
        model_user.user_id, program_id
    )

    # Store a secret, passing in the receipt that shows proof of payment
    features_store_id = await model_user.store_values(
        features, ttl_days=5, permissions=permissions
    ).invoke()

    print("Stored features: ", features_store_id)

    ##### COMPUTE
    print("-----COMPUTE")

    # Bind the parties in the computation to the client to set input and output parties
    input_bindings = [
        InputPartyBinding(model_provider_name, model_provider_user_id),
        InputPartyBinding(model_user_name, model_user.user_id),
    ]
    output_bindings = [OutputPartyBinding(model_user_name, [model_user.user_id])]

    # Create a computation time secret to use
    compute_time_values = {
        # "my_int2": SecretInteger(10)
    }

    # Compute, passing in the compute time values as well as the previously uploaded value.
    print(
        f"Invoking computation using program {program_id} and values id {model_store_id}, {features_store_id}"
    )
    compute_id = await model_user.compute(
        program_id,
        input_bindings,
        output_bindings,
        values=compute_time_values,
        value_ids=[model_store_id, features_store_id],
    ).invoke()

    # Print compute result
    print(f"The computation was sent to the network. compute_id: {compute_id}")
    result = await model_user.retrieve_compute_results(compute_id).invoke()
    print(f"✅  Compute complete for compute_id {compute_id}")
    print(f"🖥️  The result is {result}")
    return result


if __name__ == "__main__":
    asyncio.run(main(ARGS.features_path, ARGS.in_path))
