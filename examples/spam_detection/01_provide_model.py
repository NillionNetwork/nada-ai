"""Provide model weights to program"""

import asyncio
import os
import joblib
import nada_numpy as na
import nada_numpy.client as na_client
import numpy as np
import json
import torch
from torch import nn
from nada_ai.client import SklearnClient

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

PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    "--model-path",
    dest="model_path",
    type=str,
    required=True,
)
PARSER.add_argument(
    "--out-path",
    dest="out_path",
    type=str,
    required=True,
)
ARGS = PARSER.parse_args()

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

async def main(model_path: str, out_path: str) -> None:    # Use the devnet configuration generated by `nillion-devnet`
    network = Network.from_config("devnet")

    # WARNING: In a real use case, the Provider and User would never have access to the Private Key
    # This is just for demonstration purposes
    # Provider and User should only exchange their IDs
    model_provider_name = "Provider"
    model_provider = await new_client(network, 0, b'\xbf\xdf7\xa9\x1eL\x10i"\xd8\x1f\xbb\xe8\r;\x1b`\x1a\xd1\xa1;\xef\xd8\xbbf|\xf9\x12\xe9\xef\x03\xc7')
    model_user_name = "User"
    model_user = await new_client(network, 1, b'\x15\xa0\xc1\xcc\x12\xb5r\xf9\xcb\x89\x95\x8d\x94\xfb\xfe)\xdf\xfe\xbd3\x00\x18\x80\xc1\xd9W\x8b\xf7\xc0\x92S\xe9')
    program_name = "spam_detection"
    program_mir_path = f"./target/{program_name}.nada.bin"

    ##### STORE PROGRAM
    print("-----STORE PROGRAM")

    # Store program
    program_mir = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), program_mir_path), "rb").read()
    program_id = await model_provider.store_program(program_name, program_mir).invoke()

    # Print details about stored program
    print(f"Stored program_id: {program_id}")

    ##### STORE SECRETS
    print("-----STORE SECRETS (MODEL)")

    # Create a secret
    classifier = joblib.load(model_path)

    model_client = SklearnClient(classifier)

    model_secrets = model_client.export_state_as_secrets("my_model", na_client.SecretRational)
    print(model_secrets)
    # Create a permissions object to attach to the stored secret
    permissions = Permissions.defaults_for_user(model_provider.user_id).allow_compute(
        model_user.user_id, program_id
    )

    # Store a secret, passing in the receipt that shows proof of payment
    model_store_id = await model_provider.store_values(
        model_secrets, ttl_days=1, permissions=permissions
    ).invoke()

    print(f"Stored program with id: {program_id} {type(program_id)}")
    print(f"Stored model with id: {model_store_id} {type(model_store_id)}")
    print(f"Stored model_provider_user_id with id: {model_provider.user_id} {type(model_provider.user_id)}")

    # This information is needed by the model user
    with open(out_path, "w") as provider_variables_file:
        provider_variables = {
            "program_id": str(program_id),
            "model_store_id": model_store_id.hex,
            "model_provider_user_id": str(model_provider.user_id),
        }
        json.dump(provider_variables, provider_variables_file)


if __name__ == "__main__":
    asyncio.run(main(ARGS.model_path, ARGS.out_path))

