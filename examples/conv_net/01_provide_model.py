"""Provide model weights to program"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import argparse
import asyncio
import json

import nada_numpy as na
import py_nillion_client as nillion
import torch
from common.utils import store_program, store_secrets
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nillion_python_helpers import (create_nillion_client,
                                    create_payments_config)
from py_nillion_client import NodeKey, UserKey
from torch import nn

from nada_ai.client import TorchClient

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


async def main(model_path: str, out_path: str) -> None:
    """Main nada program"""

    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    grpc_endpoint = os.getenv("NILLION_NILCHAIN_GRPC")
    chain_id = os.getenv("NILLION_NILCHAIN_CHAIN_ID")
    seed = "my_seed"
    model_provider_userkey = UserKey.from_seed((seed))
    model_provider_nodekey = NodeKey.from_seed((seed))
    model_provider_client = create_nillion_client(
        model_provider_userkey, model_provider_nodekey
    )
    model_provider_party_id = model_provider_client.party_id
    model_provider_user_id = model_provider_client.user_id

    program_name = "conv_net"
    program_mir_path = f"target/{program_name}.nada.bin"

    payments_config = create_payments_config(chain_id, grpc_endpoint)
    payments_client = LedgerClient(payments_config)
    payments_wallet = LocalWallet(
        PrivateKey(bytes.fromhex(os.getenv("NILLION_NILCHAIN_PRIVATE_KEY_0"))),
        prefix="nillion",
    )

    print("Storing program...")

    program_id = await store_program(
        model_provider_client,
        payments_wallet,
        payments_client,
        model_provider_user_id,
        cluster_id,
        program_name,
        program_mir_path,
    )

    print("Program stored successfully!")

    class MyNN(nn.Module):
        """My brand new model"""

        def __init__(self) -> None:
            """Model is a two layers and an activations"""
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels=1, out_channels=2, kernel_size=3, padding=1, stride=3
            )
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(in_features=18, out_features=2)

            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()

        def forward(self, x: na.NadaArray) -> na.NadaArray:
            """My forward pass logic"""
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.flatten(x)
            x = self.fc1(x)
            return x

    classifier = MyNN()
    classifier.load_state_dict(torch.load(model_path))

    model_client = TorchClient(classifier)

    model_secrets = nillion.NadaValues(
        model_client.export_state_as_secrets("my_nn", na.SecretRational)
    )
    permissions = nillion.Permissions.default_for_user(model_provider_client.user_id)
    permissions.add_compute_permissions({model_provider_client.user_id: {program_id}})

    print("Storing model...")

    model_store_id = await store_secrets(
        model_provider_client,
        payments_wallet,
        payments_client,
        cluster_id,
        model_secrets,
        1,
        permissions,
    )

    print("Model stored successfully!")

    # This information is needed by the model user
    with open(out_path, "w") as provider_variables_file:
        provider_variables = {
            "program_id": program_id,
            "model_store_id": model_store_id,
            "model_provider_party_id": model_provider_party_id,
        }
        json.dump(provider_variables, provider_variables_file)


if __name__ == "__main__":
    asyncio.run(main(ARGS.model_path, ARGS.out_path))
