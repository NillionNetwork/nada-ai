"""Neural net example"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

import nada_numpy as na
import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
import torch
from common.utils import compute, store_program, store_secrets
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nillion_python_helpers import (create_nillion_client,
                                    create_payments_config)
from py_nillion_client import NodeKey, UserKey

from nada_ai.client import TorchClient

home = os.getenv("HOME")
load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")


async def main() -> None:
    """Main nada program"""

    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    grpc_endpoint = os.getenv("NILLION_NILCHAIN_GRPC")
    chain_id = os.getenv("NILLION_NILCHAIN_CHAIN_ID")
    seed = "my_seed"
    userkey = UserKey.from_seed((seed))
    nodekey = NodeKey.from_seed((seed))
    client = create_nillion_client(userkey, nodekey)
    party_id = client.party_id
    user_id = client.user_id

    party_names = na_client.parties(2)
    program_name = "neural_net"
    program_mir_path = f"target/{program_name}.nada.bin"

    # Configure payments
    payments_config = create_payments_config(chain_id, grpc_endpoint)
    payments_client = LedgerClient(payments_config)
    payments_wallet = LocalWallet(
        PrivateKey(bytes.fromhex(os.getenv("NILLION_NILCHAIN_PRIVATE_KEY_0"))),
        prefix="nillion",
    )

    # Store program
    program_id = await store_program(
        client,
        payments_wallet,
        payments_client,
        user_id,
        cluster_id,
        program_name,
        program_mir_path,
    )

    # Create custom torch Module
    class MyNN(torch.nn.Module):
        """My simple neural net"""

        def __init__(self) -> None:
            """Model is a two layers and an activations"""
            super().__init__()
            self.linear_0 = torch.nn.Linear(8, 4)
            self.linear_1 = torch.nn.Linear(4, 2)
            self.relu = torch.nn.ReLU()

        def forward(self, x: torch.tensor) -> torch.tensor:
            """My forward pass logic"""
            x = self.linear_0(x)
            x = self.relu(x)
            x = self.linear_1(x)
            return x

    my_nn = MyNN()

    print("Model state is:", my_nn.state_dict())

    # Create and store model secrets via ModelClient
    model_client = TorchClient(my_nn)
    model_secrets = nillion.NadaValues(
        model_client.export_state_as_secrets("my_nn", na.SecretRational)
    )
    permissions = nillion.Permissions.default_for_user(client.user_id)
    permissions.add_compute_permissions({client.user_id: {program_id}})

    model_store_id = await store_secrets(
        client,
        payments_wallet,
        payments_client,
        cluster_id,
        model_secrets,
        1,
        permissions,
    )

    # Store inputs to perform inference for
    my_input = na_client.array(np.ones((8,)), "my_input", na.SecretRational)
    input_secrets = nillion.NadaValues(my_input)

    data_store_id = await store_secrets(
        client,
        payments_wallet,
        payments_client,
        cluster_id,
        input_secrets,
        1,
        permissions,
    )

    # Set up the compute bindings for the parties
    compute_bindings = nillion.ProgramBindings(program_id)

    for party_name in party_names:
        compute_bindings.add_input_party(party_name, party_id)
    compute_bindings.add_output_party(party_names[-1], party_id)

    print(f"Computing using program {program_id}")
    print(f"Use secret store_id: {model_store_id} {data_store_id}")

    # Create a computation time secret to use
    computation_time_secrets = nillion.NadaValues({})

    # Compute, passing all params including the receipt that shows proof of payment
    result = await compute(
        client,
        payments_wallet,
        payments_client,
        program_id,
        cluster_id,
        compute_bindings,
        [model_store_id, data_store_id],
        computation_time_secrets,
        verbose=True,
    )

    # Sort & rescale the obtained results by the quantization scale
    outputs = [
        na_client.float_from_rational(result[1])
        for result in sorted(
            result.items(),
            key=lambda x: int(x[0].replace("my_output", "").replace("_", "")),
        )
    ]

    print(f"🖥️  The processed result is {outputs} @ {na.get_log_scale()}-bit precision")

    expected = my_nn.forward(torch.ones((8,))).detach().numpy().tolist()

    print(f"🖥️  VS expected result {expected}")


if __name__ == "__main__":
    asyncio.run(main())
