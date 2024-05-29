import asyncio
import py_nillion_client as nillion
import os
import sys
import numpy as np
import time
from nada_ai.client import ModelClient
import torch
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
                file.write(f"{elapsed_time:.6f},\n")
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

    # Create custom torch Module
    class MyConvModule(torch.nn.Module):
        """My Convolutional module"""

        def __init__(self) -> None:
            """Contains some ConvNet components"""
            super(MyConvModule, self).__init__()
            self.conv = torch.nn.Conv2d(kernel_size=2, in_channels=3, out_channels=2)
            self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Takes convolution & pools"""
            return self.pool(self.conv(x))

    class PlusOne(torch.nn.Module):
        """My +1 module"""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Simply does +1"""
            return x + 1

    class MyModel(torch.nn.Module):
        """My aribitrarily specific model architecture"""

        def __init__(self) -> None:
            """Model is a collection of arbitrary custom components"""
            super(MyModel, self).__init__()
            self.conv_module = MyConvModule()
            self.plus_one = PlusOne()
            self.linear = torch.nn.Linear(4, 2)
            self.flatten = torch.nn.Flatten(start_dim=0, end_dim=-1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """My custom forward pass logic"""
            x = self.conv_module(x)
            x = self.flatten(x)
            x = self.plus_one(x)
            x = self.linear(x)
            return x

    my_model = MyModel()

    print("Model state is:", my_model.state_dict())

    # Create and store model secrets via ModelClient
    model_client = ModelClient.from_torch(my_model)
    model_secrets = nillion.Secrets(
        model_client.export_state_as_secrets("my_model", as_rational=True, scale=16)
    )

    model_store_id = await store_secrets(
        client, cluster_id, program_id, party_id, party_names[0], model_secrets
    )

    # Store inputs to perform inference for
    my_input = na_client.array(np.ones((3, 4, 3)), "my_input")
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

    # Sort & rescale the obtained results by the quantization scale (here: 16)
    outputs = [result[1] / 2**16 for result in sorted(result.items())]

    print(f"🖥️  The result is {outputs}")

    expected = my_model.forward(torch.ones((3, 4, 3))).detach().numpy().tolist()
    print(f"🖥️  VS expected plain-text result {expected}")
    return result


# Run the main function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
