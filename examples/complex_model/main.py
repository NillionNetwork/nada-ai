import asyncio
import os

import nada_numpy as na
import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
import torch
from dotenv import load_dotenv
# Import helper functions for creating nillion client and getting keys
from nillion_python_helpers import (create_nillion_client, getNodeKeyFromFile,
                                    getUserKeyFromFile)

from examples.common.utils import compute, store_program, store_secrets
from nada_ai.client import TorchClient

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
    program_name = "complex_model"
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

    class MyOperations(torch.nn.Module):
        """My operations module"""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Does some arbitrary operations for illustrative purposes"""
            return (x * 2) - 1

    class MyModel(torch.nn.Module):
        """My aribitrarily specific model architecture"""

        def __init__(self) -> None:
            """Model is a collection of arbitrary custom components"""
            super(MyModel, self).__init__()
            self.conv_module = MyConvModule()
            self.my_operations = MyOperations()
            self.linear = torch.nn.Linear(4, 2)
            self.flatten = torch.nn.Flatten(0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """My custom forward pass logic"""
            x = self.conv_module(x)
            x = self.flatten(x)
            x = self.my_operations(x)
            x = self.linear(x)
            return x

    my_model = MyModel()

    print("Model state is:", my_model.state_dict())

    # Create and store model secrets via ModelClient
    model_client = TorchClient(my_model)
    model_secrets = nillion.Secrets(model_client.export_state_as_secrets("my_model", na.SecretRational))

    model_store_id = await store_secrets(
        client, cluster_id, program_id, party_id, party_names[0], model_secrets
    )

    # Store inputs to perform inference for
    my_input = na_client.array(np.ones((3, 4, 3)), "my_input", na.SecretRational)
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
    outputs = outputs = [
        na_client.float_from_rational(result[1])
        for result in sorted(
            result.items(),
            key=lambda x: int(x[0].replace("my_output", "").replace("_", "")),
        )
    ]

    print(f"🖥️  The result is {outputs}")

    expected = my_model.forward(torch.ones((3, 4, 3))).detach().numpy().tolist()
    print(f"🖥️  VS expected plain-text result {expected}")
    return result


# Run the main function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
