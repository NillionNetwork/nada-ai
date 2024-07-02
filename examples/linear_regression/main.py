"""Linear regression example"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

import nada_numpy as na
import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
from sklearn.linear_model import LinearRegression
from common.utils import compute, store_program, store_secrets
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nada_ai.client import SklearnClient
from nillion_python_helpers import (create_nillion_client,
                                    create_payments_config)
from py_nillion_client import NodeKey, UserKey

home = os.getenv("HOME")
load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")

NUM_FEATS = 10


# Main asynchronous function to coordinate the process
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
    program_name = "linear_regression"
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
    model_secrets = nillion.NadaValues(
        model_client.export_state_as_secrets("my_model", na.SecretRational)
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
    my_input = na_client.array(np.ones((NUM_FEATS,)), "my_input", na.SecretRational)
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

    # Rescale the obtained result by the quantization scale
    outputs = [na_client.float_from_rational(result["my_output"])]
    print(f"🖥️  The result is {outputs} @ {na.get_log_scale()}-bit precision")

    expected = fit_model.predict(np.ones((NUM_FEATS,)).reshape(1, -1))
    print(f"🖥️  VS expected result {expected}")


if __name__ == "__main__":
    asyncio.run(main())
