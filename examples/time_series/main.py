"""Time series forecasting example"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

import nada_numpy as na
import nada_numpy.client as na_client
import numpy as np
import pandas as pd
import py_nillion_client as nillion
from common.utils import compute, store_program, store_secrets
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nillion_python_helpers import (create_nillion_client,
                                    create_payments_config)
from prophet import Prophet
from py_nillion_client import NodeKey, UserKey

from nada_ai.client import ProphetClient

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
    program_name = "time_series"
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

    # Train prophet model
    model = Prophet()

    ds = pd.date_range("2024-05-01", "2024-05-17").tolist()
    y = np.arange(1, 18).tolist()

    fit_model = model.fit(df=pd.DataFrame({"ds": ds, "y": y}))

    print("Model params are:", fit_model.params)
    print("Number of detected changepoints:", fit_model.n_changepoints)

    # Create and store model secrets via ModelClient
    model_client = ProphetClient(fit_model)
    model_secrets = nillion.NadaValues(
        model_client.export_state_as_secrets("my_prophet", na.SecretRational)
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
    future_df = fit_model.make_future_dataframe(periods=3)
    inference_ds = fit_model.setup_dataframe(future_df.copy())

    my_input = {}
    my_input.update(
        na_client.array(inference_ds["floor"].to_numpy(), "floor", na.SecretRational)
    )
    my_input.update(
        na_client.array(inference_ds["t"].to_numpy(), "t", na.SecretRational)
    )

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

    expected = fit_model.predict(inference_ds)["yhat"].to_numpy()

    print(f"🖥️  VS expected result {expected}")


if __name__ == "__main__":
    asyncio.run(main())
