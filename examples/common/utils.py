"""General utils functions"""

import os
import time
from typing import Any, Callable, Dict, List

import nada_numpy as na
import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from nillion_python_helpers import get_quote, get_quote_and_pay, pay_with_quote


def async_timer(file_path: os.PathLike) -> Callable:
    """
    Decorator function to measure and log the execution time of asynchronous functions.

    Args:
        file_path (os.PathLike): File to write performance metrics to.

    Returns:
        Callable: Wrapped function with timer.
    """

    def decorator(func: Callable) -> Callable:
        """
        Decorator function.

        Args:
            func (Callable): Function to decorate.

        Returns:
            Callable: Decorated function.
        """

        async def wrapper(*args, **kwargs) -> Any:
            """
            Returns function result and writes execution time to file.

            Returns:
                Any: Function result.
            """
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            with open(file_path, "a") as file:
                file.write(f"{elapsed_time:.6f},\n")
            return result

        return wrapper

    return decorator


async def store_program(
    client: nillion.NillionClient,
    payments_wallet: LocalWallet,
    payments_client: LedgerClient,
    user_id: str,
    cluster_id: str,
    program_name: str,
    program_mir_path: str,
    verbose: bool = True,
) -> str:
    """
    Asynchronous function to store a program on the nillion client.

    Args:
        client (nillion.NillionClient): Nillion client.
        user_id (str): User ID.
        cluster_id (str): Cluster ID.
        program_name (str): Program name.
        program_mir_path (str): Path to program MIR.
        verbose (bool, optional): Verbosity level. Defaults to True.

    Returns:
        str: Program ID.
    """
    quote_store_program = await get_quote(
        client, nillion.Operation.store_program(program_mir_path), cluster_id
    )

    receipt_store_program = await pay_with_quote(
        quote_store_program, payments_wallet, payments_client
    )

    action_id = await client.store_program(
        cluster_id, program_name, program_mir_path, receipt_store_program
    )

    program_id = f"{user_id}/{program_name}"
    if verbose:
        print("Stored program. action_id:", action_id)
        print("Stored program_id:", program_id)

    return program_id


async def store_secret_array(
    client: nillion.NillionClient,
    payments_wallet: LocalWallet,
    payments_client: LedgerClient,
    cluster_id: str,
    secret_array: np.ndarray,
    secret_name: str,
    nada_type: Any,
    ttl_days: int = 1,
    permissions: nillion.Permissions = None,
):
    """
    Asynchronous function to store secret arrays on the nillion client.

    Args:
        client (nillion.NillionClient): Nillion client.
        cluster_id (str): Cluster ID.
        party_id (str): Party ID.
        party_name (str): Party name.
        secret_array (np.ndarray): Secret array.
        name (str): Secrets name.
        nada_type (Any): Nada type.
        permissions (nillion.Permissions): Optional Permissions.


    Returns:
        str: Store ID.
    """
    stored_secret = nillion.NadaValues(
        na_client.array(secret_array, secret_name, nada_type)
    )

    store_id = await store_secrets(
        client,
        payments_wallet,
        payments_client,
        cluster_id,
        stored_secret,
        ttl_days,
        permissions,
    )

    return store_id


async def store_secret_value(
    client: nillion.NillionClient,
    payments_wallet: LocalWallet,
    payments_client: LedgerClient,
    cluster_id: str,
    secret_value: Any,
    secret_name: str,
    nada_type: Any,
    ttl_days: int = 1,
    permissions: nillion.Permissions = None,
):
    """
    Asynchronous function to store secret values on the nillion client.

    Args:
        client (nillion.NillionClient): Nillion client.
        cluster_id (str): Cluster ID.
        party_id (str): Party ID.
        party_name (str): Party name.
        secret_value (Any): Secret single value.
        name (str): Secrets name.
        nada_type (Any): Nada type.
        permissions (nillion.Permissions): Optional Permissions.

    Returns:
        str: Store ID.
    """
    if nada_type == na.Rational:
        secret_value = round(secret_value * 2 ** na.get_log_scale())
        nada_type = nillion.Integer
    elif nada_type == na.SecretRational:
        secret_value = round(secret_value * 2 ** na.get_log_scale())
        nada_type = nillion.SecretInteger

    stored_secret = nillion.NadaValues(
        {
            secret_name: nada_type(secret_value),
        }
    )

    store_id = await store_secrets(
        client,
        payments_wallet,
        payments_client,
        cluster_id,
        stored_secret,
        ttl_days,
        permissions,
    )

    return store_id


async def store_secrets(
    client: nillion.NillionClient,
    payments_wallet: LocalWallet,
    payments_client: LedgerClient,
    cluster_id: str,
    secret: nillion.NadaValues,
    ttl_days: int = 1,
    permissions: nillion.Permissions = None,
):
    """
    Asynchronous function to store secrets on the nillion client.

    Args:
        client (nillion.NillionClient): Nillion client.
        cluster_id (str): Cluster ID.
        party_id (str): Party ID.
        party_name (str): Party name.
        secret (nillion.NadaValues): Stored secret.
        permissions (nillion.Permissions): Optional Permissions.


    Returns:
        str: Store ID.
    """
    receipt_store = await get_quote_and_pay(
        client,
        nillion.Operation.store_values(secret, ttl_days=ttl_days),
        payments_wallet,
        payments_client,
        cluster_id,
    )

    store_id = await client.store_values(cluster_id, secret, permissions, receipt_store)

    return store_id


async def compute(
    client: nillion.NillionClient,
    payments_wallet: LocalWallet,
    payments_client: LedgerClient,
    program_id: str,
    cluster_id: str,
    compute_bindings: nillion.ProgramBindings,
    store_ids: List[str],
    computation_time_secrets: nillion.NadaValues,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Asynchronous function to perform computation on the nillion client.

    Args:
        client (nillion.NillionClient): Nillion client.
        cluster_id (str): Cluster ID.
        compute_bindings (nillion.ProgramBindings): Compute bindings.
        store_ids (List[str]): List of data store IDs.
        computation_time_secrets (nillion.Secrets): Computation time secrets.
        verbose (bool, optional): Verbosity level. Defaults to True.

    Returns:
        Dict[str, Any]: Result of computation.
    """
    receipt_compute = await get_quote_and_pay(
        client,
        nillion.Operation.compute(program_id, computation_time_secrets),
        payments_wallet,
        payments_client,
        cluster_id,
    )

    _ = await client.compute(
        cluster_id,
        compute_bindings,
        store_ids,
        computation_time_secrets,
        receipt_compute,
    )

    while True:
        compute_event = await client.next_compute_event()
        if isinstance(compute_event, nillion.ComputeFinishedEvent):
            if verbose:
                print(f"✅ Compute complete for compute_id {compute_event.uuid}")
                print(f"🖥️  The result is {compute_event.result.value}")
            return compute_event.result.value
