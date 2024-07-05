import os
import subprocess

import pytest

TESTS = [
    "load_state",
    "parameters",
    "nested_modules",
    "linear_layers",
    "activations",
    "flatten",
    "conv",
    "pool",
    "linear_regression",
    "logistic_regression",
    "end-to-end",
    "distance",
    "prophet",
]

EXAMPLES = [
    "complex_model",
    "linear_regression",
    "multi_layer_perceptron",
    "neural_net",
    "spam_detection",
    "time_series",
]

TESTS = [("tests/nada-tests/", test) for test in TESTS] + [
    ("examples/" + test, test) for test in EXAMPLES
]


@pytest.fixture(params=TESTS)
def testname(request):
    return request.param


def build_nada(test_dir):
    result = subprocess.run(
        ["nada", "build", test_dir[1]], cwd=test_dir[0], capture_output=True, text=True
    )
    err = result.stderr.lower() + result.stdout.lower()
    if result.returncode != 0 or "error" in err or "fail" in err:
        pytest.fail(f"Build {test_dir}:\n{result.stdout + result.stderr}")


def run_nada(test_dir):
    result = subprocess.run(
        ["nada", "test", test_dir[1]], cwd=test_dir[0], capture_output=True, text=True
    )
    err = result.stderr.lower() + result.stdout.lower()
    if result.returncode != 0 or "error" in err or "fail" in err:
        pytest.fail(f"Run {test_dir}:\n{result.stdout + result.stderr}")


class TestSuite:

    def test_build(self, testname):
        # Get current working directory
        cwd = os.getcwd()
        try:
            # Build Nada Program
            build_nada(testname)
        finally:
            # Return to initial directory
            os.chdir(cwd)

    def test_run(self, testname):
        # Get current working directory
        cwd = os.getcwd()
        try:
            # Build Nada Program
            build_nada(testname)
        finally:
            # Return to initial directory
            os.chdir(cwd)


def test_client():
    import nada_numpy.client as na_client  # For use with Python Client
    import numpy as np
    import py_nillion_client as nillion

    parties = na_client.parties(3)

    assert parties is not None

    secrets = nillion.NadaValues(
        na_client.concat(
            [
                na_client.array(np.ones((3, 3)), "A", nillion.SecretInteger),
                na_client.array(np.ones((3, 3)), "B", nillion.SecretUnsignedInteger),
            ]
        )
    )

    assert secrets is not None

    public_variables = nillion.NadaValues(
        na_client.concat(
            [
                na_client.array(np.zeros((4, 4)), "C", nillion.Integer),
                na_client.array(np.zeros((3, 3)), "D", nillion.UnsignedInteger),
            ]
        )
    )

    assert public_variables is not None
