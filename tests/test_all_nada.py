import os
import pytest
import subprocess

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
    "end-to-end",
    "prophet",
]

EXAMPLES = [
    "complex_model",
    "linear_regression",
    "neural_net",
    # "time_series",
]

TESTS = [("tests/nada-tests/", test) for test in TESTS] + [
    ("examples/" + test, test) for test in EXAMPLES
]


@pytest.fixture(params=TESTS)
def testname(request):
    return request.param


def build_nada(test_dir):
    print(test_dir)
    result = subprocess.run(
        ["nada", "build", test_dir[1]], cwd=test_dir[0], capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.fail(f"Build failed: {result.stderr}")


def run_nada(test_dir):
    result = subprocess.run(
        ["nada", "test", test_dir[1]], cwd=test_dir[0], capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.fail(f"Tests failed: {result.stderr}")


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
    import nada_algebra.client as na_client  # For use with Python Client
    import py_nillion_client as nillion
    import numpy as np

    parties = na_client.parties(3)

    assert parties is not None

    secrets = nillion.Secrets(
        na_client.concat(
            [
                na_client.array(np.ones((3, 3)), "A", nillion.SecretInteger),
                na_client.array(np.ones((3, 3)), "B", nillion.SecretUnsignedInteger),
            ]
        )
    )

    assert secrets is not None

    public_variables = nillion.PublicVariables(
        na_client.concat(
            [
                na_client.array(np.zeros((4, 4)), "C", nillion.PublicVariableInteger),
                na_client.array(
                    np.zeros((3, 3)), "D", nillion.PublicVariableUnsignedInteger
                ),
            ]
        )
    )

    assert public_variables is not None
