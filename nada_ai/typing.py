"""Stores common typing traits"""

import nada_algebra as na
import py_nillion_client as nillion
from typing import Union, Sequence
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
)
from nada_dsl import (
    SecretInteger,
    SecretUnsignedInteger,
    PublicInteger,
    PublicUnsignedInteger,
)

__all__ = ["NillionType", "LinearModel", "ShapeLike", "NadaInteger"]

NillionType = Union[
    na.Rational,
    na.SecretRational,
    nillion.SecretInteger,
    nillion.SecretUnsignedInteger,
    nillion.PublicVariableInteger,
    nillion.PublicVariableUnsignedInteger,
]

LinearModel = Union[LinearRegression, LogisticRegression, LogisticRegressionCV]

ShapeLike = Union[int, Sequence[int]]

NadaInteger = Union[
    SecretInteger,
    SecretUnsignedInteger,
    PublicInteger,
    PublicUnsignedInteger,
    na.Rational,
    na.SecretRational,
]
