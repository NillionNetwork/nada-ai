"""Stores common typing traits"""

from typing import Sequence, Tuple, Union

import nada_algebra as na
import py_nillion_client as nillion
from nada_dsl import (PublicInteger, PublicUnsignedInteger, SecretInteger,
                      SecretUnsignedInteger)
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  LogisticRegressionCV)

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
ShapeLike2d = Union[int, Tuple[int, int]]

NadaInteger = Union[
    SecretInteger,
    SecretUnsignedInteger,
    PublicInteger,
    PublicUnsignedInteger,
    na.Rational,
    na.SecretRational,
]
