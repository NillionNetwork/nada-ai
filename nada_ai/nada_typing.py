"""Stores common typing traits"""

from typing import Sequence, Tuple, Union

import nada_dsl as dsl
import nada_numpy as na
# pylint:disable=no-name-in-module
from py_nillion_client import (Integer, SecretInteger, SecretUnsignedInteger,
                               UnsignedInteger)
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  LogisticRegressionCV)

__all__ = ["NillionType", "LinearModel", "ShapeLike", "NadaInteger"]

NillionType = Union[
    na.Rational,
    na.SecretRational,
    SecretInteger,
    SecretUnsignedInteger,
    Integer,
    UnsignedInteger,
]

LinearModel = Union[LinearRegression, LogisticRegression, LogisticRegressionCV]

ShapeLike = Union[int, Sequence[int]]
ShapeLike2d = Union[int, Tuple[int, int]]

NadaInteger = Union[
    dsl.SecretInteger,
    dsl.SecretUnsignedInteger,
    dsl.PublicInteger,
    dsl.PublicUnsignedInteger,
    na.Rational,
    na.SecretRational,
]

AnyNadaType = Union[
    dsl.SecretInteger,
    dsl.SecretUnsignedInteger,
    dsl.PublicInteger,
    dsl.PublicUnsignedInteger,
    dsl.Integer,
    dsl.UnsignedInteger,
    dsl.Boolean,
    dsl.SecretBoolean,
    na.Rational,
    na.SecretRational,
    na.PublicBoolean,
    na.SecretBoolean,
]

NadaCleartextType = Union[
    dsl.Integer,
    dsl.UnsignedInteger,
    dsl.Boolean,
    na.Rational,
]
