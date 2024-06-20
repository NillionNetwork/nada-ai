"""Stores common typing traits"""

from typing import Sequence, Tuple, Union

import nada_numpy as na
import nada_dsl as dsl
# pylint:disable=no-name-in-module
from py_nillion_client import (PublicVariableInteger,
                               PublicVariableUnsignedInteger, SecretInteger,
                               SecretUnsignedInteger)
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  LogisticRegressionCV)

__all__ = ["NillionType", "LinearModel", "ShapeLike", "NadaInteger"]

NillionType = Union[
    na.Rational,
    na.SecretRational,
    SecretInteger,
    SecretUnsignedInteger,
    PublicVariableInteger,
    PublicVariableUnsignedInteger,
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
