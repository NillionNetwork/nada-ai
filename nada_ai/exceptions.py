"""Custom exceptions"""

__all__ = ["MismatchedShapesException"]


class MismatchedShapesException(Exception):
    """Raised when NadaArray shapes are incompatible"""
