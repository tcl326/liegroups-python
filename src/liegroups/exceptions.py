"""
Custom exceptions for the Lie groups library
"""


class LieGrouopError(Exception):
    """Base Exception for Errors related to the Lie groups library"""

    pass


class LieGroupMismatch(LieGrouopError):
    """Raised when an operation is performed between two elements of different Lie groups"""

    pass

