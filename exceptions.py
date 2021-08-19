"""Exception objects for error handling"""

from typing import Tuple


SHAPE_ERROR_MSG = """
The `shape` parameter must be 2-tuple of positive integers >= 2.
"""

BOUNDS_ERROR_MSG = """
The `bounds` parameter must be a string with value 'fixed' or 'periodic'.
"""


class LifeParamsError(Exception):
    """Base exception for Life object parameter errors"""
    def __init__(self, msg: str):
        super().__init__(msg)


class ShapeError(LifeParamsError):
    def __init__(self, msg: str):
        super().__init__(f"{msg}\n{SHAPE_ERROR_MSG}")


class BoundsError(LifeParamsError):
    def __init__(self, msg: str):
        super().__init__(f"{msg}\n{BOUNDS_ERROR_MSG}")


def validate_args(shape, bounds) -> Tuple[int, str]:
    """Ensure Life parameters are correct"""
    if (t := type(shape)) != tuple:
        raise ShapeError(f"Invalid `shape` of type '{t.__name__}'.")
    if not all(type(val) == int for val in shape):
        vals = ", ".join(f"{type(v).__name__}" for v in shape)
        raise ShapeError(f"Invalid `shape` of type 'Tuple[{vals}]'")
    if not all(val >= 2 for val in shape):
        raise ShapeError(f"Invalid `shape` with values '{shape}'")
    if (t := type(bounds)) != str or bounds not in ("fixed", "periodic"):
        raise BoundsError(f"Invalid `bounds` of type '{t.__name__}'.")
    return shape, bounds
