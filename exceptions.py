from typing import Tuple


SIZE_ERROR_MSG = """
The `size` parameter must be a 2-tuple of positive
integers >= 2 but got '{}' instead.
"""

BOUNDARY_ERROR_MSG = """
The `boundary` parameter must be a string with
value 'fixed' or 'periodic' but got '{}' instead.
"""


class LifeError(Exception):
    """Base exception for Life object parameter errors"""
    def __init__(self, msg: str):
        super().__init__(msg.strip("\n").replace("\n", " "))


class SizeError(LifeError):
    """Exception for Life `size` parameter errors"""
    def __init__(self, msg: str):
        super().__init__(SIZE_ERROR_MSG.format(msg))


class BoundaryError(LifeError):
    """Exception for Life `boundary` parameter errors"""
    def __init__(self, msg: str):
        super().__init__(BOUNDARY_ERROR_MSG.format(msg))


def validate_args(size, boundary) -> Tuple[Tuple[int, int], str]:
    if type(size) != tuple:
        raise SizeError(type(size).__name__)
    if not all(dim > 1 for dim in size):
        raise SizeError(f"{size}")
    if not all((types := tuple(type(d) == int for d in size))):
        raise SizeError(f"{[t.__name__ for t in types]}")
    if type(boundary) != str:
        raise BoundaryError(type(boundary).__name__)
    if boundary not in ("fixed", "periodic"):
        raise BoundaryError(boundary)
    return size, boundary
