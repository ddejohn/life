"""Exception objects for error handling"""

from typing import Tuple


N_ERROR_MSG = """
The `n` parameter must be a positive integer >= 2
but got '{t}' with value '{v}' instead.
"""

BOUNDS_ERROR_MSG = """
The `bounds` parameter must be a string with value 'fixed' or 'periodic'
but got '{t}' with value '{v}' instead.
"""


class LifeParamsError(Exception):
    """Base exception for Life object parameter errors"""
    def __init__(self, msg: str):
        super().__init__(msg.strip("\n").replace("\n", " "))


def validate_args(n, bounds) -> Tuple[int, str]:
    """Ensure Life parameters are correct"""
    if (t := type(n)) != int or n < 2:
        raise LifeParamsError(N_ERROR_MSG.format(t=t.__name__, v=n))
    if (t := type(bounds)) != str or bounds not in ("fixed", "periodic"):
        raise LifeParamsError(BOUNDS_ERROR_MSG.format(t=t.__name__, v=bounds))
    return n, bounds
