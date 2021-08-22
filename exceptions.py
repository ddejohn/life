"""Exception objects for error handling"""

from typing import Any


SIZE_ERROR_MSG = """
The `size` parameter must be a positive integer between 10.
"""


SIZE_WARNING_MSG = """
!!! WARNING !!!

While Life can iterate arrays of size {n}, it may take several
minutes to render the animated gif!

It is recommended that you stick to arrays smaller than 200x200
if you intend to create an animation.
"""


BOUNDS_ERROR_MSG = """
The `bounds` parameter must be a string with value 'fixed' or 'periodic'.
"""


PATTERN_ERROR_MSG = """
The `pattern_type` parameter must be a string with value 'noise' or 'tiles'.
"""


class LifeParamsError(Exception):
    """Base exception for Life object parameter errors"""
    def __init__(self, msg: str):
        super().__init__(msg)


class SizeTypeError(LifeParamsError):
    def __init__(self, msg: str):
        super().__init__(f"{msg}\n{SIZE_ERROR_MSG}")


class SizeValueError(LifeParamsError):
    def __init__(self, msg: str):
        super().__init__(f"{msg}\n{SIZE_ERROR_MSG}")


class BoundsTypeError(LifeParamsError):
    def __init__(self, msg: str):
        super().__init__(f"{msg}\n{BOUNDS_ERROR_MSG}")


class BoundsValueError(LifeParamsError):
    def __init__(self, msg: str):
        super().__init__(f"{msg}\n{BOUNDS_ERROR_MSG}")


class PatternTypeError(LifeParamsError):
    def __init__(self, msg: str):
        super().__init__(f"{msg}\n{PATTERN_ERROR_MSG}")


class PatternValueError(LifeParamsError):
    def __init__(self, msg: str):
        super().__init__(f"{msg}\n{PATTERN_ERROR_MSG}")


def validate_args(size: Any, bounds: Any, pattern_type: Any) -> None:
    """Ensure Life parameters are correct"""
    if (t := type(size)) != int:
        err = f"Invalid `size` type '{t.__name__}'."
        raise SizeTypeError(err)
    if size < 10:
        err = f"Invalid `size` value '{size}'."
        raise SizeValueError(err)
    if size > 200:
        print(SIZE_WARNING_MSG.format(n=size))

    if (t := type(bounds)) != str:
        err = f"Invalid `bounds` of type '{t.__name__}'."
        raise BoundsTypeError(err)
    if bounds not in ("fixed", "periodic"):
        err = f"Invalid `bounds` value '{bounds}'."
        raise BoundsValueError(err)

    if (t := type(pattern_type)) != str:
        err = f"Invalid `pattern_type` of type '{t.__name__}'."
        raise PatternTypeError(err)
    if pattern_type not in ("tiles", "noise"):
        err = f"Invalid `pattern_type` value '{pattern_type}'."
        raise PatternValueError(err)
