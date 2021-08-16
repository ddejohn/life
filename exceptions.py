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
