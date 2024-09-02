"""
This type stub file was generated by pyright.
"""

class DegenerateDataWarning(RuntimeWarning):
    """Warns when data is degenerate and results may not be reliable."""
    def __init__(self, msg=...) -> None:
        ...



class ConstantInputWarning(DegenerateDataWarning):
    """Warns when all values in data are exactly equal."""
    def __init__(self, msg=...) -> None:
        ...



class NearConstantInputWarning(DegenerateDataWarning):
    """Warns when all values in data are nearly equal."""
    def __init__(self, msg=...) -> None:
        ...



class FitError(RuntimeError):
    """Represents an error condition when fitting a distribution to data."""
    def __init__(self, msg=...) -> None:
        ...

