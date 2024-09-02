"""
This type stub file was generated by pyright.
"""

import numpy as np
import scipy.sparse as sp
from typing import Any, Literal, Optional, Union

""" Utility functions for sparse matrix module
"""
__all__ = ['upcast', 'getdtype', 'getdata', 'isscalarlike', 'isintlike', 'isshape', 'issequence', 'isdense', 'ismatrix', 'get_sum_dtype']
supported_dtypes = ...
_upcast_memo = ...
def upcast(*args):
    """Returns the nearest supported sparse dtype for the
    combination of one or more types.

    upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

    Examples
    --------
    >>> from scipy.sparse._sputils import upcast
    >>> upcast('int32')
    <type 'numpy.int32'>
    >>> upcast('bool')
    <type 'numpy.bool_'>
    >>> upcast('int32','float32')
    <type 'numpy.float64'>
    >>> upcast('bool',complex,float)
    <type 'numpy.complex128'>

    """
    ...

def upcast_char(*args):
    """Same as `upcast` but taking dtype.char as input (faster)."""
    ...

def upcast_scalar(dtype, scalar):
    """Determine data type for binary operation between an array of
    type `dtype` and a scalar.
    """
    ...

def downcast_intp_index(arr):
    """
    Down-cast index array to np.intp dtype if it is of a larger dtype.

    Raise an error if the array contains a value that is too large for
    intp.
    """
    ...

def to_native(A): # -> NDArray[Any]:
    """
    Ensure that the data type of the NumPy array `A` has native byte order.

    `A` must be a NumPy array.  If the data type of `A` does not have native
    byte order, a copy of `A` with a native byte order is returned. Otherwise
    `A` is returned.
    """
    ...

def getdtype(dtype, a=..., default=...): # -> dtype[Any]:
    """Function used to simplify argument processing. If 'dtype' is not
    specified (is None), returns a.dtype; otherwise returns a np.dtype
    object created from the specified dtype argument. If 'dtype' and 'a'
    are both None, construct a data type out of the 'default' parameter.
    Furthermore, 'dtype' must be in 'allowed' set.
    """
    ...

def getdata(obj, dtype=..., copy=...) -> np.ndarray:
    """
    This is a wrapper of `np.array(obj, dtype=dtype, copy=copy)`
    that will generate a warning if the result is an object array.
    """
    ...

def get_index_dtype(arrays=..., maxval=..., check_contents=...): # -> int64 | int32:
    """
    Based on input (integer) arrays `a`, determine a suitable index data
    type that can hold the data in the arrays.

    Parameters
    ----------
    arrays : tuple of array_like
        Input arrays whose types/contents to check
    maxval : float, optional
        Maximum value needed
    check_contents : bool, optional
        Whether to check the values in the arrays and not just their types.
        Default: False (check only the types)

    Returns
    -------
    dtype : dtype
        Suitable index data type (int32 or int64)

    """
    ...

def get_sum_dtype(dtype: np.dtype) -> np.dtype:
    """Mimic numpy's casting for np.sum"""
    ...

def isscalarlike(x) -> bool:
    """Is x either a scalar, an array scalar, or a 0-dim array?"""
    ...

def isintlike(x) -> bool:
    """Is x appropriate as an index into a sparse matrix? Returns True
    if it can be cast safely to a machine int.
    """
    ...

def isshape(x, nonneg=..., *, allow_1d=...) -> bool:
    """Is x a valid tuple of dimensions?

    If nonneg, also checks that the dimensions are non-negative.
    If allow_1d, shapes of length 1 or 2 are allowed.
    """
    ...

def issequence(t) -> bool:
    ...

def ismatrix(t) -> bool:
    ...

def isdense(x) -> bool:
    ...

def validateaxis(axis) -> None:
    ...

def check_shape(args, current_shape=..., *, allow_1d=...) -> tuple[int, ...]:
    """Imitate numpy.matrix handling of shape arguments

    Parameters
    ----------
    args : array_like
        Data structures providing information about the shape of the sparse array.
    current_shape : tuple, optional
        The current shape of the sparse array or matrix.
        If None (default), the current shape will be inferred from args.
    allow_1d : bool, optional
        If True, then 1-D or 2-D arrays are accepted.
        If False (default), then only 2-D arrays are accepted and an error is
        raised otherwise.

    Returns
    -------
    new_shape: tuple
        The new shape after validation.
    """
    ...

def check_reshape_kwargs(kwargs): # -> tuple[Any, Any]:
    """Unpack keyword arguments for reshape function.

    This is useful because keyword arguments after star arguments are not
    allowed in Python 2, but star keyword arguments are. This function unpacks
    'order' and 'copy' from the star keyword arguments (with defaults) and
    throws an error for any remaining.
    """
    ...

def is_pydata_spmatrix(m) -> bool:
    """
    Check whether object is pydata/sparse matrix, avoiding importing the module.
    """
    ...

def convert_pydata_sparse_to_scipy(arg: Any, target_format: Optional[Literal["csc", "csr"]] = ...) -> Union[Any, sp.spmatrix]:
    """
    Convert a pydata/sparse array to scipy sparse matrix,
    pass through anything else.
    """
    ...

def matrix(*args, **kwargs):
    ...

def asmatrix(data, dtype=...): # -> matrix[Any, Any]:
    ...
