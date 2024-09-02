"""
This type stub file was generated by pyright.
"""

class spmatrix:
    """This class provides a base class for all sparse matrix classes.

    It cannot be instantiated.  Most of the work is provided by subclasses.
    """
    def __mul__(self, other):
        ...

    def __rmul__(self, other):
        ...

    def __pow__(self, power): # -> coo_array | coo_matrix | dia_array | Any | dia_matrix:
        ...

    def set_shape(self, shape): # -> None:
        """Set the shape of the matrix in-place"""
        ...

    def get_shape(self):
        """Get the shape of the matrix"""
        ...

    shape = ...
    def asfptype(self):
        """Upcast matrix to a floating point format (if necessary)"""
        ...

    def getmaxprint(self):
        """Maximum number of elements to display when printed."""
        ...

    def getformat(self):
        """Matrix storage format"""
        ...

    def getnnz(self, axis=...):
        """Number of stored values, including explicit zeros.

        Parameters
        ----------
        axis : None, 0, or 1
            Select between the number of values across the whole array, in
            each column, or in each row.
        """
        ...

    def getH(self):
        """Return the Hermitian transpose of this matrix.

        See Also
        --------
        numpy.matrix.getH : NumPy's implementation of `getH` for matrices
        """
        ...

    def getcol(self, j):
        """Returns a copy of column j of the matrix, as an (m x 1) sparse
        matrix (column vector).
        """
        ...

    def getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n) sparse
        matrix (row vector).
        """
        ...

