"""
This type stub file was generated by pyright.
"""

from ._miobase import MatFileReader, docfiller

''' Classes for read / write of matlab (TM) 4 files
'''
__all__ = ['MatFile4Reader', 'MatFile4Writer', 'SYS_LITTLE_ENDIAN', 'VarHeader4', 'VarReader4', 'VarWriter4', 'arr_to_2d', 'mclass_info', 'mdtypes_template', 'miDOUBLE', 'miINT16', 'miINT32', 'miSINGLE', 'miUINT16', 'miUINT8', 'mxCHAR_CLASS', 'mxFULL_CLASS', 'mxSPARSE_CLASS', 'np_to_mtypes', 'order_codes']
SYS_LITTLE_ENDIAN = ...
miDOUBLE = ...
miSINGLE = ...
miINT32 = ...
miINT16 = ...
miUINT16 = ...
miUINT8 = ...
mdtypes_template = ...
np_to_mtypes = ...
mxFULL_CLASS = ...
mxCHAR_CLASS = ...
mxSPARSE_CLASS = ...
order_codes = ...
mclass_info = ...
class VarHeader4:
    is_logical = ...
    is_global = ...
    def __init__(self, name, dtype, mclass, dims, is_complex) -> None:
        ...



class VarReader4:
    ''' Class to read matlab 4 variables '''
    def __init__(self, file_reader) -> None:
        ...

    def read_header(self): # -> VarHeader4:
        ''' Read and return header for variable '''
        ...

    def array_from_header(self, hdr, process=...): # -> coo_matrix | NDArray[complexfloating[Any, Any]] | ndarray[Any, Any]:
        ...

    def read_sub_array(self, hdr, copy=...): # -> ndarray[Any, Any]:
        ''' Mat4 read using header `hdr` dtype and dims

        Parameters
        ----------
        hdr : object
           object with attributes ``dtype``, ``dims``. dtype is assumed to be
           the correct endianness
        copy : bool, optional
           copies array before return if True (default True)
           (buffer is usually read only)

        Returns
        -------
        arr : ndarray
            of dtype given by `hdr` ``dtype`` and shape given by `hdr` ``dims``
        '''
        ...

    def read_full_array(self, hdr): # -> NDArray[complexfloating[Any, Any]] | ndarray[Any, Any]:
        ''' Full (rather than sparse) matrix getter

        Read matrix (array) can be real or complex

        Parameters
        ----------
        hdr : ``VarHeader4`` instance

        Returns
        -------
        arr : ndarray
            complex array if ``hdr.is_complex`` is True, otherwise a real
            numeric array
        '''
        ...

    def read_char_array(self, hdr): # -> ndarray[Any, Any]:
        ''' latin-1 text matrix (char matrix) reader

        Parameters
        ----------
        hdr : ``VarHeader4`` instance

        Returns
        -------
        arr : ndarray
            with dtype 'U1', shape given by `hdr` ``dims``
        '''
        ...

    def read_sparse_array(self, hdr): # -> coo_matrix:
        ''' Read and return sparse matrix type

        Parameters
        ----------
        hdr : ``VarHeader4`` instance

        Returns
        -------
        arr : ``scipy.sparse.coo_matrix``
            with dtype ``float`` and shape read from the sparse matrix data

        Notes
        -----
        MATLAB 4 real sparse arrays are saved in a N+1 by 3 array format, where
        N is the number of non-zero values. Column 1 values [0:N] are the
        (1-based) row indices of the each non-zero value, column 2 [0:N] are the
        column indices, column 3 [0:N] are the (real) values. The last values
        [-1,0:2] of the rows, column indices are shape[0] and shape[1]
        respectively of the output matrix. The last value for the values column
        is a padding 0. mrows and ncols values from the header give the shape of
        the stored matrix, here [N+1, 3]. Complex data are saved as a 4 column
        matrix, where the fourth column contains the imaginary component; the
        last value is again 0. Complex sparse data do *not* have the header
        ``imagf`` field set to True; the fact that the data are complex is only
        detectable because there are 4 storage columns.
        '''
        ...

    def shape_from_header(self, hdr): # -> tuple[()] | tuple[int, ...] | tuple[int, int]:
        '''Read the shape of the array described by the header.
        The file position after this call is unspecified.
        '''
        ...



class MatFile4Reader(MatFileReader):
    ''' Reader for Mat4 files '''
    @docfiller
    def __init__(self, mat_stream, *args, **kwargs) -> None:
        ''' Initialize matlab 4 file reader

    %(matstream_arg)s
    %(load_args)s
        '''
        ...

    def guess_byte_order(self): # -> Literal['<', '>']:
        ...

    def initialize_read(self): # -> None:
        ''' Run when beginning read of variables

        Sets up readers from parameters in `self`
        '''
        ...

    def read_var_header(self): # -> tuple[VarHeader4, Any]:
        ''' Read and return header, next position

        Parameters
        ----------
        None

        Returns
        -------
        header : object
           object that can be passed to self.read_var_array, and that
           has attributes ``name`` and ``is_global``
        next_position : int
           position in stream of next variable
        '''
        ...

    def read_var_array(self, header, process=...): # -> coo_matrix | NDArray[complexfloating[Any, Any]] | ndarray[Any, Any]:
        ''' Read array, given `header`

        Parameters
        ----------
        header : header object
           object with fields defining variable header
        process : {True, False}, optional
           If True, apply recursive post-processing during loading of array.

        Returns
        -------
        arr : array
           array with post-processing applied or not according to
           `process`.
        '''
        ...

    def get_variables(self, variable_names=...): # -> dict[Any, Any]:
        ''' get variables from stream as dictionary

        Parameters
        ----------
        variable_names : None or str or sequence of str, optional
            variable name, or sequence of variable names to get from Mat file /
            file stream. If None, then get all variables in file.
        '''
        ...

    def list_variables(self): # -> list[Any]:
        ''' list variables from stream '''
        ...



def arr_to_2d(arr, oned_as=...):
    ''' Make ``arr`` exactly two dimensional

    If `arr` has more than 2 dimensions, raise a ValueError

    Parameters
    ----------
    arr : array
    oned_as : {'row', 'column'}, optional
       Whether to reshape 1-D vectors as row vectors or column vectors.
       See documentation for ``matdims`` for more detail

    Returns
    -------
    arr2d : array
       2-D version of the array
    '''
    ...

class VarWriter4:
    def __init__(self, file_writer) -> None:
        ...

    def write_bytes(self, arr): # -> None:
        ...

    def write_string(self, s): # -> None:
        ...

    def write_header(self, name, shape, P=..., T=..., imagf=...): # -> None:
        ''' Write header for given data options

        Parameters
        ----------
        name : str
            name of variable
        shape : sequence
            Shape of array as it will be read in matlab
        P : int, optional
            code for mat4 data type, one of ``miDOUBLE, miSINGLE, miINT32,
            miINT16, miUINT16, miUINT8``
        T : int, optional
            code for mat4 matrix class, one of ``mxFULL_CLASS, mxCHAR_CLASS,
            mxSPARSE_CLASS``
        imagf : int, optional
            flag indicating complex
        '''
        ...

    def write(self, arr, name): # -> None:
        ''' Write matrix `arr`, with name `name`

        Parameters
        ----------
        arr : array_like
           array to write
        name : str
           name in matlab workspace
        '''
        ...

    def write_numeric(self, arr, name): # -> None:
        ...

    def write_char(self, arr, name): # -> None:
        ...

    def write_sparse(self, arr, name): # -> None:
        ''' Sparse matrices are 2-D

        See docstring for VarReader4.read_sparse_array
        '''
        ...



class MatFile4Writer:
    ''' Class for writing matlab 4 format files '''
    def __init__(self, file_stream, oned_as=...) -> None:
        ...

    def put_variables(self, mdict, write_header=...): # -> None:
        ''' Write variables in `mdict` to stream

        Parameters
        ----------
        mdict : mapping
           mapping with method ``items`` return name, contents pairs
           where ``name`` which will appeak in the matlab workspace in
           file load, and ``contents`` is something writeable to a
           matlab file, such as a NumPy array.
        write_header : {None, True, False}
           If True, then write the matlab file header before writing the
           variables. If None (the default) then write the file header
           if we are at position 0 in the stream. By setting False
           here, and setting the stream position to the end of the file,
           you can append variables to a matlab file
        '''
        ...

