"""
This type stub file was generated by pyright.
"""

from abc import ABC, abstractmethod

class VertexBase(ABC):
    """
    Base class for a vertex.
    """
    def __init__(self, x, nn=..., index=...) -> None:
        """
        Initiation of a vertex object.

        Parameters
        ----------
        x : tuple or vector
            The geometric location (domain).
        nn : list, optional
            Nearest neighbour list.
        index : int, optional
            Index of vertex.
        """
        ...

    def __hash__(self) -> int:
        ...

    def __getattr__(self, item): # -> NDArray[Any] | None:
        ...

    @abstractmethod
    def connect(self, v):
        ...

    @abstractmethod
    def disconnect(self, v):
        ...

    def star(self): # -> set[Any]:
        """Returns the star domain ``st(v)`` of the vertex.

        Parameters
        ----------
        v :
            The vertex ``v`` in ``st(v)``

        Returns
        -------
        st : set
            A set containing all the vertices in ``st(v)``
        """
        ...



class VertexScalarField(VertexBase):
    """
    Add homology properties of a scalar field f: R^n --> R associated with
    the geometry built from the VertexBase class
    """
    def __init__(self, x, field=..., nn=..., index=..., field_args=..., g_cons=..., g_cons_args=...) -> None:
        """
        Parameters
        ----------
        x : tuple,
            vector of vertex coordinates
        field : callable, optional
            a scalar field f: R^n --> R associated with the geometry
        nn : list, optional
            list of nearest neighbours
        index : int, optional
            index of the vertex
        field_args : tuple, optional
            additional arguments to be passed to field
        g_cons : callable, optional
            constraints on the vertex
        g_cons_args : tuple, optional
            additional arguments to be passed to g_cons

        """
        ...

    def connect(self, v): # -> None:
        """Connects self to another vertex object v.

        Parameters
        ----------
        v : VertexBase or VertexScalarField object
        """
        ...

    def disconnect(self, v): # -> None:
        ...

    def minimiser(self): # -> bool:
        """Check whether this vertex is strictly less than all its
           neighbours"""
        ...

    def maximiser(self): # -> bool:
        """
        Check whether this vertex is strictly greater than all its
        neighbours.
        """
        ...



class VertexVectorField(VertexBase):
    """
    Add homology properties of a scalar field f: R^n --> R^m associated with
    the geometry built from the VertexBase class.
    """
    def __init__(self, x, sfield=..., vfield=..., field_args=..., vfield_args=..., g_cons=..., g_cons_args=..., nn=..., index=...) -> None:
        ...



class VertexCacheBase:
    """Base class for a vertex cache for a simplicial complex."""
    def __init__(self) -> None:
        ...

    def __iter__(self): # -> Generator[Any, Any, None]:
        ...

    def size(self): # -> int:
        """Returns the size of the vertex cache."""
        ...

    def print_out(self): # -> None:
        ...



class VertexCube(VertexBase):
    """Vertex class to be used for a pure simplicial complex with no associated
    differential geometry (single level domain that exists in R^n)"""
    def __init__(self, x, nn=..., index=...) -> None:
        ...

    def connect(self, v): # -> None:
        ...

    def disconnect(self, v): # -> None:
        ...



class VertexCacheIndex(VertexCacheBase):
    def __init__(self) -> None:
        """
        Class for a vertex cache for a simplicial complex without an associated
        field. Useful only for building and visualising a domain complex.

        Parameters
        ----------
        """
        ...

    def __getitem__(self, x, nn=...):
        ...



class VertexCacheField(VertexCacheBase):
    def __init__(self, field=..., field_args=..., g_cons=..., g_cons_args=..., workers=...) -> None:
        """
        Class for a vertex cache for a simplicial complex with an associated
        field.

        Parameters
        ----------
        field : callable
            Scalar or vector field callable.
        field_args : tuple, optional
            Any additional fixed parameters needed to completely specify the
            field function
        g_cons : dict or sequence of dict, optional
            Constraints definition.
            Function(s) ``R**n`` in the form::
        g_cons_args : tuple, optional
            Any additional fixed parameters needed to completely specify the
            constraint functions
        workers : int  optional
            Uses `multiprocessing.Pool <multiprocessing>`) to compute the field
             functions in parallel.

        """
        ...

    def __getitem__(self, x, nn=...):
        ...

    def __getstate__(self): # -> dict[str, Any]:
        ...

    def process_pools(self): # -> None:
        ...

    def feasibility_check(self, v): # -> None:
        ...

    def compute_sfield(self, v): # -> None:
        """Compute the scalar field values of a vertex object `v`.

        Parameters
        ----------
        v : VertexBase or VertexScalarField object
        """
        ...

    def proc_gpool(self): # -> None:
        """Process all constraints."""
        ...

    def pproc_gpool(self): # -> None:
        """Process all constraints in parallel."""
        ...

    def proc_fpool_g(self): # -> None:
        """Process all field functions with constraints supplied."""
        ...

    def proc_fpool_nog(self): # -> None:
        """Process all field functions with no constraints supplied."""
        ...

    def pproc_fpool_g(self): # -> None:
        """
        Process all field functions with constraints supplied in parallel.
        """
        ...

    def pproc_fpool_nog(self): # -> None:
        """
        Process all field functions with no constraints supplied in parallel.
        """
        ...

    def proc_minimisers(self): # -> None:
        """Check for minimisers."""
        ...



class ConstraintWrapper:
    """Object to wrap constraints to pass to `multiprocessing.Pool`."""
    def __init__(self, g_cons, g_cons_args) -> None:
        ...

    def gcons(self, v_x_a): # -> bool:
        ...



class FieldWrapper:
    """Object to wrap field to pass to `multiprocessing.Pool`."""
    def __init__(self, field, field_args) -> None:
        ...

    def func(self, v_x_a): # -> float:
        ...

