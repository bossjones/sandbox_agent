"""
This type stub file was generated by pyright.
"""

"""Equality-constrained quadratic programming solvers."""
__all__ = ['eqp_kktfact', 'sphere_intersections', 'box_intersections', 'box_sphere_intersections', 'inside_box_boundaries', 'modified_dogleg', 'projected_cg']
def eqp_kktfact(H, c, A, b): # -> tuple[Any, Any]:
    """Solve equality-constrained quadratic programming (EQP) problem.

    Solve ``min 1/2 x.T H x + x.t c`` subject to ``A x + b = 0``
    using direct factorization of the KKT system.

    Parameters
    ----------
    H : sparse matrix, shape (n, n)
        Hessian matrix of the EQP problem.
    c : array_like, shape (n,)
        Gradient of the quadratic objective function.
    A : sparse matrix
        Jacobian matrix of the EQP problem.
    b : array_like, shape (m,)
        Right-hand side of the constraint equation.

    Returns
    -------
    x : array_like, shape (n,)
        Solution of the KKT problem.
    lagrange_multipliers : ndarray, shape (m,)
        Lagrange multipliers of the KKT problem.
    """
    ...

def sphere_intersections(z, d, trust_radius, entire_line=...): # -> tuple[Literal[0], Literal[0], Literal[False]] | tuple[float | Literal[0], float | Literal[1], Literal[True]] | tuple[Any | int, Any | int, bool]:
    """Find the intersection between segment (or line) and spherical constraints.

    Find the intersection between the segment (or line) defined by the
    parametric  equation ``x(t) = z + t*d`` and the ball
    ``||x|| <= trust_radius``.

    Parameters
    ----------
    z : array_like, shape (n,)
        Initial point.
    d : array_like, shape (n,)
        Direction.
    trust_radius : float
        Ball radius.
    entire_line : bool, optional
        When ``True``, the function returns the intersection between the line
        ``x(t) = z + t*d`` (``t`` can assume any value) and the ball
        ``||x|| <= trust_radius``. When ``False``, the function returns the intersection
        between the segment ``x(t) = z + t*d``, ``0 <= t <= 1``, and the ball.

    Returns
    -------
    ta, tb : float
        The line/segment ``x(t) = z + t*d`` is inside the ball for
        for ``ta <= t <= tb``.
    intersect : bool
        When ``True``, there is a intersection between the line/segment
        and the sphere. On the other hand, when ``False``, there is no
        intersection.
    """
    ...

def box_intersections(z, d, lb, ub, entire_line=...): # -> tuple[Literal[0], Literal[0], Literal[False]] | tuple[int | Any, int | Any, bool]:
    """Find the intersection between segment (or line) and box constraints.

    Find the intersection between the segment (or line) defined by the
    parametric  equation ``x(t) = z + t*d`` and the rectangular box
    ``lb <= x <= ub``.

    Parameters
    ----------
    z : array_like, shape (n,)
        Initial point.
    d : array_like, shape (n,)
        Direction.
    lb : array_like, shape (n,)
        Lower bounds to each one of the components of ``x``. Used
        to delimit the rectangular box.
    ub : array_like, shape (n, )
        Upper bounds to each one of the components of ``x``. Used
        to delimit the rectangular box.
    entire_line : bool, optional
        When ``True``, the function returns the intersection between the line
        ``x(t) = z + t*d`` (``t`` can assume any value) and the rectangular
        box. When ``False``, the function returns the intersection between the segment
        ``x(t) = z + t*d``, ``0 <= t <= 1``, and the rectangular box.

    Returns
    -------
    ta, tb : float
        The line/segment ``x(t) = z + t*d`` is inside the box for
        for ``ta <= t <= tb``.
    intersect : bool
        When ``True``, there is a intersection between the line (or segment)
        and the rectangular box. On the other hand, when ``False``, there is no
        intersection.
    """
    ...

def box_sphere_intersections(z, d, lb, ub, trust_radius, entire_line=..., extra_info=...): # -> tuple[Any, Any, bool, dict[str, Any], dict[str, Any]] | tuple[Any, Any, bool]:
    """Find the intersection between segment (or line) and box/sphere constraints.

    Find the intersection between the segment (or line) defined by the
    parametric  equation ``x(t) = z + t*d``, the rectangular box
    ``lb <= x <= ub`` and the ball ``||x|| <= trust_radius``.

    Parameters
    ----------
    z : array_like, shape (n,)
        Initial point.
    d : array_like, shape (n,)
        Direction.
    lb : array_like, shape (n,)
        Lower bounds to each one of the components of ``x``. Used
        to delimit the rectangular box.
    ub : array_like, shape (n, )
        Upper bounds to each one of the components of ``x``. Used
        to delimit the rectangular box.
    trust_radius : float
        Ball radius.
    entire_line : bool, optional
        When ``True``, the function returns the intersection between the line
        ``x(t) = z + t*d`` (``t`` can assume any value) and the constraints.
        When ``False``, the function returns the intersection between the segment
        ``x(t) = z + t*d``, ``0 <= t <= 1`` and the constraints.
    extra_info : bool, optional
        When ``True``, the function returns ``intersect_sphere`` and ``intersect_box``.

    Returns
    -------
    ta, tb : float
        The line/segment ``x(t) = z + t*d`` is inside the rectangular box and
        inside the ball for ``ta <= t <= tb``.
    intersect : bool
        When ``True``, there is a intersection between the line (or segment)
        and both constraints. On the other hand, when ``False``, there is no
        intersection.
    sphere_info : dict, optional
        Dictionary ``{ta, tb, intersect}`` containing the interval ``[ta, tb]``
        for which the line intercepts the ball. And a boolean value indicating
        whether the sphere is intersected by the line.
    box_info : dict, optional
        Dictionary ``{ta, tb, intersect}`` containing the interval ``[ta, tb]``
        for which the line intercepts the box. And a boolean value indicating
        whether the box is intersected by the line.
    """
    ...

def inside_box_boundaries(x, lb, ub):
    """Check if lb <= x <= ub."""
    ...

def reinforce_box_boundaries(x, lb, ub): # -> Any:
    """Return clipped value of x"""
    ...

def modified_dogleg(A, Y, b, trust_radius, lb, ub):
    """Approximately  minimize ``1/2*|| A x + b ||^2`` inside trust-region.

    Approximately solve the problem of minimizing ``1/2*|| A x + b ||^2``
    subject to ``||x|| < Delta`` and ``lb <= x <= ub`` using a modification
    of the classical dogleg approach.

    Parameters
    ----------
    A : LinearOperator (or sparse matrix or ndarray), shape (m, n)
        Matrix ``A`` in the minimization problem. It should have
        dimension ``(m, n)`` such that ``m < n``.
    Y : LinearOperator (or sparse matrix or ndarray), shape (n, m)
        LinearOperator that apply the projection matrix
        ``Q = A.T inv(A A.T)`` to the vector. The obtained vector
        ``y = Q x`` being the minimum norm solution of ``A y = x``.
    b : array_like, shape (m,)
        Vector ``b``in the minimization problem.
    trust_radius: float
        Trust radius to be considered. Delimits a sphere boundary
        to the problem.
    lb : array_like, shape (n,)
        Lower bounds to each one of the components of ``x``.
        It is expected that ``lb <= 0``, otherwise the algorithm
        may fail. If ``lb[i] = -Inf``, the lower
        bound for the ith component is just ignored.
    ub : array_like, shape (n, )
        Upper bounds to each one of the components of ``x``.
        It is expected that ``ub >= 0``, otherwise the algorithm
        may fail. If ``ub[i] = Inf``, the upper bound for the ith
        component is just ignored.

    Returns
    -------
    x : array_like, shape (n,)
        Solution to the problem.

    Notes
    -----
    Based on implementations described in pp. 885-886 from [1]_.

    References
    ----------
    .. [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
           "An interior point algorithm for large-scale nonlinear
           programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
    """
    ...

def projected_cg(H, c, Z, Y, b, trust_radius=..., lb=..., ub=..., tol=..., max_iter=..., max_infeasible_iter=..., return_all=...): # -> tuple[Any, dict[str, Any]]:
    """Solve EQP problem with projected CG method.

    Solve equality-constrained quadratic programming problem
    ``min 1/2 x.T H x + x.t c``  subject to ``A x + b = 0`` and,
    possibly, to trust region constraints ``||x|| < trust_radius``
    and box constraints ``lb <= x <= ub``.

    Parameters
    ----------
    H : LinearOperator (or sparse matrix or ndarray), shape (n, n)
        Operator for computing ``H v``.
    c : array_like, shape (n,)
        Gradient of the quadratic objective function.
    Z : LinearOperator (or sparse matrix or ndarray), shape (n, n)
        Operator for projecting ``x`` into the null space of A.
    Y : LinearOperator,  sparse matrix, ndarray, shape (n, m)
        Operator that, for a given a vector ``b``, compute smallest
        norm solution of ``A x + b = 0``.
    b : array_like, shape (m,)
        Right-hand side of the constraint equation.
    trust_radius : float, optional
        Trust radius to be considered. By default, uses ``trust_radius=inf``,
        which means no trust radius at all.
    lb : array_like, shape (n,), optional
        Lower bounds to each one of the components of ``x``.
        If ``lb[i] = -Inf`` the lower bound for the i-th
        component is just ignored (default).
    ub : array_like, shape (n, ), optional
        Upper bounds to each one of the components of ``x``.
        If ``ub[i] = Inf`` the upper bound for the i-th
        component is just ignored (default).
    tol : float, optional
        Tolerance used to interrupt the algorithm.
    max_iter : int, optional
        Maximum algorithm iterations. Where ``max_inter <= n-m``.
        By default, uses ``max_iter = n-m``.
    max_infeasible_iter : int, optional
        Maximum infeasible (regarding box constraints) iterations the
        algorithm is allowed to take.
        By default, uses ``max_infeasible_iter = n-m``.
    return_all : bool, optional
        When ``true``, return the list of all vectors through the iterations.

    Returns
    -------
    x : array_like, shape (n,)
        Solution of the EQP problem.
    info : Dict
        Dictionary containing the following:

            - niter : Number of iterations.
            - stop_cond : Reason for algorithm termination:
                1. Iteration limit was reached;
                2. Reached the trust-region boundary;
                3. Negative curvature detected;
                4. Tolerance was satisfied.
            - allvecs : List containing all intermediary vectors (optional).
            - hits_boundary : True if the proposed step is on the boundary
              of the trust region.

    Notes
    -----
    Implementation of Algorithm 6.2 on [1]_.

    In the absence of spherical and box constraints, for sufficient
    iterations, the method returns a truly optimal result.
    In the presence of those constraints, the value returned is only
    a inexpensive approximation of the optimal value.

    References
    ----------
    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
           "On the solution of equality constrained quadratic
            programming problems arising in optimization."
            SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
    """
    ...
