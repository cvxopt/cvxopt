.. _c-solvers:

*****************************
Nonlinear Convex Optimization
*****************************

In this chapter we consider nonlinear convex optimization problems of the
form

.. math::

    \begin{array}{ll}
    \mbox{minimize}   & f_0(x) \\
    \mbox{subject to} & f_k(x) \leq 0, \quad k=1,\ldots,m \\
                      & G x \preceq h  \\
                      & A x = b.
    \end{array}

The functions :math:`f_k` are convex and twice differentiable and the
linear inequalities are generalized inequalities with respect to a proper
convex cone, defined as a product of a nonnegative orthant, second-order
cones, and positive semidefinite cones.

The basic functions are :func:`cp <cvxopt.solvers.cp>` and
:func:`cpl <cvxopt.solvers.cpl>`, described in the sections
:ref:`s-cp` and :ref:`s-cpl`.   A simpler interface for geometric
programming problems is discussed in the section :ref:`s-gp`.
In the section :ref:`s-nlcp` we explain how custom solvers can be
implemented that exploit structure in specific classes of problems.
The last section
describes the algorithm parameters that control the solvers.


.. _s-cp:

Problems with Nonlinear Objectives
==================================

.. function:: cvxopt.solvers.cp(F[, G, h[, dims[, A, b[, kktsolver]]]])

    Solves a convex optimization problem

    .. math::
        :label: e-nlcp

        \begin{array}{ll}
            \mbox{minimize}   & f_0(x) \\
            \mbox{subject to} & f_k(x) \leq 0, \quad k=1,\ldots,m \\
                              & G x \preceq h  \\
                              & A x = b.
        \end{array}

    The argument ``F`` is a function that evaluates the objective and
    nonlinear constraint functions.  It must handle the following calling
    sequences.


    * ``F()`` returns a tuple (``m``, ``x0``), where :math:`m` is
      the number of nonlinear constraints and :math:`x_0` is a point in
      the domain of :math:`f`.  ``x0`` is a dense real matrix of size
      (:math:`n`, 1).

    * ``F(x)``, with ``x`` a dense real matrix of size (:math:`n`, 1),
      returns a tuple (``f``, ``Df``).  ``f`` is a dense real matrix of
      size (:math:`m+1`, 1), with ``f[k]`` equal to :math:`f_k(x)`.
      (If :math:`m` is zero, ``f`` can also be returned as a number.)
      ``Df`` is a dense or sparse real matrix of size (:math:`m` + 1,
      :math:`n`) with ``Df[k,:]`` equal to the transpose of the
      gradient :math:`\nabla f_k(x)`.  If :math:`x` is not in the domain
      of :math:`f`, ``F(x)`` returns :const:`None` or a tuple
      (:const:`None`, :const:`None`).

    * ``F(x,z)``, with ``x`` a dense real matrix of size (:math:`n`, 1)
      and ``z`` a positive dense real matrix of size (:math:`m` + 1, 1)
      returns a tuple (``f``, ``Df``, ``H``).  ``f`` and ``Df`` are
      defined as above.  ``H`` is a square dense or sparse real matrix of
      size (:math:`n`, :math:`n`), whose lower triangular part contains
      the lower triangular part of

      .. math::

          z_0 \nabla^2f_0(x) + z_1 \nabla^2f_1(x) + \cdots +
              z_m \nabla^2f_m(x).

      If ``F`` is called with two arguments, it can be assumed that
      :math:`x` is in the domain of :math:`f`.

    The linear inequalities are with respect to a cone :math:`C` defined
    as a Cartesian product of a nonnegative orthant, a number of
    second-order cones, and a number of positive semidefinite cones:

    .. math::

        C = C_0 \times C_1 \times \cdots \times C_M \times C_{M+1} \times
            \cdots \times C_{M+N}

    with

    .. math::


        \newcommand{\reals}{{\mbox{\bf R}}}
        \newcommand{\svec}{\mathop{\mathbf{vec}}}
        \newcommand{\symm}{{\mbox{\bf S}}}
        \begin{split}
            C_0 & =
                \{ u \in \reals^l \;| \; u_k \geq 0, \; k=1, \ldots,l\},\\
            C_{k+1} & = \{ (u_0, u_1) \in \reals \times \reals^{r_{k}-1}
                \; | \; u_0 \geq \|u_1\|_2 \},  \quad k=0,\ldots, M-1, \\
            C_{k+M+1} & = \left\{ \svec(u) \; | \; u \in \symm^{t_k}_+
                \right\}, \quad k=0,\ldots,N-1.
        \end{split}

    Here :math:`\mathbf{vec}(u)` denotes a symmetric matrix :math:`u`
    stored as a vector in column major order.

    The arguments ``h`` and ``b`` are real single-column dense matrices.
    ``G`` and ``A`` are real dense or sparse matrices.  The default values
    for ``A`` and ``b`` are sparse matrices with zero rows, meaning that
    there are no equality constraints.  The number of rows of ``G`` and
    ``h`` is equal to

    .. math::

        K = l + \sum_{k=0}^{M-1} r_k + \sum_{k=0}^{N-1} t_k^2.

    The columns of ``G`` and ``h`` are vectors in

    .. math::

        \newcommand{\reals}{{\mbox{\bf R}}}
        \reals^l \times \reals^{r_0} \times \cdots \times
        \reals^{r_{M-1}} \times \reals^{t_0^2}  \times \cdots \times
        \reals^{t_{N-1}^2},

    where the last :math:`N` components represent symmetric matrices stored
    in column major order.  The strictly upper triangular entries of these
    matrices are not accessed (i.e., the symmetric matrices are stored
    in the :const:`'L'`-type column major order used in the :mod:`blas`
    and :mod:`lapack` modules).

    The argument ``dims`` is a dictionary with the dimensions of the cones.
    It has three fields.

    ``dims['l']``:
        :math:`l`, the dimension of the nonnegative orthant (a nonnegative
        integer).

    ``dims['q']``:
        :math:`[r_0, \ldots, r_{M-1}]`, a list with the dimensions of the
        second-order cones (positive integers).

    ``dims['s']``:
        :math:`[t_0, \ldots, t_{N-1}]`, a list with the dimensions of the
        positive semidefinite cones (nonnegative integers).

    The default value of ``dims`` is
    ``{'l': h.size[0], 'q': [], 's': []}``, i.e., the default
    assumption is that the linear inequalities are componentwise
    inequalities.

    The role of the optional argument ``kktsolver`` is explained in the
    section :ref:`s-nlcp`.

    :func:`cp` returns a dictionary that contains the result and
    information about the accuracy of the solution.  The most important
    fields have keys :const:`'status'`, :const:`'x'`, :const:`'snl'`,
    :const:`'sl'`, :const:`'y'`, :const:`'znl'`, :const:`'zl'`.  The
    possible values of the :const:`'status'` key are:

    :const:`'optimal'`
        In this case the :const:`'x'` entry of the dictionary is the primal
        optimal solution, the :const:`'snl'` and :const:`'sl'` entries are
        the corresponding slacks in the nonlinear and linear inequality
        constraints, and the :const:`'znl'`, :const:`'zl'` and :const:`'y'`
        entries are the optimal values of the dual variables associated
        with the nonlinear inequalities, the linear inequalities, and the
        linear equality constraints.  These vectors approximately satisfy
        the Karush-Kuhn-Tucker (KKT) conditions

        .. math::

            \nabla f_0(x) +  D\tilde f(x)^T z_\mathrm{nl} +
            G^T z_\mathrm{l} + A^T y = 0,

            \tilde f(x) + s_\mathrm{nl} = 0, \quad k=1,\ldots,m, \qquad
            Gx + s_\mathrm{l} = h, \qquad Ax = b,

            s_\mathrm{nl}\succeq 0, \qquad s_\mathrm{l}\succeq 0, \qquad
            z_\mathrm{nl} \succeq 0, \qquad z_\mathrm{l} \succeq 0,

            s_\mathrm{nl}^T z_\mathrm{nl} +
            s_\mathrm{l}^T z_\mathrm{l} = 0

        where :math:`\tilde f = (f_1,\ldots, f_m)`.


    :const:`'unknown'`
        This indicates that the algorithm terminated before a solution was
        found, due to numerical difficulties or because the maximum number
        of iterations was reached.  The :const:`'x'`, :const:`'snl'`,
        :const:`'sl'`, :const:`'y'`, :const:`'znl'`, and :const:`'zl'`
        entries contain the iterates when the algorithm terminated.

    :func:`cp` solves the problem by applying
    :func:`cpl <cvxopt.solvers.cpl>` to the epigraph
    form problem

    .. math::
        \begin{array}{ll}
        \mbox{minimize}   & t \\
        \mbox{subject to} & f_0(x) \leq t  \\
                          & f_k(x) \leq 0, \quad k =1, \ldots, m \\
                          & Gx \preceq h \\
                          & Ax = b.
        \end{array}

    The other entries in the output dictionary of :func:`cp` describe
    the accuracy of the solution and are copied from the output of
    :func:`cpl <cvxopt.solvers.cpl>` applied to this epigraph form
    problem.

    :func:`cp` requires that the problem is strictly primal and dual
    feasible and that

    .. math::

        \newcommand{\Rank}{\mathop{\bf rank}}
        \Rank(A) = p, \qquad
        \Rank \left( \left[ \begin{array}{cccccc}
             \sum_{k=0}^m z_k \nabla^2 f_k(x) & A^T &
              \nabla f_1(x) & \cdots \nabla f_m(x) & G^T
              \end{array} \right] \right) = n,

    for all :math:`x` and all positive :math:`z`.


**Example: equality constrained analytic centering**
    The equality constrained analytic centering problem is defined as

    .. math::

        \begin{array}{ll}
        \mbox{minimize} & -\sum\limits_{i=1}^m \log x_i \\
        \mbox{subject to} & Ax = b.
        \end{array}

    The function :func:`acent` defined  below solves the problem, assuming
    it is solvable.

    ::

        from cvxopt import solvers, matrix, spdiag, log

        def acent(A, b):
            m, n = A.size
            def F(x=None, z=None):
                if x is None: return 0, matrix(1.0, (n,1))
                if min(x) <= 0.0: return None
                f = -sum(log(x))
                Df = -(x**-1).T
                if z is None: return f, Df
                H = spdiag(z[0] * x**-2)
                return f, Df, H
            return solvers.cp(F, A=A, b=b)['x']


**Example: robust least-squares**
    The function :func:`robls` defined below solves the unconstrained
    problem

    .. math::

        \begin{array}{ll}
        \mbox{minimize} &  \sum\limits_{k=1}^m \phi((Ax-b)_k),
        \end{array}
        \qquad \phi(u) = \sqrt{\rho + u^2},

    where :math:`A \in\mathbf{R}^{m\times n}`.

    ::

        from cvxopt import solvers, matrix, spdiag, sqrt, div

        def robls(A, b, rho):
            m, n = A.size
            def F(x=None, z=None):
                if x is None: return 0, matrix(0.0, (n,1))
                y = A*x-b
                w = sqrt(rho + y**2)
                f = sum(w)
                Df = div(y, w).T * A
                if z is None: return f, Df
                H = A.T * spdiag(z[0]*rho*(w**-3)) * A
                return f, Df, H
            return solvers.cp(F)['x']


**Example: analytic centering with cone constraints**

    .. math::

         \begin{array}{ll}
         \mbox{minimize}
             & -\log(1-x_1^2) -\log(1-x_2^2) -\log(1-x_3^2) \\
         \mbox{subject to}
             & \|x\|_2 \leq 1 \\
             & x_1 \left[\begin{array}{rrr}
                   -21 & -11 & 0 \\ -11 & 10 & 8 \\ 0 & 8 & 5
                    \end{array}\right] +
               x_2 \left[\begin{array}{rrr}
                    0 & 10 & 16 \\ 10 & -10 & -10 \\ 16 & -10 & 3
                   \end{array}\right] +
               x_3 \left[\begin{array}{rrr}
                   -5 & 2 & -17 \\ 2 & -6 & 8 \\ -17 & -7 & 6
                   \end{array}\right]
               \preceq \left[\begin{array}{rrr}
                   20 & 10 & 40 \\ 10 & 80 & 10 \\ 40 & 10 & 15
                   \end{array}\right].
         \end{array}

    ::

        from cvxopt import matrix, log, div, spdiag, solvers

        def F(x = None, z = None):
             if x is None:  return 0, matrix(0.0, (3,1))
             if max(abs(x)) >= 1.0:  return None
             u = 1 - x**2
             val = -sum(log(u))
             Df = div(2*x, u).T
             if z is None:  return val, Df
             H = spdiag(2 * z[0] * div(1 + x**2, u**2))
             return val, Df, H

        G = matrix([ [0., -1.,  0.,  0., -21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.],
                     [0.,  0., -1.,  0.,   0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.],
                     [0.,  0.,  0., -1.,  -5.,   2., -17.,   2.,  -6.,   8., -17.,  -7., 6.] ])
        h = matrix([1.0, 0.0, 0.0, 0.0, 20., 10., 40., 10., 80., 10., 40., 10., 15.])
        dims = {'l': 0, 'q': [4], 's':  [3]}
        sol = solvers.cp(F, G, h, dims)
        print(sol['x'])
        [ 4.11e-01]
        [ 5.59e-01]
        [-7.20e-01]


.. _s-cpl:

Problems with Linear Objectives
===============================

.. function:: cvxopt.solvers.cpl(c, F[, G, h[, dims[, A, b[, kktsolver]]]])

    Solves a convex optimization problem with a linear objective

    .. math::

        \begin{array}{ll}
        \mbox{minimize}   & c^T x \\
        \mbox{subject to} & f_k(x) \leq 0, \quad k=0,\ldots,m-1 \\
                          & G x \preceq h  \\
                          & A x = b.
        \end{array}

    ``c`` is a real single-column dense matrix.

    ``F`` is a function that evaluates the nonlinear constraint functions.
    It must handle the following calling sequences.

    * ``F()`` returns a tuple (``m``, ``x0``), where ``m`` is the
      number of nonlinear constraints and ``x0`` is a point in the domain
      of :math:`f`.  ``x0`` is a dense real matrix of size (:math:`n`, 1).

    * ``F(x)``, with ``x`` a dense real matrix of size (:math:`n`, 1),
      returns a tuple (``f``, ``Df``).  ``f`` is a dense real matrix of
      size (:math:`m`, 1), with ``f[k]`` equal to :math:`f_k(x)`.
      ``Df`` is a dense or sparse real matrix of size (:math:`m`,
      :math:`n`) with ``Df[k,:]`` equal to the transpose of the
      gradient :math:`\nabla f_k(x)`.  If :math:`x` is not in the domain
      of :math:`f`, ``F(x)`` returns :const:`None` or a tuple
      (:const:`None`, :const:`None`).

    * ``F(x,z)``, with ``x`` a dense real matrix of size (:math:`n`, 1)
      and ``z`` a positive dense real matrix of size (:math:`m`, 1)
      returns a tuple (``f``, ``Df``, ``H``).  ``f`` and ``Df`` are defined
      as above.  ``H`` is a square dense or sparse real matrix of size
      (:math:`n`, :math:`n`), whose lower triangular part contains the
      lower triangular part of

      .. math::

          z_0 \nabla^2f_0(x) + z_1 \nabla^2f_1(x) + \cdots +
          z_{m-1} \nabla^2f_{m-1}(x).

      If ``F`` is called with two arguments, it can be assumed that
      :math:`x` is in the domain of :math:`f`.

    The linear inequalities are with respect to a cone :math:`C` defined as
    a Cartesian product of a nonnegative orthant, a number of second-order
    cones, and a number of positive semidefinite cones:

    .. math::

        C = C_0 \times C_1 \times \cdots \times C_M \times C_{M+1} \times
        \cdots \times C_{M+N}

    with

    .. math::

        \newcommand{\reals}{{\mbox{\bf R}}}
        \newcommand{\svec}{\mathop{\mathbf{vec}}}
        \newcommand{\symm}{{\mbox{\bf S}}}
        \begin{split}
        C_0 &= \{ u \in \reals^l \;| \; u_k \geq 0, \; k=1, \ldots,l\}, \\
        C_{k+1} &= \{ (u_0, u_1) \in \reals \times \reals^{r_{k}-1} \; | \;
            u_0 \geq \|u_1\|_2 \},  \quad k=0,\ldots, M-1, \\
        C_{k+M+1} &= \left\{ \svec(u) \; | \;
            u \in \symm^{t_k}_+ \right\}, \quad k=0,\ldots,N-1.
        \end{split}

    Here :math:`\mathbf{vec}(u)` denotes a symmetric matrix :math:`u`
    stored as a vector in column major order.

    The arguments ``h`` and ``b`` are real single-column dense matrices.
    ``G`` and ``A`` are real dense or sparse matrices.  The default values
    for ``A`` and ``b`` are sparse matrices with zero rows, meaning that
    there are no equality constraints.  The number of rows of ``G`` and
    ``h`` is equal to

    .. math::

        K = l + \sum_{k=0}^{M-1} r_k + \sum_{k=0}^{N-1} t_k^2.

    The columns of ``G`` and ``h`` are vectors in

    .. math::

        \newcommand{\reals}{{\mbox{\bf R}}}
        \reals^l \times \reals^{r_0} \times \cdots \times
        \reals^{r_{M-1}} \times \reals^{t_0^2}  \times \cdots \times
        \reals^{t_{N-1}^2},

    where the last :math:`N` components represent symmetric matrices stored
    in column major order.  The strictly upper triangular entries of these
    matrices are not accessed (i.e., the symmetric matrices are stored
    in the :const:`'L'`-type column major order used in the :mod:`blas` and
    :mod:`lapack` modules.

    The argument ``dims`` is a dictionary with the dimensions of the cones.
    It has three fields.

    ``dims['l']``:
        :math:`l`, the dimension of the nonnegative orthant (a nonnegative
        integer).

    ``dims['q']``:
        :math:`[r_0, \ldots, r_{M-1}]`, a list with the dimensions of the
        second-order cones (positive integers).

    ``dims['s']``:
        :math:`[t_0, \ldots, t_{N-1}]`, a list with the dimensions of the
        positive semidefinite cones (nonnegative integers).

    The default value of ``dims`` is
    ``{'l': h.size[0], 'q': [], 's': []}``, i.e., the default
    assumption is that the linear inequalities are componentwise
    inequalities.

    The role of the optional argument ``kktsolver`` is explained in the
    section :ref:`s-nlcp`.

    :func:`cpl` returns a dictionary that contains the result and
    information about the accuracy of the solution.  The most important
    fields have keys :const:`'status'`, :const:`'x'`, :const:`'snl'`,
    :const:`'sl'`, :const:`'y'`, :const:`'znl'`, :const:`'zl'`.
    The possible values of the :const:`'status'` key are:

    :const:`'optimal'`
        In this case the :const:`'x'` entry of the dictionary is the primal
        optimal solution, the :const:`'snl'` and :const:`'sl'` entries are
        the corresponding slacks in the nonlinear and linear inequality
        constraints, and the :const:`'znl'`, :const:`'zl'`, and
        :const:`'y'` entries are the optimal values of the dual variables
        associated with the nonlinear inequalities, the linear
        inequalities, and the linear equality constraints.  These vectors
        approximately satisfy the Karush-Kuhn-Tucker (KKT) conditions

        .. math::

            c +  Df(x)^T z_\mathrm{nl} + G^T z_\mathrm{l} + A^T y = 0,

            f(x) + s_\mathrm{nl} = 0, \quad k=1,\ldots,m, \qquad
                Gx + s_\mathrm{l} = h, \qquad Ax = b,


            s_\mathrm{nl}\succeq 0, \qquad s_\mathrm{l}\succeq 0, \qquad
                z_\mathrm{nl} \succeq 0, \qquad z_\mathrm{l} \succeq 0,

            s_\mathrm{nl}^T z_\mathrm{nl} +  s_\mathrm{l}^T z_\mathrm{l}
                = 0.

    :const:`'unknown'`
        This indicates that the algorithm terminated before a solution was
        found, due to numerical difficulties or because the maximum number
        of iterations was reached.  The :const:`'x'`, :const:`'snl'`,
        :const:`'sl'`, :const:`'y'`, :const:`'znl'`, and :const:`'zl'`
        entries contain the iterates when the algorithm terminated.

    The other entries in the output dictionary describe the accuracy
    of the solution.  The entries :const:`'primal objective'`,
    :const:`'dual objective'`, :const:`'gap'`, and :const:`'relative gap'`     give the primal objective :math:`c^Tx`, the dual objective, calculated
    as

    .. math::

        c^Tx + z_\mathrm{nl}^T f(x) + z_\mathrm{l}^T (Gx - h) + y^T(Ax-b),

    the duality gap

    .. math::

        s_\mathrm{nl}^T z_\mathrm{nl} +  s_\mathrm{l}^T z_\mathrm{l},

    and the relative gap.  The relative gap is defined as

    .. math::

        \frac{\mbox{gap}}{-\mbox{primal objective}}
            \quad \mbox{if\ } \mbox{primal objective} < 0, \qquad
        \frac{\mbox{gap}}{\mbox{dual objective}}
            \quad \mbox{if\ } \mbox{dual objective} > 0,

    and :const:`None` otherwise.  The entry with key
    :const:`'primal infeasibility'` gives the residual in the primal
    constraints,

    .. math::

        \newcommand{\ones}{{\bf 1}}
        \frac{\| ( f(x) + s_{\mathrm{nl}},  Gx + s_\mathrm{l} - h,
            Ax-b ) \|_2} {\max\{1, \| ( f(x_0) + \ones,
        Gx_0 + \ones-h, Ax_0-b) \|_2 \}}

    where :math:`x_0` is the point returned by ``F()``.  The entry
    with key :const:`'dual infeasibility'` gives the residual

    .. math::

        \newcommand{\ones}{{\bf 1}}
        \frac
            { \| c +  Df(x)^Tz_\mathrm{nl} + G^Tz_\mathrm{l} + A^T y \|_2}
            { \max\{ 1, \| c + Df(x_0)^T\ones + G^T\ones \|_2 \} }.

    :func:`cpl` requires that the problem is strictly primal and dual
    feasible and that

    .. math::

        \newcommand{\Rank}{\mathop{\bf rank}}
        \Rank(A) = p, \qquad
        \Rank\left(\left[\begin{array}{cccccc}
            \sum_{k=0}^{m-1} z_k \nabla^2 f_k(x) & A^T &
            \nabla f_0(x) & \cdots \nabla f_{m-1}(x) & G^T
            \end{array}\right]\right) = n,

    for all :math:`x` and all positive :math:`z`.


**Example: floor planning**
    This example is the floor planning problem of section 8.8.2 in the book
    `Convex Optimization <http://www.stanford.edu/~boyd/cvxbook>`_:

    .. math::

        \begin{array}{ll}
        \mbox{minimize}    & W + H \\
        \mbox{subject to}
            & A_{\mathrm{min}, k}/h_k - w_k \leq 0, \quad k=1,\ldots, 5 \\
            & x_1 \geq 0, \quad x_2 \geq 0,  \quad x_4 \geq 0 \\
            & x_1 + w_1 + \rho \leq x_3, \quad x_2 + w_2 + \rho \leq x_3,
              \quad x_3 + w_3 + \rho \leq x_5,  \\
            & x_4 + w_4 + \rho \leq x_5, \quad x_5 + w_5 \leq W \\
            & y_2 \geq 0,  \quad y_3 \geq 0, \quad y_5 \geq 0  \\
            & y_2 + h_2 + \rho \leq y_1, \quad y_1 + h_1 + \rho \leq y_4,
              y_3 + h_3 + \rho \leq y_4, \\
            & y_4 + h_4 \leq H, \quad y_5 + h_5 \leq H \\
            & h_k/\gamma \leq w_k \leq \gamma h_k, \quad k=1,\ldots,5.
        \end{array}

    This problem has 22 variables

    .. math::

        \newcommand{\reals}{{\mbox{\bf R}}}
        W, \qquad H, \qquad x\in\reals^5, \qquad y\in\reals^5, \qquad
        w\in\reals^5, \qquad h\in\reals^5,

    5 nonlinear inequality constraints, and 26 linear inequality
    constraints.  The code belows defines a function :func:`floorplan`
    that solves the problem by calling :func:`cp`, then applies it to
    4 instances, and creates a figure.

    ::

        import pylab
        from cvxopt import solvers, matrix, spmatrix, mul, div

        def floorplan(Amin):

            #     minimize    W+H
            #     subject to  Amink / hk <= wk, k = 1,..., 5
            #                 x1 >= 0,  x2 >= 0, x4 >= 0
            #                 x1 + w1 + rho <= x3
            #                 x2 + w2 + rho <= x3
            #                 x3 + w3 + rho <= x5
            #                 x4 + w4 + rho <= x5
            #                 x5 + w5 <= W
            #                 y2 >= 0,  y3 >= 0,  y5 >= 0
            #                 y2 + h2 + rho <= y1
            #                 y1 + h1 + rho <= y4
            #                 y3 + h3 + rho <= y4
            #                 y4 + h4 <= H
            #                 y5 + h5 <= H
            #                 hk/gamma <= wk <= gamma*hk,  k = 1, ..., 5
            #
            # 22 Variables W, H, x (5), y (5), w (5), h (5).
            #
            # W, H:  scalars; bounding box width and height
            # x, y:  5-vectors; coordinates of bottom left corners of blocks
            # w, h:  5-vectors; widths and heigths of the 5 blocks

            rho, gamma = 1.0, 5.0   # min spacing, min aspect ratio

            # The objective is to minimize W + H.  There are five nonlinear
            # constraints
            #
            #     -wk + Amink / hk <= 0,  k = 1, ..., 5

            c = matrix(2*[1.0] + 20*[0.0])

            def F(x=None, z=None):
                if x is None:  return 5, matrix(17*[0.0] + 5*[1.0])
                if min(x[17:]) <= 0.0:  return None
                f = -x[12:17] + div(Amin, x[17:])
                Df = matrix(0.0, (5,22))
                Df[:,12:17] = spmatrix(-1.0, range(5), range(5))
                Df[:,17:] = spmatrix(-div(Amin, x[17:]**2), range(5), range(5))
                if z is None: return f, Df
                H = spmatrix( 2.0* mul(z, div(Amin, x[17::]**3)), range(17,22), range(17,22) )
                return f, Df, H

            G = matrix(0.0, (26,22))
            h = matrix(0.0, (26,1))
            G[0,2] = -1.0                                       # -x1 <= 0
            G[1,3] = -1.0                                       # -x2 <= 0
            G[2,5] = -1.0                                       # -x4 <= 0
            G[3, [2, 4, 12]], h[3] = [1.0, -1.0, 1.0], -rho     # x1 - x3 + w1 <= -rho
            G[4, [3, 4, 13]], h[4] = [1.0, -1.0, 1.0], -rho     # x2 - x3 + w2 <= -rho
            G[5, [4, 6, 14]], h[5] = [1.0, -1.0, 1.0], -rho     # x3 - x5 + w3 <= -rho
            G[6, [5, 6, 15]], h[6] = [1.0, -1.0, 1.0], -rho     # x4 - x5 + w4 <= -rho
            G[7, [0, 6, 16]] = -1.0, 1.0, 1.0                   # -W + x5 + w5 <= 0
            G[8,8] = -1.0                                       # -y2 <= 0
            G[9,9] = -1.0                                       # -y3 <= 0
            G[10,11] = -1.0                                     # -y5 <= 0
            G[11, [7, 8, 18]], h[11] = [-1.0, 1.0, 1.0], -rho   # -y1 + y2 + h2 <= -rho
            G[12, [7, 10, 17]], h[12] = [1.0, -1.0, 1.0], -rho  #  y1 - y4 + h1 <= -rho
            G[13, [9, 10, 19]], h[13] = [1.0, -1.0, 1.0], -rho  #  y3 - y4 + h3 <= -rho
            G[14, [1, 10, 20]] = -1.0, 1.0, 1.0                 # -H + y4 + h4 <= 0
            G[15, [1, 11, 21]] = -1.0, 1.0, 1.0                 # -H + y5 + h5 <= 0
            G[16, [12, 17]] = -1.0, 1.0/gamma                   # -w1 + h1/gamma <= 0
            G[17, [12, 17]] = 1.0, -gamma                       #  w1 - gamma * h1 <= 0
            G[18, [13, 18]] = -1.0, 1.0/gamma                   # -w2 + h2/gamma <= 0
            G[19, [13, 18]] = 1.0, -gamma                       #  w2 - gamma * h2 <= 0
            G[20, [14, 19]] = -1.0, 1.0/gamma                   # -w3 + h3/gamma <= 0
            G[21, [14, 19]] = 1.0, -gamma                       #  w3 - gamma * h3 <= 0
            G[22, [15, 20]] = -1.0, 1.0/gamma                   # -w4  + h4/gamma <= 0
            G[23, [15, 20]] = 1.0, -gamma                       #  w4 - gamma * h4 <= 0
            G[24, [16, 21]] = -1.0, 1.0/gamma                   # -w5 + h5/gamma <= 0
            G[25, [16, 21]] = 1.0, -gamma                       #  w5 - gamma * h5 <= 0.0

            # solve and return W, H, x, y, w, h
            sol = solvers.cpl(c, F, G, h)
            return  sol['x'][0], sol['x'][1], sol['x'][2:7], sol['x'][7:12], sol['x'][12:17], sol['x'][17:]

        pylab.figure(facecolor='w')
        pylab.subplot(221)
        Amin = matrix([100., 100., 100., 100., 100.])
        W, H, x, y, w, h =  floorplan(Amin)
        for k in range(5):
            pylab.fill([x[k], x[k], x[k]+w[k], x[k]+w[k]],
                       [y[k], y[k]+h[k], y[k]+h[k], y[k]], facecolor = '#D0D0D0')
            pylab.text(x[k]+.5*w[k], y[k]+.5*h[k], "%d" %(k+1))
        pylab.axis([-1.0, 26, -1.0, 26])
        pylab.xticks([])
        pylab.yticks([])

        pylab.subplot(222)
        Amin = matrix([20., 50., 80., 150., 200.])
        W, H, x, y, w, h =  floorplan(Amin)
        for k in range(5):
            pylab.fill([x[k], x[k], x[k]+w[k], x[k]+w[k]],
                       [y[k], y[k]+h[k], y[k]+h[k], y[k]], 'facecolor = #D0D0D0')
            pylab.text(x[k]+.5*w[k], y[k]+.5*h[k], "%d" %(k+1))
        pylab.axis([-1.0, 26, -1.0, 26])
        pylab.xticks([])
        pylab.yticks([])

        pylab.subplot(223)
        Amin = matrix([180., 80., 80., 80., 80.])
        W, H, x, y, w, h =  floorplan(Amin)
        for k in range(5):
            pylab.fill([x[k], x[k], x[k]+w[k], x[k]+w[k]],
                       [y[k], y[k]+h[k], y[k]+h[k], y[k]], 'facecolor = #D0D0D0')
            pylab.text(x[k]+.5*w[k], y[k]+.5*h[k], "%d" %(k+1))
        pylab.axis([-1.0, 26, -1.0, 26])
        pylab.xticks([])
        pylab.yticks([])

        pylab.subplot(224)
        Amin = matrix([20., 150., 20., 200., 110.])
        W, H, x, y, w, h =  floorplan(Amin)
        for k in range(5):
            pylab.fill([x[k], x[k], x[k]+w[k], x[k]+w[k]],
                       [y[k], y[k]+h[k], y[k]+h[k], y[k]], 'facecolor = #D0D0D0')
            pylab.text(x[k]+.5*w[k], y[k]+.5*h[k], "%d" %(k+1))
        pylab.axis([-1.0, 26, -1.0, 26])
        pylab.xticks([])
        pylab.yticks([])

        pylab.show()


    .. image:: floorplan.png
       :width: 600px


.. _s-gp:

Geometric Programming
=====================

.. function:: cvxopt.solvers.gp(K, F, g[, G, h[, A, b]])

    Solves a geometric program in convex form

    .. math::

        \newcommand{\lse}{\mathop{\mathbf{lse}}}
        \begin{array}{ll}
        \mbox{minimize}   & f_0(x) = \lse(F_0x+g_0) \\
        \mbox{subject to} & f_i(x) = \lse(F_ix+g_i) \leq 0,
                            \quad i=1,\ldots,m \\
                          & Gx \preceq h \\
                          & Ax=b
        \end{array}

    where

    .. math::

        \newcommand{\lse}{\mathop{\mathbf{lse}}}
        \lse(u) = \log \sum_k \exp(u_k), \qquad
        F = \left[ \begin{array}{cccc}
             F_0^T & F_1^T & \cdots & F_m^T
            \end{array}\right]^T, \qquad
        g = \left[ \begin{array}{cccc}
             g_0^T & g_1^T & \cdots & g_m^T
            \end{array}\right]^T,

    and the vector inequality denotes componentwise inequality.
    ``K`` is a list of :math:`m` + 1 positive integers with ``K[i]``
    equal to the number of rows in :math:`F_i`.  ``F`` is a dense or
    sparse real matrix of size ``(sum(K), n)``.
    ``g`` is a dense real matrix with one column and the same number of
    rows as ``F``.
    ``G`` and ``A`` are dense or sparse real matrices.  Their default
    values are sparse matrices with zero rows.
    ``h`` and ``b`` are dense real matrices with one column.  Their
    default values are matrices of size (0, 1).

    :func:`gp` returns a dictionary with keys :const:`'status'`,
    :const:`'x'`, :const:`'snl'`, :const:`'sl'`, :const:`'y'`,
    :const:`'znl'`, and :const:`'zl'`.  The possible values of the
    :const:`'status'` key are:

    :const:`'optimal'`
        In this case the :const:`'x'` entry is the primal optimal solution,
        the :const:`'snl'` and :const:`'sl'` entries are the corresponding
        slacks in the nonlinear and linear inequality constraints.  The
        :const:`'znl'`, :const:`'zl'`, and :const:`'y'` entries are the
        optimal values of the dual variables associated with the nonlinear
        and linear inequality constraints and the linear equality
        constraints.  These values approximately satisfy

        .. math::

            \nabla f_0(x) + \sum_{k=1}^m z_{\mathrm{nl},k}
                \nabla f_k(x) + G^T z_\mathrm{l} + A^T y = 0,

            f_k(x) + s_{\mathrm{nl},k} = 0, \quad k = 1,\ldots,m
            \qquad Gx + s_\mathrm{l} = h, \qquad Ax = b,

            s_\mathrm{nl}\succeq 0, \qquad s_\mathrm{l}\succeq 0, \qquad
            z_\mathrm{nl} \succeq 0, \qquad z_\mathrm{l} \succeq 0,

            s_\mathrm{nl}^T z_\mathrm{nl} + s_\mathrm{l}^T z_\mathrm{l} =0.


    :const:`'unknown'`
        This indicates that the algorithm terminated before a solution was
        found, due to numerical difficulties or because the maximum number
        of iterations was reached.  The :const:`'x'`, :const:`'snl'`,
        :const:`'sl'`, :const:`'y'`, :const:`'znl'`, and :const:`'zl'`
        contain the iterates when the algorithm terminated.

    The other entries in the output dictionary describe the accuracy
    of the solution, and are taken from the output of
    :func:`cp <cvxopt.solvers.cp>`.

    :func:`gp` requires that the problem is strictly primal and dual
    feasible and that

    .. math::

        \newcommand{\Rank}{\mathop{\bf rank}}
        \Rank(A) = p, \qquad
        \Rank \left( \left[ \begin{array}{cccccc}
             \sum_{k=0}^m z_k \nabla^2 f_k(x) & A^T &
              \nabla f_1(x) & \cdots \nabla f_m(x) & G^T
              \end{array} \right] \right) = n,

    for all :math:`x` and all positive :math:`z`.

As an example, we solve the small GP of section 2.4 of the paper
`A Tutorial on Geometric Programming
<http://www.stanford.edu/~boyd/gp_tutorial.html>`_.
The  posynomial form of the problem is

.. math::

    \begin{array}{ll}
    \mbox{minimize}
        & w^{-1} h^{-1} d^{-1} \\
    \mbox{subject to}
        & (2/A_\mathrm{wall}) hw + (2/A_\mathrm{wall})hd \leq 1  \\
        &  (1/A_\mathrm{flr}) wd \leq 1 \\
        &  \alpha wh^{-1} \leq 1 \\
        &  (1/\beta) hw^{-1} \leq 1 \\
        &  \gamma wd^{-1} \leq 1 \\
        &   (1/\delta)dw^{-1} \leq 1
     \end{array}

with variables :math:`h`, :math:`w`, :math:`d`.

::

    from cvxopt import matrix, log, exp, solvers

    Aflr  = 1000.0
    Awall = 100.0
    alpha = 0.5
    beta  = 2.0
    gamma = 0.5
    delta = 2.0

    F = matrix( [[-1., 1., 1., 0., -1.,  1.,  0.,  0.],
                 [-1., 1., 0., 1.,  1., -1.,  1., -1.],
                 [-1., 0., 1., 1.,  0.,  0., -1.,  1.]])
    g = log( matrix( [1.0, 2/Awall, 2/Awall, 1/Aflr, alpha, 1/beta, gamma, 1/delta]) )
    K = [1, 2, 1, 1, 1, 1, 1]
    h, w, d = exp( solvers.gp(K, F, g)['x'] )



.. _s-nlcp:

Exploiting Structure
====================

By default, the functions :func:`cp <cvxopt.solvers.cp>` and
:func:`cpl <cvxopt.solvers.cpl>` do not exploit problem
structure.  Two mechanisms are provided for implementing customized solvers
that take advantage of problem structure.

**Providing a function for solving KKT equations**
    The most expensive step of each iteration of
    :func:`cp <cvxopt.solvers.cp>` is the
    solution of a set of linear equations (*KKT equations*) of the form

    .. math::
        :label: e-cp-kkt

        \left[\begin{array}{ccc}
            H        & A^T & \tilde G^T \\
            A        & 0   & 0  \\
            \tilde G & 0   & -W^T W
        \end{array}\right]
        \left[\begin{array}{c} u_x \\ u_y \\ u_z \end{array}\right]
        =
        \left[\begin{array}{c} b_x \\ b_y \\ b_z \end{array}\right],

    where

    .. math::

        H = \sum_{k=0}^m z_k \nabla^2f_k(x), \qquad
        \tilde G = \left[\begin{array}{cccc}
        \nabla f_1(x) & \cdots & \nabla f_m(x) & G^T \end{array}\right]^T.

    The matrix :math:`W` depends on the current iterates and is defined as
    follows.  Suppose

    .. math::

        \newcommand{\svec}{\mathop{\mathbf{vec}}}
        u = \left( u_\mathrm{nl}, \; u_\mathrm{l}, \; u_{\mathrm{q},0}, \;
            \ldots, \; u_{\mathrm{q},M-1}, \; \svec{(u_{\mathrm{s},0})}, \;
            \ldots, \; \svec{(u_{\mathrm{s},N-1})} \right), \qquad

    where

    .. math::

        \newcommand{\reals}{{\mbox{\bf R}}}
        \newcommand{\symm}{{\mbox{\bf S}}}
        u_\mathrm{nl} \in \reals^m, \qquad
        u_\mathrm{l} \in \reals^l, \qquad
        u_{\mathrm{q},k} \in \reals^{r_k}, \quad k = 0, \ldots, M-1,
        \qquad
        u_{\mathrm{s},k} \in \symm^{t_k},  \quad k = 0, \ldots, N-1.

    Then :math:`W` is a block-diagonal matrix,

    .. math::

        \newcommand{\svec}{\mathop{\mathbf{vec}}}
        Wu = \left( W_\mathrm{nl} u_\mathrm{nl}, \;
             W_\mathrm{l} u_\mathrm{l}, \;
             W_{\mathrm{q},0} u_{\mathrm{q},0}, \; \ldots, \;
             W_{\mathrm{q},M-1} u_{\mathrm{q},M-1},\;
             W_{\mathrm{s},0} \svec{(u_{\mathrm{s},0})}, \; \ldots, \;
             W_{\mathrm{s},N-1} \svec{(u_{\mathrm{s},N-1})} \right)

    with the following diagonal blocks.

    * The first block is a *positive diagonal scaling* with a vector
      :math:`d_{\mathrm{nl}}`:

      .. math::

          \newcommand{\diag}{\mbox{\bf diag}\,}
          W_\mathrm{nl} = \diag(d_\mathrm{nl}), \qquad
          W_\mathrm{nl}^{-1} = \diag(d_\mathrm{nl})^{-1}.

      This transformation is symmetric:

      .. math::

          W_\mathrm{nl}^T = W_\mathrm{nl}.

    * The second block is a *positive diagonal scaling* with a vector
      :math:`d_{\mathrm{l}}`:

      .. math::

          \newcommand{\diag}{\mbox{\bf diag}\,}
          W_\mathrm{l} = \diag(d_\mathrm{l}), \qquad
          W_\mathrm{l}^{-1} = \diag(d_\mathrm{l})^{-1}.

      This transformation is symmetric:

      .. math::

          W_\mathrm{l}^T = W_\mathrm{l}.

    * The next :math:`M` blocks are positive multiples of *hyperbolic
      Householder transformations*:

      .. math::

         W_{\mathrm{q},k} = \beta_k ( 2 v_k v_k^T - J), \qquad
         W_{\mathrm{q},k}^{-1} = \frac{1}{\beta_k} ( 2 Jv_k v_k^T J - J),
         \qquad k = 0,\ldots,M-1,

      where

      .. math::

         \beta_k > 0, \qquad v_{k0} > 0, \qquad v_k^T Jv_k = 1, \qquad
         J = \left[\begin{array}{cc} 1 & 0 \\ 0 & -I \end{array}\right].

      These transformations are also symmetric:

      .. math::

         W_{\mathrm{q},k}^T = W_{\mathrm{q},k}.


    * The last :math:`N` blocks are *congruence transformations* with
      nonsingular matrices:

      .. math::

          \newcommand{\svec}{\mathop{\mathbf{vec}}}
          W_{\mathrm{s},k} \svec{(u_{\mathrm{s},k})} =
              \svec{(r_k^T u_{\mathrm{s},k} r_k)}, \qquad
          W_{\mathrm{s},k}^{-1} \svec{(u_{\mathrm{s},k})} =
             \svec{(r_k^{-T} u_{\mathrm{s},k} r_k^{-1})}, \qquad
          k = 0,\ldots,N-1.

      In  general, this operation is not symmetric, and

      .. math::

          \newcommand{\svec}{\mathop{\mathbf{vec}}}
          W_{\mathrm{s},k}^T \svec{(u_{\mathrm{s},k})} =
              \svec{(r_k u_{\mathrm{s},k} r_k^T)}, \qquad
          \qquad
          W_{\mathrm{s},k}^{-T} \svec{(u_{\mathrm{s},k})} =
              \svec{(r_k^{-1} u_{\mathrm{s},k} r_k^{-T})}, \qquad
          k = 0,\ldots,N-1.


    It is often possible to exploit problem structure to solve
    :eq:`e-cp-kkt` faster than by standard methods.  The last argument
    ``kktsolver`` of :func:`cp <cvxopt.solvers.cp>` allows the user to
    supply a Python function
    for solving the KKT equations.  This function will be called as
    ``f = kktsolver(x, z, W)``.  The argument ``x`` is the point at
    which the derivatives in the KKT matrix are evaluated.  ``z`` is a
    positive vector of length it :math:`m` + 1, containing the coefficients
    in the 1,1 block :math:`H`.  ``W`` is a dictionary that contains the
    parameters of the scaling:

    * ``W['dnl']`` is the positive vector that defines the diagonal
      scaling for the nonlinear inequalities.  ``W['dnli']`` is its
      componentwise inverse.
    * ``W['d']`` is the positive vector that defines the diagonal
      scaling for the componentwise linear inequalities.  ``W['di']``
      is its componentwise inverse.
    * ``W['beta']`` and ``W['v']`` are lists of length :math:`M`
      with the coefficients and vectors that define the hyperbolic
      Householder transformations.
    * ``W['r']`` is a list of length :math:`N` with the matrices that
      define the the congruence transformations.  ``W['rti']`` is a
      list of length :math:`N` with the transposes of the inverses of the
      matrices in ``W['r']``.

    The function call ``f = kktsolver(x, z, W)`` should return a
    routine for solving the KKT system :eq:`e-cp-kkt` defined by ``x``,
    ``z``, ``W``.  It will be called as ``f(bx, by, bz)``.
    On entry, ``bx``, ``by``, ``bz`` contain the right-hand side.  On exit,
    they should contain the solution of the KKT system, with the last
    component scaled, i.e., on exit,

    .. math::

        b_x := u_x, \qquad b_y := u_y, \qquad b_z := W u_z.

    The role of the argument ``kktsolver`` in the function
    :func:`cpl <cvxopt.solvers.cpl>` is similar, except that in
    :eq:`e-cp-kkt`,

    .. math::

     H = \sum_{k=0}^{m-1} z_k \nabla^2f_k(x), \qquad
     \tilde G = \left[\begin{array}{cccc}
     \nabla f_0(x) & \cdots & \nabla f_{m-1}(x) & G^T \end{array}\right]^T.


**Specifying constraints via Python functions**
    In the default use of :func:`cp <cvxopt.solvers.cp>`, the arguments
    ``G`` and ``A`` are the
    coefficient matrices in the constraints of :eq:`e-cp-kkt`.  It is also
    possible to specify these matrices by providing Python functions that
    evaluate the corresponding matrix-vector products and their adjoints.

    * If the argument ``G`` of :func:`cp` is a Python function, then
      ``G(u, v[, alpha = 1.0, beta = 0.0, trans = 'N'])`` should
      evaluates the matrix-vector products

        .. math::

            v := \alpha Gu + \beta v \quad
                (\mathrm{trans} = \mathrm{'N'}), \qquad
            v := \alpha G^T u + \beta v \quad
               (\mathrm{trans} = \mathrm{'T'}).


    * Similarly, if the argument ``A`` is a Python function, then
      ``A(u, v[, alpha = 1.0, beta = 0.0, trans = 'N'])`` should
      evaluate the matrix-vector products

        .. math::

           v \alpha Au + \beta v \quad
               (\mathrm{trans} = \mathrm{'N'}), \qquad
           v := \alpha A^T u + \beta v \quad
               (\mathrm{trans} = \mathrm{'T'}).

    * In a similar way, when the first argument ``F`` of
      :func:`cp <cvxopt.solvers.cp>` returns matrices of first
      derivatives or second derivatives ``Df``, ``H``, these matrices can
      be specified as Python functions.  If ``Df`` is a Python function,
      then ``Df(u, v[, alpha = 1.0, beta = 0.0, trans = 'N'])`` should
      evaluate the matrix-vector products

        .. math::

            v := \alpha Df(x) u + \beta v \quad
                (\mathrm{trans} = \mathrm{'N'}), \qquad
            v := \alpha Df(x)^T u + \beta v \quad
                (\mathrm{trans} = \mathrm{'T'}).

      If ``H`` is a Python function, then ``H(u, v[, alpha, beta])`` should
      evaluate the matrix-vector product

        .. math::

            v := \alpha H u + \beta v.

    If ``G``, ``A``, ``Df``, or ``H`` are Python functions, then the
    argument ``kktsolver`` must also be provided.


As an example, we consider the unconstrained problem

.. math::

    \begin{array}{ll}
    \mbox{minimize} & (1/2)\|Ax-b\|_2^2 - \sum_{i=1}^n \log(1-x_i^2)
    \end{array}

where :math:`A` is an :math:`m` by :math:`n` matrix with :math:`m` less
than :math:`n`.  The Hessian of the objective is diagonal plus a low-rank
term:

.. math::

    \newcommand{\diag}{\mbox{\bf diag}\,}
    H = A^TA + \diag(d), \qquad d_i = \frac{2(1+x_i^2)}{(1-x_i^2)^2}.

We can exploit this property when solving :eq:`e-cp-kkt` by applying
the matrix inversion lemma. We first solve

.. math::

    \newcommand{\diag}{\mbox{\bf diag}\,}
    (A \diag(d)^{-1}A^T + I) v = (1/z_0) A \diag(d)^{-1}b_x, \qquad

and then obtain

.. math::

    \newcommand{\diag}{\mbox{\bf diag}\,}
    u_x = \diag(d)^{-1}(b_x/z_0 - A^T v).

The following code follows this method.  It also uses BLAS functions
for matrix-matrix and matrix-vector products.

::

    from cvxopt import matrix, spdiag, mul, div, log, blas, lapack, solvers, base

    def l2ac(A, b):
        """
        Solves

            minimize  (1/2) * ||A*x-b||_2^2 - sum log (1-xi^2)

        assuming A is m x n with m << n.
        """

        m, n = A.size
        def F(x = None, z = None):
            if x is None:
                return 0, matrix(0.0, (n,1))
            if max(abs(x)) >= 1.0:
                return None
            # r = A*x - b
            r = -b
            blas.gemv(A, x, r, beta = -1.0)
            w = x**2
            f = 0.5 * blas.nrm2(r)**2  - sum(log(1-w))
            # gradf = A'*r + 2.0 * x ./ (1-w)
            gradf = div(x, 1.0 - w)
            blas.gemv(A, r, gradf, trans = 'T', beta = 2.0)
            if z is None:
                return f, gradf.T
            else:
                def Hf(u, v, alpha = 1.0, beta = 0.0):
                   # v := alpha * (A'*A*u + 2*((1+w)./(1-w)).*u + beta *v
                   v *= beta
                   v += 2.0 * alpha * mul(div(1.0+w, (1.0-w)**2), u)
                   blas.gemv(A, u, r)
                   blas.gemv(A, r, v, alpha = alpha, beta = 1.0, trans = 'T')
                return f, gradf.T, Hf


        # Custom solver for the Newton system
        #
        #     z[0]*(A'*A + D)*x = bx
        #
        # where D = 2 * (1+x.^2) ./ (1-x.^2).^2.  We apply the matrix inversion
        # lemma and solve this as
        #
        #     (A * D^-1 *A' + I) * v = A * D^-1 * bx / z[0]
        #     D * x = bx / z[0] - A'*v.

        S = matrix(0.0, (m,m))
        v = matrix(0.0, (m,1))
        def Fkkt(x, z, W):
            ds = (2.0 * div(1 + x**2, (1 - x**2)**2))**-0.5
            Asc = A * spdiag(ds)
            blas.syrk(Asc, S)
            S[::m+1] += 1.0
            lapack.potrf(S)
            a = z[0]
            def g(x, y, z):
                x[:] = mul(x, ds) / a
                blas.gemv(Asc, x, v)
                lapack.potrs(S, v)
                blas.gemv(Asc, v, x, alpha = -1.0, beta = 1.0, trans = 'T')
                x[:] = mul(x, ds)
            return g

        return solvers.cp(F, kktsolver = Fkkt)['x']


.. _s-parameters2:

Algorithm Parameters
====================

The following algorithm control parameters are accessible via the
dictionary :attr:`solvers.options`.  By default the dictionary
is empty and the default values of the parameters are used.

One can change the parameters in the default solvers by
adding entries with the following key values.

:const:`'show_progress'`
    :const:`True` or :const:`False`; turns the output to the screen on or
    off (default: :const:`True`).

:const:`'maxiters'`
    maximum number of iterations (default: :const:`100`).

:const:`'abstol'`
    absolute accuracy (default: :const:`1e-7`).

:const:`'reltol'`
    relative accuracy (default: :const:`1e-6`).

:const:`'feastol'`
    tolerance for feasibility conditions (default: :const:`1e-7`).

:const:`'refinement'`
    number of iterative refinement steps when solving KKT equations
    (default: :const:`1`).

For example the command

>>> from cvxopt import solvers
>>> solvers.options['show_progress'] = False

turns off the screen output during calls to the solvers.  The tolerances
:const:`abstol`, :const:`reltol` and :const:`feastol` have the
following meaning in :func:`cpl <cvxopt.solvers.cpl>`.

:func:`cpl` returns with status :const:`'optimal'` if

.. math::

    \newcommand{\ones}{{\bf 1}}
    \frac{\| c +  Df(x)^Tz_\mathrm{nl} + G^Tz_\mathrm{l} + A^T y \|_2 }
    {\max\{ 1, \| c + Df(x_0)^T\ones + G^T\ones \|_2 \}}
    \leq \epsilon_\mathrm{feas}, \qquad
    \frac{\| ( f(x) + s_{\mathrm{nl}},  Gx + s_\mathrm{l} - h,
     Ax-b ) \|_2}
    {\max\{1, \| ( f(x_0) + \ones,
    Gx_0 + \ones-h, Ax_0-b) \|_2 \}} \leq \epsilon_\mathrm{feas}

where :math:`x_0` is the point returned by ``F()``, and

.. math::

    \mathrm{gap} \leq \epsilon_\mathrm{abs}
    \qquad \mbox{or} \qquad \left( c^Tx < 0, \quad
    \frac{\mathrm{gap}} {-c^Tx} \leq \epsilon_\mathrm{rel} \right)
    \qquad \mbox{or} \qquad
    \left( L(x,y,z) > 0, \quad \frac{\mathrm{gap}}
    {L(x,y,z)} \leq \epsilon_\mathrm{rel} \right)

where

.. math::

    \mathrm{gap} =
    \left[\begin{array}{c} s_\mathrm{nl} \\ s_\mathrm{l}
    \end{array}\right]^T
    \left[\begin{array}{c} z_\mathrm{nl} \\ z_\mathrm{l}
    \end{array}\right],
    \qquad
    L(x,y,z) = c^Tx + z_\mathrm{nl}^T f(x) +
        z_\mathrm{l}^T (Gx-h) + y^T(Ax-b).

The functions :func:`cp <cvxopt.solvers.cp>` and
:func:`gp <cvxopt.solvers.gp>` call :func:`cpl` and hence use the
same stopping criteria (with :math:`x_0 = 0` for :func:`gp`).
