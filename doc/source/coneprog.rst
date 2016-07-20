.. _c-coneprog:

****************
Cone Programming
****************

In this chapter we consider convex optimization problems of the form

.. math:: 

    \begin{array}{ll}
    \mbox{minimize}   & (1/2) x^TPx + q^T x \\
    \mbox{subject to} & G x \preceq h \\ 
                      & Ax = b.
    \end{array}

The linear inequality is a generalized inequality with respect to a 
proper convex cone.  It may include componentwise vector inequalities, 
second-order cone inequalities, and linear matrix inequalities.  

The main solvers are :func:`conelp <cvxopt.solvers.conelp>` and 
:func:`coneqp <cvxopt.solvers.coneqp>`, described in the
sections :ref:`s-conelp` and :ref:`s-coneqp`.  The function 
:func:`conelp` is restricted to problems with linear cost functions, and 
can detect primal and dual infeasibility.  The function :func:`coneqp` 
solves the general quadratic problem, but requires the problem to be 
strictly primal and dual feasible.  For convenience (and backward 
compatibility), simpler interfaces to these function are also provided 
that handle pure linear programs, quadratic programs, second-order cone 
programs, and semidefinite programs.  These are described in the sections
:ref:`s-lpsolver`, :ref:`s-qp`, :ref:`s-socpsolver`, :ref:`s-sdpsolver`.  
In the section :ref:`s-conelp-struct` we explain how custom solvers can 
be implemented that exploit structure in cone programs.  The last two 
sections describe optional interfaces to external solvers, and the 
algorithm parameters that control the cone programming solvers.


.. _s-conelp:

Linear Cone Programs 
====================

.. function:: cvxopt.solvers.conelp(c, G, h[, dims[, A, b[, primalstart[, dualstart[, kktsolver]]]]])

    Solves a pair of primal and dual cone programs

    .. math::
    
        \begin{array}[t]{ll}
        \mbox{minimize}   & c^T x \\
        \mbox{subject to} & G x + s = h \\ 
                          & Ax = b \\ 
                          & s \succeq 0
        \end{array}
        \qquad\qquad
        \begin{array}[t]{ll}
        \mbox{maximize}   & -h^T z - b^T y \\
        \mbox{subject to} & G^T z + A^T y + c = 0 \\
                          & z \succeq 0.
        \end{array}
   
    The primal variables are :math:`x` and :math:`s`.  The dual variables  
    are :math:`y`, :math:`z`.  The inequalities are interpreted as 
    :math:`s \in C`, :math:`z\in C`, where :math:`C` is a cone defined as 
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
            C_0 & = 
                \{ u \in \reals^l \;| \; u_k \geq 0, \; k=1, \ldots,l\}, \\
            C_{k+1} & = \{ (u_0, u_1) \in \reals \times \reals^{r_{k}-1} 
                \; | \; u_0 \geq \|u_1\|_2 \},  \quad k=0,\ldots, M-1, \\
            C_{k+M+1} &= \left\{ \svec(u) \; | \; u \in \symm^{t_k}_+ 
                \right\}, \quad k=0,\ldots,N-1.
        \end{split}

    In this definition, :math:`\mathbf{vec}(u)` denotes a symmetric matrix
    :math:`u` stored as a vector in column major order.  The structure of 
    :math:`C` is specified by ``dims``.  This argument is a dictionary with
    three fields. 
   
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
    ``{'l': G.size[0], 'q': [], 's': []}``, 
    i.e., by default the 
    inequality is interpreted as a componentwise vector inequality. 

    The arguments ``c``, ``h``, and ``b`` are real single-column dense 
    matrices.  ``G`` and ``A`` are real dense or sparse matrices.  The 
    number of rows of ``G`` and ``h`` is equal to

    .. math::
    
        K = l + \sum_{k=0}^{M-1} r_k + \sum_{k=0}^{N-1} t_k^2.
   
    The columns of ``G`` and ``h`` are vectors in

    .. math::

        \newcommand{\reals}{{\mbox{\bf R}}}
        \reals^l \times \reals^{r_0} \times \cdots \times 
        \reals^{r_{M-1}} \times \reals^{t_0^2}  \times \cdots \times 
        \reals^{t_{N-1}^2},

    where the last :math:`N` components represent symmetric matrices 
    stored in column major order.  The strictly upper triangular entries 
    of these matrices are not accessed (i.e.,  the symmetric matrices are 
    stored in the :const:`'L'`-type column major order used in the 
    :mod:`blas` and :mod:`lapack` modules).  The default values for ``A`` 
    and ``b`` are matrices with zero rows, meaning that there are no 
    equality constraints.  

    ``primalstart`` is a dictionary with keys :const:`'x'` and 
    :const:`'s'`, used as an optional primal starting point.   
    ``primalstart['x']`` and ``primalstart['s']`` are real dense 
    matrices of size (:math:`n`, 1) and (:math:`K`, 1), respectively, 
    where :math:`n` is the length of ``c``.  The vector 
    ``primalstart['s']`` must be strictly positive with respect
    to the cone :math:`C`.

    ``dualstart`` is a dictionary with keys :const:`'y'` and :const:`'z'`, 
    used as an optional dual starting point.  ``dualstart['y']`` and 
    ``dualstart['z']`` are real dense matrices of size (:math:`p`, 1) 
    and (:math:`K`, 1), respectively, where :math:`p` is the number of 
    rows in ``A``.  The vector ``dualstart['s']`` must be strictly 
    positive with respect to the cone :math:`C`.

    The role of the optional argument ``kktsolver`` is explained in 
    the section :ref:`s-conelp-struct`.  

    :func:`conelp` returns a dictionary that contains the result and 
    information about the accuracy of the solution.  The most important 
    fields have keys :const:`'status'`, :const:`'x'`, :const:`'s'`, 
    :const:`'y'`, :const:`'z'`.  The :const:`'status'` field  is a string 
    with possible values :const:`'optimal'`, :const:`'primal infeasible'`,
    :const:`'dual infeasible'`, and :const:`'unknown'`.  The meaning of 
    the :const:`'x'`, :const:`'s'`, :const:`'y'`, :const:`'z'` fields 
    depends on the value of :const:`'status'`.


    :const:`'optimal'` 
        In this case the :const:`'x'`, :const:`'s'`, :const:`'y'`, and 
        :const:`'z'` entries contain the primal and dual solutions, which 
        approximately satisfy

        .. math::
        
            Gx + s = h, \qquad Ax = b, \qquad G^T z  + A^T y + c = 0,
         
            s \succeq 0, \qquad z \succeq 0, \qquad  s^T z = 0.

        The other entries in the output dictionary summarize the accuracy
        with which these optimality conditions are satisfied.  The fields 
        :const:`'primal objective'`, :const:`'dual objective'`, and
        :const:`'gap'` give the primal objective :math:`c^Tx`, dual 
        objective :math:`-h^Tz - b^Ty`, and the gap :math:`s^Tz`.  The 
        field :const:`'relative gap'` is the relative gap, defined as
        
        .. math::

            \frac{ s^Tz }{ \max\{ -c^Tx, -h^Tz-b^Ty \} } 
            \quad \mbox{if} \quad \max\{ -c^Tx, -h^Tz-b^Ty \} > 0
        
        and :const:`None` otherwise.  The fields 
        :const:`'primal infeasibility'` and :const:`'dual infeasibility'` 
        are the residuals in the primal and dual equality constraints, 
        defined as

        .. math::
        
            \max\{ \frac{ \|Gx+s-h\|_2 }{ \max\{1, \|h\|_2\} }, 
                \frac{ \|Ax-b\|_2 }{ \max\{1,\|b\|_2\} } \}, \qquad
            \frac{ \|G^Tz + A^Ty + c\|_2 }{ \max\{1, \|c\|_2\} }, 
        
        respectively.

    :const:`'primal infeasible'`
        The :const:`'x'` and :const:`'s'` entries are :const:`None`, and 
        the :const:`'y'`, :const:`'z'` entries provide an approximate 
        certificate of infeasibility, i.e., vectors that approximately 
        satisfy

        .. math::
        
            G^T z + A^T y = 0, \qquad h^T z + b^T y = -1, \qquad 
            z \succeq 0.
        
        The field :const:`'residual as primal infeasibility certificate'` 
        gives the residual 

        .. math::
        
            \frac{ \|G^Tz + A^Ty\|_2 }{ \max\{1, \|c\|_2\} }.
        

    :const:`'dual infeasible'`  
        The :const:`'y'` and :const:`'z'` entries are :const:`None`, and 
        the :const:`'x'` and :const:`'s'` entries contain an approximate 
        certificate of dual infeasibility 

        .. math::
        
            Gx + s = 0, \qquad Ax=0, \qquad  c^T x = -1, \qquad 
            s \succeq 0.
        
        The field :const:`'residual as dual infeasibility certificate'` 
        gives the residual 

        .. math::
        
            \max\{ \frac{ \|Gx + s\|_2 }{ \max\{1, \|h\|_2\} },
            \frac{ \|Ax\|_2 }{ \max\{1, \|b\|_2\} } \}.
        

    :const:`'unknown'` 
        This indicates that the algorithm terminated early due to 
        numerical difficulties or because the maximum number of iterations 
        was reached.  The :const:`'x'`, :const:`'s'`, :const:`'y'`, 
        :const:`'z'` entries contain the iterates when the algorithm 
        terminated.  Whether these entries are useful, as approximate 
        solutions or certificates of primal and dual infeasibility, can be
        determined from the other fields in the dictionary.

        The fields :const:`'primal objective'`, :const:`'dual objective'`, 
        :const:`'gap'`, :const:`'relative gap'`, 
        :const:`'primal infeasibility'`,
        :const:`'dual infeasibility'` are defined as when :const:`'status'`
        is :const:`'optimal'`.  The field 
        :const:`'residual as primal infeasibility certificate'` is defined
        as
        
        .. math::

            \frac{ \|G^Tz+A^Ty\|_2 }{ -(h^Tz + b^Ty) \max\{1, \|h\|_2 \} }.
        
        if :math:`h^Tz+b^Ty < 0`, and :const:`None` otherwise.  A small 
        value of this residual indicates that :math:`y` and :math:`z`, 
        divided by :math:`-h^Tz-b^Ty`, are an approximate proof of primal 
        infeasibility.  The field 
        :const:`'residual as dual infeasibility certificate'` is defined as

        .. math::
        
            \max\{ \frac{ \|Gx+s\|_2 }{ -c^Tx \max\{ 1, \|h\|_2 \} }, 
            \frac{ \|Ax\|_2 }{ -c^Tx \max\{1,\|b\|_2\} }\}
        
        if :math:`c^Tx < 0`, and as :const:`None` otherwise.  A small value
        indicates that :math:`x` and :math:`s`, divided by :math:`-c^Tx` 
        are an approximate proof of dual infeasibility.  

    It is required that 

    .. math::
    
        \newcommand{\Rank}{\mathop{\bf rank}}
        \Rank(A) = p, \qquad 
        \Rank(\left[\begin{array}{c} G \\ A \end{array}\right]) = n,
   
    where :math:`p` is the number or rows of :math:`A` and :math:`n` is 
    the number of columns of :math:`G` and :math:`A`.


As an example we solve the problem

.. math::

    \begin{array}{ll}
    \mbox{minimize}   &  -6x_1 - 4x_2 - 5x_3 \\*[1ex]
    \mbox{subject to} 
        & 16x_1 - 14x_2 + 5x_3 \leq -3 \\*[1ex]
        & 7x_1 + 2x_2 \leq 5 \\*[1ex]
        & \left\| \left[ \begin{array}{c}
             8x_1 + 13x_2 - 12x_3 - 2  \\
            -8x_1 + 18x_2 +  6x_3 - 14 \\
              x_1 -  3x_2 - 17x_3 - 13 \end{array}\right] \right\|_2
            \leq -24x_1 - 7x_2 + 15x_3 + 12 \\*[3ex]
        & \left\| \left[ 
          \begin{array}{c} x_1 \\ x_2 \\ x_3 \end{array}
          \right] \right\|_2 \leq 10 \\*[3ex]
        & \left[\begin{array}{ccc}
           7x_1 +  3x_2 + 9x_3 & -5x_1 + 13x_2 + 6x_3 &   
               x_1 - 6x_2 - 6x_3\\
          -5x_1 + 13x_2 + 6x_3 &   x_1 + 12x_2 - 7x_3 & 
              -7x_1 -10x_2 - 7x_3\\
           x_1 - 6x_2 -6x_3 & -7x_1 -10x_2 -7 x_3 & 
              -4x_1 -28 x_2 -11x_3 
           \end{array}\right]  
    \preceq \left[\begin{array}{ccc}
        68  & -30 & -19 \\
       -30 & 99  &  23 \\
       -19 & 23  & 10 \end{array}\right].
    \end{array} 


>>> from cvxopt import matrix, solvers
>>> c = matrix([-6., -4., -5.])
>>> G = matrix([[ 16., 7.,  24.,  -8.,   8.,  -1.,  0., -1.,  0.,  0.,   
                   7.,  -5.,   1.,  -5.,   1.,  -7.,   1.,   -7.,  -4.], 
                [-14., 2.,   7., -13., -18.,   3.,  0.,  0., -1.,  0.,   
                   3.,  13.,  -6.,  13.,  12., -10.,  -6.,  -10., -28.],
                [  5., 0., -15.,  12.,  -6.,  17.,  0.,  0.,  0., -1.,   
                   9.,   6.,  -6.,   6.,  -7.,  -7.,  -6.,   -7., -11.]])
>>> h = matrix( [ -3., 5.,  12.,  -2., -14., -13., 10.,  0.,  0.,  0.,  
                  68., -30., -19., -30.,  99.,  23., -19.,   23.,  10.] )
>>> dims = {'l': 2, 'q': [4, 4], 's': [3]}
>>> sol = solvers.conelp(c, G, h, dims)
>>> sol['status']
'optimal'
>>> print(sol['x'])
[-1.22e+00]
[ 9.66e-02]
[ 3.58e+00]
>>> print(sol['z'])
[ 9.30e-02]
[ 2.04e-08]
[ 2.35e-01]
[ 1.33e-01]
[-4.74e-02]
[ 1.88e-01]
[ 2.79e-08]
[ 1.85e-09]
[-6.32e-10]
[-7.59e-09]
[ 1.26e-01]
[ 8.78e-02]
[-8.67e-02]
[ 8.78e-02]
[ 6.13e-02]
[-6.06e-02]
[-8.67e-02]
[-6.06e-02]
[ 5.98e-02]


Only the entries of ``G`` and ``h`` defining the lower triangular portions
of the coefficients in the linear matrix inequalities are accessed.  We 
obtain the same result if we define ``G`` and ``h`` as below. 


>>> G = matrix([[ 16., 7.,  24.,  -8.,   8.,  -1.,  0., -1.,  0.,  0.,   
                   7.,  -5.,   1.,  0.,   1.,  -7.,  0.,  0.,  -4.], 
                [-14., 2.,   7., -13., -18.,   3.,  0.,  0., -1.,  0.,   
                   3.,  13.,  -6.,  0.,  12., -10.,  0.,  0., -28.],
                [  5., 0., -15.,  12.,  -6.,  17.,  0.,  0.,  0., -1.,   
                   9.,   6.,  -6.,  0.,  -7.,  -7.,  0.,  0., -11.]])
>>> h = matrix( [ -3., 5.,  12.,  -2., -14., -13., 10.,  0.,  0.,  0.,  
                  68., -30., -19.,  0.,  99.,  23.,  0.,  0.,  10.] )


.. _s-coneqp: 

Quadratic Cone Programs 
=======================

.. function:: cvxopt.solvers.coneqp(P, q[, G, h[, dims[, A, b[, initvals[, kktsolver]]]]])

    Solves a pair of primal and dual quadratic cone programs

    .. math::

        \begin{array}[t]{ll}
        \mbox{minimize}   & (1/2) x^T Px + q^T x \\
        \mbox{subject to} & G x + s = h \\ 
                          & Ax = b \\ 
                          & s \succeq 0
        \end{array}

    and
        
    .. math::

        \newcommand{\Range}{\mbox{\textrm{range}}}
        \begin{array}[t]{ll}
        \mbox{maximize}   & -(1/2) (q+G^Tz+A^Ty)^T P^\dagger
                           (q+G^Tz+A^Ty) -h^T z - b^T y \\
        \mbox{subject to} & q + G^T z + A^T y \in \Range(P) \\ 
                          & z \succeq 0.
        \end{array}

    The primal variables are :math:`x` and the slack variable :math:`s`.  
    The dual variables are :math:`y` and :math:`z`.  The inequalities are
    interpreted as :math:`s \in C`, :math:`z\in C`, where :math:`C` is a 
    cone defined as a Cartesian product of a nonnegative orthant, a number
    of second-order cones, and a number of positive semidefinite cones:

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
                \{ u \in \reals^l \;| \; u_k \geq 0, \; k=1, \ldots,l\}, \\
            C_{k+1} & = \{ (u_0, u_1) \in \reals \times \reals^{r_{k}-1} 
                \; | \; u_0 \geq \|u_1\|_2 \},  \quad k=0,\ldots, M-1, \\
            C_{k+M+1} &= \left\{ \svec(u) \; | \; u \in \symm^{t_k}_+ 
                \right\}, \quad k=0,\ldots,N-1.
        \end{split}


    In this definition, :math:`\mathbf{vec}(u)` denotes a symmetric matrix
    :math:`u` stored as a vector in column major order.  The structure of 
    :math:`C` is specified by ``dims``.  This argument is a dictionary with
    three fields. 

    ``dims['l']``:
        :math:`l`, the dimension of the nonnegative orthant (a nonnegative
        integer).

    ``dims['q']``: 
        :math:`[r_0, \ldots, r_{M-1}]`, a list with the dimensions of the 
        second-order cones (positive integers).

    ``dims['s']``: 
        :math:`[t_0, \ldots, t_{N-1}]`, a list with the dimensions of the 
        positive semidefinite cones (nonnegative integers).

    The default value of ``dims`` is ``{'l': G.size[0], 'q': [], 
    's': []}``, i.e., by default the inequality is interpreted as a 
    componentwise vector inequality. 

    ``P`` is a square dense or sparse real matrix, representing a positive 
    semidefinite symmetric matrix in :const:`'L'` storage, i.e., only the 
    lower triangular part of ``P`` is referenced.  ``q`` is a real 
    single-column dense matrix.

    The arguments ``h`` and ``b`` are real single-column dense matrices.  
    ``G`` and ``A`` are real dense or sparse matrices.  The number of rows
    of ``G`` and ``h`` is equal to

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
    matrices are not accessed (i.e.,  the symmetric matrices are stored
    in the :const:`'L'`-type column major order used in the :mod:`blas` and
    :mod:`lapack` modules).  The default values for ``G``, ``h``, ``A``,
    and ``b`` are matrices with zero rows, meaning that there are no 
    inequality or equality constraints.  

    ``initvals`` is a dictionary with keys :const:`'x'`, :const:`'s'`, 
    :const:`'y'`, :const:`'z'` used as an optional starting point.  The 
    vectors ``initvals['s']`` and ``initvals['z']`` must be 
    strictly positive with respect to the cone :math:`C`.  If the argument
    ``initvals`` or any the four entries in it are missing, default 
    starting points are used for the corresponding variables.

    The role of the optional argument ``kktsolver`` is explained in the
    section :ref:`s-conelp-struct`.  

    :func:`coneqp` returns a dictionary that contains the result and 
    information about the accuracy of the solution.  The most important 
    fields have keys :const:`'status'`, :const:`'x'`, :const:`'s'`, 
    :const:`'y'`, :const:`'z'`.  The :const:`'status'` field  is a string 
    with possible values :const:`'optimal'` and :const:`'unknown'`.  

    :const:`'optimal'` 
        In this case the :const:`'x'`, :const:`'s'`, :const:`'y'`, and 
        :const:`'z'` entries contain primal and dual solutions, which 
        approximately satisfy

        .. math::
        
            Gx+s = h, \qquad Ax = b, \qquad Px + G^Tz + A^T y + q = 0,

            s \succeq 0, \qquad z \succeq 0, \qquad s^T z  = 0.
  

    :const:`'unknown'` 
        This indicates that the algorithm terminated early due to numerical
        difficulties or because the maximum number of iterations was 
        reached.  The :const:`'x'`, :const:`'s'`, :const:`'y'`, 
        :const:`'z'` entries contain the iterates when the algorithm 
        terminated.

    The other entries in the output dictionary summarize the accuracy
    with which the optimality conditions are satisfied.  The fields 
    :const:`'primal objective'`, :const:`'dual objective'`, and
    :const:`'gap'` give the primal objective :math:`c^Tx`, the dual 
    objective calculated as 

    .. math::

        (1/2) x^TPx + q^T x + z^T(Gx-h) + y^T(Ax-b)
    
    and the gap :math:`s^Tz`.  The field :const:`'relative gap'` is the 
    relative gap, defined as

    .. math::
    
        \frac{s^Tz}{-\mbox{primal objective}}
        \quad \mbox{if\ } \mbox{primal objective} < 0, \qquad
        \frac{s^Tz}{\mbox{dual objective}}
        \quad \mbox{if\ } \mbox{dual objective} > 0, \qquad

    and :const:`None` otherwise.  The fields 
    :const:`'primal infeasibility'` and :const:`'dual infeasibility'` are 
    the residuals in the primal and dual equality constraints, defined as

    .. math::

        \max\{ \frac{\|Gx+s-h\|_2}{\max\{1, \|h\|_2\}}, 
        \frac{\|Ax-b\|_2}{\max\{1,\|b\|_2\}} \}, \qquad
        \frac{\|Px + G^Tz + A^Ty + q\|_2}{\max\{1, \|q\|_2\}}, 

    respectively.

    It is required that the problem is solvable and that 

    .. math::

        \newcommand{\Rank}{\mathop{\bf rank}}
        \Rank(A) = p, \qquad 
        \Rank(\left[\begin{array}{c} P \\ G \\ A \end{array}\right]) = n,
    
    where :math:`p` is the number or rows of :math:`A` and :math:`n` is the
    number of columns of :math:`G` and :math:`A`.


As an example, we solve a constrained least-squares problem

.. math::

    \begin{array}{ll}
    \mbox{minimize}   & \|Ax - b\|_2^2 \\
    \mbox{subject to} &  x \succeq 0 \\
                      & \|x\|_2 \leq 1 
    \end{array}

with 

.. math::
    A = \left[ \begin{array}{rrr}
        0.3 &  0.6 & -0.3 \\
       -0.4 &  1.2 &  0.0 \\
       -0.2 & -1.7 &  0.6 \\
       -0.4 &  0.3 & -1.2 \\
        1.3 & -0.3 & -2.0 
       \end{array} \right], \qquad 
    b = \left[ \begin{array}{r} 1.5 \\ 0.0 \\ -1.2 \\ -0.7 \\ 0.0 
        \end{array} \right]. 

>>> from cvxopt import matrix, solvers
>>> A = matrix([ [ .3, -.4,  -.2,  -.4,  1.3 ], 
                 [ .6, 1.2, -1.7,   .3,  -.3 ],
                 [-.3,  .0,   .6, -1.2, -2.0 ] ])
>>> b = matrix([ 1.5, .0, -1.2, -.7, .0])
>>> m, n = A.size
>>> I = matrix(0.0, (n,n))
>>> I[::n+1] = 1.0
>>> G = matrix([-I, matrix(0.0, (1,n)), I])
>>> h = matrix(n*[0.0] + [1.0] + n*[0.0])
>>> dims = {'l': n, 'q': [n+1], 's': []}
>>> x = solvers.coneqp(A.T*A, -A.T*b, G, h, dims)['x']
>>> print(x)
[ 7.26e-01]
[ 6.18e-01]
[ 3.03e-01]


.. _s-lpsolver:

Linear Programming
==================

The function :func:`lp <cvxopt.solvers.lp>` is an interface to 
:func:`conelp <cvxopt.solvers.conelp>` for linear 
programs.  It also provides the option of using the linear programming 
solvers from GLPK or MOSEK.

.. function:: cvxopt.solvers.lp(c, G, h[, A, b[, solver[, primalstart[, dualstart]]]])

    Solves the pair of primal and dual linear programs

    .. math::
    
        \begin{array}[t]{ll}
        \mbox{minimize}   & c^T x \\
        \mbox{subject to} & G x + s = h \\ 
                          & Ax = b \\ 
                          & s \succeq 0
        \end{array}
        \qquad\qquad
        \begin{array}[t]{ll}
        \mbox{maximize}   & -h^T z - b^T y \\
        \mbox{subject to} & G^T z + A^T y + c = 0 \\
                          & z \succeq 0.
        \end{array}
   
    The inequalities are componentwise vector inequalities.

    The ``solver`` argument is used to choose among three solvers.  When 
    it is omitted or :const:`None`, the CVXOPT function 
    :func:`conelp <cvxopt.solvers.conelp>` is 
    used.  The external solvers GLPK and MOSEK (if installed) can be 
    selected by setting ``solver`` to :const:`'glpk'` or :const:`'mosek'`; 
    see the section :ref:`s-external`.  The meaning of the other 
    arguments and the return value are the same as for 
    :func:`conelp` called with 
    ``dims`` equal to ``{'l': G.size[0], 'q': [], 's': []}``. 

    The initial values are ignored when ``solver`` is :const:`'mosek'` or 
    :const:`'glpk'`.  With the GLPK option, the solver does not return 
    certificates of primal or dual infeasibility: if the status is
    :const:`'primal infeasible'` or :const:`'dual infeasible'`, all entries
    of the output dictionary are :const:`None`.  If the GLPK or MOSEK 
    solvers are used, and the code returns with status :const:`'unknown'`, 
    all the other fields in the output dictionary are :const:`None`.

As a simple example we solve the LP

.. math::

    \begin{array}[t]{ll}
    \mbox{minimize}   & -4x_1 - 5x_2 \\
    \mbox{subject to} &  2x_1 + x_2 \leq 3 \\
                      & x_1 + 2x_2 \leq 3 \\
                      & x_1 \geq 0, \quad x_2 \geq 0.
    \end{array} 


>>> from cvxopt import matrix, solvers
>>> c = matrix([-4., -5.])
>>> G = matrix([[2., 1., -1., 0.], [1., 2., 0., -1.]])
>>> h = matrix([3., 3., 0., 0.])
>>> sol = solvers.lp(c, G, h)
>>> print(sol['x'])
[ 1.00e+00]
[ 1.00e+00]


.. _s-qp:

Quadratic Programming
=====================

The function :func:`qp <cvxopt.solvers.qp>` is an interface to 
:func:`coneqp <cvxopt.solvers.coneqp>` for quadratic 
programs.  It also provides the option of using the quadratic programming 
solver from MOSEK.

.. function:: cvxopt.solvers.qp(P, q[, G, h[, A, b[, solver[, initvals]]]])

    Solves the pair of primal and dual convex quadratic programs 

    .. math::
    
        \begin{array}[t]{ll}
        \mbox{minimize} & (1/2) x^TPx + q^T x \\
        \mbox{subject to} & Gx \preceq h \\ & Ax = b
        \end{array}

    and

    .. math::
    
        \newcommand{\Range}{\mbox{\textrm{range}}}
        \begin{array}[t]{ll}
        \mbox{maximize}   & -(1/2) (q+G^Tz+A^Ty)^T P^\dagger
                             (q+G^Tz+A^Ty) -h^T z - b^T y \\
        \mbox{subject to} & q + G^T z + A^T y \in \Range(P) \\ 
                          & z \succeq 0.
        \end{array}
  
    The inequalities are componentwise vector inequalities.

    The default CVXOPT solver is used when the ``solver`` argument is 
    absent or :const:`None`.  The MOSEK solver (if installed) can be 
    selected by setting ``solver`` to :const:`'mosek'`; see the 
    section :ref:`s-external`.  The meaning of the other arguments and the
    return value is the same as for 
    :func:`coneqp <cvxopt.solvers.coneqp>` called with `dims` 
    equal to ``{'l': G.size[0], 'q': [], 's': []}``.

    When ``solver`` is :const:`'mosek'`, the initial values are ignored,
    and the :const:`'status'` string in the solution dictionary can take 
    four possible values: :const:`'optimal'`, :const:`'unknown'`.
    :const:`'primal infeasible'`, :const:`'dual infeasible'`. 

    :const:`'primal infeasible'`  
        This means that a certificate of primal infeasibility has been 
        found.  The :const:`'x'` and :const:`'s'` entries are 
        :const:`None`, and the :const:`'z'` and :const:`'y'` entries are 
        vectors that approximately satisfy

        .. math:: 
        
            G^Tz + A^T y = 0, \qquad h^Tz + b^Ty = -1, \qquad z \succeq 0.
        

    :const:`'dual infeasible'`  
        This means that a certificate of dual infeasibility has been found.
        The :const:`'z'` and :const:`'y'` entries are :const:`None`, and 
        the :const:`'x'` and :const:`'s'` entries are vectors that 
        approximately satisfy

        .. math:: 
        
            Px = 0, \qquad q^Tx = -1, \qquad Gx + s = 0, \qquad Ax=0, 
            \qquad s \succeq  0.
        

As an example we compute the trade-off curve on page 187 of the book 
`Convex Optimization <http://www.stanford.edu/~boyd/cvxbook>`_,
by solving the quadratic program 

.. math::

    \newcommand{\ones}{{\bf 1}}
    \begin{array}{ll}
    \mbox{minimize}   & -\bar p^T x + \mu x^T S x \\
    \mbox{subject to} & \ones^T x = 1, \quad x \succeq 0
    \end{array}

for a sequence of positive values of :math:`\mu`.  The code below computes 
the trade-off curve and produces two figures using the 
`Matplotlib <http://matplotlib.sourceforge.net>`_ package.

.. image:: portfolio2.png
   :width: 400px

.. image:: portfolio1.png
   :width: 400px

::

    from math import sqrt
    from cvxopt import matrix
    from cvxopt.blas import dot 
    from cvxopt.solvers import qp
    import pylab

    # Problem data.
    n = 4
    S = matrix([[ 4e-2,  6e-3, -4e-3,    0.0 ], 
                [ 6e-3,  1e-2,  0.0,     0.0 ],
                [-4e-3,  0.0,   2.5e-3,  0.0 ],
                [ 0.0,   0.0,   0.0,     0.0 ]])
    pbar = matrix([.12, .10, .07, .03])
    G = matrix(0.0, (n,n))
    G[::n+1] = -1.0
    h = matrix(0.0, (n,1))
    A = matrix(1.0, (1,n))
    b = matrix(1.0)

    # Compute trade-off.
    N = 100
    mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
    portfolios = [ qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]
    returns = [ dot(pbar,x) for x in portfolios ]
    risks = [ sqrt(dot(x, S*x)) for x in portfolios ]

    # Plot trade-off curve and optimal allocations.
    pylab.figure(1, facecolor='w')
    pylab.plot(risks, returns)
    pylab.xlabel('standard deviation')
    pylab.ylabel('expected return')
    pylab.axis([0, 0.2, 0, 0.15])
    pylab.title('Risk-return trade-off curve (fig 4.12)')
    pylab.yticks([0.00, 0.05, 0.10, 0.15])

    pylab.figure(2, facecolor='w')
    c1 = [ x[0] for x in portfolios ] 
    c2 = [ x[0] + x[1] for x in portfolios ]
    c3 = [ x[0] + x[1] + x[2] for x in portfolios ] 
    c4 = [ x[0] + x[1] + x[2] + x[3] for x in portfolios ]
    pylab.fill(risks + [.20], c1 + [0.0], '#F0F0F0') 
    pylab.fill(risks[-1::-1] + risks, c2[-1::-1] + c1, facecolor = '#D0D0D0') 
    pylab.fill(risks[-1::-1] + risks, c3[-1::-1] + c2, facecolor = '#F0F0F0') 
    pylab.fill(risks[-1::-1] + risks, c4[-1::-1] + c3, facecolor = '#D0D0D0') 
    pylab.axis([0.0, 0.2, 0.0, 1.0])
    pylab.xlabel('standard deviation')
    pylab.ylabel('allocation')
    pylab.text(.15,.5,'x1')
    pylab.text(.10,.7,'x2')
    pylab.text(.05,.7,'x3')
    pylab.text(.01,.7,'x4')
    pylab.title('Optimal allocations (fig 4.12)')
    pylab.show()



.. _s-socpsolver:

Second-Order Cone Programming
=============================

The function :func:`socp <cvxopt.solvers.socp>` is a simpler interface to 
:func:`conelp <cvxopt.solvers.conelp>` for 
cone programs with no linear matrix inequality constraints.

.. function:: cvxopt.solvers.socp(c[, Gl, hl[, Gq, hq[, A, b[, solver[, primalstart[, dualstart]]]]]])

    Solves the pair of primal and dual second-order cone programs
    
    .. math::

        \begin{array}[t]{ll}
        \mbox{minimize}   & c^T x \\
        \mbox{subject to} & G_k x + s_k = h_k, \quad k = 0, \ldots, M  \\ 
                          & Ax = b \\ 
                          & s_0 \succeq 0 \\
                          & s_{k0} \geq \|s_{k1}\|_2, \quad k = 1,\ldots,M
        \end{array}
     
    and

    .. math::

        \begin{array}[t]{ll}
        \mbox{maximize}   & - \sum_{k=0}^M h_k^Tz_k - b^T y \\
        \mbox{subject to} & \sum_{k=0}^M G_k^T z_k + A^T y + c = 0 \\
                          & z_0 \succeq 0 \\
                          & z_{k0} \geq \|z_{k1}\|_2, \quad k=1,\ldots,M.
        \end{array}

    The inequalities 

    .. math::

        s_0 \succeq 0, \qquad z_0 \succeq 0

    are componentwise vector inequalities.  In the other inequalities, it 
    is assumed that the variables are partitioned as

    .. math::
    
        \newcommand{\reals}{{\mbox{\bf R}}}
        s_k = (s_{k0}, s_{k1}) \in\reals\times\reals^{r_{k}-1}, \qquad 
        z_k = (z_{k0}, z_{k1}) \in\reals\times\reals^{r_{k}-1}, \qquad
        k=1,\ldots,M.

    The input argument ``c`` is a real single-column dense matrix.  The 
    arguments ``Gl`` and ``hl`` are the coefficient matrix :math:`G_0` and 
    the right-hand side :math:`h_0` of the componentwise inequalities.
    ``Gl`` is a real dense or sparse matrix; ``hl`` is a real single-column
    dense matrix.  The default values for ``Gl`` and ``hl`` are matrices 
    with zero rows.

    The argument ``Gq`` is a list of :math:`M` dense or sparse matrices 
    :math:`G_1`, ..., :math:`G_M`.  The argument ``hq`` is a list of 
    :math:`M` dense single-column matrices :math:`h_1`, \ldots, 
    :math:`h_M`.  The elements of ``Gq`` and ``hq`` must have at least one
    row.  The default values of ``Gq`` and ``hq`` are empty lists.

    ``A`` is dense or sparse matrix and ``b`` is a single-column dense 
    matrix.  The default values for ``A`` and ``b`` are matrices with 
    zero rows. 

    The ``solver`` argument is used to choose between two solvers: the 
    CVXOPT :func:`conelp <cvxopt.solvers.conelp>` solver (used when 
    ``solver`` is absent or equal 
    to :const:`None` and the external solver MOSEK (``solver`` is 
    :const:`'mosek'`); see the section :ref:`s-external`.  With the 
    :const:`'mosek'` option the code does not accept problems with equality
    constraints.

    ``primalstart`` and ``dualstart`` are dictionaries with optional 
    primal, respectively, dual starting points.  ``primalstart`` has 
    elements :const:`'x'`, :const:`'sl'`, :const:`'sq'`.  
    ``primalstart['x']`` and ``primalstart['sl']`` are 
    single-column dense matrices with the initial values of :math:`x` and 
    :math:`s_0`;  ``primalstart['sq']`` is a list of single-column 
    matrices with the initial values of :math:`s_1`, \ldots, :math:`s_M`.
    The initial values must satisfy the inequalities in the primal problem 
    strictly, but not necessarily the equality constraints.

    ``dualstart`` has elements :const:`'y'`, :const:`'zl'`, :const:`'zq'`.
    ``dualstart['y']`` and ``dualstart['zl']`` are single-column 
    dense matrices with the initial values of :math:`y` and :math:`z_0`.
    ``dualstart['zq']`` is a list of single-column matrices with the 
    initial values of :math:`z_1`, \ldots, :math:`z_M`.  These values must
    satisfy the dual inequalities strictly, but not necessarily the 
    equality constraint.

    The arguments ``primalstart`` and ``dualstart`` are ignored when the 
    MOSEK solver is used.

    :func:`socp` returns a dictionary that include entries with keys 
    :const:`'status'`, :const:`'x'`, :const:`'sl'`, :const:`'sq'`, 
    :const:`'y'`, :const:`'zl'`, :const:`'zq'`.  The :const:`'sl'` and 
    :const:`'zl'` fields are matrices with the primal slacks and dual 
    variables associated with the componentwise linear inequalities.
    The :const:`'sq'` and :const:`'zq'` fields are lists with the primal 
    slacks and dual variables associated with the second-order cone 
    inequalities.  The other entries in the output dictionary have the 
    same meaning as in the output of 
    :func:`conelp <cvxopt.solvers.conelp>`.


As an example, we solve  the second-order cone program

.. math::
    \begin{array}{ll}
    \mbox{minimize}   & -2x_1 + x_2 + 5x_3 \\*[2ex]
    \mbox{subject to} & \left\| \left[\begin{array}{c}
        -13 x_1 +  3 x_2 + 5 x_3 - 3 \\ 
        -12 x_1 + 12 x_2 - 6 x_3 - 2 \end{array}\right] \right\|_2 
         \leq -12 x_1 - 6 x_2 + 5x_3 - 12  \\*[2ex]
                       & \left\| \left[\begin{array}{c}
         -3 x_1 +  6 x_2 + 2 x_3    \\ 
            x_1 +  9 x_2 + 2 x_3 + 3 \\ 
           -x_1 - 19 x_2 + 3 x_3 - 42 \end{array}\right] \right\|_2 
         \leq -3x_1 + 6x_2 - 10x_3 + 27.
    \end{array}


>>> from cvxopt import matrix, solvers
>>> c = matrix([-2., 1., 5.])
>>> G = [ matrix( [[12., 13., 12.], [6., -3., -12.], [-5., -5., 6.]] ) ]
>>> G += [ matrix( [[3., 3., -1., 1.], [-6., -6., -9., 19.], [10., -2., -2., -3.]] ) ]
>>> h = [ matrix( [-12., -3., -2.] ),  matrix( [27., 0., 3., -42.] ) ]
>>> sol = solvers.socp(c, Gq = G, hq = h)
>>> sol['status']
optimal
>>> print(sol['x'])
[-5.02e+00]
[-5.77e+00]
[-8.52e+00]
>>> print(sol['zq'][0])
[ 1.34e+00]
[-7.63e-02]
[-1.34e+00]
>>> print(sol['zq'][1])
[ 1.02e+00]
[ 4.02e-01]
[ 7.80e-01]
[-5.17e-01]


.. _s-sdpsolver:

Semidefinite Programming
========================

The function :func:`sdp <cvxopt.solvers.sdp>` is a simple interface to 
:func:`conelp <cvxopt.solvers.conelp>` for cone 
programs with no second-order cone constraints.  It also provides the 
option of using the DSDP semidefinite programming solver.

.. function:: cvxopt.solvers.sdp(c[, Gl, hl[, Gs, hs[, A, b[, solver[, primalstart[, dualstart]]]]]])

    Solves the pair of primal and dual semidefinite programs

    .. math::
    
        \newcommand{\svec}{\mathop{\mathbf{vec}}}
        \begin{array}[t]{ll}
        \mbox{minimize}   & c^T x \\
        \mbox{subject to} & G_0 x + s_0 = h_0 \\
                          & G_k x + \svec{(s_k)} = \svec{(h_k)}, 
                            \quad k = 1, \ldots, N  \\ 
                          & Ax = b \\ 
                          & s_0 \succeq 0 \\
                          & s_k \succeq 0, \quad k=1,\ldots,N
        \end{array}

    and

    .. math::

        \newcommand{\Tr}{\mathop{\mathbf{tr}}}
        \newcommand{\svec}{\mathop{\mathbf{vec}}}
        \begin{array}[t]{ll}
        \mbox{maximize}   & -h_0^Tz_0 - \sum_{k=1}^N \Tr(h_kz_k) - b^Ty \\
        \mbox{subject to} & G_0^Tz_0 + \sum_{k=1}^N G_k^T \svec(z_k) + 
                             A^T y + c = 0 \\
                          & z_0 \succeq 0 \\
                          & z_k \succeq 0, \quad k=1,\ldots,N.
        \end{array}

    The inequalities 
    
    .. math::
       
        s_0 \succeq 0, \qquad z_0 \succeq 0

    are componentwise vector inequalities.  The other inequalities are 
    matrix inequalities (\ie, the require the left-hand sides to be 
    positive semidefinite).  We use the notation :math:`\mathbf{vec}(z)` 
    to denote a symmetric matrix :math:`z` stored in column major order 
    as a column vector.

    The input argument ``c`` is a real single-column dense matrix.  The 
    arguments ``Gl`` and ``hl`` are the coefficient matrix :math:`G_0` and
    the right-hand side :math:`h_0` of the componentwise inequalities.
    ``Gl`` is a real dense or sparse matrix;  ``hl`` is a real 
    single-column dense matrix.   The default values for ``Gl`` and ``hl``
    are matrices with zero rows.

    ``Gs`` and ``hs`` are lists of length :math:`N` that specify the 
    linear matrix inequality constraints.  ``Gs`` is a list of :math:`N` 
    dense or sparse real matrices :math:`G_1`, \ldots, :math:`G_M`.  The 
    columns of these matrices can be interpreted as symmetric matrices 
    stored in column major order, using the BLAS :const:`'L'`-type storage
    (i.e., only the entries corresponding to lower triangular positions
    are accessed).  ``hs`` is a list of :math:`N` dense symmetric matrices
    :math:`h_1`, \ldots, :math:`h_N`.  Only the lower triangular elements 
    of these matrices are accessed.  The default values for ``Gs`` and 
    ``hs`` are empty lists.

    ``A`` is a dense or sparse matrix and ``b`` is a single-column dense 
    matrix.  The default values for ``A`` and ``b`` are matrices with zero 
    rows. 

    The ``solver`` argument is used to choose between two solvers: the 
    CVXOPT :func:`conelp <cvxopt.solvers.conelp>` solver 
    (used when ``solver`` is absent or equal 
    to :const:`None`) and the external solver DSDP5 (``solver`` is 
    :const:`'dsdp'`); see the section :ref:`s-external`.  With the 
    :const:`'dsdp'` option the code does not accept problems with equality
    constraints.

    The optional argument ``primalstart`` is a dictionary with keys 
    :const:`'x'`, :const:`'sl'`, and :const:`'ss'`, used as an optional 
    primal starting point.  ``primalstart['x']`` and 
    ``primalstart['sl']`` are single-column dense matrices with the 
    initial values of :math:`x` and :math:`s_0`; 
    ``primalstart['ss']`` is a list of square matrices with the initial
    values of :math:`s_1`, \ldots, :math:`s_N`.  The initial values must 
    satisfy the inequalities in the primal problem strictly, but not 
    necessarily the equality constraints.

    ``dualstart`` is a dictionary with keys :const:`'y'`, :const:`'zl'`, 
    :const:`'zs'`, used as an optional dual starting point.  
    ``dualstart['y']`` and ``dualstart['zl']`` are single-column 
    dense matrices with the initial values of :math:`y` and :math:`z_0`.
    ``dualstart['zs']`` is a list of square matrices with the initial 
    values of :math:`z_1`, \ldots, :math:`z_N`.  These values must satisfy
    the dual inequalities strictly, but not necessarily the equality 
    constraint.

    The arguments ``primalstart`` and ``dualstart`` are ignored when the 
    DSDP solver is used.

    :func:`sdp` returns a dictionary that includes entries with keys 
    :const:`'status'`, :const:`'x'`, :const:`'sl'`, :const:`'ss'`, 
    :const:`'y'`, :const:`'zl'`, :const:`'ss'`.  The :const:`'sl'` and 
    :const:`'zl'` fields are matrices with the primal slacks and dual  
    variables associated with the componentwise linear inequalities.
    The :const:`'ss'` and :const:`'zs'` fields are lists with the primal 
    slacks and dual variables associated with the second-order cone 
    inequalities.  The other entries in the output dictionary have the 
    same meaning as in the output of 
    :func:`conelp <cvxopt.solvers.conelp>`.

We illustrate the calling sequence with a small example.

    .. math::

        \begin{array}{ll}
        \mbox{minimize}   & x_1 - x_2 + x_3 \\
        \mbox{subject to} 
            & x_1 \left[ \begin{array}{cc} 
                      -7 &  -11 \\ -11 &  3
                  \end{array}\right] + 
              x_2 \left[ \begin{array}{cc}
                      7 & -18 \\ -18 & 8 
                  \end{array}\right] + 
              x_3 \left[ \begin{array}{cc}
                      -2 & -8 \\ -8 & 1 
                  \end{array}\right] \preceq  
              \left[ \begin{array}{cc} 
                      33 & -9 \\ -9 & 26 
                  \end{array}\right] \\*[1ex]
              & x_1 \left[ \begin{array}{ccc} 
                      -21 & -11 & 0 \\ 
                      -11 &  10 & 8 \\ 
                        0 &   8 & 5
                  \end{array}\right] + 
              x_2 \left[ \begin{array}{ccc} 
                        0 &  10 &  16 \\
                       10 & -10 & -10 \\
                       16 & -10 & 3 
                  \end{array}\right] + 
              x_3 \left[ \begin{array}{ccc} 
                       -5  & 2 & -17 \\
                        2  & -6 & 8 \\
                       -17 & 8 & 6 
                   \end{array}\right]  \preceq  
              \left[ \begin{array}{ccc}
                       14 &  9 & 40 \\
                        9  & 91 & 10 \\
                       40 & 10 & 15
                  \end{array} \right] 
        \end{array}

>>> from cvxopt import matrix, solvers
>>> c = matrix([1.,-1.,1.])
>>> G = [ matrix([[-7., -11., -11., 3.], 
                  [ 7., -18., -18., 8.], 
                  [-2.,  -8.,  -8., 1.]]) ]
>>> G += [ matrix([[-21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.], 
                   [  0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.], 
                   [ -5.,   2., -17.,   2.,  -6.,   8., -17.,  8., 6.]]) ]
>>> h = [ matrix([[33., -9.], [-9., 26.]]) ]
>>> h += [ matrix([[14., 9., 40.], [9., 91., 10.], [40., 10., 15.]]) ]
>>> sol = solvers.sdp(c, Gs=G, hs=h)  
>>> print(sol['x'])
[-3.68e-01]
[ 1.90e+00]
[-8.88e-01]
>>> print(sol['zs'][0])
[ 3.96e-03 -4.34e-03]
[-4.34e-03  4.75e-03]
>>> print(sol['zs'][1])
[ 5.58e-02 -2.41e-03  2.42e-02]
[-2.41e-03  1.04e-04 -1.05e-03]
[ 2.42e-02 -1.05e-03  1.05e-02]

Only the entries in ``Gs`` and ``hs`` that correspond to lower triangular 
entries need to be provided, so in the example ``h`` and ``G`` may also be
defined as follows.

>>> G = [ matrix([[-7., -11., 0., 3.], 
                  [ 7., -18., 0., 8.], 
                  [-2.,  -8., 0., 1.]]) ]
>>> G += [ matrix([[-21., -11.,   0., 0.,  10.,   8., 0., 0., 5.], 
                   [  0.,  10.,  16., 0., -10., -10., 0., 0., 3.], 
                   [ -5.,   2., -17., 0.,  -6.,   8., 0., 0., 6.]]) ]
>>> h = [ matrix([[33., -9.], [0., 26.]]) ]
>>> h += [ matrix([[14., 9., 40.], [0., 91., 10.], [0., 0., 15.]]) ]


.. _s-conelp-struct:

Exploiting Structure
====================

By default, the functions 
:func:`conelp <cvxopt.solvers.conelp>` and 
:func:`coneqp <cvxopt.solvers.coneqp>` exploit no 
problem structure except (to some limited extent) sparsity.  Two mechanisms
are provided for implementing customized solvers that take advantage of 
problem structure.


**Providing a function for solving KKT equations**
    The most expensive step of each iteration of 
    :func:`conelp <cvxopt.solvers.conelp>` or 
    :func:`coneqp <cvxopt.solvers.coneqp>` is the solution of a set of  
    linear equations (*KKT equations*) of the form

    .. math::
        :label: e-conelp-kkt

        \left[\begin{array}{ccc}
            P & A^T & G^T \\
            A & 0   & 0  \\
            G & 0   & -W^T W 
        \end{array}\right]
        \left[\begin{array}{c} u_x \\ u_y \\ u_z \end{array}\right]
        = 
        \left[\begin{array}{c} b_x \\ b_y \\ b_z \end{array}\right]
    
    (with :math:`P = 0` in :func:`conelp`).  The matrix :math:`W` depends 
    on the current iterates and is defined as follows.  We use the notation
    of the sections :ref:`s-conelp` and :ref:`s-coneqp`.  Let 

    .. math::

        \newcommand{\svec}{\mathop{\mathbf{vec}}}
        u = \left(u_\mathrm{l}, \; u_{\mathrm{q},0}, \; \ldots, \; 
            u_{\mathrm{q},M-1}, \; \svec{(u_{\mathrm{s},0})}, \; 
            \ldots, \; \svec{(u_{\mathrm{s},N-1})}\right), \qquad

        \newcommand{\reals}{{\mbox{\bf R}}}
        \newcommand{\symm}{{\mbox{\bf S}}}
        u_\mathrm{l} \in\reals^l, \qquad 
        u_{\mathrm{q},k} \in\reals^{r_k}, \quad k = 0,\ldots,M-1, \qquad 
        u_{\mathrm{s},k} \in\symm^{t_k},  \quad k = 0,\ldots,N-1.

    Then :math:`W` is a block-diagonal matrix, 
  
    .. math::
    
        \newcommand{\svec}{\mathop{\mathbf{vec}}}
        Wu = \left( W_\mathrm{l} u_\mathrm{l}, \;
            W_{\mathrm{q},0} u_{\mathrm{q},0}, \; \ldots, \;
            W_{\mathrm{q},M-1} u_{\mathrm{q},M-1},\; 
            W_{\mathrm{s},0} \svec{(u_{\mathrm{s},0})}, \; \ldots, \;
            W_{\mathrm{s},N-1} \svec{(u_{\mathrm{s},N-1})} \right)

    with the following diagonal blocks.

    * The first block is a *positive diagonal scaling* with a vector 
      :math:`d`:

      .. math::
    
          \newcommand{\diag}{\mbox{\bf diag}\,}
          W_\mathrm{l} = \diag(d), \qquad 
          W_\mathrm{l}^{-1} = \diag(d)^{-1}.
    
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

      In  general, this operation is not symmetric: 
 
      .. math::

          \newcommand{\svec}{\mathop{\mathbf{vec}}}
          W_{\mathrm{s},k}^T \svec{(u_{\mathrm{s},k})} = 
              \svec{(r_k u_{\mathrm{s},k} r_k^T)}, \qquad \qquad
          W_{\mathrm{s},k}^{-T} \svec{(u_{\mathrm{s},k})} = 
              \svec{(r_k^{-1} u_{\mathrm{s},k} r_k^{-T})}, \qquad \qquad
          k = 0,\ldots,N-1.

    It is often possible to exploit problem structure to solve 
    :eq:`e-conelp-kkt` faster than by standard methods.  The last argument
    ``kktsolver`` of :func:`conelp <cvxopt.solvers.conelp>` and 
    :func:`coneqp <cvxopt.solvers.coneqp>` allows the user to 
    supply a Python  function for solving the KKT equations.  This 
    function will be called as ``f = kktsolver(W)``, where ``W`` is a 
    dictionary that contains the parameters of the scaling:

    * ``W['d']`` is the positive vector that defines the diagonal
      scaling.   ``W['di']`` is its componentwise inverse.

    * ``W['beta']`` and ``W['v']`` are lists of length :math:`M` 
      with the coefficients and vectors that define the hyperbolic 
      Householder transformations.

    * ``W['r']`` is a list of length :math:`N` with the matrices that
      define the the congruence transformations.  ``W['rti']`` is a 
      list of length :math:`N` with the transposes of the inverses of the 
      matrices in ``W['r']``.

    The function call ``f = kktsolver(W)`` should return a routine for
    solving the KKT system :eq:`e-conelp-kkt` defined by ``W``.  It will 
    be called as ``f(bx, by, bz)``.  On entry, ``bx``, ``by``, ``bz`` 
    contain the right-hand side.  On exit, they should contain the solution
    of the KKT system, with the last component scaled, i.e., on exit,
    
    .. math::

        b_x := u_x, \qquad b_y := u_y, \qquad b_z := W u_z.

    In other words, the function returns the solution of

    .. math::

        \left[\begin{array}{ccc}
            P & A^T & G^TW^{-1} \\
            A & 0   & 0  \\
            G & 0   & -W^T 
        \end{array}\right]
        \left[\begin{array}{c} 
            \hat u_x \\ \hat u_y \\ \hat u_z 
        \end{array}\right]
        = 
        \left[\begin{array}{c} 
            b_x \\ b_y \\ b_z 
        \end{array}\right].


**Specifying constraints via Python functions**
    In the default use of :func:`conelp <cvxopt.solvers.conelp>` and 
    :func:`coneqp <cvxopt.solvers.coneqp>`, the linear 
    constraints and the quadratic term in the objective are parameterized 
    by CVXOPT matrices ``G``, ``A``, ``P``.  It is possible to specify 
    these parameters via Python functions that evaluate the corresponding 
    matrix-vector products and their adjoints.

    * If the argument ``G`` of :func:`conelp` or :func:`coneqp` is a 
      Python function, then 
      ``G(x, y[, alpha = 1.0, beta = 0.0, trans = 'N'])`` 
      should evaluate the matrix-vector products

        .. math::

            y := \alpha Gx + \beta y \quad 
                (\mathrm{trans} = \mathrm{'N'}), \qquad
            y := \alpha G^T x + \beta y \quad 
                (\mathrm{trans} = \mathrm{'T'}).

    * Similarly, if the argument ``A`` is a Python function, then 
      ``A(x, y[, alpha = 1.0, beta = 0.0, trans = 'N'])`` 
      should evaluate the matrix-vector products

        .. math::

            y := \alpha Ax + \beta y \quad 
                (\mathrm{trans} = \mathrm{'N'}), \qquad
            y := \alpha A^T x + \beta y \quad 
                (\mathrm{trans} = \mathrm{'T'}).

    * If the argument ``P`` of :func:`coneqp` is a Python function, then 
      ``P(x, y[, alpha = 1.0, beta = 0.0])`` 
      should evaluate the matrix-vector products

        .. math::

            y := \alpha Px + \beta y.

    If ``G``, ``A``, or ``P`` are Python functions, then the argument 
    ``kktsolver`` must also be provided.


We illustrate these features with three applications.

**Example: 1-norm approximation**

    The optimization problem

    .. math::

        \begin{array}{ll}
        \mbox{minimize} & \|Pu-q\|_1
        \end{array}

    can be formulated as a linear program

    .. math::

        \newcommand{\ones}{{\bf 1}}
        \begin{array}{ll}
        \mbox{minimize} & \ones^T v \\
        \mbox{subject to} & -v \preceq Pu - q  \preceq v.
        \end{array}

    By exploiting the structure in the inequalities, the cost of an 
    iteration of an interior-point method can be reduced to the cost of 
    least-squares problem of the same dimensions.  (See section 11.8.2 in 
    the book 
    `Convex Optimization <http://www.stanford.edu/~boyd/cvxbook>`_.) 
    The code below takes advantage of this fact.

    :: 

        from cvxopt import blas, lapack, solvers, matrix, spmatrix, mul, div

        def l1(P, q):
            """

            Returns the solution u, w of the l1 approximation problem

                (primal) minimize    ||P*u - q||_1       
            
                (dual)   maximize    q'*w
                         subject to  P'*w = 0
                                     ||w||_infty <= 1.
            """

            m, n = P.size

            # Solve the equivalent LP 
            #
            #     minimize    [0; 1]' * [u; v]
            #     subject to  [P, -I; -P, -I] * [u; v] <= [q; -q]
            #
            #     maximize    -[q; -q]' * z 
            #     subject to  [P', -P']*z  = 0
            #                 [-I, -I]*z + 1 = 0 
            #                 z >= 0.
            
            c = matrix(n*[0.0] + m*[1.0])

            def G(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):    

                if trans=='N':
                    # y := alpha * [P, -I; -P, -I] * x + beta*y
                    u = P*x[:n]
                    y[:m] = alpha * ( u - x[n:]) + beta * y[:m]
                    y[m:] = alpha * (-u - x[n:]) + beta * y[m:]

                else:
                    # y := alpha * [P', -P'; -I, -I] * x + beta*y
                    y[:n] =  alpha * P.T * (x[:m] - x[m:]) + beta * y[:n]
                    y[n:] = -alpha * (x[:m] + x[m:]) + beta * y[n:]

            h = matrix([q, -q])
            dims = {'l': 2*m, 'q': [], 's': []}

            def F(W): 

                """
                Returns a function f(x, y, z) that solves
               
                    [ 0  0  P'      -P'      ] [ x[:n] ]   [ bx[:n] ]
                    [ 0  0 -I       -I       ] [ x[n:] ]   [ bx[n:] ]
                    [ P -I -D1^{-1}  0       ] [ z[:m] ] = [ bz[:m] ]
                    [-P -I  0       -D2^{-1} ] [ z[m:] ]   [ bz[m:] ]
                
                where D1 = diag(di[:m])^2, D2 = diag(di[m:])^2 and di = W['di'].
                """
                
                # Factor A = 4*P'*D*P where D = d1.*d2 ./(d1+d2) and 
                # d1 = di[:m].^2, d2 = di[m:].^2.

                di = W['di']
                d1, d2 = di[:m]**2, di[m:]**2
                D = div( mul(d1,d2), d1+d2 )  
                A = P.T * spmatrix(4*D, range(m), range(m)) * P
                lapack.potrf(A)

                def f(x, y, z):

                    """
                    On entry bx, bz are stored in x, z.  On exit x, z contain the solution, 
                    with z scaled: z./di is returned instead of z. 
                    """"

                    # Solve for x[:n]:
                    #
                    #    A*x[:n] = bx[:n] + P' * ( ((D1-D2)*(D1+D2)^{-1})*bx[n:]
                    #              + (2*D1*D2*(D1+D2)^{-1}) * (bz[:m] - bz[m:]) ).

                    x[:n] += P.T * ( mul(div(d1-d2, d1+d2), x[n:]) + mul(2*D, z[:m]-z[m:]) )
                    lapack.potrs(A, x)

                    # x[n:] := (D1+D2)^{-1} * (bx[n:] - D1*bz[:m] - D2*bz[m:] + (D1-D2)*P*x[:n])

                    u = P*x[:n]
                    x[n:] =  div(x[n:] - mul(d1, z[:m]) - mul(d2, z[m:]) + mul(d1-d2, u), d1+d2)

                    # z[:m] := d1[:m] .* ( P*x[:n] - x[n:] - bz[:m])
                    # z[m:] := d2[m:] .* (-P*x[:n] - x[n:] - bz[m:]) 

                    z[:m] = mul(di[:m],  u - x[n:] - z[:m])
                    z[m:] = mul(di[m:], -u - x[n:] - z[m:])

                return f

            sol = solvers.conelp(c, G, h, dims, kktsolver = F) 
            return sol['x'][:n],  sol['z'][m:] - sol['z'][:m]    


**Example: SDP with diagonal linear term**

    The SDP

    .. math::

        \newcommand{\diag}{\mbox{\bf diag}\,}
        \newcommand{\ones}{{\bf 1}}
        \begin{array}{ll}
        \mbox{minimize} & \ones^T x \\
        \mbox{subject to} & W + \diag(x) \succeq 0 
        \end{array} 

    can be solved efficiently by exploiting properties of the diag 
    operator.

    :: 

        from cvxopt import blas, lapack, solvers, matrix

        def mcsdp(w):
            """
            Returns solution x, z to 

                (primal)  minimize    sum(x)
                          subject to  w + diag(x) >= 0

                (dual)    maximize    -tr(w*z)
                          subject to  diag(z) = 1
                                      z >= 0.
            """

            n = w.size[0]
            c = matrix(1.0, (n,1))

            def G(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):
                """
                    y := alpha*(-diag(x)) + beta*y.   
                """

                if trans=='N':
                    # x is a vector; y is a symmetric matrix in column major order.
                    y *= beta
                    y[::n+1] -= alpha * x

                else:   
                    # x is a symmetric matrix in column major order; y is a vector.
                    y *= beta
                    y -= alpha * x[::n+1] 
                 

            def cngrnc(r, x, alpha = 1.0):
                """
                Congruence transformation

                    x := alpha * r'*x*r.

                r and x are square matrices.
                """

                # Scale diagonal of x by 1/2.  
                x[::n+1] *= 0.5
            
                # a := tril(x)*r 
                a = +r
                tx = matrix(x, (n,n))
                blas.trmm(tx, a, side = 'L')

                # x := alpha*(a*r' + r*a') 
                blas.syr2k(r, a, tx, trans = 'T', alpha = alpha)
                x[:] = tx[:]

            dims = {'l': 0, 'q': [], 's': [n]}

            def F(W):
                """
                Returns a function f(x, y, z) that solves 

                              -diag(z)     = bx
                    -diag(x) - r*r'*z*r*r' = bz

                where r = W['r'][0] = W['rti'][0]^{-T}.
                """
           
                rti = W['rti'][0]

                # t = rti*rti' as a nonsymmetric matrix.
                t = matrix(0.0, (n,n))
                blas.gemm(rti, rti, t, transB = 'T') 

                # Cholesky factorization of tsq = t.*t.
                tsq = t**2
                lapack.potrf(tsq)

                def f(x, y, z):
                    """
                    On entry, x contains bx, y is empty, and z contains bz stored 
                    in column major order.
                    On exit, they contain the solution, with z scaled 
                    (vec(r'*z*r) is returned instead of z).

                    We first solve 
                    
                       ((rti*rti') .* (rti*rti')) * x = bx - diag(t*bz*t)
                   
                    and take z = - rti' * (diag(x) + bz) * rti.
                    """

                    # tbst := t * bz * t
                    tbst = +z
                    cngrnc(t, tbst) 

                    # x := x - diag(tbst) = bx - diag(rti*rti' * bz * rti*rti')
                    x -= tbst[::n+1]

                    # x := (t.*t)^{-1} * x = (t.*t)^{-1} * (bx - diag(t*bz*t))
                    lapack.potrs(tsq, x)

                    # z := z + diag(x) = bz + diag(x)
                    z[::n+1] += x 

                    # z := -vec(rti' * z * rti) 
                    #    = -vec(rti' * (diag(x) + bz) * rti 
                    cngrnc(rti, z, alpha = -1.0)

                return f

            sol = solvers.conelp(c, G, w[:], dims, kktsolver = F) 
            return sol['x'], sol['z']


**Example: Minimizing 1-norm subject to a 2-norm constraint**
    In the second example, we use a similar trick to solve the problem

    .. math::

        \begin{array}{ll}
        \mbox{minimize}   & \|u\|_1 \\
        \mbox{subject to} & \|Au - b\|_2 \leq 1.
        \end{array}

    The code below is efficient, if we assume that the number of rows in 
    :math:`A` is greater than or equal to the number of columns.

    ::

        def qcl1(A, b):
            """
            Returns the solution u, z of

                (primal)  minimize    || u ||_1       
                          subject to  || A * u - b ||_2  <= 1

                (dual)    maximize    b^T z - ||z||_2
                          subject to  || A'*z ||_inf <= 1.

            Exploits structure, assuming A is m by n with m >= n. 
            """

            m, n = A.size

            # Solve equivalent cone LP with variables x = [u; v].
            #
            #     minimize    [0; 1]' * x 
            #     subject to  [ I  -I ] * x <=  [  0 ]   (componentwise)
            #                 [-I  -I ] * x <=  [  0 ]   (componentwise)
            #                 [ 0   0 ] * x <=  [  1 ]   (SOC)
            #                 [-A   0 ]         [ -b ]
            #
            #     maximize    -t + b' * w
            #     subject to  z1 - z2 = A'*w
            #                 z1 + z2 = 1
            #                 z1 >= 0,  z2 >=0,  ||w||_2 <= t.
             
            c = matrix(n*[0.0] + n*[1.0])
            h = matrix( 0.0, (2*n + m + 1, 1))
            h[2*n] = 1.0
            h[2*n+1:] = -b

            def G(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):    
                y *= beta
                if trans=='N':
                    # y += alpha * G * x 
                    y[:n] += alpha * (x[:n] - x[n:2*n]) 
                    y[n:2*n] += alpha * (-x[:n] - x[n:2*n]) 
                    y[2*n+1:] -= alpha * A*x[:n] 

                else:
                    # y += alpha * G'*x 
                    y[:n] += alpha * (x[:n] - x[n:2*n] - A.T * x[-m:])  
                    y[n:] -= alpha * (x[:n] + x[n:2*n]) 


            def Fkkt(W): 
                """
                Returns a function f(x, y, z) that solves
                
                    [ 0   G'   ] [ x ] = [ bx ]
                    [ G  -W'*W ] [ z ]   [ bz ].
                """

                # First factor 
                #
                #     S = G' * W**-1 * W**-T * G
                #       = [0; -A]' * W3^-2 * [0; -A] + 4 * (W1**2 + W2**2)**-1 
                #
                # where
                #
                #     W1 = diag(d1) with d1 = W['d'][:n] = 1 ./ W['di'][:n]  
                #     W2 = diag(d2) with d2 = W['d'][n:] = 1 ./ W['di'][n:]  
                #     W3 = beta * (2*v*v' - J),  W3^-1 = 1/beta * (2*J*v*v'*J - J)  
                #        with beta = W['beta'][0], v = W['v'][0], J = [1, 0; 0, -I].
          
                # As = W3^-1 * [ 0 ; -A ] = 1/beta * ( 2*J*v * v' - I ) * [0; A]
                beta, v = W['beta'][0], W['v'][0]
                As = 2 * v * (v[1:].T * A)
                As[1:,:] *= -1.0
                As[1:,:] -= A
                As /= beta
              
                # S = As'*As + 4 * (W1**2 + W2**2)**-1
                S = As.T * As 
                d1, d2 = W['d'][:n], W['d'][n:]       
                d = 4.0 * (d1**2 + d2**2)**-1
                S[::n+1] += d
                lapack.potrf(S)

                def f(x, y, z):

                    # z := - W**-T * z 
                    z[:n] = -div( z[:n], d1 )
                    z[n:2*n] = -div( z[n:2*n], d2 )
                    z[2*n:] -= 2.0*v*( v[0]*z[2*n] - blas.dot(v[1:], z[2*n+1:]) ) 
                    z[2*n+1:] *= -1.0
                    z[2*n:] /= beta

                    # x := x - G' * W**-1 * z
                    x[:n] -= div(z[:n], d1) - div(z[n:2*n], d2) + As.T * z[-(m+1):]
                    x[n:] += div(z[:n], d1) + div(z[n:2*n], d2) 

                    # Solve for x[:n]:
                    #
                    #    S*x[:n] = x[:n] - (W1**2 - W2**2)(W1**2 + W2**2)^-1 * x[n:]
                    
                    x[:n] -= mul( div(d1**2 - d2**2, d1**2 + d2**2), x[n:]) 
                    lapack.potrs(S, x)
                    
                    # Solve for x[n:]:
                    #
                    #    (d1**-2 + d2**-2) * x[n:] = x[n:] + (d1**-2 - d2**-2)*x[:n]
                     
                    x[n:] += mul( d1**-2 - d2**-2, x[:n])
                    x[n:] = div( x[n:], d1**-2 + d2**-2)

                    # z := z + W^-T * G*x 
                    z[:n] += div( x[:n] - x[n:2*n], d1) 
                    z[n:2*n] += div( -x[:n] - x[n:2*n], d2) 
                    z[2*n:] += As*x[:n]

                return f

            dims = {'l': 2*n, 'q': [m+1], 's': []}
            sol = solvers.conelp(c, G, h, dims, kktsolver = Fkkt)
            if sol['status'] == 'optimal':
                return sol['x'][:n],  sol['z'][-m:]
            else:
                return None, None


**Example: 1-norm regularized least-squares** 
    As an example that illustrates how structure can be exploited in 
    :func:`coneqp <cvxopt.solvers.coneqp>`, we consider the 1-norm 
    regularized least-squares problem

    .. math::

        \begin{array}{ll}
        \mbox{minimize} & \|Ax - y\|_2^2 + \|x\|_1
        \end{array}

    with variable :math:`x`.  The problem is equivalent to the quadratic 
    program

    .. math::

        \newcommand{\ones}{{\bf 1}}
        \begin{array}{ll}
        \mbox{minimize} & \|Ax - y\|_2^2 + \ones^T u \\
        \mbox{subject to} & -u \preceq x \preceq u
        \end{array}

    with variables :math:`x` and :math:`u`.  The implementation below is 
    efficient when :math:`A` has many more columns than rows. 

    ::

        from cvxopt import matrix, spdiag, mul, div, blas, lapack, solvers, sqrt
        import math

        def l1regls(A, y):
            """
            
            Returns the solution of l1-norm regularized least-squares problem
          
                minimize || A*x - y ||_2^2  + || x ||_1.

            """

            m, n = A.size
            q = matrix(1.0, (2*n,1))
            q[:n] = -2.0 * A.T * y

            def P(u, v, alpha = 1.0, beta = 0.0 ):
                """
                    v := alpha * 2.0 * [ A'*A, 0; 0, 0 ] * u + beta * v 
                """
                v *= beta
                v[:n] += alpha * 2.0 * A.T * (A * u[:n])


            def G(u, v, alpha=1.0, beta=0.0, trans='N'):
                """
                    v := alpha*[I, -I; -I, -I] * u + beta * v  (trans = 'N' or 'T')
                """

                v *= beta
                v[:n] += alpha*(u[:n] - u[n:])
                v[n:] += alpha*(-u[:n] - u[n:])

            h = matrix(0.0, (2*n,1))


            # Customized solver for the KKT system 
            #
            #     [  2.0*A'*A  0    I      -I     ] [x[:n] ]     [bx[:n] ]
            #     [  0         0   -I      -I     ] [x[n:] ]  =  [bx[n:] ].
            #     [  I        -I   -D1^-1   0     ] [zl[:n]]     [bzl[:n]]
            #     [ -I        -I    0      -D2^-1 ] [zl[n:]]     [bzl[n:]]
            #
            # where D1 = W['di'][:n]**2, D2 = W['di'][n:]**2.
            #    
            # We first eliminate zl and x[n:]:
            #
            #     ( 2*A'*A + 4*D1*D2*(D1+D2)^-1 ) * x[:n] = 
            #         bx[:n] - (D2-D1)*(D1+D2)^-1 * bx[n:] + 
            #         D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bzl[:n] - 
            #         D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bzl[n:]           
            #
            #     x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bzl[:n]  - D2*bzl[n:] ) 
            #         - (D2-D1)*(D1+D2)^-1 * x[:n]         
            #
            #     zl[:n] = D1 * ( x[:n] - x[n:] - bzl[:n] )
            #     zl[n:] = D2 * (-x[:n] - x[n:] - bzl[n:] ).
            #
            # The first equation has the form
            #
            #     (A'*A + D)*x[:n]  =  rhs
            #
            # and is equivalent to
            #
            #     [ D    A' ] [ x:n] ]  = [ rhs ]
            #     [ A   -I  ] [ v    ]    [ 0   ].
            #
            # It can be solved as 
            #
            #     ( A*D^-1*A' + I ) * v = A * D^-1 * rhs
            #     x[:n] = D^-1 * ( rhs - A'*v ).

            S = matrix(0.0, (m,m))
            Asc = matrix(0.0, (m,n))
            v = matrix(0.0, (m,1))

            def Fkkt(W):

                # Factor 
                #
                #     S = A*D^-1*A' + I 
                #
                # where D = 2*D1*D2*(D1+D2)^-1, D1 = d[:n]**-2, D2 = d[n:]**-2.

                d1, d2 = W['di'][:n]**2, W['di'][n:]**2

                # ds is square root of diagonal of D
                ds = math.sqrt(2.0) * div( mul( W['di'][:n], W['di'][n:]), sqrt(d1+d2) )
                d3 =  div(d2 - d1, d1 + d2)
             
                # Asc = A*diag(d)^-1/2
                Asc = A * spdiag(ds**-1)

                # S = I + A * D^-1 * A'
                blas.syrk(Asc, S)
                S[::m+1] += 1.0 
                lapack.potrf(S)

                def g(x, y, z):

                    x[:n] = 0.5 * ( x[:n] - mul(d3, x[n:]) + 
                        mul(d1, z[:n] + mul(d3, z[:n])) - mul(d2, z[n:] - 
                        mul(d3, z[n:])) )
                    x[:n] = div( x[:n], ds) 

                    # Solve
                    #
                    #     S * v = 0.5 * A * D^-1 * ( bx[:n] - 
                    #         (D2-D1)*(D1+D2)^-1 * bx[n:] + 
                    #         D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bzl[:n] - 
                    #         D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bzl[n:] )
                        
                    blas.gemv(Asc, x, v)
                    lapack.potrs(S, v)
                    
                    # x[:n] = D^-1 * ( rhs - A'*v ).
                    blas.gemv(Asc, v, x, alpha=-1.0, beta=1.0, trans='T')
                    x[:n] = div(x[:n], ds)

                    # x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bzl[:n]  - D2*bzl[n:] ) 
                    #         - (D2-D1)*(D1+D2)^-1 * x[:n]         
                    x[n:] = div( x[n:] - mul(d1, z[:n]) - mul(d2, z[n:]), d1+d2 )\
                        - mul( d3, x[:n] )
                        
                    # zl[:n] = D1^1/2 * (  x[:n] - x[n:] - bzl[:n] )
                    # zl[n:] = D2^1/2 * ( -x[:n] - x[n:] - bzl[n:] ).
                    z[:n] = mul( W['di'][:n],  x[:n] - x[n:] - z[:n] ) 
                    z[n:] = mul( W['di'][n:], -x[:n] - x[n:] - z[n:] ) 

                return g

            return solvers.coneqp(P, q, G, h, kktsolver = Fkkt)['x'][:n]


.. _s-external:

Optional Solvers
================

CVXOPT includes optional interfaces to several other optimization 
libraries.

**GLPK** 
    :func:`lp <cvxopt.solvers.lp>` with the ``solver`` option set to 
    :const:`'glpk'` uses the 
    simplex algorithm in `GLPK (GNU Linear Programming Kit) 
    <http://www.gnu.org/software/glpk/glpk.html>`_.

**MOSEK** 
    :func:`lp <cvxopt.solvers.lp>`, :func:`socp <cvxopt.solvers.socp>`,
    and :func:`qp <cvxopt.solvers.qp>` with the ``solver`` option
    set to :const:`'mosek'` option use `MOSEK <http://www.mosek.com>`_
    version 5.

**DSDP** 
    :func:`sdp <cvxopt.solvers.sdp>` with the ``solver`` option set to 
    :const:`'dsdp'` uses 
    the `DSDP5.8 <http://www-unix.mcs.anl.gov/DSDP>`_.  

GLPK, MOSEK and DSDP are not included in the CVXOPT distribution and 
need to be installed separately.  


.. _s-parameters:

Algorithm Parameters
====================

In this section we list some algorithm control parameters that can be 
modified without editing the source code.  These control parameters are 
accessible via the dictionary :attr:`solvers.options`.  By default the 
dictionary is empty and the default values of the parameters are
used. 

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
    (default: :const:`0` if the problem has no second-order cone or matrix 
    inequality constraints; :const:`1` otherwise).

For example the command

>>> from cvxopt import solvers
>>> solvers.options['show_progress'] = False

turns off the screen output during calls to the solvers.

The tolerances :const:`'abstol'`, :const:`'reltol'` and :const:`'feastol'` 
have the following meaning.  :func:`conelp <cvxopt.solvers.conelp>` 
terminates with status :const:`'optimal'` if

.. math::

    s \succeq 0, \qquad 
    \frac{\|Gx + s - h\|_2} {\max\{1,\|h\|_2\}} \leq 
        \epsilon_\mathrm{feas}, \qquad 
    \frac{\|Ax-b\|_2}{\max\{1,\|b\|_2\}} \leq \epsilon_\mathrm{feas}, 
        \qquad

and

.. math::

    z \succeq 0, \qquad
    \frac{\|G^Tz +  A^Ty + c\|_2}{\max\{1,\|c\|_2\}} \leq 
        \epsilon_\mathrm{feas}, 

and

.. math::

    s^T z \leq \epsilon_\mathrm{abs} \qquad \mbox{or} \qquad
    \left( \min\left\{c^Tx,  h^T z + b^Ty \right\} < 0 \quad 
        \mbox{and} \quad
    \frac{s^Tz} {-\min\{c^Tx, h^Tz + b^T y\}} \leq \epsilon_\mathrm{rel} 
    \right).

It returns with status :const:`'primal infeasible'` if 

.. math::

    z \succeq 0, \qquad \qquad 
    \frac{\|G^Tz +A^Ty\|_2}{\max\{1, \|c\|_2\}} \leq 
        \epsilon_\mathrm{feas}, \qquad 
    h^Tz +b^Ty = -1.

It returns with status :const:`'dual infeasible'` if 

.. math::

    s \succeq 0, \qquad \qquad
    \frac{\|Gx+s\|_2}{\max\{1, \|h\|_2\}} \leq \epsilon_\mathrm{feas}, 
    \qquad
    \frac{\|Ax\|_2}{\max\{1, \|b\|_2\}} \leq \epsilon_\mathrm{feas}, 
    \qquad c^Tx = -1.

The functions :func:`lp <cvxopt.solvers.lp`, 
:func:`socp <cvxopt.solvers.socp>` and 
:func:`sdp <cvxopt.solvers.sdp>` call :func:`conelp` 
and hence use the same stopping criteria.

The function :func:`coneqp <cvxopt.solvers.coneqp>` terminates with 
status :const:`'optimal'` if

.. math::

    s \succeq 0, \qquad 
    \frac{\|Gx + s - h\|_2} {\max\{1,\|h\|_2\}} \leq 
        \epsilon_\mathrm{feas}, \qquad 
    \frac{\|Ax-b\|_2}{\max\{1,\|b\|_2\}} \leq \epsilon_\mathrm{feas}, 

and

.. math::
    z \succeq 0, \qquad 
    \frac{\|Px + G^Tz +  A^Ty + q\|_2}{\max\{1,\|q\|_2\}} \leq 
        \epsilon_\mathrm{feas}, 

and at least one of the following three conditions is satisfied:

.. math:: 

    s^T z \leq \epsilon_\mathrm{abs} 

or

.. math::

    \left( \frac{1}{2}x^TPx + q^Tx < 0, \quad 
    \mbox{and}\quad \frac{s^Tz} {-(1/2)x^TPx - q^Tx} \leq 
        \epsilon_\mathrm{rel} \right) 

or

.. math::
    \left( L(x,y,z) > 0 \quad \mbox{and} \quad \frac{s^Tz}
        {L(x,y,z)} \leq \epsilon_\mathrm{rel} \right).

Here

.. math::

    L(x,y,z) = \frac{1}{2}x^TPx + q^Tx  + z^T (Gx-h) + y^T(Ax-b).

The function :func:`qp <cvxopt.solvers.qp>` calls 
:func:`coneqp` and hence uses the same 
stopping criteria.

The control parameters listed in the GLPK documentation are set 
to their default values and can be customized by making an entry
in :attr:`solvers.options['glpk']`.  The entry must be a 
dictionary in which the key/value pairs are GLPK parameter names 
and values.  For example, the command 

>>> from cvxopt import solvers 
>>> solvers.options['glpk'] = {'msg_lev' : 'GLP_MSG_OFF'}

turns off the screen output in subsequent 
:func:`lp <cvxopt.solvers.lp>` calls with the :const:`'glpk'` option.

The MOSEK interior-point algorithm parameters are set to their default 
values.  They can be modified by adding an entry 
:attr:`solvers.options['MOSEK']`.  This entry is a dictionary with 
MOSEK parameter/value pairs, with the parameter names imported from
:mod:`mosek`.  For details see Section 15 of the MOSEK Python API Manual.

For example, the commands

>>> from cvxopt import solvers 
>>> import mosek
>>> solvers.options['MOSEK'] = {mosek.iparam.log: 0}

turn off the screen output during calls of 
:func:`lp` or :func:`socp` with
the :const:`'mosek'` option.

The following control parameters in :attr:`solvers.options['dsdp']` affect the 
execution of the DSDP algorithm:

:const:`'DSDP_Monitor'` 
    the interval (in number of iterations) at which output is printed to 
    the screen (default: :const:`0`).

:const:`'DSDP_MaxIts'` 
    maximum number of iterations.

:const:`'DSDP_GapTolerance'` 
    relative accuracy (default: :const:`1e-5`).

It is also possible to override the options specified in the
dictionary :attr:`solvers.options` by passing a dictionary with
options as a keyword argument. For example, the commands

>>> from cvxopt import solvers
>>> opts = {'maxiters' : 50}
>>> solvers.conelp(c, G, h, options = opts)

override the options specified in the dictionary
:attr:`solvers.options` and use the options in the dictionary
:attr:`opts` instead. This is useful e.g. when several problem
instances should be solved in parallel, but using different options.
