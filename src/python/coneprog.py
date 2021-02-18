"""
Solver for linear and quadratic cone programs.
"""

# Copyright 2012-2021 M. Andersen and L. Vandenberghe.
# Copyright 2010-2011 L. Vandenberghe.
# Copyright 2004-2009 J. Dahl and L. Vandenberghe.
#
# This file is part of CVXOPT.
#
# CVXOPT is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# CVXOPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
if sys.version > '3': long = int

__all__ = []
options = {}


def conelp(c, G, h, dims = None, A = None, b = None, primalstart = None,
    dualstart = None, kktsolver = None, xnewcopy = None, xdot = None,
    xaxpy = None, xscal = None, ynewcopy = None, ydot = None, yaxpy = None,
    yscal = None, **kwargs):

    """
    Solves a pair of primal and dual cone programs

        minimize    c'*x
        subject to  G*x + s = h
                    A*x = b
                    s >= 0

        maximize    -h'*z - b'*y
        subject to  G'*z + A'*y + c = 0
                    z >= 0.

    The inequalities are with respect to a cone C defined as the Cartesian
    product of N + M + 1 cones:

        C = C_0 x C_1 x .... x C_N x C_{N+1} x ... x C_{N+M}.

    The first cone C_0 is the nonnegative orthant of dimension ml.
    The next N cones are second order cones of dimension mq[0], ...,
    mq[N-1].  The second order cone of dimension m is defined as

        { (u0, u1) in R x R^{m-1} | u0 >= ||u1||_2 }.

    The next M cones are positive semidefinite cones of order ms[0], ...,
    ms[M-1] >= 0.


    Input arguments (basic usage).

        c is a dense 'd' matrix of size (n,1).

        dims is a dictionary with the dimensions of the components of C.
        It has three fields.
        - dims['l'] = ml, the dimension of the nonnegative orthant C_0.
          (ml >= 0.)
        - dims['q'] = mq = [ mq[0], mq[1], ..., mq[N-1] ], a list of N
          integers with the dimensions of the second order cones C_1, ...,
          C_N.  (N >= 0 and mq[k] >= 1.)
        - dims['s'] = ms = [ ms[0], ms[1], ..., ms[M-1] ], a list of M
          integers with the orders of the semidefinite cones C_{N+1}, ...,
          C_{N+M}.  (M >= 0 and ms[k] >= 0.)
        The default value of dims is {'l': G.size[0], 'q': [], 's': []}.

        G is a dense or sparse 'd' matrix of size (K,n), where

            K = ml + mq[0] + ... + mq[N-1] + ms[0]**2 + ... + ms[M-1]**2.

        Each column of G describes a vector

            v = ( v_0, v_1, ..., v_N, vec(v_{N+1}), ..., vec(v_{N+M}) )

        in V = R^ml x R^mq[0] x ... x R^mq[N-1] x S^ms[0] x ... x S^ms[M-1]
        stored as a column vector

            [ v_0; v_1; ...; v_N; vec(v_{N+1}); ...; vec(v_{N+M}) ].

        Here, if u is a symmetric matrix of order m, then vec(u) is the
        matrix u stored in column major order as a vector of length m**2.
        We use BLAS unpacked 'L' storage, i.e., the entries in vec(u)
        corresponding to the strictly upper triangular entries of u are
        not referenced.

        h is a dense 'd' matrix of size (K,1), representing a vector in V,
        in the same format as the columns of G.

        A is a dense or sparse 'd' matrix of size (p,n).  The default value
        is a sparse 'd' matrix of size (0,n).

        b is a dense 'd' matrix of size (p,1).   The default value is a
        dense 'd' matrix of size (0,1).

        The argument primalstart is a dictionary with keys 'x', 's'.  It
        specifies an optional primal starting point.
        - primalstart['x'] is a dense 'd' matrix of size (n,1).
        - primalstart['s'] is a dense 'd' matrix of size (K,1),
          representing a vector that is strictly positive with respect
          to the cone C.

        The argument dualstart is a dictionary with keys 'y', 'z'.  It
        specifies an optional dual starting point.
        - dualstart['y'] is a dense 'd' matrix of size (p,1).
        - dualstart['z'] is a dense 'd' matrix of size (K,1), representing
          a vector that is strictly positive with respect to the cone C.

        It is assumed that rank(A) = p and rank([A; G]) = n.

        The other arguments are normally not needed.  They make it possible
        to exploit certain types of structure, as described below.

    Output arguments.

        Returns a dictionary with keys 'status', 'x', 's', 'z', 'y',
        'primal objective', 'dual objective', 'gap', 'relative gap',
        'primal infeasibility', 'dual infeasibility', 'primal slack',
        'dual slack', 'residual as primal infeasibility certificate',
        'residual as dual infeasibility certificate', 'iterations'.

        The 'status' field has values 'optimal', 'primal infeasible',
        'dual infeasible', or 'unknown'.  The 'iterations' field is the
        number of iterations taken.  The values of the other fields depend
        on the exit status.

        Status 'optimal'.
        - 'x', 's', 'y', 'z' are an approximate solution of the primal and
          dual optimality conditions

              G*x + s = h,  A*x = b
              G'*z + A'*y + c = 0
              s >= 0, z >= 0
              s'*z = 0.

        - 'primal objective': the primal objective c'*x.
        - 'dual objective': the dual objective -h'*z - b'*y.
        - 'gap': the duality gap s'*z.
        - 'relative gap': the relative gap, defined as s'*z / -c'*x if
          the primal objective is negative, s'*z / -(h'*z + b'*y) if the
          dual objective is positive, and None otherwise.
        - 'primal infeasibility': the residual in the primal constraints,
          defined as the maximum of the residual in the inequalities

              || G*x + s + h || / max(1, ||h||)

          and the residual in the equalities

              || A*x - b || / max(1, ||b||).

        - 'dual infeasibility': the residual in the dual constraints,
          defined as

              || G'*z + A'*y + c || / max(1, ||c||).

        - 'primal slack': the smallest primal slack, sup {t | s >= t*e },
           where

              e = ( e_0, e_1, ..., e_N, e_{N+1}, ..., e_{M+N} )

          is the identity vector in C.  e_0 is an ml-vector of ones,
          e_k, k = 1,..., N, are unit vectors (1,0,...,0) of length mq[k],
          and e_k = vec(I) where I is the identity matrix of order ms[k].
        - 'dual slack': the smallest dual slack, sup {t | z >= t*e }.
        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate': None.
        The primal infeasibility is guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  The dual infeasibility
        is guaranteed to be less than solvers.options['feastol']
        (default 1e-7).  The gap is less than solvers.options['abstol']
        (default 1e-7) or the relative gap is less than
        solvers.options['reltol'] (default 1e-6).

        Status 'primal infeasible'.
        - 'x', 's': None.
        - 'y', 'z' are an approximate certificate of infeasibility

              -h'*z - b'*y = 1,  G'*z + A'*y = 0,  z >= 0.

        - 'primal objective': None.
        - 'dual objective': 1.0.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': None.
        - 'dual slack': the smallest dual slack, sup {t | z >= t*e }.
        - 'residual as primal infeasibility certificate': the residual in
          the condition of the infeasibility certificate, defined as

              || G'*z + A'*y || / max(1, ||c||).

        - 'residual as dual infeasibility certificate': None.
        The residual as primal infeasiblity certificate is guaranteed
        to be less than solvers.options['feastol'] (default 1e-7).

        Status 'dual infeasible'.
        - 'x', 's' are an approximate proof of dual infeasibility

              c'*x = -1,  G*x + s = 0,  A*x = 0,  s >= 0.

        - 'y', 'z': None.
        - 'primal objective': -1.0.
        - 'dual objective': None.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': the smallest primal slack, sup {t | s >= t*e}.
        - 'dual slack': None.
        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate: the residual in
          the conditions of the infeasibility certificate, defined as
          the maximum of

              || G*x + s || / max(1, ||h||) and || A*x || / max(1, ||b||).

        The residual as dual infeasiblity certificate is guaranteed
        to be less than solvers.options['feastol'] (default 1e-7).

        Status 'unknown'.
        - 'x', 'y', 's', 'z' are the last iterates before termination.
          These satisfy s > 0 and z > 0, but are not necessarily feasible.
        - 'primal objective': the primal cost c'*x.
        - 'dual objective': the dual cost -h'*z - b'*y.
        - 'gap': the duality gap s'*z.
        - 'relative gap': the relative gap, defined as s'*z / -c'*x if the
          primal cost is negative, s'*z / -(h'*z + b'*y) if the dual cost
          is positive, and None otherwise.
        - 'primal infeasibility ': the residual in the primal constraints,
          defined as the maximum of the residual in the inequalities

              || G*x + s + h || / max(1, ||h||)

          and the residual in the equalities

              || A*x - b || / max(1, ||b||).

        - 'dual infeasibility': the residual in the dual constraints,
          defined as

              || G'*z + A'*y + c || / max(1, ||c||).

        - 'primal slack': the smallest primal slack, sup {t | s >= t*e}.
        - 'dual slack': the smallest dual slack, sup {t | z >= t*e}.
        - 'residual as primal infeasibility certificate': None if
           h'*z + b'*y >= 0; the residual

              || G'*z + A'*y || / ( -(h'*z + b'*y) * max(1, ||c||) )

          otherwise.
        - 'residual as dual infeasibility certificate':
          None if c'*x >= 0; the maximum of the residuals

              || G*x + s || / ( -c'*x * max(1, ||h||) )

          and

              || A*x || / ( -c'*x * max(1, ||b||) )

          otherwise.
        Termination with status 'unknown' indicates that the algorithm
        failed to find a solution that satisfies the specified tolerances.
        In some cases, the returned solution may be fairly accurate.  If
        the primal and dual infeasibilities, the gap, and the relative gap
        are small, then x, y, s, z are close to optimal.  If the residual
        as primal infeasibility certificate is small, then

            y / (-h'*z - b'*y),   z / (-h'*z - b'*y)

        provide an approximate certificate of primal infeasibility.  If
        the residual as certificate of dual infeasibility is small, then

            x / (-c'*x),   s / (-c'*x)

        provide an approximate proof of dual infeasibility.


    Advanced usage.

        Three mechanisms are provided to express problem structure.

        1.  The user can provide a customized routine for solving linear
        equations (`KKT systems')

            [ 0  A'  G'   ] [ ux ]   [ bx ]
            [ A  0   0    ] [ uy ] = [ by ].
            [ G  0  -W'*W ] [ uz ]   [ bz ]

        W is a scaling matrix, a block diagonal mapping

           W*z = ( W0*z_0, ..., W_{N+M}*z_{N+M} )

        defined as follows.

        - For the 'l' block (W_0):

              W_0 = diag(d),

          with d a positive vector of length ml.

        - For the 'q' blocks (W_{k+1}, k = 0, ..., N-1):

              W_{k+1} = beta_k * ( 2 * v_k * v_k' - J )

          where beta_k is a positive scalar, v_k is a vector in R^mq[k]
          with v_k[0] > 0 and v_k'*J*v_k = 1, and J = [1, 0; 0, -I].

        - For the 's' blocks (W_{k+N}, k = 0, ..., M-1):

              W_k * x = vec(r_k' * mat(x) * r_k)

          where r_k is a nonsingular matrix of order ms[k], and mat(x) is
          the inverse of the vec operation.

        The optional argument kktsolver is a Python function that will be
        called as f = kktsolver(W), where W is a dictionary that contains
        the parameters of the scaling:

        - W['d'] is a positive 'd' matrix of size (ml,1).
        - W['di'] is a positive 'd' matrix with the elementwise inverse of
          W['d'].
        - W['beta'] is a list [ beta_0, ..., beta_{N-1} ]
        - W['v'] is a list [ v_0, ..., v_{N-1} ]
        - W['r'] is a list [ r_0, ..., r_{M-1} ]
        - W['rti'] is a list [ rti_0, ..., rti_{M-1} ], with rti_k the
          inverse of the transpose of r_k.

        The call f = kktsolver(W) should return a function f that solves
        the KKT system by f(x, y, z).  On entry, x, y, z contain the
        righthand side bx, by, bz.  On exit, they contain the solution,
        with uz scaled: the argument z contains W*uz.  In other words,
        on exit, x, y, z are the solution of

            [ 0  A'  G'*W^{-1} ] [ ux ]   [ bx ]
            [ A  0   0         ] [ uy ] = [ by ].
            [ G  0  -W'        ] [ uz ]   [ bz ]


        2.  The linear operators G*u and A*u can be specified by providing
        Python functions instead of matrices.  This can only be done in
        combination with 1. above, i.e., it requires the kktsolver
        argument.

        If G is a function, the call G(u, v, alpha, beta, trans)
        should evaluate the matrix-vector products

            v := alpha * G * u + beta * v  if trans is 'N'
            v := alpha * G' * u + beta * v  if trans is 'T'.

        The arguments u and v are required.  The other arguments have
        default values alpha = 1.0, beta = 0.0, trans = 'N'.

        If A is a function, the call A(u, v, alpha, beta, trans) should
        evaluate the matrix-vectors products

            v := alpha * A * u + beta * v if trans is 'N'
            v := alpha * A' * u + beta * v if trans is 'T'.

        The arguments u and v are required.  The other arguments
        have default values alpha = 1.0, beta = 0.0, trans = 'N'.


        3.  Instead of using the default representation of the primal
        variable x and the dual variable y as one-column 'd' matrices,
        we can represent these variables and the corresponding parameters
        c and b by arbitrary Python objects (matrices, lists, dictionaries,
        etc.).  This can only be done in combination with 1. and 2. above,
        i.e., it requires a user-provided KKT solver and an operator
        description of the linear mappings.  It also requires the arguments
        xnewcopy, xdot, xscal, xaxpy, ynewcopy, ydot, yscal, yaxpy.  These
        arguments are functions defined as follows.

        If X is the vector space of primal variables x, then:
        - xnewcopy(u) creates a new copy of the vector u in X.
        - xdot(u, v) returns the inner product of two vectors u and v in X.
        - xscal(alpha, u) computes u := alpha*u, where alpha is a scalar
          and u is a vector in X.
        - xaxpy(u, v, alpha = 1.0) computes v := alpha*u + v for a scalar
          alpha and two vectors u and v in X.
        If this option is used, the argument c must be in the same format
        as x, the argument G must be a Python function, the argument A
        must be a Python function or None, and the argument kktsolver is
        required.

        If Y is the vector space of primal variables y:
        - ynewcopy(u) creates a new copy of the vector u in Y.
        - ydot(u, v) returns the inner product of two vectors u and v in Y.
        - yscal(alpha, u) computes u := alpha*u, where alpha is a scalar
          and u is a vector in Y.
        - yaxpy(u, v, alpha = 1.0) computes v := alpha*u + v for a scalar
          alpha and two vectors u and v in Y.
        If this option is used, the argument b must be in the same format
        as y, the argument A must be a Python function or None, and the
        argument kktsolver is required.


    Control parameters.

        The following control parameters can be modified by adding an
        entry to the dictionary options.

        options['show_progress'] True/False (default: True)
        options['maxiters'] positive integer (default: 100)
        options['refinement'] positive integer (default: 0 for problems
            with no second-order cone and matrix inequality constraints;
            1 otherwise)
        options['abstol'] scalar (default: 1e-7 )
        options['reltol'] scalar (default: 1e-6)
        options['feastol'] scalar (default: 1e-7).

    """
    import math
    from cvxopt import base, blas, misc, matrix, spmatrix

    EXPON = 3
    STEP = 0.99

    options = kwargs.get('options',globals()['options'])

    DEBUG = options.get('debug', False)

    KKTREG = options.get('kktreg',None)
    if KKTREG is None:
        pass
    elif not isinstance(KKTREG,(float,int,long)) or KKTREG < 0.0:
        raise ValueError("options['kktreg'] must be a nonnegative scalar")

    MAXITERS = options.get('maxiters',100)
    if not isinstance(MAXITERS,(int,long)) or MAXITERS < 1:
        raise ValueError("options['maxiters'] must be a positive integer")

    ABSTOL = options.get('abstol',1e-7)
    if not isinstance(ABSTOL,(float,int,long)):
        raise ValueError("options['abstol'] must be a scalar")

    RELTOL = options.get('reltol',1e-6)
    if not isinstance(RELTOL,(float,int,long)):
        raise ValueError("options['reltol'] must be a scalar")

    if RELTOL <= 0.0 and ABSTOL <= 0.0 :
        raise ValueError("at least one of options['reltol'] and " \
            "options['abstol'] must be positive")

    FEASTOL = options.get('feastol',1e-7)
    if not isinstance(FEASTOL,(float,int,long)) or FEASTOL <= 0.0:
        raise ValueError("options['feastol'] must be a positive scalar")

    show_progress = options.get('show_progress', True)

    if kktsolver is None:
        if dims and (dims['q'] or dims['s']):
            kktsolver = 'qr'
        else:
            kktsolver = 'chol2'
    defaultsolvers = ('ldl', 'ldl2', 'qr', 'chol', 'chol2')
    if isinstance(kktsolver,str) and kktsolver not in defaultsolvers:
        raise ValueError("'%s' is not a valid value for kktsolver" \
            %kktsolver)

    # Argument error checking depends on level of customization.
    customkkt = not isinstance(kktsolver,str)
    matrixG = isinstance(G, (matrix, spmatrix))
    matrixA = isinstance(A, (matrix, spmatrix))
    if (not matrixG or (not matrixA and A is not None)) and not customkkt:
        raise ValueError("use of function valued G, A requires a "\
            "user-provided kktsolver")
    customx = (xnewcopy != None or xdot != None or xaxpy != None or
        xscal != None)
    if customx and (matrixG or matrixA or not customkkt):
        raise ValueError("use of non-vector type for x requires "\
            "function valued G, A and user-provided kktsolver")
    customy = (ynewcopy != None or ydot != None or yaxpy != None or
        yscal != None)
    if customy and (matrixA or not customkkt):
        raise ValueError("use of non-vector type for y requires "\
            "function valued A and user-provided kktsolver")


    if not customx and (not isinstance(c,matrix) or c.typecode != 'd' or c.size[1] != 1):
        raise TypeError("'c' must be a 'd' matrix with one column")

    if not isinstance(h,matrix) or h.typecode != 'd' or h.size[1] != 1:
        raise TypeError("'h' must be a 'd' matrix with 1 column")

    if not dims: dims = {'l': h.size[0], 'q': [], 's': []}
    if not isinstance(dims['l'],(int,long)) or dims['l'] < 0:
        raise TypeError("'dims['l']' must be a nonnegative integer")
    if [ k for k in dims['q'] if not isinstance(k,(int,long)) or k < 1 ]:
        raise TypeError("'dims['q']' must be a list of positive integers")
    if [ k for k in dims['s'] if not isinstance(k,(int,long)) or k < 0 ]:
        raise TypeError("'dims['s']' must be a list of nonnegative " \
            "integers")

    refinement = options.get('refinement',None)
    if refinement is None:
        if dims['q'] or dims['s']:
            refinement = 1
        else:
            refinement = 0
    elif not isinstance(refinement,(int,long)) or refinement < 0:
        raise ValueError("options['refinement'] must be a nonnegative integer")

    cdim = dims['l'] + sum(dims['q']) + sum([k**2 for k in dims['s']])
    cdim_pckd = dims['l'] + sum(dims['q']) + sum([k*(k+1)/2 for k in
        dims['s']])
    cdim_diag = dims['l'] + sum(dims['q']) + sum(dims['s'])

    if h.size[0] != cdim:
        raise TypeError("'h' must be a 'd' matrix of size (%d,1)" %cdim)

    # Data for kth 'q' constraint are found in rows indq[k]:indq[k+1] of G.
    indq = [ dims['l'] ]
    for k in dims['q']:  indq = indq + [ indq[-1] + k ]

    # Data for kth 's' constraint are found in rows inds[k]:inds[k+1] of G.
    inds = [ indq[-1] ]
    for k in dims['s']:  inds = inds + [ inds[-1] + k**2 ]

    if matrixG:
        if G.typecode != 'd' or G.size != (cdim, c.size[0]):
            raise TypeError("'G' must be a 'd' matrix of size (%d, %d)"\
                %(cdim, c.size[0]))
        def Gf(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
            misc.sgemv(G, x, y, dims, trans = trans, alpha = alpha,
                beta = beta)
    else:
        Gf = G

    if A is None:
        if customx or customy:
            def A(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
                if trans == 'N': pass
                else: xscal(beta, y)
        else:
            A = spmatrix([], [], [], (0, c.size[0]))
            matrixA = True
    if matrixA:
        if A.typecode != 'd' or A.size[1] != c.size[0]:
            raise TypeError("'A' must be a 'd' matrix with %d columns "\
                %c.size[0])
        def Af(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
            base.gemv(A, x, y, trans = trans, alpha = alpha, beta = beta)
    else:
        Af = A

    if not customy:
        if b is None: b = matrix(0.0, (0,1))
        if not isinstance(b,matrix) or b.typecode != 'd' or b.size[1] != 1:
            raise TypeError("'b' must be a 'd' matrix with one column")
        if matrixA and b.size[0] != A.size[0]:
            raise TypeError("'b' must have length %d" %A.size[0])
    else:
        if b is None:
            raise ValueError("use of non vector type for y requires b")


    # kktsolver(W) returns a routine for solving 3x3 block KKT system
    #
    #     [ 0   A'  G'*W^{-1} ] [ ux ]   [ bx ]
    #     [ A   0   0         ] [ uy ] = [ by ].
    #     [ G   0   -W'       ] [ uz ]   [ bz ]

    if kktsolver in defaultsolvers:
        if KKTREG is None and (b.size[0] > c.size[0] or b.size[0] + cdim_pckd < c.size[0]):
           raise ValueError("Rank(A) < p or Rank([G; A]) < n")
        if kktsolver == 'ldl':
            factor = misc.kkt_ldl(G, dims, A, kktreg = KKTREG)
        elif kktsolver == 'ldl2':
            factor = misc.kkt_ldl2(G, dims, A)
        elif kktsolver == 'qr':
            factor = misc.kkt_qr(G, dims, A)
        elif kktsolver == 'chol':
            factor = misc.kkt_chol(G, dims, A)
        else:
            factor = misc.kkt_chol2(G, dims, A)
        def kktsolver(W):
            return factor(W)


    # res() evaluates residual in 5x5 block KKT system
    #
    #     [ vx   ]    [ 0         ]   [ 0   A'  G'  c ] [ ux        ]
    #     [ vy   ]    [ 0         ]   [-A   0   0   b ] [ uy        ]
    #     [ vz   ] += [ W'*us     ] - [-G   0   0   h ] [ W^{-1}*uz ]
    #     [ vtau ]    [ dg*ukappa ]   [-c' -b' -h'  0 ] [ utau/dg   ]
    #
    #           vs += lmbda o (dz + ds)
    #       vkappa += lmbdg * (dtau + dkappa).

    ws3, wz3 = matrix(0.0, (cdim,1)), matrix(0.0, (cdim,1))
    def res(ux, uy, uz, utau, us, ukappa, vx, vy, vz, vtau, vs, vkappa, W,
        dg, lmbda):

        # vx := vx - A'*uy - G'*W^{-1}*uz - c*utau/dg
        Af(uy, vx, alpha = -1.0, beta = 1.0, trans = 'T')
        blas.copy(uz, wz3)
        misc.scale(wz3, W, inverse = 'I')
        Gf(wz3, vx, alpha = -1.0, beta = 1.0, trans = 'T')
        xaxpy(c, vx, alpha = -utau[0]/dg)

        # vy := vy + A*ux - b*utau/dg
        Af(ux, vy, alpha = 1.0, beta = 1.0)
        yaxpy(b, vy, alpha = -utau[0]/dg)

        # vz := vz + G*ux - h*utau/dg + W'*us
        Gf(ux, vz, alpha = 1.0, beta = 1.0)
        blas.axpy(h, vz, alpha = -utau[0]/dg)
        blas.copy(us, ws3)
        misc.scale(ws3, W, trans = 'T')
        blas.axpy(ws3, vz)

        # vtau := vtau + c'*ux + b'*uy + h'*W^{-1}*uz + dg*ukappa
        vtau[0] += dg*ukappa[0] + xdot(c,ux) + ydot(b,uy) + \
            misc.sdot(h, wz3, dims)

        # vs := vs + lmbda o (uz + us)
        blas.copy(us, ws3)
        blas.axpy(uz, ws3)
        misc.sprod(ws3, lmbda, dims, diag = 'D')
        blas.axpy(ws3, vs)

        # vkappa += vkappa + lmbdag * (utau + ukappa)
        vkappa[0] += lmbda[-1] * (utau[0] + ukappa[0])


    if xnewcopy is None: xnewcopy = matrix
    if xdot is None: xdot = blas.dot
    if xaxpy is None: xaxpy = blas.axpy
    if xscal is None: xscal = blas.scal
    def xcopy(x, y):
        xscal(0.0, y)
        xaxpy(x, y)
    if ynewcopy is None: ynewcopy = matrix
    if ydot is None: ydot = blas.dot
    if yaxpy is None: yaxpy = blas.axpy
    if yscal is None: yscal = blas.scal
    def ycopy(x, y):
        yscal(0.0, y)
        yaxpy(x, y)

    resx0 = max(1.0, math.sqrt(xdot(c,c)))
    resy0 = max(1.0, math.sqrt(ydot(b,b)))
    resz0 = max(1.0, misc.snrm2(h, dims))

    # Select initial points.

    x = xnewcopy(c);  xscal(0.0, x)
    y = ynewcopy(b);  yscal(0.0, y)
    s, z = matrix(0.0, (cdim,1)), matrix(0.0, (cdim,1))
    dx, dy = xnewcopy(c), ynewcopy(b)
    ds, dz = matrix(0.0, (cdim,1)), matrix(0.0, (cdim,1))
    dkappa, dtau = matrix(0.0, (1,1)), matrix(0.0, (1,1))

    if primalstart is None or dualstart is None:

        # Factor
        #
        #     [ 0   A'  G' ]
        #     [ A   0   0  ].
        #     [ G   0  -I  ]

        W = {}
        W['d'] = matrix(1.0, (dims['l'], 1))
        W['di'] = matrix(1.0, (dims['l'], 1))
        W['v'] = [ matrix(0.0, (m,1)) for m in dims['q'] ]
        W['beta'] = len(dims['q']) * [ 1.0 ]
        for v in W['v']: v[0] = 1.0
        W['r'] = [ matrix(0.0, (m,m)) for m in dims['s'] ]
        W['rti'] = [ matrix(0.0, (m,m)) for m in dims['s'] ]
        for r in W['r']: r[::r.size[0]+1 ] = 1.0
        for rti in W['rti']: rti[::rti.size[0]+1 ] = 1.0
        try: f = kktsolver(W)
        except ArithmeticError:
            raise ValueError("Rank(A) < p or Rank([G; A]) < n")

    if primalstart is None:

        # minimize    || G * x - h ||^2
        # subject to  A * x = b
        #
        # by solving
        #
        #     [ 0   A'  G' ]   [ x  ]   [ 0 ]
        #     [ A   0   0  ] * [ dy ] = [ b ].
        #     [ G   0  -I  ]   [ -s ]   [ h ]

        xscal(0.0, x)
        ycopy(b, dy)
        blas.copy(h, s)
        try: f(x, dy, s)
        except ArithmeticError:
            raise ValueError("Rank(A) < p or Rank([G; A]) < n")
        blas.scal(-1.0, s)

    else:
        xcopy(primalstart['x'], x)
        blas.copy(primalstart['s'], s)

    # ts = min{ t | s + t*e >= 0 }
    ts = misc.max_step(s, dims)
    if ts >= 0 and primalstart:
        raise ValueError("initial s is not positive")


    if dualstart is None:

        # minimize   || z ||^2
        # subject to G'*z + A'*y + c = 0
        #
        # by solving
        #
        #     [ 0   A'  G' ] [ dx ]   [ -c ]
        #     [ A   0   0  ] [ y  ] = [  0 ].
        #     [ G   0  -I  ] [ z  ]   [  0 ]

        xcopy(c, dx);
        xscal(-1.0, dx)
        yscal(0.0, y)
        blas.scal(0.0, z)
        try: f(dx, y, z)
        except ArithmeticError:
            raise ValueError("Rank(A) < p or Rank([G; A]) < n")

    else:
        if 'y' in dualstart: ycopy(dualstart['y'], y)
        blas.copy(dualstart['z'], z)

    # tz = min{ t | z + t*e >= 0 }
    tz = misc.max_step(z, dims)
    if tz >= 0 and dualstart:
        raise ValueError("initial z is not positive")

    nrms = misc.snrm2(s, dims)
    nrmz = misc.snrm2(z, dims)

    if primalstart is None and dualstart is None:

        gap = misc.sdot(s, z, dims)
        pcost = xdot(c,x)
        dcost = -ydot(b,y) - misc.sdot(h, z, dims)
        if pcost < 0.0:
            relgap = gap / -pcost
        elif dcost > 0.0:
            relgap = gap / dcost
        else:
            relgap = None

        if (ts <= 0 and tz <= 0 and (gap <= ABSTOL or ( relgap is not None
            and relgap <= RELTOL ))) and KKTREG is None:

            # The initial points we constructed happen to be feasible and
            # optimal.

            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(s, m, ind)
                misc.symm(z, m, ind)
                ind += m**2

            # rx = A'*y + G'*z + c
            rx = xnewcopy(c)
            Af(y, rx, beta = 1.0, trans = 'T')
            Gf(z, rx, beta = 1.0, trans = 'T')
            resx = math.sqrt( xdot(rx, rx) )

            # ry = b - A*x
            ry = ynewcopy(b)
            Af(x, ry, alpha = -1.0, beta = 1.0)
            resy = math.sqrt( ydot(ry, ry) )

            # rz = s + G*x - h
            rz = matrix(0.0, (cdim,1))
            Gf(x, rz)
            blas.axpy(s, rz)
            blas.axpy(h, rz, alpha = -1.0)
            resz = misc.snrm2(rz, dims)

            pres = max(resy/resy0, resz/resz0)
            dres = resx/resx0
            cx, by, hz = xdot(c,x), ydot(b,y), misc.sdot(h, z, dims)

            if show_progress:
                print("Optimal solution found.")
            return { 'x': x, 'y': y, 's': s, 'z': z,
                'status': 'optimal',
                'gap': gap,
                'relative gap': relgap,
                'primal objective': cx,
                'dual objective': -(by + hz),
                'primal infeasibility': pres,
                'primal slack': -ts,
                'dual slack': -tz,
                'dual infeasibility': dres,
                'residual as primal infeasibility certificate': None,
                'residual as dual infeasibility certificate': None,
                'iterations': 0 }

        if ts >= -1e-8 * max(nrms, 1.0):
            a = 1.0 + ts
            s[:dims['l']] += a
            s[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                s[ind : ind+m*m : m+1] += a
                ind += m**2

        if tz >= -1e-8 * max(nrmz, 1.0):
            a = 1.0 + tz
            z[:dims['l']] += a
            z[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                z[ind : ind+m*m : m+1] += a
                ind += m**2

    elif primalstart is None and dualstart is not None:

        if ts >= -1e-8 * max(nrms, 1.0):
            a = 1.0 + ts
            s[:dims['l']] += a
            s[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                s[ind : ind+m*m : m+1] += a
                ind += m**2

    elif primalstart is not None and dualstart is None:

        if tz >= -1e-8 * max(nrmz, 1.0):
            a = 1.0 + tz
            z[:dims['l']] += a
            z[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                z[ind : ind+m*m : m+1] += a
                ind += m**2


    tau, kappa = 1.0, 1.0

    rx, hrx = xnewcopy(c), xnewcopy(c)
    ry, hry = ynewcopy(b), ynewcopy(b)
    rz, hrz = matrix(0.0, (cdim,1)), matrix(0.0, (cdim,1))
    sigs = matrix(0.0, (sum(dims['s']), 1))
    sigz = matrix(0.0, (sum(dims['s']), 1))
    lmbda = matrix(0.0, (cdim_diag + 1, 1))
    lmbdasq = matrix(0.0, (cdim_diag + 1, 1))

    gap = misc.sdot(s, z, dims)

    for iters in range(MAXITERS+1):

        # hrx = -A'*y - G'*z
        Af(y, hrx, alpha = -1.0, trans = 'T')
        Gf(z, hrx, alpha = -1.0, beta = 1.0, trans = 'T')
        hresx = math.sqrt( xdot(hrx, hrx) )

        # rx = hrx - c*tau
        #    = -A'*y - G'*z - c*tau
        xcopy(hrx, rx)
        xaxpy(c, rx, alpha = -tau)
        resx = math.sqrt( xdot(rx, rx) ) / tau

        # hry = A*x
        Af(x, hry)
        hresy = math.sqrt( ydot(hry, hry) )

        # ry = hry - b*tau
        #    = A*x - b*tau
        ycopy(hry, ry)
        yaxpy(b, ry, alpha = -tau)
        resy = math.sqrt( ydot(ry, ry) ) / tau

        # hrz = s + G*x
        Gf(x, hrz)
        blas.axpy(s, hrz)
        hresz = misc.snrm2(hrz, dims)

        # rz = hrz - h*tau
        #    = s + G*x - h*tau
        blas.scal(0, rz)
        blas.axpy(hrz, rz)
        blas.axpy(h, rz, alpha = -tau)
        resz = misc.snrm2(rz, dims) / tau

        # rt = kappa + c'*x + b'*y + h'*z
        cx, by, hz = xdot(c,x), ydot(b,y), misc.sdot(h, z, dims)
        rt = kappa + cx + by + hz

        # Statistics for stopping criteria.
        pcost, dcost = cx / tau, -(by + hz) / tau
        if pcost < 0.0:
            relgap = gap / -pcost
        elif dcost > 0.0:
            relgap = gap / dcost
        else:
            relgap = None
        pres = max(resy/resy0, resz/resz0)
        dres = resx/resx0
        if hz + by < 0.0:
           pinfres =  hresx / resx0 / (-hz - by)
        else:
           pinfres =  None
        if cx < 0.0:
           dinfres = max(hresy / resy0, hresz/resz0) / (-cx)
        else:
           dinfres = None

        if show_progress:
            if iters == 0:
                print("% 10s% 12s% 10s% 8s% 7s % 5s" %("pcost", "dcost",
                    "gap", "pres", "dres", "k/t"))
            print("%2d: % 8.4e % 8.4e % 4.0e% 7.0e% 7.0e% 7.0e" \
                %(iters, pcost, dcost, gap, pres, dres, kappa/tau))


        if ( pres <= FEASTOL and dres <= FEASTOL and ( gap <= ABSTOL or
            (relgap is not None and relgap <= RELTOL) ) ) or \
            iters == MAXITERS:
            xscal(1.0/tau, x)
            yscal(1.0/tau, y)
            blas.scal(1.0/tau, s)
            blas.scal(1.0/tau, z)
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(s, m, ind)
                misc.symm(z, m, ind)
                ind += m**2
            ts = misc.max_step(s, dims)
            tz = misc.max_step(z, dims)
            if iters == MAXITERS:
                if show_progress:
                    print("Terminated (maximum number of iterations "\
                        "reached).")
                return { 'x': x, 'y': y, 's': s, 'z': z,
                    'status': 'unknown',
                    'gap': gap,
                    'relative gap': relgap,
                    'primal objective': pcost,
                    'dual objective' : dcost,
                    'primal infeasibility': pres,
                    'dual infeasibility': dres,
                    'primal slack': -ts,
                    'dual slack': -tz,
                    'residual as primal infeasibility certificate':
                        pinfres,
                    'residual as dual infeasibility certificate':
                        dinfres,
                    'iterations': iters}

            else:
                if show_progress:
                    print("Optimal solution found.")
                return { 'x': x, 'y': y, 's': s, 'z': z,
                    'status': 'optimal',
                    'gap': gap,
                    'relative gap': relgap,
                    'primal objective': pcost,
                    'dual objective' : dcost,
                    'primal infeasibility': pres,
                    'dual infeasibility': dres,
                    'primal slack': -ts,
                    'dual slack': -tz,
                    'residual as primal infeasibility certificate': None,
                    'residual as dual infeasibility certificate': None,
                    'iterations': iters }

        elif pinfres is not None and pinfres <= FEASTOL:
            yscal(1.0/(-hz - by), y)
            blas.scal(1.0/(-hz - by), z)
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(z, m, ind)
                ind += m**2
            tz = misc.max_step(z, dims)
            if show_progress:
                print("Certificate of primal infeasibility found.")
            return { 'x': None, 'y': y, 's': None, 'z': z,
                'status': 'primal infeasible',
                'gap': None,
                'relative gap': None,
                'primal objective': None,
                'dual objective' : 1.0,
                'primal infeasibility': None,
                'dual infeasibility': None,
                'primal slack': None,
                'dual slack': -tz,
                'residual as primal infeasibility certificate': pinfres,
                'residual as dual infeasibility certificate': None,
                'iterations': iters }

        elif dinfres is not None and dinfres <= FEASTOL:
            xscal(1.0/(-cx), x)
            blas.scal(1.0/(-cx), s)
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(s, m, ind)
                ind += m**2
            y, z = None, None
            ts = misc.max_step(s, dims)
            if show_progress:
                print("Certificate of dual infeasibility found.")
            return {'x': x, 'y': None, 's': s, 'z': None,
                'status': 'dual infeasible',
                'gap': None,
                'relative gap': None,
                'primal objective': -1.0,
                'dual objective' : None,
                'primal infeasibility': None,
                'dual infeasibility': None,
                'primal slack': -ts,
                'dual slack': None,
                'residual as primal infeasibility certificate': None,
                'residual as dual infeasibility certificate': dinfres,
                'iterations': iters }


        # Compute initial scaling W:
        #
        #     W * z = W^{-T} * s = lambda
        #     dg * tau = 1/dg * kappa = lambdag.

        if iters == 0:

            W = misc.compute_scaling(s, z, lmbda, dims, mnl = 0)

            #     dg = sqrt( kappa / tau )
            #     dgi = sqrt( tau / kappa )
            #     lambda_g = sqrt( tau * kappa )
            #
            # lambda_g is stored in the last position of lmbda.

            dg = math.sqrt( kappa / tau )
            dgi = math.sqrt( tau / kappa )
            lmbda[-1] = math.sqrt( tau * kappa )

        # lmbdasq := lmbda o lmbda
        misc.ssqr(lmbdasq, lmbda, dims)
        lmbdasq[-1] = lmbda[-1]**2


        # f3(x, y, z) solves
        #
        #     [ 0  A'  G'   ] [ ux        ]   [ bx ]
        #     [ A  0   0    ] [ uy        ] = [ by ].
        #     [ G  0  -W'*W ] [ W^{-1}*uz ]   [ bz ]
        #
        # On entry, x, y, z contain bx, by, bz.
        # On exit, they contain ux, uy, uz.
        #
        # Also solve
        #
        #     [ 0   A'  G'    ] [ x1        ]          [ c ]
        #     [-A   0   0     ]*[ y1        ] = -dgi * [ b ].
        #     [-G   0   W'*W  ] [ W^{-1}*z1 ]          [ h ]


        try:
            f3 = kktsolver(W)
            if iters == 0:
                x1, y1 = xnewcopy(c), ynewcopy(b)
                z1 = matrix(0.0, (cdim,1))
            xcopy(c, x1);  xscal(-1, x1)
            ycopy(b, y1)
            blas.copy(h, z1)
            f3(x1, y1, z1)
            xscal(dgi, x1)
            yscal(dgi, y1)
            blas.scal(dgi, z1)
        except ArithmeticError:
            if iters == 0 and primalstart and dualstart:
                raise ValueError("Rank(A) < p or Rank([G; A]) < n")
            else:
                xscal(1.0/tau, x)
                yscal(1.0/tau, y)
                blas.scal(1.0/tau, s)
                blas.scal(1.0/tau, z)
                ind = dims['l'] + sum(dims['q'])
                for m in dims['s']:
                    misc.symm(s, m, ind)
                    misc.symm(z, m, ind)
                    ind += m**2
                ts = misc.max_step(s, dims)
                tz = misc.max_step(z, dims)
                if show_progress:
                    print("Terminated (singular KKT matrix).")
                return { 'x': x, 'y': y, 's': s, 'z': z,
                    'status': 'unknown',
                    'gap': gap,
                    'relative gap': relgap,
                    'primal objective': pcost,
                    'dual objective' : dcost,
                    'primal infeasibility': pres,
                    'dual infeasibility': dres,
                    'primal slack': -ts,
                    'dual slack': -tz,
                    'residual as primal infeasibility certificate':
                        pinfres,
                    'residual as dual infeasibility certificate':
                        dinfres,
                    'iterations': iters }


        # f6_no_ir(x, y, z, tau, s, kappa) solves
        #
        #     [ 0         ]   [  0   A'  G'  c ] [ ux        ]    [ bx   ]
        #     [ 0         ]   [ -A   0   0   b ] [ uy        ]    [ by   ]
        #     [ W'*us     ] - [ -G   0   0   h ] [ W^{-1}*uz ] = -[ bz   ]
        #     [ dg*ukappa ]   [ -c' -b' -h'  0 ] [ utau/dg   ]    [ btau ]
        #
        #     lmbda o (uz + us) = -bs
        #     lmbdag * (utau + ukappa) = -bkappa.
        #
        # On entry, x, y, z, tau, s, kappa contain bx, by, bz, btau,
        # bkappa.  On exit, they contain ux, uy, uz, utau, ukappa.

        # th = W^{-T} * h
        if iters == 0: th = matrix(0.0, (cdim,1))
        blas.copy(h, th)
        misc.scale(th, W, trans = 'T', inverse = 'I')

        def f6_no_ir(x, y, z, tau, s, kappa):

            # Solve
            #
            #     [  0   A'  G'    0   ] [ ux        ]
            #     [ -A   0   0     b   ] [ uy        ]
            #     [ -G   0   W'*W  h   ] [ W^{-1}*uz ]
            #     [ -c' -b' -h'    k/t ] [ utau/dg   ]
            #
            #           [ bx                    ]
            #           [ by                    ]
            #         = [ bz - W'*(lmbda o\ bs) ]
            #           [ btau - bkappa/tau     ]
            #
            #     us = -lmbda o\ bs - uz
            #     ukappa = -bkappa/lmbdag - utau.


            # First solve
            #
            #     [ 0  A' G'   ] [ ux        ]   [  bx                    ]
            #     [ A  0  0    ] [ uy        ] = [ -by                    ]
            #     [ G  0 -W'*W ] [ W^{-1}*uz ]   [ -bz + W'*(lmbda o\ bs) ]

            # y := -y = -by
            yscal(-1.0, y)

            # s := -lmbda o\ s = -lmbda o\ bs
            misc.sinv(s, lmbda, dims)
            blas.scal(-1.0, s)

            # z := -(z + W'*s) = -bz + W'*(lambda o\ bs)
            blas.copy(s, ws3)
            misc.scale(ws3, W, trans = 'T')
            blas.axpy(ws3, z)
            blas.scal(-1.0, z)

            # Solve system.
            f3(x, y, z)

            # Combine with solution of
            #
            #     [ 0   A'  G'    ] [ x1         ]          [ c ]
            #     [-A   0   0     ] [ y1         ] = -dgi * [ b ]
            #     [-G   0   W'*W  ] [ W^{-1}*dzl ]          [ h ]
            #
            # to satisfy
            #
            #     -c'*x - b'*y - h'*W^{-1}*z + dg*tau = btau - bkappa/tau.

            # kappa[0] := -kappa[0] / lmbd[-1] = -bkappa / lmbdag
            kappa[0] = -kappa[0] / lmbda[-1]

            # tau[0] = tau[0] + kappa[0] / dgi = btau[0] - bkappa / tau
            tau[0] += kappa[0] / dgi

            tau[0] = dgi * ( tau[0] + xdot(c,x) + ydot(b,y) +
                misc.sdot(th, z, dims) ) / (1.0 + misc.sdot(z1, z1, dims))
            xaxpy(x1, x, alpha = tau[0])
            yaxpy(y1, y, alpha = tau[0])
            blas.axpy(z1, z, alpha = tau[0])

            # s := s - z = - lambda o\ bs - z
            blas.axpy(z, s, alpha = -1)

            kappa[0] -= tau[0]


        # f6(x, y, z, tau, s, kappa) solves the same system as f6_no_ir,
        # but applies iterative refinement.

        if iters == 0:
            if refinement or DEBUG:
                wx, wy = xnewcopy(c), ynewcopy(b)
                wz, ws = matrix(0.0, (cdim, 1)), matrix(0.0, (cdim, 1))
                wtau, wkappa = matrix(0.0), matrix(0.0)
            if refinement:
                wx2, wy2 = xnewcopy(c), ynewcopy(b)
                wz2, ws2 = matrix(0.0, (cdim, 1)), matrix(0.0, (cdim, 1))
                wtau2, wkappa2 = matrix(0.0), matrix(0.0)

        def f6(x, y, z, tau, s, kappa):
            if refinement or DEBUG:
                xcopy(x, wx)
                ycopy(y, wy)
                blas.copy(z, wz)
                wtau[0] = tau[0]
                blas.copy(s, ws)
                wkappa[0] = kappa[0]
            f6_no_ir(x, y, z, tau, s, kappa)
            for i in range(refinement):
                xcopy(wx, wx2)
                ycopy(wy, wy2)
                blas.copy(wz, wz2)
                wtau2[0] = wtau[0]
                blas.copy(ws, ws2)
                wkappa2[0] = wkappa[0]
                res(x, y, z, tau, s, kappa, wx2, wy2, wz2, wtau2, ws2,
                    wkappa2, W, dg, lmbda)
                f6_no_ir(wx2, wy2, wz2, wtau2, ws2, wkappa2)
                xaxpy(wx2, x)
                yaxpy(wy2, y)
                blas.axpy(wz2, z)
                tau[0] += wtau2[0]
                blas.axpy(ws2, s)
                kappa[0] += wkappa2[0]
            if DEBUG:
                res(x, y, z, tau, s, kappa, wx, wy, wz, wtau, ws, wkappa,
                    W, dg, lmbda)
                print("KKT residuals")
                print("    'x': %e" %math.sqrt(xdot(wx, wx)))
                print("    'y': %e" %math.sqrt(ydot(wy, wy)))
                print("    'z': %e" %misc.snrm2(wz, dims))
                print("    'tau': %e" %abs(wtau[0]))
                print("    's': %e" %misc.snrm2(ws, dims))
                print("    'kappa': %e" %abs(wkappa[0]))


        mu = blas.nrm2(lmbda)**2 / (1 + cdim_diag)
        sigma = 0.0
        for i in [0,1]:

            # Solve
            #
            #     [ 0         ]   [  0   A'  G'  c ] [ dx        ]
            #     [ 0         ]   [ -A   0   0   b ] [ dy        ]
            #     [ W'*ds     ] - [ -G   0   0   h ] [ W^{-1}*dz ]
            #     [ dg*dkappa ]   [ -c' -b' -h'  0 ] [ dtau/dg   ]
            #
            #                       [ rx   ]
            #                       [ ry   ]
            #         = - (1-sigma) [ rz   ]
            #                       [ rtau ]
            #
            #     lmbda o (dz + ds) = -lmbda o lmbda + sigma*mu*e
            #     lmbdag * (dtau + dkappa) = - kappa * tau + sigma*mu


            # ds = -lmbdasq if i is 0
            #    = -lmbdasq - dsa o dza + sigma*mu*e if i is 1
            # dkappa = -lambdasq[-1] if i is 0
            #        = -lambdasq[-1] - dkappaa*dtaua + sigma*mu if i is 1.

            blas.copy(lmbdasq, ds, n = dims['l'] + sum(dims['q']))
            ind = dims['l'] + sum(dims['q'])
            ind2 = ind
            blas.scal(0.0, ds, offset = ind)
            for m in dims['s']:
                blas.copy(lmbdasq, ds, n = m, offsetx = ind2,
                    offsety = ind, incy = m+1)
                ind += m*m
                ind2 += m
            dkappa[0] = lmbdasq[-1]
            if i == 1:
                blas.axpy(ws3, ds)
                ds[:dims['l']] -= sigma*mu
                ds[indq[:-1]] -= sigma*mu
                ind = dims['l'] + sum(dims['q'])
                ind2 = ind
                for m in dims['s']:
                    ds[ind : ind+m*m : m+1] -= sigma*mu
                    ind += m*m
                dkappa[0] += wkappa3 - sigma*mu

            # (dx, dy, dz, dtau) = (1-sigma)*(rx, ry, rz, rt)
            xcopy(rx, dx);  xscal(1.0 - sigma, dx)
            ycopy(ry, dy);  yscal(1.0 - sigma, dy)
            blas.copy(rz, dz);  blas.scal(1.0 - sigma, dz)
            dtau[0] = (1.0 - sigma) * rt

            f6(dx, dy, dz, dtau, ds, dkappa)

            # Save ds o dz and dkappa * dtau for Mehrotra correction
            if i == 0:
                blas.copy(ds, ws3)
                misc.sprod(ws3, dz, dims)
                wkappa3 = dtau[0] * dkappa[0]

            # Maximum step to boundary.
            #
            # If i is 1, also compute eigenvalue decomposition of the 's'
            # blocks in ds, dz.  The eigenvectors Qs, Qz are stored in
            # dsk, dzk.  The eigenvalues are stored in sigs, sigz.

            misc.scale2(lmbda, ds, dims)
            misc.scale2(lmbda, dz, dims)
            if i == 0:
                ts = misc.max_step(ds, dims)
                tz = misc.max_step(dz, dims)
            else:
                ts = misc.max_step(ds, dims, sigma = sigs)
                tz = misc.max_step(dz, dims, sigma = sigz)
            tt = -dtau[0] / lmbda[-1]
            tk = -dkappa[0] / lmbda[-1]
            t = max([ 0.0, ts, tz, tt, tk ])
            if t == 0.0:
                step = 1.0
            else:
                if i == 0:
                    step = min(1.0, 1.0 / t)
                else:
                    step = min(1.0, STEP / t)
            if i == 0:
                sigma = (1.0 - step)**EXPON


        # Update x, y.
        xaxpy(dx, x, alpha = step)
        yaxpy(dy, y, alpha = step)


        # Replace 'l' and 'q' blocks of ds and dz with the updated
        # variables in the current scaling.
        # Replace 's' blocks of ds and dz with the factors Ls, Lz in a
        # factorization Ls*Ls', Lz*Lz' of the updated variables in the
        # current scaling.

        # ds := e + step*ds for 'l' and 'q' blocks.
        # dz := e + step*dz for 'l' and 'q' blocks.
        blas.scal(step, ds, n = dims['l'] + sum(dims['q']))
        blas.scal(step, dz, n = dims['l'] + sum(dims['q']))
        ds[:dims['l']] += 1.0
        dz[:dims['l']] += 1.0
        ds[indq[:-1]] += 1.0
        dz[indq[:-1]] += 1.0

        # ds := H(lambda)^{-1/2} * ds and dz := H(lambda)^{-1/2} * dz.
        #
        # This replaces the 'l' and 'q' components of ds and dz with the
        # updated variables in the current scaling.
        # The 's' components of ds and dz are replaced with
        #
        #     diag(lmbda_k)^{1/2} * Qs * diag(lmbda_k)^{1/2}
        #     diag(lmbda_k)^{1/2} * Qz * diag(lmbda_k)^{1/2}
        #
        misc.scale2(lmbda, ds, dims, inverse = 'I')
        misc.scale2(lmbda, dz, dims, inverse = 'I')

        # sigs := ( e + step*sigs ) ./ lambda for 's' blocks.
        # sigz := ( e + step*sigz ) ./ lambda for 's' blocks.
        blas.scal(step, sigs)
        blas.scal(step, sigz)
        sigs += 1.0
        sigz += 1.0
        blas.tbsv(lmbda, sigs, n = sum(dims['s']), k = 0, ldA = 1,
            offsetA = dims['l'] + sum(dims['q']))
        blas.tbsv(lmbda, sigz, n = sum(dims['s']), k = 0, ldA = 1,
            offsetA = dims['l'] + sum(dims['q']))

        # dsk := Ls = dsk * sqrt(sigs).
        # dzk := Lz = dzk * sqrt(sigz).
        ind2, ind3 = dims['l'] + sum(dims['q']), 0
        for k in range(len(dims['s'])):
            m = dims['s'][k]
            for i in range(m):
                blas.scal(math.sqrt(sigs[ind3+i]), ds, offset = ind2 + m*i,
                    n = m)
                blas.scal(math.sqrt(sigz[ind3+i]), dz, offset = ind2 + m*i,
                    n = m)
            ind2 += m*m
            ind3 += m


        # Update lambda and scaling.

        misc.update_scaling(W, lmbda, ds, dz)

        # For kappa, tau block:
        #
        #     dg := sqrt( (kappa + step*dkappa) / (tau + step*dtau) )
        #         = dg * sqrt( (1 - step*tk) / (1 - step*tt) )
        #
        #     lmbda[-1] := sqrt((tau + step*dtau) * (kappa + step*dkappa))
        #                = lmbda[-1] * sqrt(( 1 - step*tt) * (1 - step*tk))

        dg *= math.sqrt(1.0 - step*tk) / math.sqrt(1.0 - step*tt)
        dgi = 1.0 / dg
        lmbda[-1] *= math.sqrt(1.0 - step*tt) * math.sqrt(1.0 - step*tk)


        # Unscale s, z, tau, kappa (unscaled variables are used only to
        # compute feasibility residuals).

        blas.copy(lmbda, s, n = dims['l'] + sum(dims['q']))
        ind = dims['l'] + sum(dims['q'])
        ind2 = ind
        for m in dims['s']:
            blas.scal(0.0, s, offset = ind2)
            blas.copy(lmbda, s, offsetx = ind, offsety = ind2, n = m,
                incy = m+1)
            ind += m
            ind2 += m*m
        misc.scale(s, W, trans = 'T')

        blas.copy(lmbda, z, n = dims['l'] + sum(dims['q']))
        ind = dims['l'] + sum(dims['q'])
        ind2 = ind
        for m in dims['s']:
            blas.scal(0.0, z, offset = ind2)
            blas.copy(lmbda, z, offsetx = ind, offsety = ind2, n = m,
                    incy = m+1)
            ind += m
            ind2 += m*m
        misc.scale(z, W, inverse = 'I')

        kappa, tau = lmbda[-1]/dgi, lmbda[-1]*dgi
        gap = ( blas.nrm2(lmbda, n = lmbda.size[0]-1) / tau )**2



def coneqp(P, q, G = None, h = None, dims = None, A = None, b = None,
    initvals = None, kktsolver = None, xnewcopy = None, xdot = None,
    xaxpy = None, xscal = None, ynewcopy = None, ydot = None, yaxpy = None,
    yscal = None, **kwargs):
    """

    Solves a pair of primal and dual convex quadratic cone programs

        minimize    (1/2)*x'*P*x + q'*x
        subject to  G*x + s = h
                    A*x = b
                    s >= 0

        maximize    -(1/2)*(q + G'*z + A'*y)' * pinv(P) * (q + G'*z + A'*y)
                    - h'*z - b'*y
        subject to  q + G'*z + A'*y in range(P)
                    z >= 0.

    The inequalities are with respect to a cone C defined as the Cartesian
    product of N + M + 1 cones:

        C = C_0 x C_1 x .... x C_N x C_{N+1} x ... x C_{N+M}.

    The first cone C_0 is the nonnegative orthant of dimension ml.
    The next N cones are 2nd order cones of dimension mq[0], ..., mq[N-1].
    The second order cone of dimension m is defined as

        { (u0, u1) in R x R^{m-1} | u0 >= ||u1||_2 }.

    The next M cones are positive semidefinite cones of order ms[0], ...,
    ms[M-1] >= 0.


    Input arguments (basic usage).

        P is a dense or sparse 'd' matrix of size (n,n) with the lower
        triangular part of the Hessian of the objective stored in the
        lower triangle.  Must be positive semidefinite.

        q is a dense 'd' matrix of size (n,1).

        dims is a dictionary with the dimensions of the components of C.
        It has three fields.
        - dims['l'] = ml, the dimension of the nonnegative orthant C_0.
          (ml >= 0.)
        - dims['q'] = mq = [ mq[0], mq[1], ..., mq[N-1] ], a list of N
          integers with the dimensions of the second order cones
          C_1, ..., C_N.  (N >= 0 and mq[k] >= 1.)
        - dims['s'] = ms = [ ms[0], ms[1], ..., ms[M-1] ], a list of M
          integers with the orders of the semidefinite cones
          C_{N+1}, ..., C_{N+M}.  (M >= 0 and ms[k] >= 0.)
        The default value of dims = {'l': G.size[0], 'q': [], 's': []}.

        G is a dense or sparse 'd' matrix of size (K,n), where

            K = ml + mq[0] + ... + mq[N-1] + ms[0]**2 + ... + ms[M-1]**2.

        Each column of G describes a vector

            v = ( v_0, v_1, ..., v_N, vec(v_{N+1}), ..., vec(v_{N+M}) )

        in V = R^ml x R^mq[0] x ... x R^mq[N-1] x S^ms[0] x ... x S^ms[M-1]
        stored as a column vector

            [ v_0; v_1; ...; v_N; vec(v_{N+1}); ...; vec(v_{N+M}) ].

        Here, if u is a symmetric matrix of order m, then vec(u) is the
        matrix u stored in column major order as a vector of length m**2.
        We use BLAS unpacked 'L' storage, i.e., the entries in vec(u)
        corresponding to the strictly upper triangular entries of u are
        not referenced.

        h is a dense 'd' matrix of size (K,1), representing a vector in V,
        in the same format as the columns of G.

        A is a dense or sparse 'd' matrix of size (p,n).   The default
        value is a sparse 'd' matrix of size (0,n).

        b is a dense 'd' matrix of size (p,1).  The default value is a
        dense 'd' matrix of size (0,1).

        initvals is a dictionary with optional primal and dual starting
        points initvals['x'], initvals['s'], initvals['y'], initvals['z'].
        - initvals['x'] is a dense 'd' matrix of size (n,1).
        - initvals['s'] is a dense 'd' matrix of size (K,1), representing
          a vector that is strictly positive with respect to the cone C.
        - initvals['y'] is a dense 'd' matrix of size (p,1).
        - initvals['z'] is a dense 'd' matrix of size (K,1), representing
          a vector that is strictly positive with respect to the cone C.
        A default initialization is used for the variables that are not
        specified in initvals.

        It is assumed that rank(A) = p and rank([P; A; G]) = n.

        The other arguments are normally not needed.  They make it possible
        to exploit certain types of structure, as described below.


    Output arguments.

        Returns a dictionary with keys 'status', 'x', 's', 'z', 'y',
        'primal objective', 'dual objective', 'gap', 'relative gap',
        'primal infeasibility', 'dual infeasibility', 'primal slack',
        'dual slack', 'iterations'.

        The 'status' field has values 'optimal' or 'unknown'.  'iterations'
        is the number of iterations taken.

        If the status is 'optimal', 'x', 's', 'y', 'z' are an approximate
        solution of the primal and dual optimality conditions

              G*x + s = h,  A*x = b
              P*x + G'*z + A'*y + q = 0
              s >= 0,  z >= 0
              s'*z = 0.

        If the status is 'unknown', 'x', 'y', 's', 'z' are the last
        iterates before termination.  These satisfy s > 0 and z > 0,
        but are not necessarily feasible.

        The values of the other fields are defined as follows.

        - 'primal objective': the primal objective (1/2)*x'*P*x + q'*x.

        - 'dual objective': the dual objective

              L(x,y,z) = (1/2)*x'*P*x + q'*x + z'*(G*x - h) + y'*(A*x-b).

        - 'gap': the duality gap s'*z.

        - 'relative gap': the relative gap, defined as

              gap / -primal objective

          if the primal objective is negative,

              gap / dual objective

          if the dual objective is positive, and None otherwise.

        - 'primal infeasibility': the residual in the primal constraints,
          defined as the maximum of the residual in the inequalities

              || G*x + s + h || / max(1, ||h||)

          and the residual in the equalities

              || A*x - b || / max(1, ||b||).


        - 'dual infeasibility': the residual in the dual constraints,
          defined as

              || P*x + G'*z + A'*y + q || / max(1, ||q||).


        - 'primal slack': the smallest primal slack, sup {t | s >= t*e },
           where

              e = ( e_0, e_1, ..., e_N, e_{N+1}, ..., e_{M+N} )

          is the identity vector in C.  e_0 is an ml-vector of ones,
          e_k, k = 1,..., N, is the unit vector (1,0,...,0) of length
          mq[k], and e_k = vec(I) where I is the identity matrix of order
          ms[k].

        - 'dual slack': the smallest dual slack, sup {t | z >= t*e }.

        If the exit status is 'optimal', then the primal and dual
        infeasibilities are guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  The gap is less than
        solvers.options['abstol'] (default 1e-7) or the relative gap is
        less than solvers.options['reltol'] (default 1e-6).

        Termination with status 'unknown' indicates that the algorithm
        failed to find a solution that satisfies the specified tolerances.
        In some cases, the returned solution may be fairly accurate.  If
        the primal and dual infeasibilities, the gap, and the relative gap
        are small, then x, y, s, z are close to optimal.


    Advanced usage.

        Three mechanisms are provided to express problem structure.

        1.  The user can provide a customized routine for solving linear
        equations (`KKT systems')

            [ P   A'  G'    ] [ ux ]   [ bx ]
            [ A   0   0     ] [ uy ] = [ by ].
            [ G   0   -W'*W ] [ uz ]   [ bz ]

        W is a scaling matrix, a block diagonal mapping

           W*u = ( W0*u_0, ..., W_{N+M}*u_{N+M} )

        defined as follows.

        - For the 'l' block (W_0):

              W_0 = diag(d),

          with d a positive vector of length ml.

        - For the 'q' blocks (W_{k+1}, k = 0, ..., N-1):

              W_{k+1} = beta_k * ( 2 * v_k * v_k' - J )

          where beta_k is a positive scalar, v_k is a vector in R^mq[k]
          with v_k[0] > 0 and v_k'*J*v_k = 1, and J = [1, 0; 0, -I].

        - For the 's' blocks (W_{k+N}, k = 0, ..., M-1):

              W_k * u = vec(r_k' * mat(u) * r_k)

          where r_k is a nonsingular matrix of order ms[k], and mat(x) is
          the inverse of the vec operation.

        The optional argument kktsolver is a Python function that will be
        called as g = kktsolver(W).  W is a dictionary that contains
        the parameters of the scaling:

        - W['d'] is a positive 'd' matrix of size (ml,1).
        - W['di'] is a positive 'd' matrix with the elementwise inverse of
          W['d'].
        - W['beta'] is a list [ beta_0, ..., beta_{N-1} ]
        - W['v'] is a list [ v_0, ..., v_{N-1} ]
        - W['r'] is a list [ r_0, ..., r_{M-1} ]
        - W['rti'] is a list [ rti_0, ..., rti_{M-1} ], with rti_k the
          inverse of the transpose of r_k.

        The call g = kktsolver(W) should return a function g that solves
        the KKT system by g(x, y, z).  On entry, x, y, z contain the
        righthand side bx, by, bz.  On exit, they contain the solution,
        with uz scaled, the argument z contains W*uz.  In other words,
        on exit x, y, z are the solution of

            [ P   A'  G'*W^{-1} ] [ ux ]   [ bx ]
            [ A   0   0         ] [ uy ] = [ by ].
            [ G   0   -W'       ] [ uz ]   [ bz ]


        2.  The linear operators P*u, G*u and A*u can be specified
        by providing Python functions instead of matrices.  This can only
        be done in combination with 1. above, i.e., it requires the
        kktsolver argument.

        If P is a function, the call P(u, v, alpha, beta) should evaluate
        the matrix-vectors product

            v := alpha * P * u + beta * v.

        The arguments u and v are required.  The other arguments have
        default values alpha = 1.0, beta = 0.0.

        If G is a function, the call G(u, v, alpha, beta, trans) should
        evaluate the matrix-vector products

            v := alpha * G * u + beta * v  if trans is 'N'
            v := alpha * G' * u + beta * v  if trans is 'T'.

        The arguments u and v are required.  The other arguments have
        default values alpha = 1.0, beta = 0.0, trans = 'N'.

        If A is a function, the call A(u, v, alpha, beta, trans) should
        evaluate the matrix-vectors products

            v := alpha * A * u + beta * v if trans is 'N'
            v := alpha * A' * u + beta * v if trans is 'T'.

        The arguments u and v are required.  The other arguments
        have default values alpha = 1.0, beta = 0.0, trans = 'N'.


        3.  Instead of using the default representation of the primal
        variable x and the dual variable y as one-column 'd' matrices,
        we can represent these variables and the corresponding parameters
        q and b by arbitrary Python objects (matrices, lists, dictionaries,
        etc).  This can only be done in combination with 1. and 2. above,
        i.e., it requires a user-provided KKT solver and an operator
        description of the linear mappings.   It also requires the
        arguments xnewcopy, xdot, xscal, xaxpy, ynewcopy, ydot, yscal,
        yaxpy.  These arguments are functions defined as follows.

        If X is the vector space of primal variables x, then:
        - xnewcopy(u) creates a new copy of the vector u in X.
        - xdot(u, v) returns the inner product of two vectors u and v in X.
        - xscal(alpha, u) computes u := alpha*u, where alpha is a scalar
          and u is a vector in X.
        - xaxpy(u, v, alpha = 1.0) computes v := alpha*u + v for a scalar
          alpha and two vectors u and v in X.
        If this option is used, the argument q must be in the same format
        as x, the argument P must be a Python function, the arguments A
        and G must be Python functions or None, and the argument
        kktsolver is required.

        If Y is the vector space of primal variables y:
        - ynewcopy(u) creates a new copy of the vector u in Y.
        - ydot(u, v) returns the inner product of two vectors u and v in Y.
        - yscal(alpha, u) computes u := alpha*u, where alpha is a scalar
          and u is a vector in Y.
        - yaxpy(u, v, alpha = 1.0) computes v := alpha*u + v for a scalar
          alpha and two vectors u and v in Y.
        If this option is used, the argument b must be in the same format
        as y, the argument A must be a Python function or None, and the
        argument kktsolver is required.


    Control parameters.

       The following control parameters can be modified by adding an
       entry to the dictionary options.

       options['show_progress'] True/False (default: True)
       options['maxiters'] positive integer (default: 100)
       options['refinement'] nonnegative integer (default: 0 for problems
           with no second-order cone and matrix inequality constraints;
           1 otherwise)
       options['abstol'] scalar (default: 1e-7)
       options['reltol'] scalar (default: 1e-6)
       options['feastol'] scalar (default: 1e-7).

    """
    import math
    from cvxopt import base, blas, misc
    from cvxopt.base import matrix, spmatrix

    STEP = 0.99
    EXPON = 3

    options = kwargs.get('options',globals()['options'])

    DEBUG = options.get('debug',False)

    KKTREG = options.get('kktreg',None)
    if KKTREG is None:
        pass
    elif not isinstance(KKTREG,(float,int,long)) or KKTREG < 0.0:
        raise ValueError("options['kktreg'] must be a nonnegative scalar")

    # Use Mehrotra correction or not.
    correction = options.get('use_correction', True)

    MAXITERS = options.get('maxiters',100)
    if not isinstance(MAXITERS,(int,long)) or MAXITERS < 1:
        raise ValueError("options['maxiters'] must be a positive integer")

    ABSTOL = options.get('abstol',1e-7)
    if not isinstance(ABSTOL,(float,int,long)):
        raise ValueError("options['abstol'] must be a scalar")

    RELTOL = options.get('reltol',1e-6)
    if not isinstance(RELTOL,(float,int,long)):
        raise ValueError("options['reltol'] must be a scalar")

    if RELTOL <= 0.0 and ABSTOL <= 0.0 :
        raise ValueError("at least one of options['reltol'] and " \
            "options['abstol'] must be positive")

    FEASTOL = options.get('feastol',1e-7)
    if not isinstance(FEASTOL,(float,int,long)) or FEASTOL <= 0.0:
        raise ValueError("options['feastol'] must be a positive scalar")

    show_progress = options.get('show_progress',True)

    if kktsolver is None:
        if dims and (dims['q'] or dims['s']):
            kktsolver = 'chol'
        else:
            kktsolver = 'chol2'
    defaultsolvers = ('ldl', 'ldl2', 'chol', 'chol2')
    if isinstance(kktsolver,str) and kktsolver not in defaultsolvers:
        raise ValueError("'%s' is not a valid value for kktsolver" \
            %kktsolver)

    # Argument error checking depends on level of customization.
    customkkt = not isinstance(kktsolver,str)
    matrixP = isinstance(P, (matrix, spmatrix))
    matrixG = isinstance(G, (matrix, spmatrix))
    matrixA = isinstance(A, (matrix, spmatrix))
    if (not matrixP or (not matrixG and G is not None) or
        (not matrixA and A is not None)) and not customkkt:
        raise ValueError("use of function valued P, G, A requires a "\
            "user-provided kktsolver")
    customx = (xnewcopy != None or xdot != None or xaxpy != None or
        xscal != None)
    if customx and (matrixP or matrixG or matrixA or not customkkt):
        raise ValueError("use of non-vector type for x requires "\
            "function valued P, G, A and user-provided kktsolver")
    customy = (ynewcopy != None or ydot != None or yaxpy != None or
        yscal != None)
    if customy and (matrixA or not customkkt):
        raise ValueError("use of non vector type for y requires "\
            "function valued A and user-provided kktsolver")


    if not customx and (not isinstance(q,matrix) or q.typecode != 'd' or q.size[1] != 1):
        raise TypeError("'q' must be a 'd' matrix with one column")

    if matrixP:
        if P.typecode != 'd' or P.size != (q.size[0], q.size[0]):
            raise TypeError("'P' must be a 'd' matrix of size (%d, %d)"\
                %(q.size[0], q.size[0]))
        def fP(x, y, alpha = 1.0, beta = 0.0):
            base.symv(P, x, y, alpha = alpha, beta = beta)
    else:
        fP = P


    if h is None: h = matrix(0.0, (0,1))
    if not isinstance(h, matrix) or h.typecode != 'd' or h.size[1] != 1:
        raise TypeError("'h' must be a 'd' matrix with one column")

    if not dims: dims = {'l': h.size[0], 'q': [], 's': []}
    if not isinstance(dims['l'],(int,long)) or dims['l'] < 0:
        raise TypeError("'dims['l']' must be a nonnegative integer")
    if [ k for k in dims['q'] if not isinstance(k,(int,long)) or k < 1 ]:
        raise TypeError("'dims['q']' must be a list of positive integers")
    if [ k for k in dims['s'] if not isinstance(k,(int,long)) or k < 0 ]:
        raise TypeError("'dims['s']' must be a list of nonnegative " \
            "integers")

    try: refinement = options['refinement']
    except KeyError:
        if dims['q'] or dims['s']: refinement = 1
        else: refinement = 0
    else:
        if not isinstance(refinement,(int,long)) or refinement < 0:
            raise ValueError("options['refinement'] must be a "\
                "nonnegative integer")


    cdim = dims['l'] + sum(dims['q']) + sum([ k**2 for k in dims['s'] ])
    if h.size[0] != cdim:
        raise TypeError("'h' must be a 'd' matrix of size (%d,1)" %cdim)

    # Data for kth 'q' constraint are found in rows indq[k]:indq[k+1] of G.
    indq = [ dims['l'] ]
    for k in dims['q']:  indq = indq + [ indq[-1] + k ]

    # Data for kth 's' constraint are found in rows inds[k]:inds[k+1] of G.
    inds = [ indq[-1] ]
    for k in dims['s']:  inds = inds + [ inds[-1] + k**2 ]

    if G is None:
        if customx:
            def G(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
                if trans == 'N': pass
                else: xscal(beta, y)
        else:
            G = spmatrix([], [], [], (0, q.size[0]))
            matrixG = True
    if matrixG:
        if G.typecode != 'd' or G.size != (cdim, q.size[0]):
            raise TypeError("'G' must be a 'd' matrix of size (%d, %d)"\
                %(cdim, q.size[0]))
        def fG(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
            misc.sgemv(G, x, y, dims, trans = trans, alpha = alpha,
                beta = beta)
    else:
        fG = G


    if A is None:
        if customx or customy:
            def A(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
                if trans == 'N': pass
                else: xscal(beta, y)
        else:
            A = spmatrix([], [], [], (0, q.size[0]))
            matrixA = True
    if matrixA:
        if A.typecode != 'd' or A.size[1] != q.size[0]:
            raise TypeError("'A' must be a 'd' matrix with %d columns" \
                %q.size[0])
        def fA(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
            base.gemv(A, x, y, trans = trans, alpha = alpha, beta = beta)
    else:
        fA = A
    if not customy:
        if b is None: b = matrix(0.0, (0,1))
        if not isinstance(b, matrix) or b.typecode != 'd' or b.size[1] != 1:
            raise TypeError("'b' must be a 'd' matrix with one column")
        if matrixA and b.size[0] != A.size[0]:
            raise TypeError("'b' must have length %d" %A.size[0])
    if b is None and customy:
        raise ValueEror("use of non-vector type for y requires b")


    ws3, wz3 = matrix(0.0, (cdim,1 )), matrix(0.0, (cdim,1 ))
    def res(ux, uy, uz, us, vx, vy, vz, vs, W, lmbda):

        # Evaluates residual in Newton equations:
        #
        #      [ vx ]    [ vx ]   [ 0     ]   [ P  A'  G' ]   [ ux        ]
        #      [ vy ] := [ vy ] - [ 0     ] - [ A  0   0  ] * [ uy        ]
        #      [ vz ]    [ vz ]   [ W'*us ]   [ G  0   0  ]   [ W^{-1}*uz ]
        #
        #      vs := vs - lmbda o (uz + us).

        # vx := vx - P*ux - A'*uy - G'*W^{-1}*uz
        fP(ux, vx, alpha = -1.0, beta = 1.0)
        fA(uy, vx, alpha = -1.0, beta = 1.0, trans = 'T')
        blas.copy(uz, wz3)
        misc.scale(wz3, W, inverse = 'I')
        fG(wz3, vx, alpha = -1.0, beta = 1.0, trans = 'T')

        # vy := vy - A*ux
        fA(ux, vy, alpha = -1.0, beta = 1.0)

        # vz := vz - G*ux - W'*us
        fG(ux, vz, alpha = -1.0, beta = 1.0)
        blas.copy(us, ws3)
        misc.scale(ws3, W, trans = 'T')
        blas.axpy(ws3, vz, alpha = -1.0)

        # vs := vs - lmbda o (uz + us)
        blas.copy(us, ws3)
        blas.axpy(uz, ws3)
        misc.sprod(ws3, lmbda, dims, diag = 'D')
        blas.axpy(ws3, vs, alpha = -1.0)


    # kktsolver(W) returns a routine for solving
    #
    #     [ P   A'  G'*W^{-1} ] [ ux ]   [ bx ]
    #     [ A   0   0         ] [ uy ] = [ by ].
    #     [ G   0   -W'       ] [ uz ]   [ bz ]

    if kktsolver in defaultsolvers:
         if KKTREG is None and b.size[0] > q.size[0]:
             raise ValueError("Rank(A) < p or Rank([P; G; A]) < n")
         if kktsolver == 'ldl':
             factor = misc.kkt_ldl(G, dims, A, kktreg = KKTREG)
         elif kktsolver == 'ldl2':
             factor = misc.kkt_ldl2(G, dims, A)
         elif kktsolver == 'chol':
             factor = misc.kkt_chol(G, dims, A)
         else:
             factor = misc.kkt_chol2(G, dims, A)
         def kktsolver(W):
             return factor(W, P)

    if xnewcopy is None: xnewcopy = matrix
    if xdot is None: xdot = blas.dot
    if xaxpy is None: xaxpy = blas.axpy
    if xscal is None: xscal = blas.scal
    def xcopy(x, y):
        xscal(0.0, y)
        xaxpy(x, y)
    if ynewcopy is None: ynewcopy = matrix
    if ydot is None: ydot = blas.dot
    if yaxpy is None: yaxpy = blas.axpy
    if yscal is None: yscal = blas.scal
    def ycopy(x, y):
        yscal(0.0, y)
        yaxpy(x, y)

    resx0 = max(1.0, math.sqrt(xdot(q,q)))
    resy0 = max(1.0, math.sqrt(ydot(b,b)))
    resz0 = max(1.0, misc.snrm2(h, dims))

    if cdim == 0:

        # Solve
        #
        #     [ P  A' ] [ x ]   [ -q ]
        #     [       ] [   ] = [    ].
        #     [ A  0  ] [ y ]   [  b ]

        try: f3 = kktsolver({'d': matrix(0.0, (0,1)), 'di':
            matrix(0.0, (0,1)), 'beta': [], 'v': [], 'r': [], 'rti': []})
        except ArithmeticError:
            raise ValueError("Rank(A) < p or Rank([P; A; G]) < n")
        x = xnewcopy(q)
        xscal(-1.0, x)
        y = ynewcopy(b)
        f3(x, y, matrix(0.0, (0,1)))

        # dres = || P*x + q + A'*y || / resx0
        rx = xnewcopy(q)
        fP(x, rx, beta = 1.0)
        pcost = 0.5 * (xdot(x, rx) + xdot(x, q))
        fA(y, rx, beta = 1.0, trans = 'T')
        dres = math.sqrt(xdot(rx, rx)) / resx0

        # pres = || A*x - b || / resy0
        ry = ynewcopy(b)
        fA(x, ry, alpha = 1.0, beta = -1.0)
        pres = math.sqrt(ydot(ry, ry)) / resy0

        if pcost == 0.0: relgap = None
        else: relgap = 0.0

        return { 'status': 'optimal', 'x': x,  'y': y, 'z':
            matrix(0.0, (0,1)), 's': matrix(0.0, (0,1)),
            'gap': 0.0, 'relative gap': 0.0,
            'primal objective': pcost,
            'dual objective': pcost,
            'primal slack': 0.0, 'dual slack': 0.0,
            'primal infeasibility': pres, 'dual infeasibility': dres,
            'iterations': 0 }


    x, y = xnewcopy(q), ynewcopy(b)
    s, z = matrix(0.0, (cdim, 1)), matrix(0.0, (cdim, 1))

    if initvals is None:

        # Factor
        #
        #     [ P   A'  G' ]
        #     [ A   0   0  ].
        #     [ G   0  -I  ]

        W = {}
        W['d'] = matrix(1.0, (dims['l'], 1))
        W['di'] = matrix(1.0, (dims['l'], 1))
        W['v'] = [ matrix(0.0, (m,1)) for m in dims['q'] ]
        W['beta'] = len(dims['q']) * [ 1.0 ]
        for v in W['v']: v[0] = 1.0
        W['r'] = [ matrix(0.0, (m,m)) for m in dims['s'] ]
        W['rti'] = [ matrix(0.0, (m,m)) for m in dims['s'] ]
        for r in W['r']: r[::r.size[0]+1 ] = 1.0
        for rti in W['rti']: rti[::rti.size[0]+1 ] = 1.0
        try: f = kktsolver(W)
        except ArithmeticError:
            raise ValueError("Rank(A) < p or Rank([P; A; G]) < n")


        # Solve
        #
        #     [ P   A'  G' ]   [ x ]   [ -q ]
        #     [ A   0   0  ] * [ y ] = [  b ].
        #     [ G   0  -I  ]   [ z ]   [  h ]

        xcopy(q, x)
        xscal(-1.0, x)
        ycopy(b, y)
        blas.copy(h, z)
        try: f(x, y, z)
        except ArithmeticError:
            raise ValueError("Rank(A) < p or Rank([P; G; A]) < n")
        blas.copy(z, s)
        blas.scal(-1.0, s)

        nrms = misc.snrm2(s, dims)
        ts = misc.max_step(s, dims)
        if ts >= -1e-8 * max(nrms, 1.0):
            a = 1.0 + ts
            s[:dims['l']] += a
            s[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                s[ind : ind+m*m : m+1] += a
                ind += m**2

        nrmz = misc.snrm2(z, dims)
        tz = misc.max_step(z, dims)
        if tz >= -1e-8 * max(nrmz, 1.0):
            a = 1.0 + tz
            z[:dims['l']] += a
            z[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                z[ind : ind+m*m : m+1] += a
                ind += m**2


    else:

        if 'x' in initvals:
            xcopy(initvals['x'], x)
        else:
            xscal(0.0, x)

        if 's' in initvals:
            blas.copy(initvals['s'], s)
            # ts = min{ t | s + t*e >= 0 }
            if misc.max_step(s, dims) >= 0:
                raise ValueError("initial s is not positive")
        else:
            s[: dims['l']] = 1.0
            ind = dims['l']
            for m in dims['q']:
                s[ind] = 1.0
                ind += m
            for m in dims['s']:
                s[ind : ind + m*m : m+1] = 1.0
                ind += m**2

        if 'y' in initvals:
            ycopy(initvals['y'], y)
        else:
            yscal(0.0, y)

        if 'z' in initvals:
            blas.copy(initvals['z'], z)
            # tz = min{ t | z + t*e >= 0 }
            if misc.max_step(z, dims) >= 0:
                raise ValueError("initial z is not positive")
        else:
            z[: dims['l']] = 1.0
            ind = dims['l']
            for m in dims['q']:
                z[ind] = 1.0
                ind += m
            for m in dims['s']:
                z[ind : ind + m*m : m+1] = 1.0
                ind += m**2


    rx, ry, rz = xnewcopy(q), ynewcopy(b), matrix(0.0, (cdim, 1))
    dx, dy = xnewcopy(x), ynewcopy(y)
    dz, ds = matrix(0.0, (cdim, 1)), matrix(0.0, (cdim, 1))
    lmbda = matrix(0.0, (dims['l'] + sum(dims['q']) + sum(dims['s']), 1))
    lmbdasq = matrix(0.0, (dims['l'] + sum(dims['q']) + sum(dims['s']), 1))
    sigs = matrix(0.0, (sum(dims['s']), 1))
    sigz = matrix(0.0, (sum(dims['s']), 1))


    if show_progress:
        print("% 10s% 12s% 10s% 8s% 7s" %("pcost", "dcost", "gap", "pres",
            "dres"))

    gap = misc.sdot(s, z, dims)

    for iters in range(MAXITERS + 1):

        # f0 = (1/2)*x'*P*x + q'*x + r and  rx = P*x + q + A'*y + G'*z.
        xcopy(q, rx)
        fP(x, rx, beta = 1.0)
        f0 = 0.5 * (xdot(x, rx) + xdot(x, q))
        fA(y, rx, beta = 1.0, trans = 'T')
        fG(z, rx, beta = 1.0, trans = 'T')
        resx = math.sqrt(xdot(rx, rx))

        # ry = A*x - b
        ycopy(b, ry)
        fA(x, ry, alpha = 1.0, beta = -1.0)
        resy = math.sqrt(ydot(ry, ry))

        # rz = s + G*x - h
        blas.copy(s, rz)
        blas.axpy(h, rz, alpha = -1.0)
        fG(x, rz, beta = 1.0)
        resz = misc.snrm2(rz, dims)


        # Statistics for stopping criteria.

        # pcost = (1/2)*x'*P*x + q'*x
        # dcost = (1/2)*x'*P*x + q'*x + y'*(A*x-b) + z'*(G*x-h)
        #       = (1/2)*x'*P*x + q'*x + y'*(A*x-b) + z'*(G*x-h+s) - z'*s
        #       = (1/2)*x'*P*x + q'*x + y'*ry + z'*rz - gap
        pcost = f0
        dcost = f0 + ydot(y, ry) + misc.sdot(z, rz, dims) - gap
        if pcost < 0.0:
            relgap = gap / -pcost
        elif dcost > 0.0:
            relgap = gap / dcost
        else:
            relgap = None
        pres = max(resy/resy0, resz/resz0)
        dres = resx/resx0

        if show_progress:
            print("%2d: % 8.4e % 8.4e % 4.0e% 7.0e% 7.0e" \
                %(iters, pcost, dcost, gap, pres, dres))

        if ( pres <= FEASTOL and dres <= FEASTOL and ( gap <= ABSTOL or
            (relgap is not None and relgap <= RELTOL) )) or \
            iters == MAXITERS:
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(s, m, ind)
                misc.symm(z, m, ind)
                ind += m**2
            ts = misc.max_step(s, dims)
            tz = misc.max_step(z, dims)
            if iters == MAXITERS:
                if show_progress:
                    print("Terminated (maximum number of iterations "\
                        "reached).")
                status = 'unknown'
            else:
                if show_progress:
                    print("Optimal solution found.")
                status = 'optimal'
            return { 'x': x,  'y': y,  's': s,  'z': z,  'status': status,
                    'gap': gap,  'relative gap': relgap,
                    'primal objective': pcost,  'dual objective': dcost,
                    'primal infeasibility': pres,
                    'dual infeasibility': dres, 'primal slack': -ts,
                    'dual slack': -tz , 'iterations': iters }


        # Compute initial scaling W and scaled iterates:
        #
        #     W * z = W^{-T} * s = lambda.
        #
        # lmbdasq = lambda o lambda.

        if iters == 0:  W = misc.compute_scaling(s, z, lmbda, dims)
        misc.ssqr(lmbdasq, lmbda, dims)


        # f3(x, y, z) solves
        #
        #    [ P   A'  G'    ] [ ux        ]   [ bx ]
        #    [ A   0   0     ] [ uy        ] = [ by ].
        #    [ G   0   -W'*W ] [ W^{-1}*uz ]   [ bz ]
        #
        # On entry, x, y, z containg bx, by, bz.
        # On exit, they contain ux, uy, uz.

        try: f3 = kktsolver(W)
        except ArithmeticError:
            if iters == 0:
                raise ValueError("Rank(A) < p or Rank([P; A; G]) < n")
            else:
                ind = dims['l'] + sum(dims['q'])
                for m in dims['s']:
                    misc.symm(s, m, ind)
                    misc.symm(z, m, ind)
                    ind += m**2
                ts = misc.max_step(s, dims)
                tz = misc.max_step(z, dims)
                if show_progress:
                    print("Terminated (singular KKT matrix).")
                return { 'x': x,  'y': y,  's': s,  'z': z,
                    'status': 'unknown', 'gap': gap,
                    'relative gap': relgap, 'primal objective': pcost,
                    'dual objective': dcost, 'primal infeasibility': pres,
                    'dual infeasibility': dres, 'primal slack': -ts,
                    'dual slack': -tz, 'iterations': iters }

        # f4_no_ir(x, y, z, s) solves
        #
        #     [ 0     ]   [ P  A'  G' ]   [ ux        ]   [ bx ]
        #     [ 0     ] + [ A  0   0  ] * [ uy        ] = [ by ]
        #     [ W'*us ]   [ G  0   0  ]   [ W^{-1}*uz ]   [ bz ]
        #
        #     lmbda o (uz + us) = bs.
        #
        # On entry, x, y, z, s contain bx, by, bz, bs.
        # On exit, they contain ux, uy, uz, us.

        def f4_no_ir(x, y, z, s):

            # Solve
            #
            #     [ P A' G'   ] [ ux        ]    [ bx                    ]
            #     [ A 0  0    ] [ uy        ] =  [ by                    ]
            #     [ G 0 -W'*W ] [ W^{-1}*uz ]    [ bz - W'*(lmbda o\ bs) ]
            #
            #     us = lmbda o\ bs - uz.
            #
            # On entry, x, y, z, s  contains bx, by, bz, bs.
            # On exit they contain x, y, z, s.

            # s := lmbda o\ s
            #    = lmbda o\ bs
            misc.sinv(s, lmbda, dims)

            # z := z - W'*s
            #    = bz - W'*(lambda o\ bs)
            blas.copy(s, ws3)
            misc.scale(ws3, W, trans = 'T')
            blas.axpy(ws3, z, alpha = -1.0)

            # Solve for ux, uy, uz
            f3(x, y, z)

            # s := s - z
            #    = lambda o\ bs - uz.
            blas.axpy(z, s, alpha = -1.0)


        # f4(x, y, z, s) solves the same system as f4_no_ir, but applies
        # iterative refinement.

        if iters == 0:
            if refinement or DEBUG:
                wx, wy = xnewcopy(q), ynewcopy(b)
                wz, ws = matrix(0.0, (cdim,1)), matrix(0.0, (cdim,1))
            if refinement:
                wx2, wy2 = xnewcopy(q), ynewcopy(b)
                wz2, ws2 = matrix(0.0, (cdim,1)), matrix(0.0, (cdim,1))

        def f4(x, y, z, s):
            if refinement or DEBUG:
                xcopy(x, wx)
                ycopy(y, wy)
                blas.copy(z, wz)
                blas.copy(s, ws)
            f4_no_ir(x, y, z, s)
            for i in range(refinement):
                xcopy(wx, wx2)
                ycopy(wy, wy2)
                blas.copy(wz, wz2)
                blas.copy(ws, ws2)
                res(x, y, z, s, wx2, wy2, wz2, ws2, W, lmbda)
                f4_no_ir(wx2, wy2, wz2, ws2)
                xaxpy(wx2, x)
                yaxpy(wy2, y)
                blas.axpy(wz2, z)
                blas.axpy(ws2, s)
            if DEBUG:
                res(x, y, z, s, wx, wy, wz, ws, W, lmbda)
                print("KKT residuals:")
                print("    'x': %e" %math.sqrt(xdot(wx, wx)))
                print("    'y': %e" %math.sqrt(ydot(wy, wy)))
                print("    'z': %e" %misc.snrm2(wz, dims))
                print("    's': %e" %misc.snrm2(ws, dims))


        mu = gap / (dims['l'] + len(dims['q']) + sum(dims['s']))
        sigma, eta = 0.0, 0.0

        for i in [0, 1]:

            # Solve
            #
            #     [ 0     ]   [ P  A' G' ]   [ dx        ]
            #     [ 0     ] + [ A  0  0  ] * [ dy        ] = -(1 - eta) * r
            #     [ W'*ds ]   [ G  0  0  ]   [ W^{-1}*dz ]
            #
            #     lmbda o (dz + ds) = -lmbda o lmbda + sigma*mu*e (i=0)
            #     lmbda o (dz + ds) = -lmbda o lmbda - dsa o dza
            #                         + sigma*mu*e (i=1) where dsa, dza
            #                         are the solution for i=0.

            # ds = -lmbdasq + sigma * mu * e  (if i is 0)
            #    = -lmbdasq - dsa o dza + sigma * mu * e  (if i is 1),
            #     where ds, dz are solution for i is 0.
            blas.scal(0.0, ds)
            if correction and i == 1:
                blas.axpy(ws3, ds, alpha = -1.0)
            blas.axpy(lmbdasq, ds, n = dims['l'] + sum(dims['q']),
                alpha = -1.0)
            ds[:dims['l']] += sigma*mu
            ind = dims['l']
            for m in dims['q']:
                ds[ind] += sigma*mu
                ind += m
            ind2 = ind
            for m in dims['s']:
                blas.axpy(lmbdasq, ds, n = m, offsetx = ind2, offsety =
                    ind, incy = m + 1, alpha = -1.0)
                ds[ind : ind + m*m : m+1] += sigma*mu
                ind += m*m
                ind2 += m


            # (dx, dy, dz) := -(1 - eta) * (rx, ry, rz)
            xscal(0.0, dx);  xaxpy(rx, dx, alpha = -1.0 + eta)
            yscal(0.0, dy);  yaxpy(ry, dy, alpha = -1.0 + eta)
            blas.scal(0.0, dz)
            blas.axpy(rz, dz, alpha = -1.0 + eta)

            try: f4(dx, dy, dz, ds)
            except ArithmeticError:
                if iters == 0:
                    raise ValueError("Rank(A) < p or Rank([P; A; G]) < n")
                else:
                    ind = dims['l'] + sum(dims['q'])
                    for m in dims['s']:
                        misc.symm(s, m, ind)
                        misc.symm(z, m, ind)
                        ind += m**2
                    ts = misc.max_step(s, dims)
                    tz = misc.max_step(z, dims)
                    if show_progress:
                        print("Terminated (singular KKT matrix).")
                    return { 'x': x,  'y': y,  's': s,  'z': z,
                        'status': 'unknown', 'gap': gap,
                        'relative gap': relgap, 'primal objective': pcost,
                        'dual objective': dcost,
                        'primal infeasibility': pres,
                        'dual infeasibility': dres, 'primal slack': -ts,
                        'dual slack': -tz, 'iterations': iters }

            dsdz = misc.sdot(ds, dz, dims)

            # Save ds o dz for Mehrotra correction
            if correction and i == 0:
                blas.copy(ds, ws3)
                misc.sprod(ws3, dz, dims)


            # Maximum steps to boundary.
            #
            # If i is 1, also compute eigenvalue decomposition of the
            # 's' blocks in ds,dz.  The eigenvectors Qs, Qz are stored in
            # dsk, dzk.  The eigenvalues are stored in sigs, sigz.

            misc.scale2(lmbda, ds, dims)
            misc.scale2(lmbda, dz, dims)
            if i == 0:
                ts = misc.max_step(ds, dims)
                tz = misc.max_step(dz, dims)
            else:
                ts = misc.max_step(ds, dims, sigma = sigs)
                tz = misc.max_step(dz, dims, sigma = sigz)
            t = max([ 0.0, ts, tz ])
            if t == 0:
                step = 1.0
            else:
                if i == 0:
                    step = min(1.0, 1.0 / t)
                else:
                    step = min(1.0, STEP / t)
            if i == 0:
                sigma = min(1.0, max(0.0,
                    1.0 - step + dsdz/gap * step**2))**EXPON
                eta = 0.0


        xaxpy(dx, x, alpha = step)
        yaxpy(dy, y, alpha = step)


        # We will now replace the 'l' and 'q' blocks of ds and dz with
        # the updated iterates in the current scaling.
        # We also replace the 's' blocks of ds and dz with the factors
        # Ls, Lz in a factorization Ls*Ls', Lz*Lz' of the updated variables
        # in the current scaling.

        # ds := e + step*ds for nonlinear, 'l' and 'q' blocks.
        # dz := e + step*dz for nonlinear, 'l' and 'q' blocks.
        blas.scal(step, ds, n = dims['l'] + sum(dims['q']))
        blas.scal(step, dz, n = dims['l'] + sum(dims['q']))
        ind = dims['l']
        ds[:ind] += 1.0
        dz[:ind] += 1.0
        for m in dims['q']:
            ds[ind] += 1.0
            dz[ind] += 1.0
            ind += m

        # ds := H(lambda)^{-1/2} * ds and dz := H(lambda)^{-1/2} * dz.
        #
        # This replaced the 'l' and 'q' components of ds and dz with the
        # updated iterates in the current scaling.
        # The 's' components of ds and dz are replaced with
        #
        #     diag(lmbda_k)^{1/2} * Qs * diag(lmbda_k)^{1/2}
        #     diag(lmbda_k)^{1/2} * Qz * diag(lmbda_k)^{1/2}
        #
        misc.scale2(lmbda, ds, dims, inverse = 'I')
        misc.scale2(lmbda, dz, dims, inverse = 'I')

        # sigs := ( e + step*sigs ) ./ lambda for 's' blocks.
        # sigz := ( e + step*sigz ) ./ lmabda for 's' blocks.
        blas.scal(step, sigs)
        blas.scal(step, sigz)
        sigs += 1.0
        sigz += 1.0
        blas.tbsv(lmbda, sigs, n = sum(dims['s']), k = 0, ldA = 1, offsetA
            = dims['l'] + sum(dims['q']))
        blas.tbsv(lmbda, sigz, n = sum(dims['s']), k = 0, ldA = 1, offsetA
            = dims['l'] + sum(dims['q']))

        # dsk := Ls = dsk * sqrt(sigs).
        # dzk := Lz = dzk * sqrt(sigz).
        ind2, ind3 = dims['l'] + sum(dims['q']), 0
        for k in range(len(dims['s'])):
            m = dims['s'][k]
            for i in range(m):
                blas.scal(math.sqrt(sigs[ind3+i]), ds, offset = ind2 + m*i,
                    n = m)
                blas.scal(math.sqrt(sigz[ind3+i]), dz, offset = ind2 + m*i,
                    n = m)
            ind2 += m*m
            ind3 += m


        # Update lambda and scaling.
        misc.update_scaling(W, lmbda, ds, dz)


        # Unscale s, z (unscaled variables are used only to compute
        # feasibility residuals).

        blas.copy(lmbda, s, n = dims['l'] + sum(dims['q']))
        ind = dims['l'] + sum(dims['q'])
        ind2 = ind
        for m in dims['s']:
            blas.scal(0.0, s, offset = ind2)
            blas.copy(lmbda, s, offsetx = ind, offsety = ind2, n = m,
                incy = m+1)
            ind += m
            ind2 += m*m
        misc.scale(s, W, trans = 'T')

        blas.copy(lmbda, z, n = dims['l'] + sum(dims['q']))
        ind = dims['l'] + sum(dims['q'])
        ind2 = ind
        for m in dims['s']:
            blas.scal(0.0, z, offset = ind2)
            blas.copy(lmbda, z, offsetx = ind, offsety = ind2, n = m,
                incy = m+1)
            ind += m
            ind2 += m*m
        misc.scale(z, W, inverse = 'I')

        gap = blas.dot(lmbda, lmbda)


def lp(c, G, h, A = None, b = None, kktsolver = None, solver = None, primalstart = None,
    dualstart = None, **kwargs):
    """

    Solves a pair of primal and dual LPs

        minimize    c'*x
        subject to  G*x + s = h
                    A*x = b
                    s >= 0

        maximize    -h'*z - b'*y
        subject to  G'*z + A'*y + c = 0
                    z >= 0.


    Input arguments.

        c is n x 1, G is m x n, h is m x 1, A is p x n, b is p x 1.  G and
        A must be dense or sparse 'd' matrices.  c, h and b are dense 'd'
        matrices with one column.  The default values for A and b are
        empty matrices with zero rows.

        solver is None, 'glpk' or 'mosek'.  The default solver (None)
        uses the cvxopt conelp() function.  The 'glpk' solver is the
        simplex LP solver from GLPK.  The 'mosek' solver is the LP solver
        from MOSEK.

        The arguments primalstart and dualstart are ignored when solver
        is 'glpk' or 'mosek', and are optional when solver is None.
        The argument primalstart is a dictionary with keys 'x' and 's',
        and specifies a primal starting point.  primalstart['x'] must
        be a dense 'd' matrix of length n;  primalstart['s'] must be a
        positive dense 'd' matrix of length m.
        The argument dualstart is a dictionary with keys 'z' and 'y',
        and specifies a dual starting point.   dualstart['y'] must
        be a dense 'd' matrix of length p;  dualstart['z'] must be a
        positive dense 'd' matrix of length m.

        When solver is None, we require n >= 1, Rank(A) = p and
        Rank([G; A]) = n


    Output arguments.

        Returns a dictionary with keys 'status', 'x', 's', 'z', 'y',
        'primal objective', 'dual objective', 'gap', 'relative gap',
        'primal infeasibility', 'dual infeasibility', 'primal slack',
        'dual slack', 'residual as primal infeasibility certificate',
        'residual as dual infeasibility certificate'.

        The 'status' field has values 'optimal', 'primal infeasible',
        'dual infeasible', or 'unknown'.  The values of the other fields
        depend on the exit status and the solver used.

        Status 'optimal'.
        - 'x', 's', 'y', 'z' are an approximate solution of the primal and
          dual optimality conditions

              G*x + s = h,  A*x = b
              G'*z + A'*y + c = 0
              s >= 0, z >= 0
              s'*z = 0.

        - 'primal objective': the primal objective c'*x.
        - 'dual objective': the dual objective -h'*z - b'*y.
        - 'gap': the duality gap s'*z.
        - 'relative gap': the relative gap, defined as s'*z / -c'*x if
          the primal objective is negative, s'*z / -(h'*z + b'*y) if the
          dual objective is positive, and None otherwise.
        - 'primal infeasibility': the residual in the primal constraints,
          defined as the maximum of the residual in the inequalities

              || G*x + s + h || / max(1, ||h||)

          and the residual in the equalities

              || A*x - b || / max(1, ||b||).

        - 'dual infeasibility': the residual in the dual constraints,
          defined as

              || G'*z + A'*y + c || / max(1, ||c||).

        - 'primal slack': the smallest primal slack min_k s_k.
        - 'dual slack': the smallest dual slack min_k z_k.
        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate': None.
        If the default solver is used, the primal infeasibility is
        guaranteed to be less than solvers.options['feastol']
        (default 1e-7).  The dual infeasibility is guaranteed to be less
        than solvers.options['feastol'] (default 1e-7).  The gap is less
        than solvers.options['abstol'] (default 1e-7) or the relative gap
        is less than solvers.options['reltol'] (default 1e-6).
        For the other solvers, the default GLPK or MOSEK exit criteria
        apply.

        Status 'primal infeasible'.  If the GLPK solver is used, all the
        fields except the status field are None.  For the default and
        the MOSEK solvers, the values are as follows.
        - 'x', 's': None.
        - 'y', 'z' are an approximate certificate of infeasibility

              -h'*z - b'*y = 1,  G'*z + A'*y = 0,  z >= 0.

        - 'primal objective': None.
        - 'dual objective': 1.0.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': None.
        - 'dual slack': the smallest dual slack min z_k.
        - 'residual as primal infeasibility certificate': the residual in
          the condition of the infeasibility certificate, defined as

              || G'*z + A'*y || / max(1, ||c||).

        - 'residual as dual infeasibility certificate': None.
        If the default solver is used, the residual as primal infeasiblity
        certificate is guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  For the other
        solvers, the default GLPK or MOSEK exit criteria apply.

        Status 'dual infeasible'.  If the GLPK solver is used, all the
        fields except the status field are empty.  For the default and the
        MOSEK solvers, the values are as follows.
        - 'x', 's' are an approximate proof of dual infeasibility

              c'*x = -1,  G*x + s = 0,  A*x = 0,  s >= 0.

        - 'y', 'z': None.
        - 'primal objective': -1.0.
        - 'dual objective': None.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': the smallest primal slack min_k s_k .
        - 'dual slack': None.
        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate: the residual in
          the conditions of the infeasibility certificate, defined as
          the maximum of

              || G*x + s || / max(1, ||h||) and || A*x || / max(1, ||b||).

        If the default solver is used, the residual as dual infeasiblity
        certificate is guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  For the other
        solvers, the default GLPK or MOSEK exit criteria apply.

        Status 'unknown'.  If the GLPK or MOSEK solver is used, all the
        fields except the status field are None.  If the default solver
        is used, the values are as follows.
        - 'x', 'y', 's', 'z' are the last iterates before termination.
          These satisfy s > 0 and z > 0, but are not necessarily feasible.
        - 'primal objective': the primal cost c'*x.
        - 'dual objective': the dual cost -h'*z - b'*y.
        - 'gap': the duality gap s'*z.
        - 'relative gap': the relative gap, defined as s'*z / -c'*x if the
          primal cost is negative, s'*z / -(h'*z + b'*y) if the dual cost
          is positive, and None otherwise.
        - 'primal infeasibility ': the residual in the primal constraints,
          defined as the maximum of the residual in the inequalities

              || G*x + s + h || / max(1, ||h||)

          and the residual in the equalities

              || A*x - b || / max(1, ||b||).

        - 'dual infeasibility': the residual in the dual constraints,
          defined as

              || G'*z + A'*y + c || / max(1, ||c||).

        - 'primal slack': the smallest primal slack min_k s_k.
        - 'dual slack': the smallest dual slack min_k z_k.
        - 'residual as primal infeasibility certificate':
           None if h'*z + b'*y >= 0; the residual

              || G'*z + A'*y || / (-(h'*z + b'*y) * max(1, ||c||) )

          otherwise.
        - 'residual as dual infeasibility certificate':
          None if c'*x >= 0; the maximum of the residuals

              || G*x + s || / (-c'*x * max(1, ||h||))

          and

              || A*x || / (-c'*x * max(1, ||b||))

          otherwise.
        Termination with status 'unknown' indicates that the algorithm
        failed to find a solution that satisfies the specified tolerances.
        In some cases, the returned solution may be fairly accurate.  If
        the primal and dual infeasibilities, the gap, and the relative gap
        are small, then x, y, s, z are close to optimal.  If the residual
        as primal infeasibility certificate is small, then

            y / (-h'*z - b'*y),   z / (-h'*z - b'*y)

        provide an approximate certificate of primal infeasibility.  If
        the residual as certificate of dual infeasibility is small, then

            x / (-c'*x),   s / (-c'*x)

        provide an approximate proof of dual infeasibility.


    Control parameters.

        The control parameters for the different solvers can be modified
        by adding an entry to the dictionary cvxopt.solvers.options.  The
        following parameters control the execution of the default solver.

            options['show_progress'] True/False (default: True)
            options['maxiters'] positive integer (default: 100)
            options['refinement']  positive integer (default: 0)
            options['abstol'] scalar (default: 1e-7)
            options['reltol'] scalar (default: 1e-6)
            options['feastol'] scalar (default: 1e-7).

        The control parameter names for GLPK are strings with the name of
        the GLPK parameter, listed in the GLPK documentation.  The MOSEK
        parameters can me modified by adding an entry options['mosek'],
        containing a dictionary with MOSEK parameter/value pairs, as
        described in the MOSEK documentation.

        Options that are not recognized are replaced by their default
        values.
    """
    options = kwargs.get('options',globals()['options'])

    import math
    from cvxopt import base, blas, misc
    from cvxopt.base import matrix, spmatrix

    if not isinstance(c, matrix) or c.typecode != 'd' or c.size[1] != 1:
        raise TypeError("'c' must be a dense column matrix")
    n = c.size[0]
    if n < 1: raise ValueError("number of variables must be at least 1")

    if not isinstance(G, (matrix,spmatrix)) or G.typecode != 'd' or G.size[1] != n:
        raise TypeError("'G' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    m = G.size[0]
    if not isinstance(h, matrix) or h.typecode != 'd' or h.size != (m,1):
        raise TypeError("'h' must be a 'd' matrix of size (%d,1)" %m)

    if A is None:  A = spmatrix([], [], [], (0,n), 'd')
    if not isinstance(A,(matrix,spmatrix)) or A.typecode != 'd' or A.size[1] != n:
        raise TypeError("'A' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    p = A.size[0]
    if b is None: b = matrix(0.0, (0,1))
    if not isinstance(b,matrix) or b.typecode != 'd' or b.size != (p,1):
        raise TypeError("'b' must be a dense matrix of size (%d,1)" %p)

    if solver == 'glpk':
        try: from cvxopt import glpk
        except ImportError: raise ValueError("invalid option "\
            "(solver = 'glpk'): cvxopt.glpk is not installed")
        opts = options.get('glpk',None)
        if opts:
            status, x, z, y = glpk.lp(c, G, h, A, b, options = opts)
        else:
            status, x, z, y = glpk.lp(c, G, h, A, b)

        if status == 'optimal':
            resx0 = max(1.0, blas.nrm2(c))
            resy0 = max(1.0, blas.nrm2(b))
            resz0 = max(1.0, blas.nrm2(h))

            pcost = blas.dot(c,x)
            dcost = -blas.dot(h,z) - blas.dot(b,y)

            s = matrix(h)
            base.gemv(G, x, s, alpha=-1.0, beta=1.0)

            gap = blas.dot(s, z)
            if pcost < 0.0:
                relgap = gap / -pcost
            elif dcost > 0.0:
                relgap = gap / dcost
            else:
                relgap = None

            # rx = c + G'*z + A'*y
            rx = matrix(c)
            base.gemv(G, z, rx, beta = 1.0, trans = 'T')
            base.gemv(A, y, rx, beta = 1.0, trans = 'T')
            resx = blas.nrm2(rx) / resx0

            # ry = b - A*x
            ry = matrix(b)
            base.gemv(A, x, ry, alpha = -1.0, beta = 1.0)
            resy = blas.nrm2(ry) / resy0

            # rz = G*x + s - h
            rz = matrix(0.0, (m,1))
            base.gemv(G, x, rz)
            blas.axpy(s, rz)
            blas.axpy(h, rz, alpha = -1.0)
            resz = blas.nrm2(rz) / resz0

            dims = {'l': m, 's': [], 'q': []}
            pslack = -misc.max_step(s, dims)
            dslack = -misc.max_step(z, dims)

            pres, dres = max(resy, resz), resx
            pinfres, dinfres = None, None

        else:
            s = None
            pcost, dcost = None, None
            gap, relgap = None, None
            pres, dres = None, None
            pslack, dslack = None, None
            pinfres, dinfres = None, None

        return {'status': status, 'x': x, 's': s, 'y': y, 'z': z,
            'primal objective': pcost, 'dual objective': dcost,
            'gap': gap, 'relative gap': relgap,
            'primal infeasibility': pres, 'dual infeasibility': dres,
            'primal slack': pslack, 'dual slack': dslack,
            'residual as primal infeasibility certificate': pinfres,
            'residual as dual infeasibility certificate': dinfres}

    if solver == 'mosek':
        try:
            from cvxopt import msk
            import mosek
        except ImportError:
            raise ValueError("invalid option (solver = 'mosek'): "\
                "cvxopt.msk is not installed")

        opts = options.get('mosek',None)
        if opts:
            solsta, x, z, y  = msk.lp(c, G, h, A, b, options=opts)
        else:
            solsta, x, z, y  = msk.lp(c, G, h, A, b)

        resx0 = max(1.0, blas.nrm2(c))
        resy0 = max(1.0, blas.nrm2(b))
        resz0 = max(1.0, blas.nrm2(h))

        if solsta in (mosek.solsta.optimal, getattr(mosek.solsta,'near_optimal',None)):
            if solsta is mosek.solsta.optimal: status = 'optimal'
            else: status = 'near optimal'

            pcost = blas.dot(c,x)
            dcost = -blas.dot(h,z) - blas.dot(b,y)

            # s = h - G*x
            s = matrix(h)
            base.gemv(G, x, s, alpha = -1.0, beta = 1.0)

            gap = blas.dot(s, z)
            if pcost < 0.0:
                relgap = gap / -pcost
            elif dcost > 0.0:
                relgap = gap / dcost
            else:
                relgap = None

            # rx = c + G'*z + A'*y
            rx = matrix(c)
            base.gemv(G, z, rx, beta = 1.0, trans = 'T')
            base.gemv(A, y, rx, beta = 1.0, trans = 'T')
            resx = blas.nrm2(rx) / resx0

            # ry = b - A*x
            ry = matrix(b)
            base.gemv(A, x, ry, alpha = -1.0, beta = 1.0)
            resy = blas.nrm2(ry) / resy0

            # rz = G*x + s - h
            rz = matrix(0.0, (m,1))
            base.gemv(G, x, rz)
            blas.axpy(s, rz)
            blas.axpy(h, rz, alpha = -1.0)
            resz = blas.nrm2(rz) / resz0

            dims = {'l': m, 's': [], 'q': []}
            pslack = -misc.max_step(s, dims)
            dslack = -misc.max_step(z, dims)

            pres, dres = max(resy, resz), resx
            pinfres, dinfres = None, None

        elif solsta is mosek.solsta.prim_infeas_cer:
            status = 'primal infeasible'

            hz, by = blas.dot(h, z),  blas.dot(b, y)
            blas.scal(1.0 / (-hz - by), y)
            blas.scal(1.0 / (-hz - by), z)

            # rx = -A'*y - G'*z
            rx = matrix(0.0, (n,1))
            base.gemv(A, y, rx, alpha = -1.0, trans = 'T')
            base.gemv(G, z, rx, alpha = -1.0, beta = 1.0, trans = 'T')
            pinfres =  blas.nrm2(rx) / resx0
            dinfres = None

            x, s = None, None
            pres, dres = None, None
            pcost, dcost = None, 1.0
            gap, relgap = None, None

            dims = {'l': m, 's': [], 'q': []}
            dslack = -misc.max_step(z, dims)
            pslack = None

        elif solsta == mosek.solsta.dual_infeas_cer:
            status = 'dual infeasible'

            cx = blas.dot(c,x)

            blas.scal(-1.0/cx, x)
            s = matrix(0.0, (m,1))
            base.gemv(G, x, s, alpha = -1.0)

            # ry = A*x
            ry = matrix(0.0, (p,1))
            base.gemv(A, x, ry)
            resy = blas.nrm2(ry) / resy0

            # rz = s + G*x
            rz = matrix(s)
            base.gemv(G, x, rz, beta = 1.0)
            resz = blas.nrm2(rz) / resz0

            pres, dres = None, None
            dinfres, pinfres = max(resy, resz), None
            z, y = None, None
            pcost, dcost = -1.0, None
            gap, relgap = None, None

            dims = {'l': m, 's': [], 'q': []}
            pslack = -misc.max_step(s, dims)
            dslack = None

        else:
            status = 'unknown'

            s = None
            pcost, dcost = None, None
            gap, relgap = None, None
            pres, dres = None, None
            pinfres, dinfres = None, None
            pslack, dslack = None, None

        return {'status': status, 'x': x, 's': s, 'y': y, 'z': z,
            'primal objective': pcost, 'dual objective': dcost,
            'gap': gap, 'relative gap': relgap,
            'primal infeasibility': pres, 'dual infeasibility': dres,
            'residual as primal infeasibility certificate': pinfres,
            'residual as dual infeasibility certificate': dinfres,
            'primal slack': pslack, 'dual slack': dslack}

    return conelp(c, G, h, {'l': m, 'q': [], 's': []}, A,  b, primalstart,
        dualstart, kktsolver = kktsolver, options = options)


def socp(c, Gl = None, hl = None, Gq = None, hq = None, A = None, b = None,
    kktsolver = None, solver = None, primalstart = None, dualstart = None, **kwargs):

    """
    Solves a pair of primal and dual SOCPs

        minimize    c'*x
        subject to  Gl*x + sl = hl
                    Gq[k]*x + sq[k] = hq[k],  k = 0, ..., N-1
                    A*x = b
                    sl >= 0,
                    sq[k] >= 0, k = 0, ..., N-1

        maximize    -hl'*z - sum_k hq[k]'*zq[k] - b'*y
        subject to  Gl'*zl + sum_k Gq[k]'*zq[k] + A'*y + c = 0
                    zl >= 0,  zq[k] >= 0, k = 0, ..., N-1.

    The inequalities sl >= 0 and zl >= 0 are elementwise vector
    inequalities.  The inequalities sq[k] >= 0, zq[k] >= 0 are second
    order cone inequalities, i.e., equivalent to

        sq[k][0] >= || sq[k][1:] ||_2,  zq[k][0] >= || zq[k][1:] ||_2.


    Input arguments.

        Gl is a dense or sparse 'd' matrix of size (ml, n).  hl is a
        dense 'd' matrix of size (ml, 1). The default values of Gl and hl
        are matrices with zero rows.

        The argument Gq is a list of N dense or sparse 'd' matrices of
        size (m[k] n), k = 0, ..., N-1, where m[k] >= 1.  hq is a list
        of N dense 'd' matrices of size (m[k], 1), k = 0, ..., N-1.
        The default values of Gq and hq are empty lists.

        A is a dense or sparse 'd' matrix of size (p,1).  b is a dense 'd'
        matrix of size (p,1).  The default values of A and b are matrices
        with zero rows.

        solver is None or 'mosek'.  The default solver (None) uses the
        cvxopt conelp() function.  The 'mosek' solver is the SOCP solver
        from MOSEK.

        The arguments primalstart and dualstart are ignored when solver
        is 'mosek', and are optional when solver is None.

        The argument primalstart is a dictionary with keys 'x', 'sl', 'sq',
        and specifies an optional primal starting point.
        primalstart['x'] is a dense 'd' matrix of size (n,1).
        primalstart['sl'] is a positive dense 'd' matrix of size (ml,1).
        primalstart['sq'] is a list of matrices of size (m[k],1), positive
        with respect to the second order cone of order m[k].

        The argument dualstart is a dictionary with keys 'y', 'zl', 'zq',
        and specifies an optional dual starting point.
        dualstart['y'] is a dense 'd' matrix of size (p,1).
        dualstart['zl'] is a positive dense 'd' matrix of size (ml,1).
        dualstart['sq'] is a list of matrices of size (m[k],1), positive
        with respect to the second order cone of order m[k].


    Output arguments.

        Returns a dictionary with keys 'status', 'x', 'sl', 'sq', 'zl',
        'zq', 'y', 'primal objective', 'dual objective', 'gap',
        'relative gap',  'primal infeasibility', 'dual infeasibility',
        'primal slack', 'dual slack', 'residual as primal infeasibility
        certificate', 'residual as dual infeasibility certificate'.

        The 'status' field has values 'optimal', 'primal infeasible',
        'dual infeasible', or 'unknown'.  The values of the other fields
        depend on the exit status and the solver used.

        Status 'optimal'.
        - 'x', 'sl', 'sq', 'y', 'zl', 'zq' are an approximate solution of
          the primal and dual optimality conditions

              G*x + s = h,  A*x = b
              G'*z + A'*y + c = 0
              s >= 0, z >= 0
              s'*z = 0

          where

              G = [ Gl; Gq[0]; ...; Gq[N-1] ]
              h = [ hl; hq[0]; ...; hq[N-1] ]
              s = [ sl; sq[0]; ...; sq[N-1] ]
              z = [ zl; zq[0]; ...; zq[N-1] ].

        - 'primal objective': the primal objective c'*x.
        - 'dual objective': the dual objective -h'*z - b'*y.
        - 'gap': the duality gap s'*z.
        - 'relative gap': the relative gap, defined as s'*z / -c'*x if
          the primal objective is negative, s'*z / -(h'*z + b'*y) if the
          dual objective is positive, and None otherwise.
        - 'primal infeasibility': the residual in the primal constraints,
          defined as the maximum of the residual in the inequalities

              || G*x + s + h || / max(1, ||h||)

          and the residual in the equalities

              || A*x - b || / max(1, ||b||).

        - 'dual infeasibility': the residual in the dual constraints,
          defined as

              || G'*z + A'*y + c || / max(1, ||c||).

        - 'primal slack': the smallest primal slack,

              min( min_k sl_k, min_k (sq[k][0] - || sq[k][1:] ||) ).

        - 'dual slack': the smallest dual slack,

              min( min_k zl_k, min_k (zq[k][0] - || zq[k][1:] ||) ).

        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate': None.
        If the default solver is used, the primal infeasibility is
        guaranteed to be less than solvers.options['feastol']
        (default 1e-7).  The dual infeasibility is guaranteed to be less
        than solvers.options['feastol'] (default 1e-7).  The gap is less
        than solvers.options['abstol'] (default 1e-7) or the relative gap
        is less than solvers.options['reltol'] (default 1e-6).
        If the MOSEK solver is used, the default MOSEK exit criteria
        apply.

        Status 'primal infeasible'.
        - 'x', 'sl', 'sq': None.
        - 'y', 'zl', 'zq' are an approximate certificate of infeasibility

              -h'*z - b'*y = 1,  G'*z + A'*y = 0,  z >= 0.

        - 'primal objective': None.
        - 'dual objective': 1.0.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': None.
        - 'dual slack': the smallest dual slack,

              min( min_k zl_k, min_k (zq[k][0] - || zq[k][1:] ||) ).

        - 'residual as primal infeasibility certificate': the residual in
          the condition of the infeasibility certificate, defined as

              || G'*z + A'*y || / max(1, ||c||).

        - 'residual as dual infeasibility certificate': None.
        If the default solver is used, the residual as primal infeasiblity
        certificate is guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  If the MOSEK solver is
        used, the default MOSEK exit criteria apply.

        Status 'dual infeasible'.
        - 'x', 'sl', 'sq': an approximate proof of dual infeasibility

              c'*x = -1,  G*x + s = 0,  A*x = 0,  s >= 0.

        - 'y', 'zl', 'zq': None.
        - 'primal objective': -1.0.
        - 'dual objective': None.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': the smallest primal slack,

              min( min_k sl_k, min_k (sq[k][0] - || sq[k][1:] ||) ).

        - 'dual slack': None.
        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate: the residual in
          the conditions of the infeasibility certificate, defined as
          the maximum of

              || G*x + s || / max(1, ||h||) and || A*x || / max(1, ||b||).

        If the default solver is used, the residual as dual infeasiblity
        certificate is guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  If the MOSEK solver
        is used, the default MOSEK exit criteria apply.

        Status 'unknown'.  If the MOSEK solver is used, all the fields
        except the status field are empty.  If the default solver
        is used, the values are as follows.
        - 'x', 'y', 'sl', 'sq', 'zl', 'zq': the last iterates before
          termination.   These satisfy s > 0 and z > 0, but are not
          necessarily feasible.
        - 'primal objective': the primal cost c'*x.
        - 'dual objective': the dual cost -h'*z - b'*y.
        - 'gap': the duality gap s'*z.
        - 'relative gap': the relative gap, defined as s'*z / -c'*x if the
          primal cost is negative, s'*z / -(h'*z + b'*y) if the dual cost
          is positive, and None otherwise.
        - 'primal infeasibility ': the residual in the primal constraints,
          defined as the maximum of the residual in the inequalities

              || G*x + s + h || / max(1, ||h||)

          and the residual in the equalities

              || A*x - b || / max(1, ||b||).

        - 'dual infeasibility': the residual in the dual constraints,
          defined as

              || G'*z + A'*y + c || / max(1, ||c||).

        - 'primal slack': the smallest primal slack,

              min( min_k sl_k, min_k (sq[k][0] - || sq[k][1:] ||) ).

        - 'dual slack': the smallest dual slack,

              min( min_k zl_k, min_k (zq[k][0] - || zq[k][1:] ||) ).

        - 'residual as primal infeasibility certificate':
           None if h'*z + b'*y >= 0; the residual

              || G'*z + A'*y || / (-(h'*z + b'*y) * max(1, ||c||) )

          otherwise.
        - 'residual as dual infeasibility certificate':
          None if c'*x >= 0; the maximum of the residuals

              || G*x + s || / (-c'*x * max(1, ||h||))

          and

              || A*x || / (-c'*x * max(1, ||b||))

          otherwise.
        Termination with status 'unknown' indicates that the algorithm
        failed to find a solution that satisfies the specified tolerances.
        In some cases, the returned solution may be fairly accurate.  If
        the primal and dual infeasibilities, the gap, and the relative gap
        are small, then x, y, s, z are close to optimal.  If the residual
        as primal infeasibility certificate is small, then

            y / (-h'*z - b'*y),   z / (-h'*z - b'*y)

        provide an approximate certificate of primal infeasibility.  If
        the residual as certificate of dual infeasibility is small, then

            x / (-c'*x),   s / (-c'*x)

        provide an approximate proof of dual infeasibility.


    Control parameters.

        The control parameters for the different solvers can be modified
        by adding an entry to the dictionary cvxopt.solvers.options.  The
        following parameters control the execution of the default solver.

            options['show_progress'] True/False (default: True)
            options['maxiters'] positive integer (default: 100)
            options['refinement'] positive integer (default: 1)
            options['abstol'] scalar (default: 1e-7)
            options['reltol'] scalar (default: 1e-6)
            options['feastol'] scalar (default: 1e-7).

        The MOSEK parameters can me modified by adding an entry
        options['mosek'], containing a dictionary with MOSEK
        parameter/value pairs, as described in the MOSEK documentation.

        Options that are not recognized are replaced by their default
        values.
    """

    from cvxopt import base, blas
    from cvxopt.base import matrix, spmatrix

    if not isinstance(c,matrix) or c.typecode != 'd' or c.size[1] != 1:
        raise TypeError("'c' must be a dense column matrix")
    n = c.size[0]
    if n < 1: raise ValueError("number of variables must be at least 1")

    if Gl is None:  Gl = spmatrix([], [], [], (0,n), tc='d')
    if not isinstance(Gl,(matrix,spmatrix)) or Gl.typecode != 'd' or Gl.size[1] != n:
        raise TypeError("'Gl' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    ml = Gl.size[0]
    if hl is None: hl = matrix(0.0, (0,1))
    if not isinstance(hl, matrix) or hl.typecode != 'd' or \
        hl.size != (ml,1):
        raise TypeError("'hl' must be a dense 'd' matrix of " \
            "size (%d,1)" %ml)

    if Gq is None: Gq = []
    if not isinstance(Gq,list) or [ G for G in Gq if not isinstance(G,(matrix,spmatrix)) \
                                    or G.typecode != 'd' or G.size[1] != n ]:
        raise TypeError("'Gq' must be a list of sparse or dense 'd' "\
            "matrices with %d columns" %n)
    mq = [ G.size[0] for G in Gq ]
    a = [ k for k in range(len(mq)) if mq[k] == 0 ]
    if a: raise TypeError("the number of rows of Gq[%d] is zero" %a[0])
    if hq is None: hq = []
    if not isinstance(hq,list) or len(hq) != len(mq) or \
      [ h for h in hq if not isinstance(h,(matrix,spmatrix)) or h.typecode != 'd' ]:
        raise TypeError("'hq' must be a list of %d dense or sparse "\
            "'d' matrices" %len(mq))
    a = [ k for k in range(len(mq)) if hq[k].size != (mq[k], 1) ]
    if a:
        k = a[0]
        raise TypeError("'hq[%d]' has size (%d,%d).  Expected size "\
            "is (%d,1)." %(k, hq[k].size[0], hq[k].size[1], mq[k]))

    if A is None: A = spmatrix([], [], [], (0,n), 'd')
    if not isinstance(A,(matrix,spmatrix)) or A.typecode != 'd' or A.size[1] != n:
        raise TypeError("'A' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    p = A.size[0]
    if b is None: b = matrix(0.0, (0,1))
    if not isinstance(b,matrix) or b.typecode != 'd' or b.size != (p,1):
        raise TypeError("'b' must be a dense matrix of size (%d,1)" %p)

    dims = {'l': ml, 'q': mq, 's': []}
    N = ml + sum(mq)

    if solver == 'mosek':
        from cvxopt import misc
        try:
            from cvxopt import msk
            import mosek
        except ImportError:
            raise ValueError("invalid option (solver = 'mosek'): "\
                "cvxopt.msk is not installed")
        if p: raise ValueError("socp() with the solver = 'mosek' option "\
            "does not handle problems with equality constraints")

        opts = options.get('mosek',None)
        if opts:
            solsta, x, zl, zq  = msk.socp(c, Gl, hl, Gq, hq, options=opts)
        else:
            solsta, x, zl, zq  = msk.socp(c, Gl, hl, Gq, hq)

        resx0 = max(1.0, blas.nrm2(c))
        rh = matrix([ blas.nrm2(hl) ] + [ blas.nrm2(hqk) for hqk in hq ])
        resz0 = max(1.0, blas.nrm2(rh))

        if solsta in (mosek.solsta.optimal, getattr(mosek.solsta,'near_optimal')):
            if solsta is mosek.solsta.optimal: status = 'optimal'
            else: status = 'near optimal'

            y = matrix(0.0, (0,1))
            pcost = blas.dot(c,x)
            dcost = -blas.dot(hl,zl) - \
                sum([ blas.dot(hq[k],zq[k]) for k in range(len(mq))])

            sl = matrix(hl)
            base.gemv(Gl, x, sl, alpha = -1.0, beta = 1.0)
            sq = [ +hqk for hqk in hq ]
            for k in range(len(Gq)):
                base.gemv(Gq[k], x, sq[k], alpha = -1.0, beta = 1.0)

            gap = blas.dot(sl, zl) + \
                sum([blas.dot(zq[k],sq[k]) for k in range(len(mq))])
            if pcost < 0.0:
                relgap = gap / -pcost
            elif dcost > 0.0:
                relgap = gap / dcost
            else:
                relgap = None

            # rx = c + G'*z
            rx = matrix(c)
            base.gemv(Gl, zl, rx, beta = 1.0, trans = 'T')
            for k in range(len(mq)):
                base.gemv(Gq[k], zq[k], rx, beta = 1.0, trans = 'T')
            resx = blas.nrm2(rx) / resx0

            # rz = G*x + s - h
            rz = matrix(0.0, (ml + sum(mq),1))
            base.gemv(Gl, x, rz)
            blas.axpy(sl, rz)
            blas.axpy(hl, rz, alpha = -1.0)
            ind = ml
            for k in range(len(mq)):
                base.gemv(Gq[k], x, rz, offsety = ind)
                blas.axpy(sq[k], rz, offsety = ind)
                blas.axpy(hq[k], rz, alpha = -1.0, offsety = ind)
                ind += mq[k]
            resz = blas.nrm2(rz) / resz0

            s, z = matrix(0.0, (N,1)),  matrix(0.0, (N,1))
            blas.copy(sl, s)
            blas.copy(zl, z)
            ind = ml
            for k in range(len(mq)):
                blas.copy(zq[k], z, offsety = ind)
                blas.copy(sq[k], s, offsety = ind)
                ind += mq[k]
            pslack = -misc.max_step(s, dims)
            dslack = -misc.max_step(z, dims)

            pres, dres = resz, resx
            pinfres, dinfres = None, None

        elif solsta is mosek.solsta.dual_infeas_cer:
            status = 'primal infeasible'
            y = matrix(0.0, (0,1))
            hz = blas.dot(hl, zl) + sum([blas.dot(hq[k],zq[k]) for k
                in range(len(mq))])
            blas.scal(1.0 / -hz, zl)
            for k in range(len(mq)):
                blas.scal(1.0 / -hz, zq[k])

            x, sl, sq = None, None, None

            # rx = - G'*z
            rx = matrix(0.0, (n,1))
            base.gemv(Gl, zl, rx, alpha = -1.0, beta = 1.0, trans = 'T')
            for k in range(len(mq)):
                base.gemv(Gq[k], zq[k], rx, beta = 1.0, trans = 'T')
            pinfres =  blas.nrm2(rx) / resx0
            dinfres = None

            z = matrix(0.0, (N,1))
            blas.copy(zl, z)
            ind = ml
            for k in range(len(mq)):
                blas.copy(zq[k], z, offsety = ind)
                ind += mq[k]
            dslack = -misc.max_step(z, dims)
            pslack = None

            x, s = None, None
            pres, dres = None, None
            pcost, dcost = None, 1.0
            gap, relgap = None, None

        elif solsta == mosek.solsta.prim_infeas_cer:
            status = 'dual infeasible'
            cx = blas.dot(c,x)

            blas.scal(-1.0/cx, x)
            sl = matrix(0.0, (ml,1))
            base.gemv(Gl, x, sl, alpha = -1.0)
            sq = [ matrix(0.0, (mqk,1)) for mqk in mq ]
            for k in range(len(mq)):
                base.gemv(Gq[k], x, sq[k], alpha = -1.0, beta = 1.0)

            # rz = s + G*x
            rz = matrix( [sl] + [sqk for sqk in sq])
            base.gemv(Gl, x, rz, beta = 1.0)
            ind = ml
            for k in range(len(mq)):
                base.gemv(Gq[k], x, rz, beta = 1.0, offsety = ind)
                ind += mq[k]
            resz = blas.nrm2(rz) / resz0

            dims = {'l': ml, 's': [], 'q': mq}
            s = matrix(0.0, (N,1))
            blas.copy(sl, s)
            ind = ml
            for k in range(len(mq)):
                blas.copy(sq[k], s, offsety = ind)
                ind += mq[k]
            pslack = -misc.max_step(s, dims)
            dslack = None

            pres, dres = None, None
            dinfres, pinfres = resz, None
            z, y = None, None
            pcost, dcost = -1.0, None
            gap, relgap = None, None

        else:
            status = 'unknown'
            sl, sq = None, None
            zl, zq = None, None
            x, y = None, None
            pcost, dcost = None, None
            gap, relgap = None, None
            pres, dres = None, None
            pinfres, dinfres = None, None
            pslack, dslack = None, None

        return {'status': status, 'x': x, 'sl': sl, 'sq': sq, 'y': y,
            'zl': zl, 'zq': zq, 'primal objective': pcost,
            'dual objective': dcost, 'gap': gap, 'relative gap': relgap,
            'primal infeasibility': pres, 'dual infeasibility': dres,
            'residual as primal infeasibility certificate': pinfres,
            'residual as dual infeasibility certificate': dinfres,
            'primal slack': pslack, 'dual slack': dslack}

    h = matrix(0.0, (N,1))
    if isinstance(Gl,matrix) or [ Gk for Gk in Gq if isinstance(Gk,matrix) ]:
        G = matrix(0.0, (N, n))
    else:
        G = spmatrix([], [], [], (N, n), 'd')
    h[:ml] = hl
    G[:ml,:] = Gl
    ind = ml
    for k in range(len(mq)):
        h[ind : ind + mq[k]] = hq[k]
        G[ind : ind + mq[k], :] = Gq[k]
        ind += mq[k]

    if primalstart:
        ps = {}
        ps['x'] = primalstart['x']
        ps['s'] = matrix(0.0, (N,1))
        if ml: ps['s'][:ml] = primalstart['sl']
        if mq:
            ind = ml
            for k in range(len(mq)):
                ps['s'][ind : ind + mq[k]] = primalstart['sq'][k][:]
                ind += mq[k]
    else:
        ps = None

    if dualstart:
        ds = {}
        if p:  ds['y'] = dualstart['y']
        ds['z'] = matrix(0.0, (N,1))
        if ml: ds['z'][:ml] = dualstart['zl']
        if mq:
            ind = ml
            for k in range(len(mq)):
                ds['z'][ind : ind + mq[k]] = dualstart['zq'][k][:]
                ind += mq[k]
    else:
        ds = None

    sol = conelp(c, G, h, dims, A = A, b = b, primalstart = ps, dualstart = ds, kktsolver = kktsolver, options = options)
    if sol['s'] is None:
        sol['sl'] = None
        sol['sq'] = None
    else:
        sol['sl'] = sol['s'][:ml]
        sol['sq'] = [ matrix(0.0, (m,1)) for m in mq ]
        ind = ml
        for k in range(len(mq)):
            sol['sq'][k][:] = sol['s'][ind : ind+mq[k]]
            ind += mq[k]
    del sol['s']

    if sol['z'] is None:
        sol['zl'] = None
        sol['zq'] = None
    else:
        sol['zl'] = sol['z'][:ml]
        sol['zq'] = [ matrix(0.0, (m,1)) for m in mq]
        ind = ml
        for k in range(len(mq)):
            sol['zq'][k][:] = sol['z'][ind : ind+mq[k]]
            ind += mq[k]
    del sol['z']

    return sol


def sdp(c, Gl = None, hl = None, Gs = None, hs = None, A = None, b = None,
    kktsolver = None, solver = None, primalstart = None, dualstart = None, **kwargs):
    """

    Solves a pair of primal and dual SDPs

        minimize    c'*x
        subject to  Gl*x + sl = hl
                    mat(Gs[k]*x) + ss[k] = hs[k], k = 0, ..., N-1
                    A*x = b
                    sl >= 0,  ss[k] >= 0, k = 0, ..., N-1

        maximize    -hl'*z - sum_k trace(hs[k]*zs[k]) - b'*y
        subject to  Gl'*zl + sum_k Gs[k]'*vec(zs[k]) + A'*y + c = 0
                    zl >= 0,  zs[k] >= 0, k = 0, ..., N-1.

    The inequalities sl >= 0 and zl >= 0 are elementwise vector
    inequalities.  The inequalities ss[k] >= 0, zs[k] >= 0 are matrix
    inequalities, i.e., the symmetric matrices ss[k] and zs[k] must be
    positive semidefinite.  mat(Gs[k]*x) is the symmetric matrix X with
    X[:] = Gs[k]*x.  For a symmetric matrix, zs[k], vec(zs[k]) is the
    vector zs[k][:].


    Input arguments.

        Gl is a dense or sparse 'd' matrix of size (ml, n).  hl is a
        dense 'd' matrix of size (ml, 1). The default values of Gl and hl
        are matrices with zero rows.

        The argument Gs is a list of N dense or sparse 'd' matrices of
        size (m[k]**2, n), k = 0, ..., N-1.   The columns of Gs[k]
        represent symmetric matrices stored as vectors in column major
        order.  hs is a list of N dense 'd' matrices of size (m[k], m[k]),
        k = 0, ..., N-1.  The columns of Gs[k] and the matrices hs[k]
        represent symmetric matrices in 'L' storage, i.e., only the lower
        triangular elements are accessed.  The default values of Gs and
        hs are empty lists.

        A is a dense or sparse 'd' matrix of size (p,n).  b is a dense 'd'
        matrix of size (p,1).  The default values of A and b are matrices
        with zero rows.

        solver is None or 'dsdp'.  The default solver (None) calls
        cvxopt.conelp().  The 'dsdp' solver uses an interface to DSDP5.
        The 'dsdp' solver does not accept problems with equality
        constraints (A and b must have zero rows, or be absent).

        The argument primalstart is a dictionary with keys 'x', 'sl',
        'ss', and specifies an optional primal starting point.
        primalstart['x'] is a dense 'd' matrix of length n;
        primalstart['sl'] is a  positive dense 'd' matrix of length ml;
        primalstart['ss'] is a list of positive definite matrices of
        size (ms[k], ms[k]).  Only the lower triangular parts of these
        matrices will be accessed.

        The argument dualstart is a dictionary with keys 'zl', 'zs', 'y'
        and specifies an optional dual starting point.
        dualstart['y'] is a dense 'd' matrix of length p;
        dualstart['zl'] must be a positive dense 'd' matrix of length ml;
        dualstart['zs'] is a list of positive definite matrices of
        size (ms[k], ms[k]).  Only the lower triangular parts of these
        matrices will be accessed.

        The arguments primalstart and dualstart are ignored when solver
        is 'dsdp'.


    Output arguments.

        Returns a dictionary with keys 'status', 'x', 'sl', 'ss', 'zl',
        'zs', 'y', 'primal objective', 'dual objective', 'gap',
        'relative gap',  'primal infeasibility', 'dual infeasibility',
        'primal slack', 'dual slack', 'residual as primal infeasibility
        certificate', 'residual as dual infeasibility certificate'.

        The 'status' field has values 'optimal', 'primal infeasible',
        'dual infeasible', or 'unknown'.  The values of the other fields
        depend on the exit status and the solver used.

        Status 'optimal'.
        - 'x', 'sl', 'ss', 'y', 'zl', 'zs' are an approximate solution of
          the primal and dual optimality conditions

              G*x + s = h,  A*x = b
              G'*z + A'*y + c = 0
              s >= 0, z >= 0
              s'*z = 0

          where

              G = [ Gl; Gs[0][:]; ...; Gs[N-1][:] ]
              h = [ hl; hs[0][:]; ...; hs[N-1][:] ]
              s = [ sl; ss[0][:]; ...; ss[N-1][:] ]
              z = [ zl; zs[0][:]; ...; zs[N-1][:] ].

        - 'primal objective': the primal objective c'*x.
        - 'dual objective': the dual objective -h'*z - b'*y.
        - 'gap': the duality gap s'*z.
        - 'relative gap': the relative gap, defined as s'*z / -c'*x if
          the primal objective is negative, s'*z / -(h'*z + b'*y) if the
          dual objective is positive, and None otherwise.
        - 'primal infeasibility': the residual in the primal constraints,
          defined as the maximum of the residual in the inequalities

              || G*x + s + h || / max(1, ||h||)

          and the residual in the equalities

              || A*x - b || / max(1, ||b||).

        - 'dual infeasibility': the residual in the dual constraints,
          defined as

              || G'*z + A'*y + c || / max(1, ||c||).

        - 'primal slack': the smallest primal slack,

              min( min_k sl_k, min_k lambda_min(mat(ss[k])) ).

        - 'dual slack': the smallest dual slack,

              min( min_k zl_k, min_k lambda_min(mat(zs[k])) ).

        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate': None.
        If the default solver is used, the primal infeasibility is
        guaranteed to be less than solvers.options['feastol']
        (default 1e-7).  The dual infeasibility is guaranteed to be less
        than solvers.options['feastol'] (default 1e-7).  The gap is less
        than solvers.options['abstol'] (default 1e-7) or the relative gap
        is less than solvers.options['reltol'] (default 1e-6).
        If the DSDP solver is used, the default DSDP exit criteria
        apply.

        Status 'primal infeasible'.
        - 'x', 'sl', 'ss': None.
        - 'y', 'zl', 'zs' are an approximate certificate of infeasibility

              -h'*z - b'*y = 1,  G'*z + A'*y = 0,  z >= 0.

        - 'primal objective': None.
        - 'dual objective': 1.0.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': None
        - 'dual slack': the smallest dual slack,

              min( min_k zl_k, min_k lambda_min(mat(zs[k])) ).

        - 'residual as primal infeasibility certificate': the residual in
          the condition of the infeasibility certificate, defined as

              || G'*z + A'*y || / max(1, ||c||).

        - 'residual as dual infeasibility certificate': None.
        If the default solver is used, the residual as primal infeasiblity
        certificate is guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  If the DSDP solver is
        used, the default DSDP exit criteria apply.

        Status 'dual infeasible'.
        - 'x', 'sl', 'ss': an approximate proof of dual infeasibility

              c'*x = -1,  G*x + s = 0,  A*x = 0,  s >= 0.

        - 'y', 'zl', 'zs': None.
        - 'primal objective': -1.0.
        - 'dual objective': None.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': the smallest primal slack,

              min( min_k sl_k, min_k lambda_min(mat(ss[k])) ).

        - 'dual slack': None.
        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate: the residual in
          the conditions of the infeasibility certificate, defined as
          the maximum of

              || G*x + s || / max(1, ||h||) and || A*x || / max(1, ||b||).

        If the default solver is used, the residual as dual infeasiblity
        certificate is guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  If the MOSEK solver
        is used, the default MOSEK exit criteria apply.

        Status 'unknown'.  If the DSDP solver is used, all the fields
        except the status field are empty.  If the default solver
        is used, the values are as follows.
        - 'x', 'y', 'sl', 'ss', 'zl', 'zs': the last iterates before
          termination.   These satisfy s > 0 and z > 0, but are not
          necessarily feasible.
        - 'primal objective': the primal cost c'*x.
        - 'dual objective': the dual cost -h'*z - b'*y.
        - 'gap': the duality gap s'*z.
        - 'relative gap': the relative gap, defined as s'*z / -c'*x if the
          primal cost is negative, s'*z / -(h'*z + b'*y) if the dual cost
          is positive, and None otherwise.
        - 'primal infeasibility ': the residual in the primal constraints,
          defined as the maximum of the residual in the inequalities

              || G*x + s + h || / max(1, ||h||)

          and the residual in the equalities

              || A*x - b || / max(1, ||b||).

        - 'dual infeasibility': the residual in the dual constraints,
          defined as

              || G'*z + A'*y + c || / max(1, ||c||).

        - 'primal slack': the smallest primal slack,

              min( min_k sl_k, min_k lambda_min(mat(ss[k])) ).

        - 'dual slack': the smallest dual slack,

              min( min_k zl_k, min_k lambda_min(mat(zs[k])) ).

        - 'residual as primal infeasibility certificate':
           None if h'*z + b'*y >= 0; the residual

              || G'*z + A'*y || / (-(h'*z + b'*y) * max(1, ||c||) )

          otherwise.
        - 'residual as dual infeasibility certificate':
          None if c'*x >= 0; the maximum of the residuals

              || G*x + s || / (-c'*x * max(1, ||h||))

          and

              || A*x || / (-c'*x * max(1, ||b||))

          otherwise.
        Termination with status 'unknown' indicates that the algorithm
        failed to find a solution that satisfies the specified tolerances.
        In some cases, the returned solution may be fairly accurate.  If
        the primal and dual infeasibilities, the gap, and the relative gap
        are small, then x, y, s, z are close to optimal.  If the residual
        as primal infeasibility certificate is small, then

            y / (-h'*z - b'*y),   z / (-h'*z - b'*y)

        provide an approximate certificate of primal infeasibility.  If
        the residual as certificate of dual infeasibility is small, then

            x / (-c'*x),   s / (-c'*x)

        provide an approximate proof of dual infeasibility.


    Control parameters.

        The following parameters control the execution of the default
        solver.

            options['show_progress'] True/False (default: True)
            options['maxiters'] positive integer (default: 100)
            options['refinement'] positive integer (default: 1)
            options['abstol'] scalar (default: 1e-7)
            options['reltol'] scalar (default: 1e-6)
            options['feastol'] scalar (default: 1e-7).

        The execution of the 'dsdp' solver is controlled by:

            options['DSDP_Monitor'] integer (default: 0)
            options['DSDP_MaxIts'] positive integer
            options['DSDP_GapTolerance'] scalar (default: 1e-5).
    """

    options = kwargs.get('options',globals()['options'])

    import math
    from cvxopt import base, blas, misc
    from cvxopt.base import matrix, spmatrix

    if not isinstance(c,matrix) or c.typecode != 'd' or c.size[1] != 1:
        raise TypeError("'c' must be a dense column matrix")
    n = c.size[0]
    if n < 1: raise ValueError("number of variables must be at least 1")

    if Gl is None: Gl = spmatrix([], [], [], (0,n), tc='d')
    if not isinstance(Gl,(matrix,spmatrix)) or Gl.typecode != 'd' or Gl.size[1] != n:
        raise TypeError("'Gl' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    ml = Gl.size[0]
    if hl is None: hl = matrix(0.0, (0,1))
    if not isinstance(hl,matrix) or hl.typecode != 'd' or \
        hl.size != (ml,1):
        raise TypeError("'hl' must be a 'd' matrix of size (%d,1)" %ml)

    if Gs is None: Gs = []
    if not isinstance(Gs,list) or [ G for G in Gs if not isinstance(G,(matrix,spmatrix)) \
      or G.typecode != 'd' or G.size[1] != n ]:
        raise TypeError("'Gs' must be a list of sparse or dense 'd' "\
            "matrices with %d columns" %n)
    ms = [ int(math.sqrt(G.size[0])) for G in Gs ]
    a = [ k for k in range(len(ms)) if ms[k]**2 != Gs[k].size[0] ]
    if a: raise TypeError("the squareroot of the number of rows in "\
        "'Gs[%d]' is not an integer" %k)
    if hs is None: hs = []
    if not isinstance(hs,list) or len(hs) != len(ms) \
      or [ h for h in hs if not isinstance(h,(matrix,spmatrix)) or h.typecode != 'd' ]:
        raise TypeError("'hs' must be a list of %d dense or sparse "\
            "'d' matrices" %len(ms))
    a = [ k for k in range(len(ms)) if hs[k].size != (ms[k],ms[k]) ]
    if a:
        k = a[0]
        raise TypeError("hs[%d] has size (%d,%d).  Expected size is "\
            "(%d,%d)." %(k,hs[k].size[0], hs[k].size[1], ms[k], ms[k]))

    if A is None: A = spmatrix([], [], [], (0,n), 'd')
    if not isinstance(A,(matrix,spmatrix)) or A.typecode != 'd' or A.size[1] != n:
        raise TypeError("'A' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    p = A.size[0]
    if b is None: b = matrix(0.0, (0,1))
    if not isinstance(b,matrix) or b.typecode != 'd' or b.size != (p,1):
        raise TypeError("'b' must be a dense matrix of size (%d,1)" %p)

    dims = {'l': ml, 'q': [], 's': ms}
    N = ml + sum([ m**2 for m in ms ])

    if solver == 'dsdp':
        try: from cvxopt import dsdp
        except ImportError: raise ValueError("invalid option "\
            "(solver = 'dsdp'): cvxopt.dsdp is not installed")
        if p: raise ValueError("sdp() with the solver = 'dsdp' option "\
            "does not handle problems with equality constraints")
        opts = options.get('dsdp',None)
        if opts:
            dsdpstatus, x, r, zl, zs = dsdp.sdp(c, Gl, hl, Gs, hs, options = opts)
        else:
            dsdpstatus, x, r, zl, zs = dsdp.sdp(c, Gl, hl, Gs, hs)

        resx0 = max(1.0, blas.nrm2(c))
        rh = matrix([ blas.nrm2(hl) ] + [ math.sqrt(misc.sdot2(hsk, hsk))
            for hsk in hs ])
        resz0 = max(1.0, blas.nrm2(rh))

        if dsdpstatus == 'DSDP_UNBOUNDED':
            status = 'dual infeasible'
            cx = blas.dot(c,x)
            blas.scal(-1.0/cx, x)
            sl = -Gl*x
            ss = [ -matrix(Gs[k]*x, (ms[k], ms[k])) for k in
                range(len(ms)) ]
            for k in range(len(ms)):
                misc.symm(ss[k], ms[k])

            # rz = s + G*x
            rz = matrix( [sl] + [ssk[:] for ssk in ss])
            base.gemv(Gl, x, rz, beta = 1.0)
            ind = ml
            for k in range(len(ms)):
                base.gemv(Gs[k], x, rz, beta = 1.0, offsety = ind)
                ind += ms[k]**2
            dims = {'l': ml, 's': ms, 'q': []}
            resz = misc.nrm2(rz, dims) / resz0

            s = matrix(0.0, (N,1))
            blas.copy(sl, s)
            ind = ml
            for k in range(len(ms)):
                blas.copy(ss[k], s, offsety = ind)
                ind += ms[k]
            pslack = -misc.max_step(s, dims)
            sslack = None

            pres, dres = None, None
            dinfres, pinfres = resz, None
            zl, zs, y = None, None, None
            pcost, dcost = -1.0, None
            gap, relgap = None, None

        elif dsdpstatus == 'DSDP_INFEASIBLE':
            status = 'primal infeasible'
            y = matrix(0.0, (0,1))
            hz = blas.dot(hl, zl) + misc.sdot2(hs, zs)
            blas.scal(1.0 / -hz, zl)
            for k in range(len(ms)):
                blas.scal(1.0 / -hz, zs[k])
                misc.symm(zs[k], ms[k])

            # rx = -G'*z
            rx = matrix(0.0, (n,1))
            base.gemv(Gl, zl, rx, alpha = -1.0, beta = 1.0, trans = 'T')
            ind = 0
            for k in range(len(ms)):
                blas.scal(0.5, zs[k], inc=ms[k]+1)
                for j in range(ms[k]):
                    blas.scal(0.0, zs[k], offset=j+ms[k]*(j+1), inc=ms[k])
                base.gemv(Gs[k], zs[k], rx, alpha=2.0, beta=1.0, trans='T')
                blas.scal(2.0, zs[k], inc=ms[k]+1)
                ind += ms[k]
            pinfres =  blas.nrm2(rx) / resx0
            dinfres = None

            z = matrix(0.0, (N,1))
            blas.copy(zl, z)
            ind = ml
            for k in range(len(ms)):
                blas.copy(zs[k], z, offsety = ind)
                ind += ms[k]
            dslack = -misc.max_step(z, dims)
            pslack = None

            x, sl, ss = None, None, None
            pres, dres = None, None
            pcost, dcost = None, 1.0
            gap, relgap = None, None

        else:
            if dsdpstatus == 'DSDP_PDFEASIBLE':
                status = 'optimal'
            else:
                status = 'unknown'
            y = matrix(0.0, (0,1))
            sl = hl - Gl*x
            ss = [ hs[k] - matrix(Gs[k]*x, (ms[k], ms[k])) for k in
                range(len(ms)) ]
            for k in range(len(ms)):
                misc.symm(ss[k], ms[k])
                misc.symm(zs[k], ms[k])
            pcost = blas.dot(c,x)
            dcost = -blas.dot(hl,zl) - misc.sdot2(hs, zs)
            gap = blas.dot(sl, zl) + misc.sdot2(ss, zs)
            if pcost < 0.0:
                relgap = gap / -pcost
            elif dcost > 0.0:
                relgap = gap / dcost
            else:
                relgap = None

            # rx = c + G'*z
            rx = matrix(c)
            base.gemv(Gl, zl, rx, beta = 1.0, trans = 'T')
            ind = 0
            for k in range(len(ms)):
                blas.scal(0.5, zs[k], inc = ms[k]+1)
                for j in range(ms[k]):
                    blas.scal(0.0, zs[k], offset=j+ms[k]*(j+1), inc=ms[k])
                base.gemv(Gs[k], zs[k], rx, alpha=2.0, beta=1.0, trans='T')
                blas.scal(2.0, zs[k], inc=ms[k]+1)
                ind += ms[k]
            resx = blas.nrm2(rx) / resx0

            # rz = G*x + s - h
            rz = matrix(0.0, (ml + sum([msk**2 for msk in ms]), 1))
            base.gemv(Gl, x, rz)
            blas.axpy(sl, rz)
            blas.axpy(hl, rz, alpha = -1.0)
            ind = ml
            for k in range(len(ms)):
                base.gemv(Gs[k], x, rz, offsety = ind)
                blas.axpy(ss[k], rz, offsety = ind, n = ms[k]**2)
                blas.axpy(hs[k], rz, alpha = -1.0, offsety = ind,
                    n = ms[k]**2)
                ind += ms[k]**2
            resz = misc.snrm2(rz, dims) / resz0
            pres, dres = resz, resx

            s, z = matrix(0.0, (N,1)), matrix(0.0, (N,1))
            blas.copy(sl, s)
            blas.copy(zl, z)
            ind = ml
            for k in range(len(ms)):
                blas.copy(ss[k], s, offsety = ind)
                blas.copy(zs[k], z, offsety = ind)
                ind += ms[k]
            pslack = -misc.max_step(s, dims)
            dslack = -misc.max_step(z, dims)

            if status == 'optimal' or dcost <= 0.0:
                pinfres = None
            else:
                # rx = G'*z
                rx = matrix(0.0, (n,1))
                base.gemv(Gl, zl, rx, beta = 1.0, trans = 'T')
                ind = 0
                for k in range(len(ms)):
                    blas.scal(0.5, zs[k], inc = ms[k]+1)
                    for j in range(ms[k]):
                        blas.scal(0.0, zs[k], offset=j+ms[k]*(j+1),
                            inc=ms[k])
                    base.gemv(Gs[k], zs[k], rx, alpha=2.0, beta=1.0,
                        trans='T')
                    blas.scal(2.0, zs[k], inc=ms[k]+1)
                    ind += ms[k]
                pinfres = blas.nrm2(rx) / resx0 / dcost

            if status == 'optimal' or pcost >= 0.0:
                dinfres = None
            else:
                # rz = G*x + s
                rz = matrix(0.0, (ml + sum([msk**2 for msk in ms]), 1))
                base.gemv(Gl, x, rz)
                blas.axpy(sl, rz)
                ind = ml
                for k in range(len(ms)):
                    base.gemv(Gs[k], x, rz, offsety = ind)
                    blas.axpy(ss[k], rz, offsety = ind, n = ms[k]**2)
                    ind += ms[k]**2
                dims = {'l': ml, 's': ms, 'q': []}
                dinfres = misc.snrm2(rz, dims) / resz0  / -pcost

        return {'status': status, 'x': x, 'sl': sl, 'ss': ss, 'y': y,
            'zl': zl, 'zs': zs, 'primal objective': pcost,
            'dual objective': dcost, 'gap': gap, 'relative gap': relgap,
            'primal infeasibility': pres, 'dual infeasibility': dres,
            'residual as primal infeasibility certificate': pinfres,
            'residual as dual infeasibility certificate': dinfres,
            'primal slack': pslack, 'dual slack': dslack}

    h = matrix(0.0, (N,1))
    if isinstance(Gl,matrix) or [ Gk for Gk in Gs if isinstance(Gk,matrix) ]:
        G = matrix(0.0, (N, n))
    else:
        G = spmatrix([], [], [], (N, n), 'd')
    h[:ml] = hl
    G[:ml,:] = Gl
    ind = ml
    for k in range(len(ms)):
        m = ms[k]
        h[ind : ind + m*m] = hs[k][:]
        G[ind : ind + m*m, :] = Gs[k]
        ind += m**2

    if primalstart:
        ps = {}
        ps['x'] = primalstart['x']
        ps['s'] = matrix(0.0, (N,1))
        if ml: ps['s'][:ml] = primalstart['sl']
        if ms:
            ind = ml
            for k in range(len(ms)):
                m = ms[k]
                ps['s'][ind : ind + m*m] = primalstart['ss'][k][:]
                ind += m**2
    else:
        ps = None

    if dualstart:
        ds = {}
        if p:  ds['y'] = dualstart['y']
        ds['z'] = matrix(0.0, (N,1))
        if ml: ds['z'][:ml] = dualstart['zl']
        if ms:
            ind = ml
            for k in range(len(ms)):
                m = ms[k]
                ds['z'][ind : ind + m*m] = dualstart['zs'][k][:]
                ind += m**2
    else:
        ds = None

    sol = conelp(c, G, h, dims, A = A, b = b, primalstart = ps, dualstart = ds, kktsolver = kktsolver, options = options)
    if sol['s'] is None:
        sol['sl'] = None
        sol['ss'] = None
    else:
        sol['sl'] = sol['s'][:ml]
        sol['ss'] = [ matrix(0.0, (mk, mk)) for mk in ms ]
        ind = ml
        for k in range(len(ms)):
            m = ms[k]
            sol['ss'][k][:] = sol['s'][ind:ind+m*m]
            ind += m**2
    del sol['s']

    if sol['z'] is None:
        sol['zl'] = None
        sol['zs'] = None
    else:
        sol['zl'] = sol['z'][:ml]
        sol['zs'] = [ matrix(0.0, (mk, mk)) for mk in ms ]
        ind = ml
        for k in range(len(ms)):
            m = ms[k]
            sol['zs'][k][:] = sol['z'][ind:ind+m*m]
            ind += m**2
    del sol['z']

    return sol


def qp(P, q, G = None, h = None, A = None, b = None, solver = None,
    kktsolver = None, initvals = None, **kwargs):

    """
    Solves a quadratic program

        minimize    (1/2)*x'*P*x + q'*x
        subject to  G*x <= h
                    A*x = b.


    Input arguments.

        P is a n x n dense or sparse 'd' matrix with the lower triangular
        part of P stored in the lower triangle.  Must be positive
        semidefinite.

        q is an n x 1 dense 'd' matrix.

        G is an m x n dense or sparse 'd' matrix.

        h is an m x 1 dense 'd' matrix.

        A is a p x n dense or sparse 'd' matrix.

        b is a p x 1 dense 'd' matrix or None.

        solver is None or 'mosek'.

        The default values for G, h, A and b are empty matrices with
        zero rows.


    Output arguments (default solver).

        Returns a dictionary with keys 'status', 'x', 's', 'y', 'z',
        'primal objective', 'dual objective', 'gap', 'relative gap',
        'primal infeasibility, 'dual infeasibility', 'primal slack',
        'dual slack'.

        The 'status' field has values 'optimal' or 'unknown'.
        If the status is 'optimal', 'x', 's', 'y', 'z' are an approximate
        solution of the primal and dual optimal solutions

            G*x + s = h,  A*x = b
            P*x + G'*z + A'*y + q = 0
            s >= 0, z >= 0
            s'*z = o.

        If the status is 'unknown', 'x', 's', 'y', 'z' are the last
        iterates before termination.  These satisfy s > 0 and z > 0, but
        are not necessarily feasible.

        The values of the other fields are defined as follows.

        - 'primal objective': the primal objective (1/2)*x'*P*x + q'*x.

        - 'dual objective': the dual objective

              L(x,y,z) = (1/2)*x'*P*x + q'*x + z'*(G*x - h) + y'*(A*x-b).

        - 'gap': the duality gap s'*z.

        - 'relative gap': the relative gap, defined as

              gap / -primal objective

          if the primal objective is negative,

              gap / dual objective

          if the dual objective is positive, and None otherwise.

        - 'primal infeasibility': the residual in the primal constraints,
          defined as the maximum of the residual in the inequalities

              || G*x + s + h || / max(1, ||h||)

          and the residual in the equalities

              || A*x - b || / max(1, ||b||).


        - 'dual infeasibility': the residual in the dual constraints,
          defined as

              || P*x + G'*z + A'*y + q || / max(1, ||q||).

        - 'primal slack': the smallest primal slack, min_k s_k.
        - 'dual slack': the smallest dual slack, min_k z_k.

        If the exit status is 'optimal', then the primal and dual
        infeasibilities are guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  The gap is less than
        solvers.options['abstol'] (default 1e-7) or the relative gap is
        less than solvers.options['reltol'] (default 1e-6).

        Termination with status 'unknown' indicates that the algorithm
        failed to find a solution that satisfies the specified tolerances.
        In some cases, the returned solution may be fairly accurate.  If
        the primal and dual infeasibilities, the gap, and the relative gap
        are small, then x, y, s, z are close to optimal.


    Output arguments (MOSEK solver).

        The return dictionary has two additional fields
        'residual as primal infeasibility certificate' and
        'residual as dual infeasibility certificate', and 'status' field
        can also have the values 'primal infeasible' or 'dual infeasible'.

        If the exit status is 'optimal', the different fields have the
        same meaning as for the default solver, but the the magnitude of
        the residuals and duality gap is controlled by the MOSEK exit
        criteria.  The 'residual as primal infeasibility certificate' and
        'residual as dual infeasibility certificate' are None.

        Status 'primal infeasible'.
        - 'x', 's': None.
        - 'y', 'z' are an approximate certificate of infeasibility

              G'*z + A'*y = 0,  h'*z + b'*y = -1,  z >= 0.

        - 'primal objective': None.
        - 'dual objective': 1.0.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': None.
        - 'dual slack': the smallest dual slack min z_k.
        - 'residual as primal infeasibility certificate': the residual in
          the condition of the infeasibility certificate, defined as

              || G'*z + A'*y || / max(1, ||c||).

        - 'residual as dual infeasibility certificate': None.

        Status 'dual infeasible'.
        - 'x', 's' are an approximate proof of dual infeasibility

              P*x = 0,  q'*x = -1,  G*x + s = 0,  A*x = 0,  s >= 0.

        - 'y', 'z': None.
        - 'primal objective': -1.0.
        - 'dual objective': None.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': the smallest primal slack min_k s_k .
        - 'dual slack': None.
        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate: the residual in
          the conditions of the infeasibility certificate, defined as
          the maximum of

              || P*x || / max(1, ||q||),
              || G*x + s || / max(1, ||h||),
              || A*x || / max(1, ||b||).


        If status is 'unknown', all the other fields are None.


    Control parameters.

        The control parameters for the different solvers can be modified
        by adding an entry to the dictionary cvxopt.solvers.options.  The
        following parameters control the execution of the default solver.

            options['show_progress'] True/False (default: True)
            options['maxiters'] positive integer (default: 100)
            options['refinement']  positive integer (default: 0)
            options['abstol'] scalar (default: 1e-7)
            options['reltol'] scalar (default: 1e-6)
            options['feastol'] scalar (default: 1e-7).

        The MOSEK parameters can me modified by adding an entry
        options['mosek'], containing a dictionary with MOSEK
        parameter/value pairs, as described in the MOSEK documentation.

        Options that are not recognized are replaced by their default
        values.
    """

    options = kwargs.get('options',globals()['options'])

    from cvxopt import base, blas
    from cvxopt.base import matrix, spmatrix

    if solver == 'mosek':
        from cvxopt import misc
        try:
            from cvxopt import msk
            import mosek
        except ImportError: raise ValueError("invalid option "\
            "(solver='mosek'): cvxopt.msk is not installed")

        opts = options.get('mosek',None)
        if opts:
            solsta, x, z, y = msk.qp(P, q, G, h, A, b, options=opts)
        else:
            solsta, x, z, y = msk.qp(P, q, G, h, A, b)

        n = q.size[0]
        if G is None: G = spmatrix([], [], [], (0,n), 'd')
        if h is None: h = matrix(0.0, (0,1))
        if A is None: A = spmatrix([], [], [], (0,n), 'd')
        if b is None: b = matrix(0.0, (0,1))
        m = G.size[0]

        resx0 = max(1.0, blas.nrm2(q))
        resy0 = max(1.0, blas.nrm2(b))
        resz0 = max(1.0, blas.nrm2(h))

        if solsta in (mosek.solsta.optimal, getattr(mosek.solsta,'near_optimal',None)):
            if solsta is mosek.solsta.optimal: status = 'optimal'
            else: status = 'near optimal'

            s = matrix(h)
            base.gemv(G, x, s, alpha = -1.0, beta = 1.0)

            # rx = q + P*x + G'*z + A'*y
            # pcost = 0.5 * x'*P*x + q'*x
            rx = matrix(q)
            base.symv(P, x, rx, beta = 1.0)
            pcost = 0.5 * (blas.dot(x, rx) + blas.dot(x, q))
            base.gemv(A, y, rx, beta = 1.0, trans = 'T')
            base.gemv(G, z, rx, beta = 1.0, trans = 'T')
            resx = blas.nrm2(rx) / resx0

            # ry = A*x - b
            ry = matrix(b)
            base.gemv(A, x, ry, alpha = 1.0, beta = -1.0)
            resy = blas.nrm2(ry) / resy0

            # rz = G*x + s - h
            rz = matrix(0.0, (m,1))
            base.gemv(G, x, rz)
            blas.axpy(s, rz)
            blas.axpy(h, rz, alpha = -1.0)
            resz = blas.nrm2(rz) / resz0

            gap = blas.dot(s, z)
            dcost = pcost + blas.dot(y, ry) + blas.dot(z, rz) - gap
            if pcost < 0.0:
                relgap = gap / -pcost
            elif dcost > 0.0:
                relgap = gap / dcost
            else:
                relgap = None

            dims = {'l': m, 's': [], 'q': []}
            pslack = -misc.max_step(s, dims)
            dslack = -misc.max_step(z, dims)

            pres, dres = max(resy, resz), resx
            pinfres, dinfres = None, None

        elif solsta == mosek.solsta.prim_infeas_cer:
            status = 'primal infeasible'

            hz, by = blas.dot(h, z),  blas.dot(b, y)
            blas.scal(1.0 / (-hz - by), y)
            blas.scal(1.0 / (-hz - by), z)

            # rx = -A'*y - G'*z
            rx = matrix(0.0, (q.size[0],1))
            base.gemv(A, y, rx, alpha = -1.0, trans = 'T')
            base.gemv(G, z, rx, alpha = -1.0, beta = 1.0, trans = 'T')
            pinfres =  blas.nrm2(rx) / resx0
            dinfres = None

            x, s = None, None
            pres, dres = None, None
            pcost, dcost = None, 1.0
            gap, relgap = None, None

            dims = {'l': m, 's': [], 'q': []}
            dslack = -misc.max_step(z, dims)
            pslack = None

        elif solsta == mosek.solsta.dual_infeas_cer:
            status = 'dual infeasible'
            qx = blas.dot(q,x)
            blas.scal(-1.0/qx, x)
            s = matrix(0.0, (m,1))
            base.gemv(G, x, s, alpha=-1.0)
            z, y = None, None

            # rz = P*x
            rx = matrix(0.0, (q.size[0],1))
            base.symv(P, x, rx, beta = 1.0)
            resx = blas.nrm2(rx) / resx0

            # ry = A*x
            ry = matrix(0.0, (b.size[0],1))
            base.gemv(A, x, ry)
            resy = blas.nrm2(ry) / resy0

            # rz = s + G*x
            rz = matrix(s)
            base.gemv(G, x, rz, beta = 1.0)
            resz = blas.nrm2(rz) / resz0

            pres, dres = None, None
            dinfres, pinfres = max(resx, resy, resz), None
            z, y = None, None
            pcost, dcost = -1.0, None
            gap, relgap = None, None

            dims = {'l': m, 's': [], 'q': []}
            pslack = -misc.max_step(s, dims)
            dslack = None

        else:
            status = 'unknown'
            x, s, y, z = None, None, None, None
            pcost, dcost = None, None
            gap, relgap = None, None
            pres, dres = None, None
            pslack, dslack = None, None
            pinfres, dinfres = None, None

        return {'status': status, 'x': x, 's': s, 'y': y, 'z': z,
            'primal objective': pcost, 'dual objective': dcost,
            'gap': gap, 'relative gap': relgap,
            'primal infeasibility': pres, 'dual infeasibility': dres,
            'primal slack': pslack, 'dual slack': dslack,
            'residual as primal infeasibility certificate': pinfres,
            'residual as dual infeasibility certificate': dinfres}

    return coneqp(P, q, G, h, None, A,  b, initvals, kktsolver = kktsolver, options = options)
