"""
Convex programming solver.

A primal-dual interior-point solver written in Python and interfaces
for quadratic and geometric programming.  Also includes an interface
to the quadratic programming solver from MOSEK.
"""

# Copyright 2012-2020 M. Andersen and L. Vandenberghe.
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


def cpl(c, F, G = None, h = None, dims = None, A = None, b = None, 
    kktsolver = None, xnewcopy = None, xdot = None, xaxpy = None,
    xscal = None, ynewcopy = None, ydot = None, yaxpy = None, 
    yscal = None, **kwargs):

    """
    Solves a convex optimization problem with a linear objective

        minimize    c'*x 
        subject to  f(x) <= 0
                    G*x <= h
                    A*x = b.                      

    f is vector valued, convex and twice differentiable.  The linear 
    inequalities are with respect to a cone C defined as the Cartesian 
    product of N + M + 1 cones:
    
        C = C_0 x C_1 x .... x C_N x C_{N+1} x ... x C_{N+M}.

    The first cone C_0 is the nonnegative orthant of dimension ml.  The 
    next N cones are second order cones of dimension mq[0], ..., mq[N-1].
    The second order cone of dimension m is defined as
    
        { (u0, u1) in R x R^{m-1} | u0 >= ||u1||_2 }.

    The next M cones are positive semidefinite cones of order ms[0], ...,
    ms[M-1] >= 0.  


    Input arguments (basic usage).

        c is a dense 'd' matrix of size (n,1). 

        F is a function that handles the following arguments.

            F() returns a tuple (mnl, x0).  mnl is the number of nonlinear 
            inequality constraints.  x0 is a point in the domain of f.

            F(x) returns a tuple (f, Df).

                f is  a dense 'd' matrix of size (mnl, 1) containing f(x). 

                Df is a dense or sparse 'd' matrix of size (mnl, n), 
                containing the derivatives of f at x:  Df[k,:] is the 
                transpose of the gradient of fk at x.  If x is not in 
                dom f, F(x) returns None or (None, None).

            F(x, z) with z a positive 'd' matrix of size (mnl,1), returns 
            a tuple (f, Df, H).
            
                f and Df are defined as above.
                
                H is a dense or sparse 'd' matrix of size (n,n).  The 
                lower triangular part of H contains the lower triangular
                part of sum_k z[k] * Hk where Hk is the Hessian of fk at x.

                If F is called with two arguments, it can be assumed that 
                x is dom f. 

            If Df and H are returned as sparse matrices, their sparsity
            patterns must be the same for each call to F(x) or F(x,z). 

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
    
        A is a dense or sparse 'd' matrix of size (p,n).  The default
        value is a sparse 'd' matrix of size (0,n).

        b is a dense 'd' matrix of size (p,1).  The default value is a 
        dense 'd' matrix of size (0,1).

        It is assumed that rank(A) = p and rank([H; A; Df; G]) = n at all 
        x in dom f.

        The other arguments are normally not needed.  They make it possible
        to exploit certain types of structure, as described further below.


    Output arguments.

        Returns a dictionary with keys 'status', 'x', 'snl', 'sl', 'znl', 
        'zl', 'y', 'primal objective', 'dual objective', 'gap', 
        'relative gap', 'primal infeasibility', 'dual infeasibility',
        'primal slack', 'dual slack'.

        The 'status' field has values 'optimal' or 'unknown'.
        If status is 'optimal', x, snl, sl, y, znl, zl are an approximate 
        solution of the primal and dual optimality conditions 

            f(x) + snl = 0,  G*x + sl = h,  A*x = b 
            Df(x)'*znl + G'*zl + A'*y + c = 0 
            snl >= 0,  znl >= 0,  sl >= 0,  zl >= 0
            snl'*znl + sl'* zl = 0.

        If status is 'unknown', x, snl, sl, y, znl, zl are the last
        iterates before termination.  They satisfy snl > 0, znl > 0, 
        sl > 0, zl > 0, but are not necessarily feasible.

        The values of the other fields are defined as follows.

        - 'primal objective': the primal objective c'*x.

        - 'dual objective': the dual objective 

              L(x,y,znl,zl) = c'*x + znl'*f(x) + zl'*(G*x-h) + y'*(A*x-b).

        - 'gap': the duality gap snl'*znl + sl'*zl.

        - 'relative gap': the relative gap, defined as 

              gap / -primal objective

          if the primal objective is negative, 

              gap / dual objective

          if the dual objective is positive, and None otherwise.

        - 'primal infeasibility': the residual in the primal constraints,
          defined as 

              || (f(x) + snl, G*x + sl - h, A*x-b) ||_2  

          divided by 

              max(1, || (f(x0) + 1, G*x0 + 1 - h, A*x0 - b) ||_2 )

          where x0 is the point returned by F().

        - 'dual infeasibility': the residual in the dual constraints,
          defined as

              || c + Df(x)'*znl + G'*zl + A'*y ||_2
 
          divided by 

              max(1, || c + Df(x0)'*1 + G'*1 ||_2 ).

        - 'primal slack': the smallest primal slack, min( min_k sl_k,
          sup {t | sl >= te} ) where 

              e = ( e_0, e_1, ..., e_N, e_{N+1}, ..., e_{M+N} )
    
          is the identity vector in C.  e_0 is an ml-vector of ones, 
          e_k, k = 1,..., N, is the unit vector (1,0,...,0) of length
          mq[k], and e_k = vec(I) where I is the identity matrix of order
          ms[k].

        - 'dual slack': the smallest dual slack, min( min_k zl_k,
          sup {t | zl >= te} ).
               

        If the exit status is 'optimal', then the primal and dual
        infeasibilities are guaranteed to be less than 
        solvers.options['feastol'] (default 1e-7).  The gap is less than
        solvers.options['abstol'] (default 1e-7) or the relative gap is 
        less than solvers.options['reltol'] (defaults 1e-6).     

        Termination with status 'unknown' indicates that the algorithm 
        failed to find a solution that satisfies the specified tolerances.
        In some cases, the returned solution may be fairly accurate.  If
        the primal and dual infeasibilities, the gap, and the relative gap
        are small, then x, y, snl, sl, znl, zl are close to optimal.


    Advanced usage.

        Three mechanisms are provided to express problem structure.

        1.  The user can provide a customized routine for solving 
        linear equations ('KKT systems')

            [ sum_k zk*Hk(x)  A'  GG'   ] [ ux ]   [ bx ]
            [ A               0   0     ] [ uy ] = [ by ]
            [ GG              0   -W'*W ] [ uz ]   [ bz ]

        where GG = [ Df(x);  G ], uz = (uznl, uzl), bz = (bznl, bzl).  

        z is a positive vector of length mnl and x is a point in the domain
        of f.   W is a scaling matrix, i.e., a block diagonal mapping

           W*u = ( Wnl*unl, W0*u_0, ..., W_{N+M}*u_{N+M} )

        defined as follows.

        - For the nonlinear block (Wnl):

              Wnl = diag(dnl),

          with dnl a positive vector of length mnl.

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
        called as g = kktsolver(x, z, W).  W is a dictionary that contains
        the parameters of the scaling:

        - W['dnl'] is a positive 'd' matrix of size (mnl, 1).
        - W['dnli'] is a positive 'd' matrix with the elementwise inverse 
          of W['dnl'].
        - W['d'] is a positive 'd' matrix of size (ml, 1).
        - W['di'] is a positive 'd' matrix with the elementwise inverse of
          W['d'].
        - W['beta'] is a list [ beta_0, ..., beta_{N-1} ]
        - W['v'] is a list [ v_0, ..., v_{N-1} ]
        - W['r'] is a list [ r_0, ..., r_{M-1} ]
        - W['rti'] is a list [ rti_0, ..., rti_{M-1} ], with rti_k the
          inverse of the transpose of r_k.

        The call g = kktsolver(x, z, W) should return a function g that
        solves the KKT system by g(x, y, z).  On entry, x, y, z contain 
        the righthand side bx, by, bz.  On exit, they contain the 
        solution, with uz scaled: W*uz is returned instead of uz.  In other
        words, on exit x, y, z are the solution of

            [ sum_k zk*Hk(x)  A'   GG'*W^{-1} ] [ ux ]   [ bx ]
            [ A               0    0          ] [ uy ] = [ by ].
            [ GG              0   -W'         ] [ uz ]   [ bz ]


        2.  The linear operators Df*u, H*u, G*u and A*u can be specified 
        by providing Python functions instead of matrices.  This can only 
        be done in combination with 1. above, i.e., it also requires the 
        kktsolver argument.
        
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

        If Df is a function, the call Df(u, v, alpha, beta, trans) should
        evaluate the matrix-vectors products

            v := alpha * Df(x) * u + beta * v if trans is 'N'
            v := alpha * Df(x)' * u + beta * v if trans is 'T'.

        If H is a function, the call H(u, v, alpha, beta) should evaluate 
        the matrix-vectors product

            v := alpha * H * u + beta * v.


        3.  Instead of using the default representation of the primal 
        variable x and the dual variable y as one-column 'd' matrices, 
        we can represent these variables and the corresponding parameters 
        c and b by arbitrary Python objects (matrices, lists, dictionaries,
        etc).  This can only be done in combination with 1. and 2. above,
        i.e., it requires a user-provided KKT solver and a function
        description of the linear mappings.   It also requires the 
        arguments xnewcopy, xdot, xscal, xaxpy, ynewcopy, ydot, yscal, 
        yaxpy.  These arguments are functions defined as follows.
   
        If X is the vector space of primal variables x, then:
        - xnewcopy(u) creates a new copy of the vector u in X.
        - xdot(u, v) returns the inner product of two vectors u and v in X.
        - xscal(alpha, u) computes u := alpha*u, where alpha is a scalar
          and u is a vector in X.
        - xaxpy(u, v, alpha = 1.0, beta = 0.0) computes v := alpha*u + v
          for a scalar alpha and two vectors u and v in X.

        If Y is the vector space of primal variables y:
        - ynewcopy(u) creates a new copy of the vector u in Y.
        - ydot(u, v) returns the inner product of two vectors u and v in Y.
        - yscal(alpha, u) computes u := alpha*u, where alpha is a scalar
          and u is a vector in Y.
        - yaxpy(u, v, alpha = 1.0, beta = 0.0) computes v := alpha*u + v
          for a scalar alpha and two vectors u and v in Y.


    Control parameters.

       The following control parameters can be modified by adding an
       entry to the dictionary options.

       options['show_progress'] True/False (default: True)
       options['maxiters'] positive integer (default: 100)
       options['refinement'] nonnegative integer (default: 1)
       options['abstol'] scalar (default: 1e-7)
       options['reltol'] scalar (default: 1e-6)
       options['feastol'] scalar (default: 1e-7).

    """

    import math 
    from cvxopt import base, blas, misc
    from cvxopt.base import matrix, spmatrix 

    STEP = 0.99
    BETA = 0.5
    ALPHA = 0.01
    EXPON = 3
    MAX_RELAXED_ITERS = 8

    options = kwargs.get('options',globals()['options'])

    DEBUG = options.get('debug',False)

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

    refinement = options.get('refinement',1)
    if not isinstance(refinement,(int,long)) or refinement < 0:
        raise ValueError("options['refinement'] must be a nonnegative integer")

    if kktsolver is None: 
        if dims and (dims['q'] or dims['s']):  
            kktsolver = 'chol'            
        else:
            kktsolver = 'chol2'            
    defaultsolvers = ('ldl', 'ldl2', 'chol', 'chol2')
    if type(kktsolver) is str and kktsolver not in defaultsolvers:
        raise ValueError("'%s' is not a valid value for kktsolver" \
            %kktsolver)

    try: mnl, x0 = F()   
    except: raise ValueError("function call 'F()' failed")
    
    # Argument error checking depends on level of customization.
    customkkt = type(kktsolver) is not str
    operatorG = G is not None and type(G) not in (matrix, spmatrix)
    operatorA = A is not None and type(A) not in (matrix, spmatrix)
    if (operatorG or operatorA) and not customkkt:
        raise ValueError("use of function valued G, A requires a "\
            "user-provided kktsolver")
    customx = (xnewcopy != None or xdot != None or xaxpy != None or 
        xscal != None)
    if customx and (not operatorG or not operatorA or not customkkt):
        raise ValueError("use of non-vector type for x requires "\
            "function valued G, A and user-provided kktsolver")
    customy = (ynewcopy != None or ydot != None or yaxpy != None or 
        yscal != None) 
    if customy and (not operatorA or not customkkt):
        raise ValueError("use of non vector type for y requires "\
            "function valued A and user-provided kktsolver")

    if not customx:  
        if type(x0) is not matrix or x0.typecode != 'd' or x0.size[1] != 1:
            raise TypeError("'x0' must be a 'd' matrix with one column")
        if type(c) is not matrix or c.typecode != 'd' or c.size != x0.size:
            raise TypeError("'c' must be a 'd' matrix of size (%d,%d)"\
                %(x0.size[0],1))
        
    if h is None: h = matrix(0.0, (0,1))
    if type(h) is not matrix or h.typecode != 'd' or h.size[1] != 1:
        raise TypeError("'h' must be a 'd' matrix with 1 column")

    if not dims:  dims = {'l': h.size[0], 'q': [], 's': []}

    # Dimension of the product cone of the linear inequalities. with 's' 
    # components unpacked.
    cdim = dims['l'] + sum(dims['q']) + sum([ k**2 for k in dims['s'] ])
    if h.size[0] != cdim:
        raise TypeError("'h' must be a 'd' matrix of size (%d,1)" %cdim)

    if G is None:
        if customx:
            def G(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
                if trans == 'N': pass
                else: xscal(beta, y)
        else:
            G = spmatrix([], [], [], (0, c.size[0]))
    if not operatorG:
        if G.typecode != 'd' or G.size != (cdim, c.size[0]):
            raise TypeError("'G' must be a 'd' matrix with size (%d, %d)"\
                %(cdim, c.size[0]))
        def fG(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
            misc.sgemv(G, x, y, dims, trans = trans, alpha = alpha, 
                beta = beta)
    else:
        fG = G

    if A is None:
        if customx or customy:
            def A(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
                if trans == 'N': pass
                else: yscal(beta, y)
        else:
            A = spmatrix([], [], [], (0, c.size[0]))
    if not operatorA:
        if A.typecode != 'd' or A.size[1] != c.size[0]:
            raise TypeError("'A' must be a 'd' matrix with %d columns" \
                %c.size[0])
        def fA(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
            base.gemv(A, x, y, trans = trans, alpha = alpha, beta = beta)
    else:
        fA = A
    if not customy:
        if b is None: b = matrix(0.0, (0,1))
        if type(b) is not matrix or b.typecode != 'd' or b.size[1] != 1:
            raise TypeError("'b' must be a 'd' matrix with one column")
        if not operatorA and b.size[0] != A.size[0]:
            raise TypeError("'b' must have length %d" %A.size[0])
    if b is None and customy:  
        raise ValueEror("use of non vector type for y requires b")

   
    # kktsolver(x, z, W) returns a routine for solving
    #
    #     [ sum_k zk*Hk(x)  A'   GG'*W^{-1} ] [ ux ]   [ bx ]
    #     [ A               0    0          ] [ uy ] = [ by ]
    #     [ GG              0   -W'         ] [ uz ]   [ bz ]
    #
    # where G = [Df(x); G].

    if kktsolver in defaultsolvers:
         if kktsolver == 'ldl': 
             factor = misc.kkt_ldl(G, dims, A, mnl, kktreg = KKTREG)
         elif kktsolver == 'ldl2': 
             factor = misc.kkt_ldl2(G, dims, A, mnl)
         elif kktsolver == 'chol':
             factor = misc.kkt_chol(G, dims, A, mnl)
         else: 
             factor = misc.kkt_chol2(G, dims, A, mnl)
         def kktsolver(x, z, W):
             f, Df, H = F(x, z)
             return factor(W, H, Df)             


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
             

    # Initial points
    x = xnewcopy(x0)
    y = ynewcopy(b);  yscal(0.0, y)
    z, s = matrix(0.0, (mnl + cdim, 1)), matrix(0.0, (mnl + cdim, 1))
    z[: mnl+dims['l']] = 1.0 
    s[: mnl+dims['l']] = 1.0 
    ind = mnl + dims['l']
    for m in dims['q']:
        z[ind] = 1.0
        s[ind] = 1.0
        ind += m
    for m in dims['s']:
        z[ind : ind + m*m : m+1] = 1.0
        s[ind : ind + m*m : m+1] = 1.0
        ind += m**2


    rx, ry = xnewcopy(x0), ynewcopy(b)
    rznl, rzl = matrix(0.0, (mnl, 1)), matrix(0.0, (cdim, 1)), 
    dx, dy = xnewcopy(x), ynewcopy(y)   
    dz, ds = matrix(0.0, (mnl + cdim, 1)), matrix(0.0, (mnl + cdim, 1))

    lmbda = matrix(0.0, (mnl + dims['l'] + sum(dims['q']) + 
        sum(dims['s']), 1))
    lmbdasq = matrix(0.0, (mnl + dims['l'] + sum(dims['q']) + 
        sum(dims['s']), 1))
    sigs = matrix(0.0, (sum(dims['s']), 1))
    sigz = matrix(0.0, (sum(dims['s']), 1))

    dz2, ds2 = matrix(0.0, (mnl + cdim, 1)), matrix(0.0, (mnl + cdim, 1))

    newx, newy = xnewcopy(x),  ynewcopy(y)
    newz, news = matrix(0.0, (mnl + cdim, 1)), matrix(0.0, (mnl + cdim, 1))
    newrx = xnewcopy(x0)
    newrznl = matrix(0.0, (mnl, 1))

    rx0, ry0 = xnewcopy(x0), ynewcopy(b)
    rznl0, rzl0 = matrix(0.0, (mnl, 1)), matrix(0.0, (cdim, 1)), 
    x0, dx0 = xnewcopy(x), xnewcopy(dx)
    y0, dy0 = ynewcopy(y), ynewcopy(dy)
    z0 = matrix(0.0, (mnl + cdim, 1))
    dz0 = matrix(0.0, (mnl + cdim, 1))
    dz20 = matrix(0.0, (mnl + cdim, 1))
    s0 = matrix(0.0, (mnl + cdim, 1))
    ds0 = matrix(0.0, (mnl + cdim, 1))
    ds20 = matrix(0.0, (mnl + cdim, 1))
    W0 = {}
    W0['dnl'] = matrix(0.0, (mnl, 1))
    W0['dnli'] = matrix(0.0, (mnl, 1))
    W0['d'] = matrix(0.0, (dims['l'], 1))
    W0['di'] = matrix(0.0, (dims['l'], 1))
    W0['v'] = [ matrix(0.0, (m, 1)) for m in dims['q'] ]
    W0['beta'] = len(dims['q']) * [ 0.0 ]
    W0['r'] = [ matrix(0.0, (m, m)) for m in dims['s'] ]
    W0['rti'] = [ matrix(0.0, (m, m)) for m in dims['s'] ]
    lmbda0 = matrix(0.0, (mnl + dims['l'] + sum(dims['q']) + 
        sum(dims['s']), 1))
    lmbdasq0 = matrix(0.0, (mnl + dims['l'] + sum(dims['q']) + 
        sum(dims['s']), 1))
    

    if show_progress: 
        print("% 10s% 12s% 10s% 8s% 7s" %("pcost", "dcost", "gap", "pres",
            "dres"))


    relaxed_iters = 0
    for iters in range(MAXITERS + 1):  

        if refinement or DEBUG:  
            # We need H to compute residuals of KKT equations.
            f, Df, H = F(x, z[:mnl])
        else:
            f, Df = F(x)
       
        f = matrix(f, tc='d')
        if f.typecode != 'd' or f.size != (mnl, 1):
            raise TypeError("first output argument of F() must be a "\
                "'d' matrix of size (%d, %d)" %(mnl, 1))

        if type(Df) is matrix or type(Df) is spmatrix:
            if customx: raise ValueError("use of non-vector type for x "\
                "requires function valued Df")
            if Df.typecode != 'd' or Df.size != (mnl, c.size[0]):
                raise TypeError("second output argument of F() must "\
                    "be a 'd' matrix of size (%d,%d)" %(mnl, c.size[0]))
            def fDf(u, v, alpha = 1.0, beta = 0.0, trans = 'N'): 
                base.gemv(Df, u, v, alpha = alpha, beta = beta, trans = 
                    trans)
        else: 
            if not customkkt:
                raise ValueError("use of function valued Df requires "\
                    "a user-provided kktsolver")
            fDf = Df

        if refinement or DEBUG:
            if type(H) is matrix or type(H) is spmatrix:
                if customx: raise ValueError("use of non-vector type "\
                    "for  x requires function valued H")
                if H.typecode != 'd' or H.size != (c.size[0], c.size[0]):
                    raise TypeError("third output argument of F() must "\
                        "be a 'd' matrix of size (%d,%d)" \
                        %(c.size[0], c.size[0]))
                def fH(u, v, alpha = 1.0, beta = 0.0): 
                    base.symv(H, u, v, alpha = alpha, beta = beta)
            else: 
                if not customkkt:
                    raise ValueError("use of function valued H requires "\
                        "a user-provided kktsolver")
                fH = H
           

        gap = misc.sdot(s, z, dims, mnl) 

        # rx = c + A'*y + Df'*z[:mnl] + G'*z[mnl:]
        xcopy(c, rx) 
        fA(y, rx, beta = 1.0, trans = 'T')
        fDf(z[:mnl], rx, beta = 1.0, trans = 'T')
        fG(z[mnl:], rx, beta = 1.0, trans = 'T')
        resx = math.sqrt(xdot(rx, rx))
           
        # ry = A*x - b
        ycopy(b, ry)
        fA(x, ry, alpha = 1.0, beta = -1.0)
        resy = math.sqrt(ydot(ry, ry))

        # rznl = s[:mnl] + f 
        blas.copy(s[:mnl], rznl)
        blas.axpy(f, rznl)
        resznl = blas.nrm2(rznl)

        # rzl = s[mnl:] + G*x - h
        blas.copy(s[mnl:], rzl)
        blas.axpy(h, rzl, alpha = -1.0)
        fG(x, rzl, beta = 1.0)
        reszl = misc.snrm2(rzl, dims)

        # Statistics for stopping criteria.

        # pcost = c'*x
        # dcost = c'*x + y'*(A*x-b) + znl'*f(x) + zl'*(G*x-h)
        #       = c'*x + y'*(A*x-b) + znl'*(f(x)+snl) + zl'*(G*x-h+sl) 
        #         - z'*s
        #       = c'*x + y'*ry + znl'*rznl + zl'*rzl - gap
        pcost = xdot(c,x)
        dcost = pcost + ydot(y, ry) + blas.dot(z[:mnl], rznl) + \
            misc.sdot(z[mnl:], rzl, dims) - gap
        if pcost < 0.0:
            relgap = gap / -pcost
        elif dcost > 0.0:
            relgap = gap / dcost
        else:
            relgap = None
        pres = math.sqrt( resy**2 + resznl**2 + reszl**2 )
        dres = resx
        if iters == 0: 
            resx0 = max(1.0, resx)
            resznl0 = max(1.0, resznl)
            pres0 = max(1.0, pres)
            dres0 = max(1.0, dres)
            gap0 = gap
            theta1 = 1.0 / gap0
            theta2 = 1.0 / resx0
            theta3 = 1.0 / resznl0
        phi = theta1 * gap + theta2 * resx + theta3 * resznl
        pres = pres / pres0
        dres = dres / dres0

        if show_progress:
            print("%2d: % 8.4e % 8.4e % 4.0e% 7.0e% 7.0e" \
                %(iters, pcost, dcost, gap, pres, dres))

        # Stopping criteria.    
        if ( pres <= FEASTOL and dres <= FEASTOL and ( gap <= ABSTOL or 
            (relgap is not None and relgap <= RELTOL) )) or \
            iters == MAXITERS:
            sl, zl = s[mnl:], z[mnl:]
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(sl, m, ind)
                misc.symm(zl, m, ind)
                ind += m**2
            ts = misc.max_step(s, dims, mnl)
            tz = misc.max_step(z, dims, mnl)
            if iters == MAXITERS:
                if show_progress:
                    print("Terminated (maximum number of iterations "\
                        "reached).")
                status = 'unknown'
            else:
                if show_progress:
                    print("Optimal solution found.")
                status = 'optimal'

            return {'status': status, 'x': x,  'y': y, 'znl': z[:mnl],  
                'zl': zl, 'snl': s[:mnl], 'sl': sl, 'gap': gap, 
                'relative gap': relgap, 'primal objective': pcost, 
                'dual objective': dcost,  'primal slack': -ts, 
                'dual slack': -tz, 'primal infeasibility': pres,
                'dual infeasibility': dres }


        # Compute initial scaling W: 
        #
        #     W * z = W^{-T} * s = lambda.
        #
        # lmbdasq = lambda o lambda 

        if iters == 0:  
            W = misc.compute_scaling(s, z, lmbda, dims, mnl)
        misc.ssqr(lmbdasq, lmbda, dims, mnl)


        # f3(x, y, z) solves
        #
        #     [ H   A'  GG'*W^{-1} ] [ ux ]   [ bx ]
        #     [ A   0   0          ] [ uy ] = [ by ].
        #     [ GG  0  -W'         ] [ uz ]   [ bz ]
        #
        # On entry, x, y, z contain bx, by, bz.
        # On exit, they contain ux, uy, uz.
        
        try: f3 = kktsolver(x, z[:mnl], W)
        except ArithmeticError: 
            singular_kkt_matrix = False
            if iters == 0:
                raise ValueError("Rank(A) < p or "\
                    "Rank([H(x); A; Df(x); G]) < n")

            elif 0 < relaxed_iters < MAX_RELAXED_ITERS > 0:
                # The arithmetic error may be caused by a relaxed line 
                # search in the previous iteration.  Therefore we restore 
                # the last saved state and require a standard line search. 

                phi, gap = phi0, gap0
                mu = gap / ( mnl + dims['l'] + len(dims['q']) + 
                    sum(dims['s']) )
                blas.copy(W0['dnl'], W['dnl'])
                blas.copy(W0['dnli'], W['dnli'])
                blas.copy(W0['d'], W['d'])
                blas.copy(W0['di'], W['di'])
                for k in range(len(dims['q'])):
                    blas.copy(W0['v'][k], W['v'][k])
                    W['beta'][k] = W0['beta'][k]
                for k in range(len(dims['s'])):
                    blas.copy(W0['r'][k], W['r'][k])
                    blas.copy(W0['rti'][k], W['rti'][k])
                xcopy(x0, x); 
                ycopy(y0, y); 
                blas.copy(s0, s); blas.copy(z0, z)
                blas.copy(lmbda0, lmbda)
                blas.copy(lmbdasq, lmbdasq0)
                xcopy(rx0, rx); ycopy(ry0, ry)
                resx = math.sqrt(xdot(rx, rx))
                blas.copy(rznl0, rznl);  blas.copy(rzl0, rzl);
                resznl = blas.nrm2(rznl)

                relaxed_iters = -1

                try: f3 = kktsolver(x, z[:mnl], W)
                except ArithmeticError: 
                     singular_kkt_matrix = True

            else:  
                 singular_kkt_matrix = True

            if singular_kkt_matrix:
                sl, zl = s[mnl:], z[mnl:]
                ind = dims['l'] + sum(dims['q'])
                for m in dims['s']:
                    misc.symm(sl, m, ind)
                    misc.symm(zl, m, ind)
                    ind += m**2
                ts = misc.max_step(s, dims, mnl)
                tz = misc.max_step(z, dims, mnl)
                if show_progress:
                    print("Terminated (singular KKT matrix).")
                status = 'unknown'
                return {'status': status, 'x': x,  'y': y, 
                    'znl': z[:mnl],  'zl': zl, 'snl': s[:mnl], 
                    'sl': sl, 'gap': gap, 'relative gap': relgap, 
                    'primal objective': pcost, 'dual objective': dcost,  
                    'primal infeasibility': pres, 
                    'dual infeasibility': dres, 'primal slack': -ts,
                    'dual slack': -tz }


        # f4_no_ir(x, y, z, s) solves
        # 
        #     [ 0     ]   [ H   A'  GG' ] [ ux        ]   [ bx ]
        #     [ 0     ] + [ A   0   0   ] [ uy        ] = [ by ]
        #     [ W'*us ]   [ GG  0   0   ] [ W^{-1}*uz ]   [ bz ]
        #
        #     lmbda o (uz + us) = bs.
        #
        # On entry, x, y, z, x, contain bx, by, bz, bs.
        # On exit, they contain ux, uy, uz, us.

        if iters == 0:
            ws3 = matrix(0.0, (mnl + cdim, 1))
            wz3 = matrix(0.0, (mnl + cdim, 1))

        def f4_no_ir(x, y, z, s):

            # Solve 
            #
            #     [ H  A'  GG'  ] [ ux        ]   [ bx                    ]
            #     [ A  0   0    ] [ uy        ] = [ by                    ]
            #     [ GG 0  -W'*W ] [ W^{-1}*uz ]   [ bz - W'*(lmbda o\ bs) ]
            #
            #     us = lmbda o\ bs - uz.
            
            # s := lmbda o\ s 
            #    = lmbda o\ bs
            misc.sinv(s, lmbda, dims, mnl)

            # z := z - W'*s 
            #    = bz - W' * (lambda o\ bs)
            blas.copy(s, ws3)
            misc.scale(ws3, W, trans = 'T')
            blas.axpy(ws3, z, alpha = -1.0)

            # Solve for ux, uy, uz
            f3(x, y, z)

            # s := s - z 
            #    = lambda o\ bs - z.
            blas.axpy(z, s, alpha = -1.0)


        if iters == 0:
            wz2nl, wz2l = matrix(0.0, (mnl,1)), matrix(0.0, (cdim, 1))

        def res(ux, uy, uz, us, vx, vy, vz, vs):

            # Evaluates residuals in Newton equations:
            #
            #     [ vx ]     [ 0     ]   [ H  A' GG' ] [ ux        ]
            #     [ vy ] -=  [ 0     ] + [ A  0  0   ] [ uy        ]
            #     [ vz ]     [ W'*us ]   [ GG 0  0   ] [ W^{-1}*uz ]
            #
            #     vs -= lmbda o (uz + us).

            # vx := vx - H*ux - A'*uy - GG'*W^{-1}*uz
            fH(ux, vx, alpha = -1.0, beta = 1.0)
            fA(uy, vx, alpha = -1.0, beta = 1.0, trans = 'T') 
            blas.copy(uz, wz3)
            misc.scale(wz3, W, inverse = 'I')
            fDf(wz3[:mnl], vx, alpha = -1.0, beta = 1.0, trans = 'T')
            fG(wz3[mnl:], vx, alpha = -1.0, beta = 1.0, trans = 'T') 

            # vy := vy - A*ux 
            fA(ux, vy, alpha = -1.0, beta = 1.0)

            # vz := vz - W'*us - GG*ux 
            fDf(ux, wz2nl)
            blas.axpy(wz2nl, vz, alpha = -1.0)
            fG(ux, wz2l)
            blas.axpy(wz2l, vz, alpha = -1.0, offsety = mnl)
            blas.copy(us, ws3) 
            misc.scale(ws3, W, trans = 'T')
            blas.axpy(ws3, vz, alpha = -1.0)

            # vs -= lmbda o (uz + us)
            blas.copy(us, ws3)
            blas.axpy(uz, ws3)
            misc.sprod(ws3, lmbda, dims, mnl, diag = 'D')
            blas.axpy(ws3, vs, alpha = -1.0)


        # f4(x, y, z, s) solves the same system as f4_no_ir, but applies
        # iterative refinement.

        if iters == 0:
            if refinement or DEBUG:
                wx, wy = xnewcopy(c), ynewcopy(b)
                wz = matrix(0.0, (mnl + cdim, 1))
                ws = matrix(0.0, (mnl + cdim, 1))
            if refinement:
                wx2, wy2 = xnewcopy(c), ynewcopy(b)
                wz2 = matrix(0.0, (mnl + cdim, 1))
                ws2 = matrix(0.0, (mnl + cdim, 1))

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
                res(x, y, z, s, wx2, wy2, wz2, ws2) 
                f4_no_ir(wx2, wy2, wz2, ws2)
                xaxpy(wx2, x)
                yaxpy(wy2, y)
                blas.axpy(wz2, z)
                blas.axpy(ws2, s)
            if DEBUG:
                res(x, y, z, s, wx, wy, wz, ws)
                print("KKT residuals:")
                print("    'x': %e" %math.sqrt(xdot(wx, wx)))
                print("    'y': %e" %math.sqrt(ydot(wy, wy)))
                print("    'z': %e" %misc.snrm2(wz, dims, mnl))
                print("    's': %e" %misc.snrm2(ws, dims, mnl))
     

        sigma, eta = 0.0, 0.0
        for i in [0, 1]:

            # Solve
            #
            #     [ 0     ]   [ H  A' GG' ] [ dx        ]
            #     [ 0     ] + [ A  0  0   ] [ dy        ] = -(1 - eta)*r  
            #     [ W'*ds ]   [ GG 0  0   ] [ W^{-1}*dz ]
            #
            #     lmbda o (dz + ds) = -lmbda o lmbda + sigma*mu*e.
            #

            mu = gap / (mnl + dims['l'] + len(dims['q']) + sum(dims['s']))

            # ds = -lmbdasq + sigma * mu * e  
            blas.scal(0.0, ds)
            blas.axpy(lmbdasq, ds, n = mnl + dims['l'] + sum(dims['q']), 
                alpha = -1.0)
            ds[:mnl + dims['l']] += sigma*mu
            ind = mnl + dims['l']
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
       
            # (dx, dy, dz) := -(1-eta) * (rx, ry, rz)
            xscal(0.0, dx);  xaxpy(rx, dx, alpha = -1.0 + eta)
            yscal(0.0, dy);  yaxpy(ry, dy, alpha = -1.0 + eta)
            blas.scal(0.0, dz) 
            blas.axpy(rznl, dz, alpha = -1.0 + eta)
            blas.axpy(rzl, dz, alpha = -1.0 + eta, offsety = mnl)
            
            try: f4(dx, dy, dz, ds)
            except ArithmeticError: 
                if iters == 0:
                    raise ValueError("Rank(A) < p or "\
                        "Rank([H(x); A; Df(x); G]) < n")
                else:
                    sl, zl = s[mnl:], z[mnl:]
                    ind = dims['l'] + sum(dims['q'])
                    for m in dims['s']:
                        misc.symm(sl, m, ind)
                        misc.symm(zl, m, ind)
                        ind += m**2
                    ts = misc.max_step(s, dims, mnl)
                    tz = misc.max_step(z, dims, mnl)
                    if show_progress:
                        print("Terminated (singular KKT matrix).")
                    return {'status': 'unknown', 'x': x,  'y': y, 
                        'znl': z[:mnl],  'zl': zl, 'snl': s[:mnl], 
                        'sl': sl, 'gap': gap, 'relative gap': relgap, 
                        'primal objective': pcost, 'dual objective': dcost,
                        'primal infeasibility': pres, 
                        'dual infeasibility': dres, 'primal slack': -ts,
                        'dual slack': -tz }

            # Inner product ds'*dz and unscaled steps are needed in the 
            # line search.
            dsdz = misc.sdot(ds, dz, dims, mnl)
            blas.copy(dz, dz2)
            misc.scale(dz2, W, inverse = 'I')
            blas.copy(ds, ds2)
            misc.scale(ds2, W, trans = 'T')

            # Maximum steps to boundary. 
            # 
            # Also compute the eigenvalue decomposition of 's' blocks in 
            # ds, dz.  The eigenvectors Qs, Qz are stored in ds, dz.
            # The eigenvalues are stored in sigs, sigz.

            misc.scale2(lmbda, ds, dims, mnl)
            ts = misc.max_step(ds, dims, mnl, sigs)
            misc.scale2(lmbda, dz, dims, mnl)
            tz = misc.max_step(dz, dims, mnl, sigz)
            t = max([ 0.0, ts, tz ])
            if t == 0:
                step = 1.0
            else:
                step = min(1.0, STEP / t)

            # Backtrack until newx is in domain of f.
            backtrack = True
            while backtrack:
                xcopy(x, newx);  xaxpy(dx, newx, alpha = step)
                t = F(newx)
                if t is None: newf = None
                else: newf, newDf = t[0], t[1]
                if newf is not None:
                    backtrack = False
                else:
                    step *= BETA


            # Merit function 
            #
            #     phi = theta1 * gap + theta2 * norm(rx) + 
            #         theta3 * norm(rznl)
            #
            # and its directional derivative dphi.

            phi = theta1 * gap + theta2 * resx + theta3 * resznl
            if i == 0:
                dphi = -phi 
            else:
                dphi = -theta1 * (1 - sigma) * gap -  \
                    theta2 * (1 - eta) * resx - \
                    theta3 * (1 - eta) * resznl  


            # Line search.
            #
            # We use two types of line search.  In a standard iteration we
            # use is a normal backtracking line search terminating with a 
            # sufficient decrease of phi.  In a "relaxed" iteration the 
            # line search is terminated after one step, regardless of the 
            # value of phi.  We make at most MAX_RELAXED_ITERS consecutive
            # relaxed iterations.  When starting a series of relaxed 
            # iteration, we save the state at the end of the first line 
            # search in the series (scaling, primal and dual iterates, 
            # steps, sigma, eta, i.e., all information needed to resume 
            # the line search at some later point).  If a series of
            # MAX_RELAXED_ITERS relaxed iterations does not result in a 
            # sufficient decrease compared to the value of phi at the start
            # of the series, then we resume the first line search in the 
            # series as a standard line search (using the saved state).
            # On the other hand, if at some point during the series of 
            # relaxed iterations we obtain a sufficient decrease of phi 
            # compared with the value at the start of the series, then we 
            # start a new series of relaxed line searches. 
            #  
            # To implement this we use a counter relaxed_iters.
            #
            # 1. If 0 <= relaxed_iters < MAX_RELAXED_ITERS, we use a 
            #    relaxed line search (one full step).  If relaxed_iters 
            #    is 0, we save the value phi0 of the merit function at the
            #    current point, as well as the state at the end of the 
            #    line search.  If the relaxed line search results in a 
            #    sufficient decrease w.r.t. phi0, we reset relaxed_iters 
            #    to 0.  Otherwise we increment relaxed_iters.
            #
            # 2. If relaxed_iters is MAX_RELAXED_ITERS, we use a standard
            #    line search.  If this results in a sufficient decrease 
            #    of the merit function compared with phi0, we set 
            #    relaxed_iters to 0.  If phi decreased compared with phi0,
            #    but not sufficiently, we set relaxed_iters to -1.  
            #    If phi increased compared with phi0, we resume the 
            #    backtracking at the last saved iteration, and after 
            #    completing the line search, set relaxed_iters to 0.
            # 
            # 3. If relaxed_iters is -1, we use a standard line search
            #    and increment relaxed_iters to 0. 


            backtrack = True
            while backtrack:
                xcopy(x, newx);  xaxpy(dx, newx, alpha = step)
                ycopy(y, newy);  yaxpy(dy, newy, alpha = step)
                blas.copy(z, newz);  blas.axpy(dz2, newz, alpha = step) 
                blas.copy(s, news);  blas.axpy(ds2, news, alpha = step) 

                t = F(newx)
                newf, newDf = matrix(t[0], tc = 'd'), t[1]
                if type(newDf) is matrix or type(Df) is spmatrix:
                    if newDf.typecode != 'd' or \
                        newDf.size != (mnl, c.size[0]):
                            raise TypeError("second output argument "\
                                "of F() must be a 'd' matrix of size "\
                                "(%d,%d)" %(mnl, c.size[0]))
                    def newfDf(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):
                        base.gemv(newDf, u, v, alpha = alpha, beta = 
                                beta, trans = trans)
                else: 
                    newfDf = newDf

                # newrx = c + A'*newy + newDf'*newz[:mnl] + G'*newz[mnl:]
                xcopy(c, newrx) 
                fA(newy, newrx, beta = 1.0, trans = 'T')
                newfDf(newz[:mnl], newrx, beta = 1.0, trans = 'T')
                fG(newz[mnl:], newrx, beta = 1.0, trans = 'T')
                newresx = math.sqrt(xdot(newrx, newrx))
           
                # newrznl = news[:mnl] + newf 
                blas.copy(news[:mnl], newrznl)
                blas.axpy(newf, newrznl)
                newresznl = blas.nrm2(newrznl)

                newgap = (1.0 - (1.0 - sigma) * step) * gap \
                    + step**2 * dsdz
                newphi = theta1 * newgap  + theta2 * newresx + \
                    theta3 * newresznl

                if i == 0:
                    if newgap <= (1.0 - ALPHA * step) * gap and \
                        ( 0 <= relaxed_iters < MAX_RELAXED_ITERS or \
                        newphi <= phi + ALPHA * step * dphi ):
                        backtrack = False
                        sigma = min(newgap/gap, (newgap/gap) ** EXPON)
                        eta = 0.0 
                    else:
                        step *= BETA

                else:
                    if relaxed_iters == -1 or ( relaxed_iters == 0 == 
                        MAX_RELAXED_ITERS ):
                        # Do a standard line search.
                        if newphi <= phi + ALPHA * step * dphi:
                            relaxed_iters == 0
                            backtrack = False
                        else:
                            step *= BETA

                    elif relaxed_iters == 0 < MAX_RELAXED_ITERS:
                        if newphi <= phi + ALPHA * step * dphi:
                            # Relaxed l.s. gives sufficient decrease.
                            relaxed_iters = 0

                        else:
                            # Save state.
                            phi0, dphi0, gap0 = phi, dphi, gap
                            step0 = step
                            blas.copy(W['dnl'], W0['dnl'])
                            blas.copy(W['dnli'], W0['dnli'])
                            blas.copy(W['d'], W0['d'])
                            blas.copy(W['di'], W0['di'])
                            for k in range(len(dims['q'])):
                                blas.copy(W['v'][k], W0['v'][k])
                                W0['beta'][k] = W['beta'][k]
                            for k in range(len(dims['s'])):
                                blas.copy(W['r'][k], W0['r'][k])
                                blas.copy(W['rti'][k], W0['rti'][k])
                            xcopy(x, x0); xcopy(dx, dx0)
                            ycopy(y, y0); ycopy(dy, dy0)
                            blas.copy(s, s0); blas.copy(z, z0)
                            blas.copy(ds, ds0)
                            blas.copy(dz, dz0)
                            blas.copy(ds2, ds20)
                            blas.copy(dz2, dz20)
                            blas.copy(lmbda, lmbda0)
                            blas.copy(lmbdasq, lmbdasq0)
                            dsdz0 = dsdz
                            sigma0, eta0 = sigma, eta
                            xcopy(rx, rx0);  ycopy(ry, ry0)
                            blas.copy(rznl, rznl0); blas.copy(rzl, rzl0)
                            relaxed_iters = 1

                        backtrack = False

                    elif 0 <= relaxed_iters < MAX_RELAXED_ITERS > 0:
                        if newphi <= phi0 + ALPHA * step0 * dphi0:
                            # Relaxed l.s. gives sufficient decrease.
                            relaxed_iters = 0

                        else: 
                            # Relaxed line search 
                            relaxed_iters += 1

                        backtrack = False

                    elif relaxed_iters == MAX_RELAXED_ITERS > 0:
                        if newphi <= phi0 + ALPHA * step0 * dphi0:
                            # Series of relaxed line searches ends 
                            # with sufficient decrease w.r.t. phi0.
                            backtrack = False
                            relaxed_iters = 0

                        elif newphi >= phi0:
                            # Resume last saved line search.
                            phi, dphi, gap = phi0, dphi0, gap0
                            step = step0
                            blas.copy(W0['dnl'], W['dnl'])
                            blas.copy(W0['dnli'], W['dnli'])
                            blas.copy(W0['d'], W['d'])
                            blas.copy(W0['di'], W['di'])
                            for k in range(len(dims['q'])):
                                blas.copy(W0['v'][k], W['v'][k])
                                W['beta'][k] = W0['beta'][k]
                            for k in range(len(dims['s'])):
                                blas.copy(W0['r'][k], W['r'][k])
                                blas.copy(W0['rti'][k], W['rti'][k])
                            xcopy(x0, x); xcopy(dx0, dx);
                            ycopy(y0, y); ycopy(dy0, dy);
                            blas.copy(s0, s); blas.copy(z0, z)
                            blas.copy(ds0, ds)
                            blas.copy(dz0, dz)
                            blas.copy(ds20, ds2)
                            blas.copy(dz20, dz2)
                            blas.copy(lmbda0, lmbda)
                            dsdz = dsdz0
                            sigma, eta = sigma0, eta0
                            relaxed_iters = -1

                        elif newphi <= phi + ALPHA * step * dphi:
                            # Series of relaxed line searches ends with
                            # with insufficient decrease w.r.t. phi0" 
                            backtrack = False
                            relaxed_iters = -1


        # Update x, y.
        xaxpy(dx, x, alpha = step)
        yaxpy(dy, y, alpha = step)


        # Replace nonlinear, 'l' and 'q' blocks of ds and dz with the 
        # updated variables in the current scaling.
        # Replace 's' blocks of ds and dz with the factors Ls, Lz in a
        # factorization Ls*Ls', Lz*Lz' of the updated variables in the 
        # current scaling.

        # ds := e + step*ds for nonlinear, 'l' and 'q' blocks.
        # dz := e + step*dz for nonlinear, 'l' and 'q' blocks.
        blas.scal(step, ds, n = mnl + dims['l'] + sum(dims['q']))
        blas.scal(step, dz, n = mnl + dims['l'] + sum(dims['q']))
        ind = mnl + dims['l']
        ds[:ind] += 1.0
        dz[:ind] += 1.0
        for m in dims['q']:
            ds[ind] += 1.0
            dz[ind] += 1.0
            ind += m


        # ds := H(lambda)^{-1/2} * ds and dz := H(lambda)^{-1/2} * dz.
        # 
        # This replaces the nonlinear, 'l' and 'q' components of ds and dz
        # with the updated variables in the new scaling.
        # The 's' components of ds and dz are replaced with
        #
        #     diag(lmbda_k)^{1/2} * Qs * diag(lmbda_k)^{1/2}
        #     diag(lmbda_k)^{1/2} * Qz * diag(lmbda_k)^{1/2}
         
        misc.scale2(lmbda, ds, dims, mnl, inverse = 'I')
        misc.scale2(lmbda, dz, dims, mnl, inverse = 'I')

        # sigs := ( e + step*sigs ) ./ lambda for 's' blocks.
        # sigz := ( e + step*sigz ) ./ lambda for 's' blocks.
        blas.scal(step, sigs)
        blas.scal(step, sigz)
        sigs += 1.0
        sigz += 1.0
        blas.tbsv(lmbda, sigs, n = sum(dims['s']), k = 0, ldA = 1, 
            offsetA = mnl + dims['l'] + sum(dims['q']) )
        blas.tbsv(lmbda, sigz, n = sum(dims['s']), k = 0, ldA = 1, 
            offsetA = mnl + dims['l'] + sum(dims['q']) )

        # dsk := Ls = dsk * sqrt(sigs).
        # dzk := Lz = dzk * sqrt(sigz).
        ind2, ind3 = mnl + dims['l'] + sum(dims['q']), 0
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

        blas.copy(lmbda, s, n = mnl + dims['l'] + sum(dims['q']))
        ind = mnl + dims['l'] + sum(dims['q'])
        ind2 = ind
        for m in dims['s']:
            blas.scal(0.0, s, offset = ind2)
            blas.copy(lmbda, s, offsetx = ind, offsety = ind2, n = m, 
                incy = m+1)
            ind += m
            ind2 += m*m
        misc.scale(s, W, trans = 'T')

        blas.copy(lmbda, z, n = mnl + dims['l'] + sum(dims['q']))
        ind = mnl + dims['l'] + sum(dims['q'])
        ind2 = ind
        for m in dims['s']:
            blas.scal(0.0, z, offset = ind2)
            blas.copy(lmbda, z, offsetx = ind, offsety = ind2, n = m, 
                incy = m+1)
            ind += m
            ind2 += m*m
        misc.scale(z, W, inverse = 'I')

        gap = blas.dot(lmbda, lmbda) 



def cp(F, G = None, h = None, dims = None, A = None, b = None,
    kktsolver = None, xnewcopy = None, xdot = None, xaxpy = None,
    xscal = None, ynewcopy = None, ydot = None, yaxpy = None, 
    yscal = None, **kwargs):

    """
    Solves a convex optimization problem
    
        minimize    f0(x)
        subject to  fk(x) <= 0, k = 1, ..., mnl
                    G*x   <= h
                    A*x   =  b.                      

    f = (f0, f1, ..., fmnl) is convex and twice differentiable.  The linear
    inequalities are with respect to a cone C defined as the Cartesian 
    product of N + M + 1 cones:
    
        C = C_0 x C_1 x .... x C_N x C_{N+1} x ... x C_{N+M}.

    The first cone C_0 is the nonnegative orthant of dimension ml.  The 
    next N cones are second order cones of dimension mq[0], ..., mq[N-1].
    The second order cone of dimension m is defined as
    
        { (u0, u1) in R x R^{m-1} | u0 >= ||u1||_2 }.

    The next M cones are positive semidefinite cones of order ms[0], ...,
    ms[M-1] >= 0.  


    Input arguments (basic usage).

        F is a function that handles the following arguments.

            F() returns a tuple (mnl, x0).  mnl is the number of nonlinear 
            inequality constraints.  x0 is a point in the domain of f.

            F(x) returns a tuple (f, Df).

                f is  a dense 'd' matrix of size (mnl+1,1) containing f(x).

                Df is a dense or sparse 'd' matrix of size (mnl+1, n), 
                containing the derivatives of f at x:  Df[k,:] is the 
                transpose of the gradient of fk at x.  If x is not in 
                dom f, F(x) returns None or (None, None).

            F(x, z) with z a positive 'd' matrix of size (mnl+1,1), returns
            a tuple (f, Df, H).
            
                f and Df are defined as above.
                
                H is a dense or sparse 'd' matrix of size (n,n).  The lower
                triangular part of H contains the lower triangular part of
                sum_k z[k] * Hk where Hk is the Hessian of fk at x.

                If F is called with two arguments, it can be assumed that 
                x is dom f. 

            If Df and H are returned as sparse matrices, their sparsity
            patterns must be the same for each call to F(x) or F(x,z). 

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

        It is assumed that rank(A) = p and rank([H; A; Df; G]) = n at all 
        x in dom f.

        The other arguments are normally not needed.  They make it possible
        to exploit certain types of structure, as described below.


    Output arguments.

        cp() returns a dictionary with keys 'status', 'x', 'snl', 'sl',
        'znl', 'zl', 'y', 'primal objective', 'dual objective', 'gap',
        'relative gap', 'primal infeasibility', 'dual infeasibility',
        'primal slack', 'dual slack'.

        The 'status' field has values 'optimal' or 'unknown'.
        If status is 'optimal', x, snl, sl, y, znl, zl  are approximate 
        solutions of the primal and dual optimality conditions

            f(x)[1:] + snl = 0,  G*x + sl = h,  A*x = b 
            Df(x)'*[1; znl] + G'*zl + A'*y + c = 0 
            snl >= 0,  znl >= 0,  sl >= 0,  zl >= 0
            snl'*znl + sl'* zl = 0.

        If status is 'unknown', x, snl, sl, y, znl, zl are the last
        iterates before termination.  They satisfy snl > 0, znl > 0, 
        sl > 0, zl > 0, but are not necessarily feasible.

        The values of the other fields are the values returned by cpl()
        applied to the epigraph form problem

            minimize   t 
            subjec to  f0(x) <= t
                       fk(x) <= 0, k = 1, ..., mnl
                       G*x <= h
                       A*x = b.

        Termination with status 'unknown' indicates that the algorithm 
        failed to find a solution that satisfies the specified tolerances.
        In some cases, the returned solution may be fairly accurate.  If
        the primal and dual infeasibilities, the gap, and the relative gap
        are small, then x, y, snl, sl, znl, zl are close to optimal.


    Advanced usage.

        Three mechanisms are provided to express problem structure.

        1.  The user can provide a customized routine for solving linear 
        equations (`KKT systems')

            [ sum_k zk*Hk(x)  A'  GG'   ] [ ux ]   [ bx ]
            [ A               0   0     ] [ uy ] = [ by ]
            [ GG              0   -W'*W ] [ uz ]   [ bz ]

        where GG = [ Df[1:,:]; G ], uz = (uznl, uzl), bz = (bznl, bzl). 

        z is a positive vector of length mnl+1 and x is a point in the 
        domain of f.   W is a scaling matrix, a block diagonal mapping

           W*u = ( Wnl*unl, W0*u_0, ..., W_{N+M}*u_{N+M} )

        defined as follows.

        - For the nonlinear block (Wnl):

              Wnl = diag(dnl),

          with dnl a positive vector of length mnl.

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
        called as g = kktsolver(x, z, W).  W is a dictionary that contains
        the parameters of the scaling:

        - W['dnl'] is a positive 'd' matrix of size (mnl, 1).
        - W['dnli'] is a positive 'd' matrix with the elementwise inverse 
          of W['dnl'].
        - W['d'] is a positive 'd' matrix of size (ml, 1).
        - W['di'] is a positive 'd' matrix with the elementwise inverse of
          W['d'].
        - W['beta'] is a list [ beta_0, ..., beta_{N-1} ]
        - W['v'] is a list [ v_0, ..., v_{N-1} ]
        - W['r'] is a list [ r_0, ..., r_{M-1} ]
        - W['rti'] is a list [ rti_0, ..., rti_{M-1} ], with rti_k the
          inverse of the transpose of r_k.

        The call g = kktsolver(x, z, W) should return a function g that
        solves the KKT system by g(ux, uy, uz).  On entry, ux, uy, uz 
        contain the righthand side bx, by, bz.  On exit, they contain the 
        solution, with uz scaled: (Wnl*uznl, Wl*uzl) is returned instead 
        of (uznl, uzl).

        2.  The linear operators Df*u, H*u, G*u and A*u can be specified 
        by providing Python functions instead of matrices.  This can only 
        be done in combination with 1. above, i.e., it requires the 
        kktsolver argument.
        
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

        If Df is a function, the call Df(u, v, alpha, beta, trans) should
        evaluate the matrix-vectors products

            v := alpha * Df(x) * u + beta * v if trans is 'N'
            v := alpha * Df(x)' * u + beta * v if trans is 'T'.

        If H is a function, the call H(u, v, alpha, beta) should evaluate 
        the matrix-vectors product

            v := alpha * H * u + beta * v.


        3.  Instead of using the default representation of the primal 
        variable x and the dual variable y as one-column 'd' matrices, 
        we can represent these variables and the corresponding parameters 
        c and b by arbitrary Python objects (matrices, lists, dictionaries,
        etc).  This can only be done in combination with 1. and 2. above,
        i.e., it requires a user-provided KKT solver and a function
        description of the linear mappings.   It also requires the 
        arguments xnewcopy, xdot, xscal, xaxpy, ynewcopy, ydot, yscal, 
        yaxpy.  These arguments are functions defined as follows.
   
        If X is the vector space of primal variables x, then:
        - xnewcopy(u) creates a new copy of the vector u in X.
        - xdot(u, v) returns the inner product of two vectors u and v in X.
        - xscal(alpha, u) computes u := alpha*u, where alpha is a scalar
          and u is a vector in X.
        - xaxpy(u, v, alpha = 1.0, beta = 0.0) computes v := alpha*u + v
          for a scalar alpha and two vectors u and v in X.

        If Y is the vector space of primal variables y:
        - ynewcopy(u) creates a new copy of the vector u in Y.
        - ydot(u, v) returns the inner product of two vectors u and v in Y.
        - yscal(alpha, u) computes u := alpha*u, where alpha is a scalar
          and u is a vector in Y.
        - yaxpy(u, v, alpha = 1.0, beta = 0.0) computes v := alpha*u + v
          for a scalar alpha and two vectors u and v in Y.


    Control parameters.

       The following control parameters can be modified by adding an
       entry to the dictionary options.

       options['show_progress'] True/False (default: True)
       options['maxiters'] positive integer (default: 100)
       options['refinement'] nonnegative integer (default: 1)
       options['abstol'] scalar (default: 1e-7)
       options['reltol'] scalar (default: 1e-6)
       options['feastol'] scalar (default: 1e-7).

    """

    options = kwargs.get('options',globals()['options'])
    KKTREG = options.get('kktreg',None)
    
    import math 
    from cvxopt import base, blas, misc
    from cvxopt.base import matrix, spmatrix 

    mnl, x0 = F()

    # Argument error checking depends on level of customization.
    customkkt = type(kktsolver) is not str
    operatorG = G is not None and type(G) not in (matrix, spmatrix)
    operatorA = A is not None and type(A) not in (matrix, spmatrix)
    if (operatorG or operatorA) and not customkkt:
        raise ValueError("use of function valued G, A requires a "\
            "user-provided kktsolver")
    customx = (xnewcopy != None or xdot != None or xaxpy != None or
        xscal != None)
    if customx and (not operatorG or not operatorA or not customkkt):
        raise ValueError("use of non-vector type for x requires "\
            "function valued G, A and user-provided kktsolver")
    customy = (ynewcopy != None or ydot != None or yaxpy != None or 
        yscal != None)
    if customy and (not operatorA or not customkkt):
        raise ValueError("use of non vector type for y requires "\
            "function valued A and user-provided kktsolver")

    if not customx:  
        if type(x0) is not matrix or x0.typecode != 'd' or x0.size[1] != 1:
            raise TypeError("'x0' must be a 'd' matrix with one column")
        
    if h is None: h = matrix(0.0, (0,1))
    if type(h) is not matrix or h.typecode != 'd' or h.size[1] != 1:
        raise TypeError("'h' must be a 'd' matrix with one column")
    if not dims: dims = {'l': h.size[0], 'q': [], 's': []}

    # Dimension of the product cone of the linear inequalities. with 's' 
    # components unpacked.
    cdim = dims['l'] + sum(dims['q']) + sum([ k**2 for k in dims['s'] ])
    if h.size[0] != cdim:
        raise TypeError("'h' must be a 'd' matrix of size (%d,1)" %cdim)

    if G is None:
        if customx:
            def G(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
                if trans == 'N': pass
                else: xscal(beta, y)
        else:
            G = spmatrix([], [], [], (0, x0.size[0]))
    if type(G) is matrix or type(G) is spmatrix:
        if G.typecode != 'd' or G.size != (cdim, x0.size[0]):
            raise TypeError("'G' must be a 'd' matrix with size (%d, %d)"\
                %(cdim, x0.size[0]))
        def fG(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
            misc.sgemv(G, x, y, dims, trans = trans, alpha = alpha, 
                beta = beta)
    else:
        fG = G

    if A is None:
        if customy:
            def A(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
                if trans == 'N': pass
                else: xscal(beta, y)
        else:
            A = spmatrix([], [], [], (0, x0.size[0]))
    if type(A) is matrix or type(A) is spmatrix:
        if A.typecode != 'd' or A.size[1] != x0.size[0]:
            raise TypeError("'A' must be a 'd' matrix with %d columns" \
                %x0.size[0])
        def fA(x, y, trans = 'N', alpha = 1.0, beta = 0.0):
            base.gemv(A, x, y, trans = trans, alpha = alpha, beta = beta)
    else:
        fA = A
    if not customy:
        if b is None: b = matrix(0.0, (0,1))
        if type(b) is not matrix or b.typecode != 'd' or b.size[1] != 1:
            raise TypeError("'b' must be a 'd' matrix with one column")
        if not operatorA and b.size[0] != A.size[0]:
            raise TypeError("'b' must have length %d" %A.size[0])
    if b is None and customy:  
        raise ValueEror("use of non vector type for y requires b")


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
             

    # The problem is solved by applying cpl() to the epigraph form 
    #
    #     minimize   t 
    #     subject to f0(x) - t <= 0
    #                f1(x) <= 0
    #                ...
    #                fmnl(x) <= 0
    #                G*x <= h
    #                A*x = b.
    #
    # The epigraph form variable is stored as a list [x, t].

    # Epigraph form objective c = (0, 1).
    c = [ xnewcopy(x0), 1 ] 
    xscal(0.0, c[0])

    # Nonlinear inequalities for the epigraph problem
    #
    #     f_e(x,t) = (f0(x) - t, f1(x), ..., fmnl(x)).
    #     

    def F_e(x = None, z = None):

        if x is None: 
            return mnl+1, [ x0, 0.0 ]

        else:
            if z is None:
                v = F(x[0])
                if v is None or v[0] is None: return None, None
                val = matrix(v[0], tc = 'd')
                val[0] -= x[1]
                Df = v[1]
            else:
                val, Df, H = F(x[0], z)
                val = matrix(val, tc = 'd')
                val[0] -= x[1]

            if type(Df) in (matrix, spmatrix):
                def Df_e(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):  
                    if trans == 'N':
                        base.gemv(Df, u[0], v, alpha = alpha, beta = beta,
                            trans = 'N')
                        v[0] -= alpha * u[1]
                    else:
                        base.gemv(Df, u, v[0], alpha = alpha, beta = beta,
                            trans = 'T')
                        v[1] = -alpha * u[0] + beta * v[1]
            else:
                def Df_e(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):  
                    if trans == 'N':
                        Df(u[0], v, alpha = alpha, beta = beta, 
                            trans = 'N')
                        v[0] -= alpha * u[1]
                    else:
                        Df(u, v[0], alpha = alpha, beta = beta, 
                            trans = 'T')
                        v[1] = -alpha * u[0] + beta * v[1]

            if z is None:
                return val, Df_e
            else:
                if type(H) in (matrix, spmatrix):
                    def H_e(u, v, alpha = 1.0, beta = 1.0):
                        base.symv(H, u[0], v[0], alpha = alpha, 
                            beta = beta) 
                        v[1] += beta*v[1]
                else:
                    def H_e(u, v, alpha = 1.0, beta = 1.0):
                        H(u[0], v[0], alpha = alpha, beta = beta)
                        v[1] += beta*v[1]
                return val, Df_e, H_e


    # Linear inequality constraints.
    #
    #     G_e  = [ G, 0 ]
    #

    if type(G) in (matrix, spmatrix):
        def G_e(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):
            if trans == 'N':
                misc.sgemv(G, u[0], v, dims, alpha = alpha, beta = beta) 
            else:
                misc.sgemv(G, u, v[0], dims, alpha = alpha, beta = beta, 
                    trans = 'T') 
                v[1] *= beta
    else:
        def G_e(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):
            if trans == 'N':
                G(u[0], v, alpha = alpha, beta = beta) 
            else:
                G(u, v[0], alpha = alpha, beta = beta, trans = 'T') 
                v[1] *= beta


    # Linear equality constraints.
    #
    #     A_e = [ A, 0 ]
    #

    if type(A) in (matrix, spmatrix):
        def A_e(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):
            if trans == 'N':
                base.gemv(A, u[0], v, alpha = alpha, beta = beta) 
            else:
                base.gemv(A, u, v[0], alpha = alpha, beta = beta, 
                    trans = 'T') 
                v[1] *= beta
    else:
        def A_e(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):
            if trans == 'N':
                A(u[0], v, alpha = alpha, beta = beta) 
            else:
                A(u, v[0], alpha = alpha, beta = beta, trans = 'T') 
                v[1] *= beta
 

    # kktsolver(x, z, W) returns a routine for solving equations with 
    # coefficient matrix
    #
    #         [ H             A'   [Df[1:]; G]' ]
    #     K = [ A             0    0            ]. 
    #         [ [Df[1:]; G]   0    -W'*W        ]
 
    if kktsolver is None: 
        if dims and (dims['q'] or dims['s']):  
            kktsolver = 'chol'            
        else:
            kktsolver = 'chol2'            
    if kktsolver in ('ldl', 'chol', 'chol2', 'qr'):
        if kktsolver == 'ldl':
            factor = misc.kkt_ldl(G, dims, A, mnl, kktreg = KKTREG)
        elif kktsolver == 'qr':
            factor = misc.kkt_qr(G, dims, A, mnl)
        elif kktsolver == 'chol':
            factor = misc.kkt_chol(G, dims, A, mnl)
        else: 
            factor = misc.kkt_chol2(G, dims, A, mnl)
        def kktsolver(x, z, W):
            f, Df, H = F(x, z)
            return factor(W, H, Df[1:,:])             

    ux, uz = xnewcopy(x0), matrix(0.0, (mnl + cdim, 1))
    def kktsolver_e(x, znl, W):

        We = W.copy()
        We['dnl'] = W['dnl'][1:]
        We['dnli'] = W['dnli'][1:]
        g = kktsolver(x[0], znl, We)

        f, Df = F(x[0])
        if type(Df) is matrix:
            gradf0 = Df[0,:].T
        elif type(Df) is spmatrix:        
            gradf0 = matrix(Df[0,:].T)
        else:
            gradf0 = xnewcopy(x[0])
            e0 = matrix(0.0, (mnl + 1, 1))
            e0[0] = 1.0
            Df(e0, gradf0, trans = 'T')

        def solve(x, y, z):

            # Solves 
            #
            #    [ [ H   0  ]   [ A' ]  [ Df'  G'] ] [ ux ]    [ bx ]
            #    [ [ 0   0  ]   [ 0  ]  [ -e0' 0 ] ] [    ]    [    ]
            #    [                                 ] [    ]    [    ]
            #    [ [ A   0  ]   0       0          ] [ uy ] =  [ by ].
            #    [                                 ] [    ]    [    ]
            #    [ [ Df -e0 ]   0       -W'*W      ] [ uz ]    [ bz ]
            #    [ [ G   0  ]                      ] [    ]    [    ]
            # 
            # The solution is:
            #
            #     uz[0] = -bx[1] 
            #
            #     [ ux[0]  ]          [ bx[0] + bx[1] * gradf0 ]
            #     [ uy     ] = K^-1 * [ by                     ].
            #     [ uz[1:] ]          [ bz[1:]                 ]
            #
            #     ux[1] = gradf0' * ux[0] - W['dnl'][0]**2 * uz[0] - bz[0]
            #           = gradf0' * ux[0] + W['dnl'][0]**2 * bx[1] - bz[0].
            #
            # Instead of uz we return the scaled solution W*uz.

            a = z[0]
            xcopy(x[0], ux)
            xaxpy(gradf0, ux, alpha = x[1])
            blas.copy(z, uz, offsetx = 1)
            g(ux, y, uz)
            z[0] = -x[1] * W['dnl'][0]
            blas.copy(uz, z, offsety = 1)
            xcopy(ux, x[0])
            x[1] = xdot(gradf0, x[0]) + W['dnl'][0]**2 * x[1] - a

        return solve

    def xnewcopy_e(x):
        return [ xnewcopy(x[0]), x[1] ]

    def xdot_e(x, y):
        return xdot(x[0], y[0]) + x[1]*y[1]

    def xaxpy_e(x, y, alpha = 1.0):
        xaxpy(x[0], y[0], alpha = alpha)
        y[1] += alpha*x[1]

    def xscal_e(alpha, x):
        xscal(alpha, x[0])
        x[1] *= alpha

    sol = cpl(c, F_e, G_e, h, dims, A_e, b, kktsolver_e, xnewcopy_e, 
         xdot_e, xaxpy_e, xscal_e, options = options)

    sol['x'] = sol['x'][0]
    sol['znl'], sol['snl'] = sol['znl'][1:], sol['snl'][1:]
    return sol


def gp(K, F, g, G=None, h=None, A=None, b=None, kktsolver=None, **kwargs):

    """
    Solves a geometric program

        minimize    log sum exp (F0*x+g0)
        subject to  log sum exp (Fi*x+gi) <= 0,  i=1,...,m
                    G*x <= h      
                    A*x = b

    Input arguments.

        K is a list of positive integers [K0, K1, K2, ..., Km].

        F is a sum(K)xn dense or sparse 'd' matrix with block rows F0, 
        F1, ..., Fm.  Each Fi is Kixn.

        g is a sum(K)x1 dense or sparse 'd' matrix with blocks g0, g1, 
        g2, ..., gm.  Each gi is Kix1.

        G is an mxn dense or sparse 'd' matrix.

        h is an mx1 dense 'd' matrix.

        A is a pxn dense or sparse 'd' matrix.

        b is a px1 dense 'd' matrix.

        The default values for G, h, A and b are empty matrices with 
        zero rows.


    Output arguments.

        Returns a dictionary with keys 'status', 'x', 'snl', 'sl',
        'znl', 'zl', 'y', 'primal objective', 'dual objective', 'gap',
        'relative gap', 'primal infeasibility', 'dual infeasibility',
        'primal slack', 'dual slack'.

        The 'status' field has values 'optimal' or 'unknown'.
        If status is 'optimal', x, snl, sl, y, znl, zl  are approximate 
        solutions of the primal and dual optimality conditions

            f(x)[1:] + snl = 0,  G*x + sl = h,  A*x = b 
            Df(x)'*[1; znl] + G'*zl + A'*y + c = 0 
            snl >= 0,  znl >= 0,  sl >= 0,  zl >= 0
            snl'*znl + sl'* zl = 0,

        where fk(x) = log sum exp (Fk*x + gk). 

        If status is 'unknown', x, snl, sl, y, znl, zl are the last
        iterates before termination.  They satisfy snl > 0, znl > 0, 
        sl > 0, zl > 0, but are not necessarily feasible.

        The values of the other fields are the values returned by cpl()
        applied to the epigraph form problem

            minimize   t 
            subjec to  f0(x) <= t
                       fk(x) <= 0, k = 1, ..., mnl
                       G*x <= h
                       A*x = b.

        Termination with status 'unknown' indicates that the algorithm 
        failed to find a solution that satisfies the specified tolerances.
        In some cases, the returned solution may be fairly accurate.  If
        the primal and dual infeasibilities, the gap, and the relative gap
        are small, then x, y, snl, sl, znl, zl are close to optimal.


    Control parameters.

       The following control parameters can be modified by adding an
       entry to the dictionary options.

       options['show_progress'] True/False (default: True)
       options['maxiters'] positive integer (default: 100)
       options['refinement'] nonnegative integer (default: 1)
       options['abstol'] scalar (default: 1e-7)
       options['reltol'] scalar (default: 1e-6)
       options['feastol'] scalar (default: 1e-7).
    """

    options = kwargs.get('options',globals()['options'])

    import math 
    from cvxopt import base, blas, misc
    from cvxopt.base import matrix, spmatrix 

    if type(K) is not list or [ k for k in K if type(k) is not int 
        or k <= 0 ]:
        raise TypeError("'K' must be a list of positive integers")
    mnl = len(K)-1
    l = sum(K)

    if type(F) not in (matrix, spmatrix) or F.typecode != 'd' or \
        F.size[0] != l:
        raise TypeError("'F' must be a dense or sparse 'd' matrix "\
            "with %d rows" %l)
    if type(g) is not matrix or g.typecode != 'd' or g.size != (l,1): 
        raise TypeError("'g' must be a dene 'd' matrix of "\
            "size (%d,1)" %l)
    n = F.size[1]

    if G is None: G = spmatrix([], [], [], (0,n))
    if h is None: h = matrix(0.0, (0,1))
    if type(G) not in (matrix, spmatrix) or G.typecode != 'd' or \
        G.size[1] != n:
        raise TypeError("'G' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    ml = G.size[0]
    if type(h) is not matrix or h.typecode != 'd' or h.size != (ml,1):
        raise TypeError("'h' must be a dense 'd' matrix of "\
            "size (%d,1)" %ml)
    dims = {'l': ml, 's': [], 'q': []}

    if A is None: A = spmatrix([], [], [], (0,n))
    if b is None: b = matrix(0.0, (0,1))
    if type(A) not in (matrix, spmatrix) or A.typecode != 'd' or \
        A.size[1] != n:
        raise TypeError("'A' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    p = A.size[0]
    if type(b) is not matrix or b.typecode != 'd' or b.size != (p,1): 
        raise TypeError("'b' must be a dense 'd' matrix of "\
            "size (%d,1)" %p)

    y = matrix(0.0, (l,1))
    u = matrix(0.0, (max(K),1))
    Fsc = matrix(0.0, (max(K),n))

    cs1 = [ sum(K[:i]) for i in range(mnl+1) ] 
    cs2 = [ cs1[i] + K[i] for i in range(mnl+1) ]
    ind = list(zip(range(mnl+1), cs1, cs2))

    def Fgp(x = None, z = None):

        if x is None: return mnl, matrix(0.0, (n,1))
	
        f = matrix(0.0, (mnl+1,1))
        Df = matrix(0.0, (mnl+1,n))

        # y = F*x+g
        blas.copy(g, y)
        base.gemv(F, x, y, beta=1.0)

        if z is not None: H = matrix(0.0, (n,n))

        for i, start, stop in ind:

            # yi := exp(yi) = exp(Fi*x+gi) 
            ymax = max(y[start:stop])
            y[start:stop] = base.exp(y[start:stop] - ymax)

            # fi = log sum yi = log sum exp(Fi*x+gi)
            ysum = blas.asum(y, n=stop-start, offset=start)
            f[i] = ymax + math.log(ysum)

            # yi := yi / sum(yi) = exp(Fi*x+gi) / sum(exp(Fi*x+gi))
            blas.scal(1.0/ysum, y, n=stop-start, offset=start)

            # gradfi := Fi' * yi 
            #        = Fi' * exp(Fi*x+gi) / sum(exp(Fi*x+gi))
            base.gemv(F, y, Df, trans='T', m=stop-start, incy=mnl+1,
                offsetA=start, offsetx=start, offsety=i)

            if z is not None:

                # Hi = Fi' * (diag(yi) - yi*yi') * Fi 
                #    = Fisc' * Fisc
                # where 
                # Fisc = diag(yi)^1/2 * (I - 1*yi') * Fi
                #      = diag(yi)^1/2 * (Fi - 1*gradfi')

                Fsc[:K[i], :] = F[start:stop, :] 
                for k in range(start,stop):
                   blas.axpy(Df, Fsc, n=n, alpha=-1.0, incx=mnl+1,
                       incy=Fsc.size[0], offsetx=i, offsety=k-start)
                   blas.scal(math.sqrt(y[k]), Fsc, inc=Fsc.size[0],
                       offset=k-start)

                # H += z[i]*Hi = z[i] * Fisc' * Fisc
                blas.syrk(Fsc, H, trans='T', k=stop-start, alpha=z[i],
                    beta=1.0)

        if z is None: return f, Df
        else: return f, Df, H

    return cp(Fgp, G, h, dims, A, b, kktsolver = kktsolver, options = options)
