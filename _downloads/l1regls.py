from cvxopt import matrix, spdiag, mul, div, sqrt, normal, setseed
from cvxopt import blas, lapack, solvers, sparse, spmatrix
import math

try:
    import mosek
    import sys
    __MOSEK = True
except: __MOSEK = False

if __MOSEK:

    def l1regls_mosek(A, b):
        """

        Returns the solution of l1-norm regularized least-squares problem

            minimize    || A*x - b ||_2^2  + e'*u

            subject to  -u <= x <= u

        """

        m, n = A.size

        env  = mosek.Env()
        task = env.Task(0,0)
        task.set_Stream(mosek.streamtype.log, lambda x: sys.stdout.write(x))

        task.appendvars( 2*n)            # number of variables
        task.appendcons( 2*n)            # number of constraints

        # input quadratic objective
        Q = matrix(0.0, (n,n)) 
        blas.syrk(A, Q, alpha = 2.0, trans='T')

        I = []
        for i in range(n):
            I.extend(range(i,n))

        J = []
        for i in range(n):
            J.extend((n-i)*[i])

        task.putqobj(I, J, list(Q[matrix(I) + matrix(J)*n]))
        task.putclist(range(2*n), list(-2*A.T*b) + n*[1.0])  # setup linear objective

        # input constraint matrix row by row
        for i in range(n):
            task.putarow(   i, [i, n+i], [1.0, -1.0])
            task.putarow( n+i, [i, n+i], [1.0,  1.0])

        # setup bounds on constraints
        task.putboundslice(mosek.accmode.con,
                           0, n, n*[mosek.boundkey.up], n*[0.0], n*[0.0])
        task.putboundslice(mosek.accmode.con,
                           n, 2*n, n*[mosek.boundkey.lo], n*[0.0], n*[0.0])

        # setup variable bounds
        task.putboundslice(mosek.accmode.var,
                           0, 2*n, 2*n*[mosek.boundkey.fr], 2*n*[0.0], 2*n*[0.0])

        # optimize the task
        task.putobjsense(mosek.objsense.minimize)
        task.optimize()
        task.solutionsummary(mosek.streamtype.log)
        x = n*[0.0]
        task.getsolutionslice(mosek.soltype.itr, mosek.solitem.xx, 0, n, x)

        return matrix(x)

    def l1regls_mosek2(A, b):
        """

        Returns the solution of l1-norm regularized least-squares problem

            minimize     w'*w + e'*u

            subject to  -u <= x <= u

                         A*x - w = b

        """

        m, n = A.size

        env  = mosek.Env()
        task = env.Task(0,0)
        task.set_Stream(mosek.streamtype.log, lambda x: sys.stdout.write(x))

        task.appendvars(2*n + m)     # number of variables
        task.appendcons(2*n + m)     # number of constraints

        # input quadratic objective
        task.putqobj(range(2*n,2*n+m), range(2*n,2*n+m), m*[2.0])

        task.putclist(range(2*n+m), n*[0.0] + n*[1.0] + m*[0.0])  # setup linear objective

        # input constraint matrix row by row
        for i in range(n):
            task.putarow(   i, [i, n+i], [1.0, -1.0])
            task.putarow( n+i, [i, n+i], [1.0,  1.0])

        for i in range(m):
            task.putarow( 2*n+i, range(n) + [2*n+i], list(A[i,:]) + [-1.0])

        # setup bounds on constraints
        task.putboundslice(mosek.accmode.con,
                           0, n, n*[mosek.boundkey.up], n*[0.0], n*[0.0])
        task.putboundslice(mosek.accmode.con,
                           n, 2*n, n*[mosek.boundkey.lo], n*[0.0], n*[0.0])
        task.putboundslice(mosek.accmode.con,
                           2*n, 2*n+m, m*[mosek.boundkey.fx], list(b), list(b))

        # setup variable bounds
        task.putboundslice(mosek.accmode.var, 0, 2*n+m, (2*n+m)*[mosek.boundkey.fr], 
                           (2*n+m)*[0.0], (2*n+m)*[0.0])

        # optimize the task
        task.putobjsense(mosek.objsense.minimize)
        task.optimize()
        task.solutionsummary(mosek.streamtype.log)
        x = n*[0.0]
        task.getsolutionslice(mosek.soltype.itr, mosek.solitem.xx, 0, n, x)

        return matrix(x)

def l1regls(A, b):
    """
    
    Returns the solution of l1-norm regularized least-squares problem
  
        minimize || A*x - b ||_2^2  + || x ||_1.

    """

    m, n = A.size
    q = matrix(1.0, (2*n,1))
    q[:n] = -2.0 * A.T * b

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
    # where D1 = W['di'][:n]**2, D2 = W['di'][:n]**2.
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
        ds = math.sqrt(2.0) * div( mul( W['di'][:n], W['di'][n:]), 
            sqrt(d1+d2) )
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

