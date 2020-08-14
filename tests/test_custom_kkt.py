import unittest
from kvxopt import matrix, spdiag, mul, div, sqrt, normal, setseed, base, blas, lapack, solvers, sparse, spmatrix
import math

class TestCustomKKT(unittest.TestCase):

    def assertAlmostEqualLists(self,L1,L2,places=7):
        self.assertEqual(len(L1),len(L2))
        for u,v in zip(L1,L2): self.assertAlmostEqual(u,v,places)

    def test_l1(self):
        setseed(100)
        m,n = 500,250
        P = normal(m,n)
        q = normal(m,1)
        u1,st1 = l1(P,q)
        u2,st2 = l1blas(P,q)
        self.assertTrue(st1 == 'optimal')
        self.assertTrue(st2 == 'optimal')
        self.assertAlmostEqualLists(list(u1),list(u2),places=3)

    def test_l1regls(self):
        setseed(100)
        m,n = 250,500
        A = normal(m,n)
        b = normal(m,1)

        x,st = l1regls(A,b)
        self.assertTrue(st == 'optimal')
        # Check optimality conditions (list should be empty, e.g., False)
        self.assertFalse([t for t in zip(A.T*(A*x-b),x) if abs(t[1])>1e-6 and abs(t[0]) > 1.0])

def l1(P, q):

    """
    Returns the solution u of the ell-1 approximation problem

        (primal) minimize ||P*u - q||_1

        (dual)   maximize    q'*w
                 subject to  P'*w = 0
                             ||w||_infty <= 1.
    """

    m, n = P.size

    # Solve equivalent LP
    #
    #     minimize    [0; 1]' * [u; v]
    #     subject to  [P, -I; -P, -I] * [u; v] <= [q; -q]
    #
    #     maximize    -[q; -q]' * z
    #     subject to  [P', -P']*z  = 0
    #                 [-I, -I]*z + 1 = 0
    #                 z >= 0

    c = matrix(n*[0.0] + m*[1.0])
    h = matrix([q, -q])

    def Fi(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):
        if trans == 'N':
            # y := alpha * [P, -I; -P, -I] * x + beta*y
            u = P*x[:n]
            y[:m] = alpha * ( u - x[n:]) + beta*y[:m]
            y[m:] = alpha * (-u - x[n:]) + beta*y[m:]

        else:
            # y := alpha * [P', -P'; -I, -I] * x + beta*y
            y[:n] =  alpha * P.T * (x[:m] - x[m:]) + beta*y[:n]
            y[n:] = -alpha * (x[:m] + x[m:]) + beta*y[n:]


    def Fkkt(W):

        # Returns a function f(x, y, z) that solves
        #
        # [ 0  0  P'      -P'      ] [ x[:n] ]   [ bx[:n] ]
        # [ 0  0 -I       -I       ] [ x[n:] ]   [ bx[n:] ]
        # [ P -I -W1^2     0       ] [ z[:m] ] = [ bz[:m] ]
        # [-P -I  0       -W2      ] [ z[m:] ]   [ bz[m:] ]
        #
        # On entry bx, bz are stored in x, z.
        # On exit x, z contain the solution, with z scaled (W['di'] .* z is
        # returned instead of z).

        d1, d2 = W['d'][:m], W['d'][m:]
        D = 4*(d1**2 + d2**2)**-1
        A = P.T * spdiag(D) * P
        lapack.potrf(A)

        def f(x, y, z):

            x[:n] += P.T * ( mul( div(d2**2 - d1**2, d1**2 + d2**2), x[n:])
                + mul( .5*D, z[:m]-z[m:] ) )
            lapack.potrs(A, x)

            u = P*x[:n]
            x[n:] =  div( x[n:] - div(z[:m], d1**2) - div(z[m:], d2**2) +
                mul(d1**-2 - d2**-2, u), d1**-2 + d2**-2 )

            z[:m] = div(u-x[n:]-z[:m], d1)
            z[m:] = div(-u-x[n:]-z[m:], d2)

        return f


    # Initial primal and dual points from least-squares solution.

    # uls minimizes ||P*u-q||_2; rls is the LS residual.
    uls =  +q
    lapack.gels(+P, uls)
    rls = P*uls[:n] - q

    # x0 = [ uls;  1.1*abs(rls) ];   s0 = [q;-q] - [P,-I; -P,-I] * x0
    x0 = matrix( [uls[:n],  1.1*abs(rls)] )
    s0 = +h
    Fi(x0, s0, alpha=-1, beta=1)

    # z0 = [ (1+w)/2; (1-w)/2 ] where w = (.9/||rls||_inf) * rls
    # if rls is nonzero and w = 0 otherwise.
    if max(abs(rls)) > 1e-10:
        w = .9/max(abs(rls)) * rls
    else:
        w = matrix(0.0, (m,1))
    z0 = matrix([.5*(1+w), .5*(1-w)])

    dims = {'l': 2*m, 'q': [], 's': []}
    sol = solvers.conelp(c, Fi, h, dims, kktsolver = Fkkt,
        primalstart={'x': x0, 's': s0}, dualstart={'z': z0})
    return sol['x'][:n],sol['status']


def l1blas (P, q):

    """
    Returns the solution u of the ell-1 approximation problem

        (primal) minimize ||P*u - q||_1

        (dual)   maximize    q'*w
                 subject to  P'*w = 0
                             ||w||_infty <= 1.
    """

    m, n = P.size

    # Solve equivalent LP
    #
    #     minimize    [0; 1]' * [u; v]
    #     subject to  [P, -I; -P, -I] * [u; v] <= [q; -q]
    #
    #     maximize    -[q; -q]' * z
    #     subject to  [P', -P']*z  = 0
    #                 [-I, -I]*z + 1 = 0
    #                 z >= 0

    c = matrix(n*[0.0] + m*[1.0])
    h = matrix([q, -q])

    u = matrix(0.0, (m,1))
    Ps = matrix(0.0, (m,n))
    A = matrix(0.0, (n,n))

    def Fi(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):
        if trans == 'N':
            # y := alpha * [P, -I; -P, -I] * x + beta*y
            blas.gemv(P, x, u)
            y[:m] = alpha * ( u - x[n:]) + beta*y[:m]
            y[m:] = alpha * (-u - x[n:]) + beta*y[m:]

        else:
            # y := alpha * [P', -P'; -I, -I] * x + beta*y
            blas.copy(x[:m] - x[m:], u)
            blas.gemv(P, u, y, alpha = alpha, beta = beta, trans = 'T')
            y[n:] = -alpha * (x[:m] + x[m:]) + beta*y[n:]


    def Fkkt(W):

        # Returns a function f(x, y, z) that solves
        #
        # [ 0  0  P'      -P'      ] [ x[:n] ]   [ bx[:n] ]
        # [ 0  0 -I       -I       ] [ x[n:] ]   [ bx[n:] ]
        # [ P -I -D1^{-1}  0       ] [ z[:m] ] = [ bz[:m] ]
        # [-P -I  0       -D2^{-1} ] [ z[m:] ]   [ bz[m:] ]
        #
        # where D1 = diag(di[:m])^2, D2 = diag(di[m:])^2 and di = W['di'].
        #
        # On entry bx, bz are stored in x, z.
        # On exit x, z contain the solution, with z scaled (di .* z is
        # returned instead of z).

        # Factor A = 4*P'*D*P where D = d1.*d2 ./(d1+d2) and
        # d1 = d[:m].^2, d2 = d[m:].^2.

        di = W['di']
        d1, d2 = di[:m]**2, di[m:]**2
        D = div( mul(d1,d2), d1+d2 )
        Ds = spdiag(2 * sqrt(D))
        base.gemm(Ds, P, Ps)
        blas.syrk(Ps, A, trans = 'T')
        lapack.potrf(A)

        def f(x, y, z):

            # Solve for x[:n]:
            #
            #    A*x[:n] = bx[:n] + P' * ( ((D1-D2)*(D1+D2)^{-1})*bx[n:]
            #        + (2*D1*D2*(D1+D2)^{-1}) * (bz[:m] - bz[m:]) ).

            blas.copy(( mul( div(d1-d2, d1+d2), x[n:]) +
                mul( 2*D, z[:m]-z[m:] ) ), u)
            blas.gemv(P, u, x, beta = 1.0, trans = 'T')
            lapack.potrs(A, x)

            # x[n:] := (D1+D2)^{-1} * (bx[n:] - D1*bz[:m] - D2*bz[m:]
            #     + (D1-D2)*P*x[:n])

            base.gemv(P, x, u)
            x[n:] =  div( x[n:] - mul(d1, z[:m]) - mul(d2, z[m:]) +
                mul(d1-d2, u), d1+d2 )

            # z[:m] := d1[:m] .* ( P*x[:n] - x[n:] - bz[:m])
            # z[m:] := d2[m:] .* (-P*x[:n] - x[n:] - bz[m:])

            z[:m] = mul(di[:m],  u-x[n:]-z[:m])
            z[m:] = mul(di[m:], -u-x[n:]-z[m:])

        return f


    # Initial primal and dual points from least-squares solution.

    # uls minimizes ||P*u-q||_2; rls is the LS residual.
    uls =  +q
    lapack.gels(+P, uls)
    rls = P*uls[:n] - q

    # x0 = [ uls;  1.1*abs(rls) ];   s0 = [q;-q] - [P,-I; -P,-I] * x0
    x0 = matrix( [uls[:n],  1.1*abs(rls)] )
    s0 = +h
    Fi(x0, s0, alpha=-1, beta=1)

    # z0 = [ (1+w)/2; (1-w)/2 ] where w = (.9/||rls||_inf) * rls
    # if rls is nonzero and w = 0 otherwise.
    if max(abs(rls)) > 1e-10:
        w = .9/max(abs(rls)) * rls
    else:
        w = matrix(0.0, (m,1))
    z0 = matrix([.5*(1+w), .5*(1-w)])

    dims = {'l': 2*m, 'q': [], 's': []}
    sol = solvers.conelp(c, Fi, h, dims, kktsolver = Fkkt,
        primalstart={'x': x0, 's': s0}, dualstart={'z': z0})
    return sol['x'][:n],sol['status']

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

    sol = solvers.coneqp(P, q, G, h, kktsolver = Fkkt)
    return sol['x'][:n],sol['status']


if __name__ == '__main__':
    unittest.main()
