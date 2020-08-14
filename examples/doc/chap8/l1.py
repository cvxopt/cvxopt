# The 1-norm approximation example of section 8.7 (Exploiting structure).  

from kvxopt import blas, lapack, solvers
from kvxopt import matrix, spdiag, mul, div, setseed, normal
from math import sqrt

def l1(P, q):

    """
    Returns the solution u, w of the ell-1 approximation problem

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
        A = P.T * spdiag(4*D) * P
        lapack.potrf(A)

        def f(x, y, z):

            # Solve for x[:n]:
            #
            #    A*x[:n] = bx[:n] + P' * ( ((D1-D2)*(D1+D2)^{-1})*bx[n:]
            #        + (2*D1*D2*(D1+D2)^{-1}) * (bz[:m] - bz[m:]) ).

            x[:n] += P.T * ( mul( div(d1-d2, d1+d2), x[n:]) + 
                mul( 2*D, z[:m]-z[m:] ) )
            lapack.potrs(A, x)

            # x[n:] := (D1+D2)^{-1} * (bx[n:] - D1*bz[:m] - D2*bz[m:]
            #     + (D1-D2)*P*x[:n])

            u = P*x[:n]
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
    return sol['x'][:n],  sol['z'][m:] - sol['z'][:m]    

setseed()
m, n = 1000, 100
P, q = normal(m,n), normal(m,1)
x, y = l1(P,q)
