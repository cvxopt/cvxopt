# The analytic centering example at the end of chapter 4 (The LAPACK 
# interface).

from kvxopt import matrix, log, mul, div, blas, lapack, normal, uniform
from math import sqrt

def acent(A,b):
    """  
    Computes analytic center of A*x <= b with A m by n of rank n. 
    We assume that b > 0 and the feasible set is bounded.
    """

    MAXITERS = 100
    ALPHA = 0.01
    BETA = 0.5
    TOL = 1e-8

    ntdecrs = []
    m, n = A.size
    x = matrix(0.0, (n,1))
    H = matrix(0.0, (n,n))

    for iter in range(MAXITERS):
        
        # Gradient is g = A^T * (1./(b-A*x)).
        d = (b-A*x)**-1
        g = A.T * d

        # Hessian is H = A^T * diag(1./(b-A*x))^2 * A.
        Asc = mul( d[:,n*[0]], A)
        blas.syrk(Asc, H, trans='T')

        # Newton step is v = H^-1 * g.
        v = -g
        lapack.posv(H, v)

        # Directional derivative and Newton decrement.
        lam = blas.dot(g, v)
        ntdecrs += [ sqrt(-lam) ]
        print("%2d.  Newton decr. = %3.3e" %(iter,ntdecrs[-1]))
        if ntdecrs[-1] < TOL: return x, ntdecrs

        # Backtracking line search.
        y = mul(A*v, d)
        step = 1.0
        while 1-step*max(y) < 0: step *= BETA 
        while True:
            if -sum(log(1-step*y)) < ALPHA*step*lam: break
            step *= BETA
        x += step*v


# Generate an analytic centering problem  
#
#    -b1 <=  Ar*x <= b2 
#
# with random mxn Ar and random b1, b2.

m, n  = 500, 500
Ar = normal(m,n);
A = matrix([Ar, -Ar])
b = uniform(2*m,1)

x, ntdecrs = acent(A, b)  
try: 
    import pylab
except ImportError: 
    pass
else:
    pylab.semilogy(range(len(ntdecrs)), ntdecrs, 'o', 
             range(len(ntdecrs)), ntdecrs, '-')
    pylab.xlabel('Iteration number')
    pylab.ylabel('Newton decrement')
    pylab.show()
