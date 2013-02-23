# The example of section 9.4 (Exploiting structure).. 

from cvxopt import matrix, spdiag, mul, div, log, setseed, uniform, normal 
from cvxopt import blas, lapack, solvers 


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
        r = - b
        blas.gemv(A, x, r, beta = -1.0)
        w = x**2
        f = 0.5 * blas.nrm2(r)**2  - sum(log(1-w))
        gradf = div(x, 1.0 - w)
        blas.gemv(A, r, gradf, trans = 'T', beta = 2.0)
        if z is None:
            return f, gradf.T
        else:
            def Hf(u, v, alpha = 1.0, beta = 0.0):
               """
                   v := alpha * (A'*A*u + 2*((1+w)./(1-w)).*u + beta *v
               """
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

m, n = 200, 2000
setseed()
A = normal(m,n)
x = uniform(n,1)
b = A*x
x = l2ac(A, b)
