# The SDP example of section 8.7 (Exploiting structure).

from kvxopt import blas, lapack, solvers, matrix, normal

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

    def Fs(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):
        """
            y := alpha*(-diag(x)) + beta*y.   
        """
        if trans=='N':
            # x is a vector; y is a matrix.
            blas.scal(beta, y)
            blas.axpy(x, y, alpha = -alpha, incy = n+1)

        else:   
            # x is a matrix; y is a vector.
            blas.scal(beta, y)
            blas.axpy(x, y, alpha = -alpha, incx = n+1)
	 

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
        blas.trmm(tx, a, side='L')

        # x := alpha*(a*r' + r*a') 
        blas.syr2k(r, a, tx, trans = 'T', alpha = alpha)
        x[:] = tx[:]
       

    def Fkkt(W):
       
        rti = W['rti'][0]

        # t = rti*rti' as a nonsymmetric matrix.
        t = matrix(0.0, (n,n))
        blas.gemm(rti, rti, t, transB = 'T') 

        # Cholesky factorization of tsq = t.*t.
        tsq = t**2
        lapack.potrf(tsq)

        def f(x, y, z):
            """
            Solve
                          -diag(z)                           = bx
                -diag(x) - inv(rti*rti') * z * inv(rti*rti') = bs

            On entry, x and z contain bx and bs.  
            On exit, they contain the solution, with z scaled
            (inv(rti)'*z*inv(rti) is returned instead of z).

            We first solve 

                ((rti*rti') .* (rti*rti')) * x = bx - diag(t*bs*t) 

            and take z  = -rti' * (diag(x) + bs) * rti.
            """

            # tbst := t * zs * t = t * bs * t
            tbst = matrix(z, (n,n))
            cngrnc(t, tbst) 

            # x := x - diag(tbst) = bx - diag(rti*rti' * bs * rti*rti')
            x -= tbst[::n+1]

            # x := (t.*t)^{-1} * x = (t.*t)^{-1} * (bx - diag(t*bs*t))
            lapack.potrs(tsq, x)

            # z := z + diag(x) = bs + diag(x)
            z[::n+1] += x

            # z := -rti' * z * rti = -rti' * (diag(x) + bs) * rti 
            cngrnc(rti, z, alpha = -1.0)

        return f

    c = matrix(1.0, (n,1))

    # Initial feasible x:  x = 1.0 - min(lambda(w)).
    lmbda = matrix(0.0, (n,1))
    lapack.syevx(+w, lmbda, range='I', il=1, iu=1)
    x0 = matrix(-lmbda[0]+1.0, (n,1)) 
    s0 = +w
    s0[::n+1] += x0

    # Initial feasible z is identity.
    z0 = matrix(0.0, (n,n))
    z0[::n+1] = 1.0

    dims = {'l': 0, 'q': [], 's': [n]}
    sol = solvers.conelp(c, Fs, w[:], dims, kktsolver = Fkkt,
        primalstart = {'x': x0, 's': s0[:]}, dualstart = {'z': z0[:]})
    return sol['x'], matrix(sol['z'], (n,n))

n = 100
w = normal(n,n)
x, z = mcsdp(w)
