# The covariance selection example at the end of chapter 7 (Sparse linear
# equations).

from cvxopt import matrix, spmatrix, log, mul, blas, lapack, amd, cholmod
from pickle import load

def covsel(Y):
    """
    Returns the solution of
 
        minimize    -log det K + tr(KY)
        subject to  K_ij = 0  if (i,j) not in zip(I, J).

    Y is a symmetric sparse matrix with nonzero diagonal elements.
    I = Y.I,  J = Y.J.
    """

    cholmod.options['supernodal'] = 2

    I, J = Y.I, Y.J
    n, m = Y.size[0], len(I) 
    # non-zero positions for one-argument indexing 
    N = I + J*n         
    # position of diagonal elements
    D = [ k for k in range(m) if I[k]==J[k] ]  

    # starting point: symmetric identity with nonzero pattern I,J
    K = spmatrix(0.0, I, J) 
    K[::n+1] = 1.0

    # Kn is used in the line search
    Kn = spmatrix(0.0, I, J)

    # symbolic factorization of K 
    F = cholmod.symbolic(K)

    # Kinv will be the inverse of K
    Kinv = matrix(0.0, (n,n))

    for iters in range(100):

        # numeric factorization of K
        cholmod.numeric(K, F)
        d = cholmod.diag(F)

        # compute Kinv by solving K*X = I 
        Kinv[:] = 0.0
        Kinv[::n+1] = 1.0
        cholmod.solve(F, Kinv)
        
        # solve Newton system
        grad = 2 * (Y.V - Kinv[N])
        hess = 2 * ( mul(Kinv[I,J], Kinv[J,I]) + 
               mul(Kinv[I,I], Kinv[J,J]) )
        v = -grad
        lapack.posv(hess,v) 
                                                  
        # stopping criterion
        sqntdecr = -blas.dot(grad,v) 
        print("Newton decrement squared:%- 7.5e" %sqntdecr)
        if (sqntdecr < 1e-12):
            print("number of iterations: %d" %(iters+1))
            break

        # line search
        dx = +v
        dx[D] *= 2      
        f = -2.0*sum(log(d))      # f = -log det K
        s = 1
        for lsiter in range(50):
            Kn.V = K.V + s*dx
            try: 
                cholmod.numeric(Kn, F)
            except ArithmeticError: 
                s *= 0.5
            else:
                d = cholmod.diag(F)
                fn = -2.0 * sum(log(d)) + 2*s*blas.dot(v,Y.V)
                if (fn < f - 0.01*s*sqntdecr): break
                else: s *= 0.5

        K.V = Kn.V

    return K

Y = load(open("covsel.bin","rb"))
print("%d rows/columns, %d nonzeros\n" %(Y.size[0], len(Y)))
covsel(Y)
