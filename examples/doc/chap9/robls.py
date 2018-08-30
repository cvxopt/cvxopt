# The robust least-squares example of section 9.1 (Problems with nonlinear
# objectives). 

from math import sqrt, ceil, floor
from cvxopt import solvers, blas, lapack
from cvxopt import matrix, spmatrix, spdiag, sqrt, mul, div, setseed, normal

def robls(A, b, rho): 

    # Minimize  sum_k sqrt(rho + (A*x-b)_k^2).

    m, n = A.size
    def F(x=None, z=None):
        if x is None: return 0, matrix(0.0, (n,1))
        y = A*x-b
        w = sqrt(rho + y**2)
        f = sum(w)
        Df = div(y, w).T * A 
        if z is None: return f, Df 
        H = A.T * spdiag(z[0]*rho*(w**-3)) * A
        return f, Df, H

    return solvers.cp(F)['x']


setseed()
m, n  = 500, 100
A = normal(m,n)
b = normal(m,1)
xh = robls(A,b,0.1)

try: import pylab
except ImportError: pass
else:

    # Least-squares solution.
    pylab.subplot(211)
    xls = +b
    lapack.gels(+A,xls)
    rls =  A*xls[:n] - b
    pylab.hist(list(rls), m//5)
    pylab.title('Least-squares solution')
    pylab.xlabel('Residual')
    mr = ceil(max(rls))
    pylab.axis([-mr, mr, 0, 25])
 
    # Robust least-squares solution with rho = 0.01.
    pylab.subplot(212)
    rh =  A*xh - b
    pylab.hist(list(rh), m//5)
    mr = ceil(max(rh))
    pylab.title('Robust least-squares solution')
    pylab.xlabel('Residual')
    pylab.axis([-mr, mr, 0, 50])

    pylab.show()
