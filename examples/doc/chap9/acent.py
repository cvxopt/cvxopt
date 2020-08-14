# The equality constrained analytical centering example of section 9.1 
# (Problems with nonlinear objectives).

from kvxopt import matrix, spmatrix, spdiag, log, normal, uniform
from kvxopt import blas, solvers

def acent(A, b):
     
    # Returns the solution of
    #
    #     minimize    -sum log(x)
    #     subject to  A*x = b
    
    m, n = A.size
    def F(x=None, z=None):
        if x is None:  return 0, matrix(1.0, (n,1))
        if min(x) <= 0.0:  return None
        f = -sum(log(x))
        Df = -(x**-1).T 
        if z is None: return matrix(f), Df
        H = spdiag(z[0] * x**-2)
        return f, Df, H
 
    return solvers.cp(F, A=A, b=b)['x']


# Randomly generate a feasible problem

m, n = 50, 500
y = normal(m,1)

# Random A with A'*y > 0.
s = uniform(n,1)
A = normal(m,n)
r = s - A.T * y
# A = A - (1/y'*y) * y*r'
blas.ger(y, r, A, alpha = 1.0/blas.dot(y,y)) 

# Random feasible x > 0.
x = uniform(n,1)
b = A*x

x = acent(A,b)
