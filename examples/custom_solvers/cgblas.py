# Sparse CG solver and custom BLAS
#
# Dale Roberts <dale.o.roberts@gmail.com>

# inner nodes 
n = 1022

## 1. setting up the matrices and vectors

import numpy as np
import scipy.linalg as la

import cvxopt.blas as blas
import cvxopt.solvers as solvers

from cvxopt import spmatrix, matrix

# spatial mesh points (including end points)

x = np.linspace(-1.0,1.0,n+2)

# right-hand side

f = matrix(0.0, (n,1))

# P

h = 2.0/(n+1)

A = spmatrix(2.0/h, range(n), range(n))
A[1::(n+1)] = -1.0/h
A[n::(n+1)] = -1.0/h

def fP(x, y, alpha = 1.0, beta = 0.0):
    """ P is tridiagonal matrix T """
    y[:] = alpha * (A * x) + beta * y

# G

def G(x, y, alpha=1.0, beta=1.0, trans='T'):
    """ Inequality constraint: G = -I """
    y[:] = alpha * -1.0 * x + beta * y

# obstacle constraint (has to be cvxopt.matrix)

g = (1.0-4*x**2)*(x**2-0.5)+1.0
c = matrix(-g[1:-1])

# abstract matrix and vector operations

def scal(alpha, x):
    """ Scale a vector by a constant """
    x[:] = alpha * x

def newcopy(x):
    """ New copy of vector """
    return matrix(x)

def dot(x, y):
    """ Dot product """
    return blas.dot(x,y)

def axpy(x, y, alpha=1.0):
    """ Constant times a vector plus a vector """
    y[:] = alpha*x + y

# Custom solvers

ITERS = []
def cg(A, b, x0=None, eps=1e-8):
    """ Conjugate gradient solve Ax = b """
    if not x0: x0 = matrix(0.0, b.size)
    r = b - A * x0
    w = -r
    z = A * w
    a = (r.T * w) / (w.T * z)
    x = x0 + a * w
    for i in range(A.size[0]):
        r = r - a * z
        if blas.nrm2(r) < eps:
            break
        B = (r.T * z) / (w.T * z)
        w = -r + B*w
        z = A * w
        a = (r.T * w) / (w.T * z)
        x = x + a * w
    ITERS.append(i)
    return x

def fKKT(W):
    # Wit = W^{-T} 
    Wit = spmatrix(W['di'], range(n), range(n))
    # R = W^{-1} W^{-T} 
    R = Wit.T * Wit
    K = A + R
    def solve(bx, by, bz):
        x = cg(K, bx - R * bz)
        bx[:] = x
        bz[:] = matrix(Wit * (-x-bz))
    return solve

dims = {'l': n, 'q': [], 's': []}

def run():
    vi = solvers.coneqp(fP, f, G, h=c, dims=dims, kktsolver=fKKT,
            xnewcopy=newcopy, xdot=dot, xaxpy=axpy, xscal=scal)['x']

    v  = np.zeros((n+2,))
    v[1:-1] = vi[:,0].T

    print 'CG iters:', ITERS
    pretty_print(x,v)

def pretty_print(*arrays):
    print 'Solution:'
    for i in range(0,n+2):
        for a in arrays:
            print "%+.4f" % (a[i]),
        print ""

def tikz_save(basename, *arrays):
    x = arrays[0]
    for i, a in enumerate(arrays[1:]):
        filename = "%s-%d.table" % (basename, i)
        np.savetxt(filename, np.column_stack((x,a)), fmt="%.4f")

run()
#tikz_save("cg",x,v)
