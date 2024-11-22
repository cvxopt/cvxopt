# Custom KKT Solver using CG
#
# Dale Roberts <dale.o.roberts@gmail.com>

n = 1022 # inner nodes

## 1. setting up the matrices and vectors

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg.isolve import cg, cgs, bicg

import cvxopt as cvxopt
import cvxopt.blas as blas
import cvxopt.lapack as lapack
import cvxopt.solvers as solvers
import cvxopt.misc_solvers as misc_solvers
import cvxopt.umfpack as umfpack

from scipy.sparse import csc_matrix, lil_matrix

matrix = cvxopt.matrix
sparse = cvxopt.sparse
scale = misc_solvers.scale
syrk = blas.syrk
ormqr = lapack.ormqr
potrf = lapack.potrf
potrs = lapack.potrs
gemv = blas.gemv
scal = blas.scal
gesv = lapack.gesv


# spatial mesh points (including end points)

x = np.linspace(-1.0,1.0,n+2)

# right-hand side

f = matrix(np.zeros(x.size - 2))

# P

h = 2.0/(n+1)
A = la.toeplitz(np.array((2/h, -1/h) + (n-2)*(0,)))
P = matrix(A)
Ps = csc_matrix(P)

# G

I = sparse(matrix(np.eye(n)))
G = -I

# obstacle constraint

g = (1.0-4*x**2)*(x**2-0.5)+1.0
c = matrix(-g[1:-1])

# Custom solvers

def denseInverseTransposeW(W):
    Wd = matrix(cvxopt.spmatrix(list(W['di']),range(n),range(n)))
    return Wd

def sparseInverseTransposeW(W):
    Wd = lil_matrix((n,n))
    Wd.setdiag(W['di'])
    return csc_matrix(Wd)

def fKKT(W):
    """ KKT Solver using CG """

    # Wd = W^{-T}
    Wd = sparseInverseTransposeW(W)

    # R = W^{-1} W^{-T} 
    #               = Wd' * Wd
    R = Wd * Wd

    # K = P + R
    K = csc_matrix(Ps + R)
    
    def solve(bx, by, bz):
        # solve K x = bx - R * bz -> bx
        x, info = cg(K, bx - R * bz)
        bx[:] = matrix(x)
                
        # bz <- W^{-T}(-x - bz)
        bz[:] = matrix(Wd*(-bx-bz))

    return solve

solvers.options['abstol'] = 1e-5
solvers.options['reltol'] = 1e-5
solvers.options['feastol'] = 1e-5

wi = solvers.coneqp(P, f, G, c, kktsolver=fKKT)['x']
w  = np.zeros((n+2,))
w[1:-1] = wi[:,0].T

def pretty_print(*arrays):
    print 'Solution:'
    for i in range(0,n+2):
        for a in arrays:
            print "%+.4f" % (a[i]),
        print ""

def tikz_save(basename, *arrays):
    x = arrays[0]
    for i, a in enumerate(arrays[1:]):
        filename = "../Plots/%s-%d.table" % (basename, i)
        np.savetxt(filename, np.column_stack((x,a)), fmt="%.4f")

pretty_print(x,w)
tikz_save("cgkkt",x,v)
