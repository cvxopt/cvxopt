# Examples of custom KKT solvers
#
# Dale Roberts <dale.o.roberts@gmail.com>

import sys

n = 126 # inner nodes 

## 1. setting up the matrices and vectors

import numpy as np
import scipy.linalg as la

import cvxopt as cvxopt
import cvxopt.blas as blas
import cvxopt.lapack as lapack
import cvxopt.solvers as solvers
import cvxopt.misc_solvers as misc_solvers
import cvxopt.umfpack as umfpack

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
P = sparse(matrix(A))

# G

I = sparse(matrix(np.eye(n)))
G = -I

# obstacle constraint

g = (1.0-4*x**2)*(x**2-0.5)+1.0
c = matrix(-g[1:-1])

# Custom solvers

def denseInverseTransposeW(W):
	# Wd = W^{-T} in packed storage.
	Wd = matrix(I)
	scale(Wd, W, trans = 'T', inverse = 'I')
	return Wd

def denseInverseTransposeW2(W):
	# Wd = W^{-T} in packed storage.
	Wd = matrix(cvxopt.spmatrix(list(W['d']),range(n),range(n))).T
	ipiv = matrix(0,(n,1))
	lapack.getrf(Wd,ipiv)
	lapack.getri(Wd, ipiv)
	return Wd

def fKKT1(W):
	""" Custom solver using Cholesky factorization """
	# Compute 
	#
	#     K = P + W^{-1} * W^{-T}
	#
	# and take the Cholesky factorization of
	#
	#     K = P + W^{-1} * W^{-T}

	# Wd = W^{-T} in packed storage.
	Wd = denseInverseTransposeW(W)

	# R = W^{-1} W^{-T} 
	#   = Wd' * Wd
	R = matrix(0.0, (n,n))
	syrk(Wd, R, trans = 'T') # R = Wd' * Wd

	# K = P + R
	K = matrix(P + R)

	# Cholesky factorization of K.
	potrf(K)
	
	def solve(bx, by, bz):
	    # bx <- bx - W^{-1} W^{-T} * bz
            #     = bx - R * bz
            gemv(R, bz, bx, alpha=-1.0, beta=1.0)

            # solve K x = bx - R * bz -> bx
            potrs(K, bx)
            
            # bz <- W^{-T}(-x - bz)
            gemv(Wd,(-bx-bz),bz)

         return solve

def fKKT2(W):
	""" KKT Solver without factorization """

	# Wd = W^{-T} in packed storage.
	Wd = denseInverseTransposeW(W)

	# R = W^{-1} W^{-T} 
	#		= Wd' * Wd
	R = matrix(0.0, (n,n))
	syrk(Wd, R, trans = 'T') # R <- Wd' * Wd

	# K = P + R
	K = matrix(P + R)
	
	def solve(bx, by, bz):
	    # bx <- bx - W^{-1} W^{-T} * bz
            #     = bx - R * bz
            gemv(R, bz, bx, trans='N', alpha=-1.0, beta=1.0)
            
            # solve K x = bx - R * bz -> bx
            gesv(K, bx)
            
            # bz <- W^{-T}(-x - bz)
            gemv(Wd,(-bx-bz),bz)

	return solve

ui = solvers.coneqp(P, f, G, c)['x']
u  = np.zeros((n+2,))
u[1:-1] = ui[:,0].T

vi = solvers.coneqp(P, f, G, c, kktsolver=fKKT1)['x']
v  = np.zeros((n+2,))
v[1:-1] = vi[:,0].T

wi = solvers.coneqp(P, f, G, c, kktsolver=fKKT2)['x']
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
		filename = "%s-%d.table" % (basename, i)
		np.savetxt(filename, np.column_stack((x,a)), fmt="%.4f")

def mm_out(*arrays):
	x = arrays[0]
	for i, a in enumerate(arrays[1:]):
		filename = "%s-%d.table" % (basename, i)
		np.savetxt(filename, np.column_stack((x,a)), fmt="%.4f")

pretty_print(x,u,v,w)
#tikz_save("KKT0",x,v)
