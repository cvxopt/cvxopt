# 1D Poisson problem with a positivity constraint
# and an obstacle constraint.
#
# Dale Roberts <dale.o.roberts@gmail.com>

n = 126 # inner nodes 

## 1. setting up the matrices and vectors

import numpy as np
import scipy.linalg as la

x = np.linspace(-1.0,1.0,n+2)

I = -np.eye(n)

# unconstrained problem

h = 2.0/(n+1)
A = la.toeplitz(np.array((2/h, -1/h) + (n-2)*(0,)))
b = np.ones(x.size - 2) * 0.0 # f = 0

# positivity constraint

z =  np.zeros((n,))

# obstacle constraint

c = (1.0-4*x**2)*(x**2-0.5)+1.0

## 2. SOLVER  from cvxopt

import cvxopt as cv
import cvxopt.solvers as cvx

# cast to cvx format

cvA = cv.sparse(cv.matrix(A))
cvb = cv.matrix(b)
cvI = cv.matrix(I)
cvz = cv.matrix(z)
cvc = cv.matrix(-c[1:-1])

# solve unconstrained problem

sol = cvx.qp(cvA, cvb)
ui  = np.array(sol['x'])
u   = np.zeros((n+2,))
u[1:-1] = ui[:,0]

# solve with positivity constraint

sol1 = cvx.qp(cvA, cvb, cvI, cvz)
ui1  = np.array(sol1['x'])
u1  = np.zeros((n+2,))
u1[1:-1] = ui1[:,0]

# solve with obstacle constraint

sol2 = cvx.qp(cvA, cvb, cvI, cvc)
ui2  = np.array(sol2['x'])
u2  = np.zeros((n+2,))
u2[1:-1] = ui2[:,0]

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

pretty_print(x,c,u2)
tikz_save("obstacle1d",x,c,u2)

#from pylab import *
#plot(x,u)
#plot(x,u1)
#plot(x,u2)
#plot(x[1:-1], -np.array(c)[:,0])
##plot(x,z_n)
##plot(x,w_n)
#show()
