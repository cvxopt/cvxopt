from cvxopt import matrix, spmatrix, normal
import random 

def sp_rand(m,n,a):

     ''' 
     Generates an mxn sparse 'd' matrix with round(a*m*n) nonzeros.
     '''
  
     if m == 0 or n == 0: return spmatrix([], [], [], (m,n))
     nnz = min(max(0, int(round(a*m*n))), m*n)
     nz = matrix(random.sample(range(m*n), nnz), tc='i')
     J = matrix([k//m for k in nz])
     return spmatrix(normal(nnz,1), nz%m, J, (m,n))
