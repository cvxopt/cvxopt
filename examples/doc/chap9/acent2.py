# The analytic centering with cone constraints example of section 9.1 
# (Problems with nonlinear objectives).

from cvxopt import matrix, log, div, spdiag 
from cvxopt import solvers  
 
def F(x = None, z = None):  
     if x is None:  return 0, matrix(0.0, (3,1))  
     if max(abs(x)) >= 1.0:  return None  
     u = 1 - x**2  
     val = -sum(log(u))  
     Df = div(2*x, u).T  
     if z is None:  return val, Df  
     H = spdiag(2 * z[0] * div(1 + u**2, u**2))  
     return val, Df, H  
 
G = matrix([ 
    [0., -1.,  0.,  0., -21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.],
    [0.,  0., -1.,  0.,   0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.],
    [0.,  0.,  0., -1.,  -5.,   2., -17.,   2.,  -6.,   8., -17.,  -7., 6.]
    ])  
h = matrix(
    [1.0, 0.0, 0.0, 0.0, 20., 10., 40., 10., 80., 10., 40., 10., 15.])  
dims = {'l': 0, 'q': [4], 's':  [3]}  
sol = solvers.cp(F, G, h, dims)  
print("\nx = \n") 
print(sol['x'])
