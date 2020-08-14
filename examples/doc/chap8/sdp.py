# The small SDP of section 8.6 (Semidefinite programming).  

from kvxopt import matrix, solvers  
c = matrix([1.,-1.,1.])  
G = [ matrix([[-7., -11., -11., 3.],  
              [ 7., -18., -18., 8.],  
              [-2.,  -8.,  -8., 1.]]) ]  
G += [ matrix([[-21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.],  
               [  0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.],  
               [ -5.,   2., -17.,   2.,  -6.,   8., -17.,  -7., 6.]]) ]  
h = [ matrix([[33., -9.], [-9., 26.]]) ]  
h += [ matrix([[14., 9., 40.], [9., 91., 10.], [40., 10., 15.]]) ]  
sol = solvers.sdp(c, Gs=G, hs=h)  
print("\nx = \n") 
print(sol['x'])
print("zs[0] = \n")
print(sol['zs'][0])
print("zs[1] =\n")
print(sol['zs'][1])
