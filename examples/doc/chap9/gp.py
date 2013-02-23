# The small GP of section 9.3 (Geometric programming).

from cvxopt import matrix, log, exp, solvers  
 
Aflr  = 1000.0  
Awall = 100.0  
alpha = 0.5  
beta  = 2.0  
gamma = 0.5  
delta = 2.0  
 
F = matrix( [[-1., 1., 1., 0., -1.,  1.,  0.,  0.],  
             [-1., 1., 0., 1.,  1., -1.,  1., -1.],  
             [-1., 0., 1., 1.,  0.,  0., -1.,  1.]])  
g = log( matrix( [1.0, 2/Awall, 2/Awall, 1/Aflr, alpha, 1/beta, gamma, 
    1/delta]) )  
K = [1, 2, 1, 1, 1, 1, 1]  
h, w, d = exp( solvers.gp(K, F, g)['x'] )
print("\n h = %f,  w = %f, d = %f.\n" %(h,w,d))   
