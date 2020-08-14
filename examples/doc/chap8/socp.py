# The small SOCP of section 8.5 (Second-order cone programming).  

from kvxopt import matrix, solvers 
c = matrix([-2., 1., 5.])  
G = [ matrix( [[12., 13., 12.], [6., -3., -12.], [-5., -5., 6.]] ) ]  
G += [ matrix( [[3., 3., -1., 1.], [-6., -6., -9., 19.], [10., -2., -2., -3.]] ) ]  
h = [ matrix( [-12., -3., -2.] ),  matrix( [27., 0., 3., -42.] ) ]  
sol = solvers.socp(c, Gq = G, hq = h)  
print("\nx = \n") 
print(sol['x'])
print("zq[0] = \n")
print(sol['zq'][0])
print("zq[1] = \n") 
print(sol['zq'][1])
