# The 1-norm support vector classifier of section 10.5 (Examples).

from cvxopt import normal, setseed
from cvxopt.modeling import variable, op, max, sum
from cvxopt.blas import nrm2

m, n = 500, 100
A = normal(m,n) 

x = variable(A.size[1],'x')  
u = variable(A.size[0],'u')  
op(sum(abs(x)) + sum(u), [A*x >= 1-u, u >= 0]).solve()

x2 = variable(A.size[1],'x')  
op(sum(abs(x2)) + sum(max(0, 1 - A*x2))).solve() 

print("\nDifference between two solutions: %e" %nrm2(x.value - x2.value))
