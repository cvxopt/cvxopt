# The robust LP example of section 10.5 (Examples).

from cvxopt import normal, uniform  
from cvxopt.modeling import variable, dot, op, sum  
from cvxopt.blas import nrm2  
     
m, n = 500, 100  
A = normal(m,n)  
b = uniform(m)  
c = normal(n)  
     
x = variable(n)  
op(dot(c,x), A*x+sum(abs(x)) <= b).solve()  
     
x2 = variable(n)  
y = variable(n)  
op(dot(c,x2), [A*x2+sum(y) <= b, -y <= x2, x2 <= y]).solve()

print("\nDifference between two solutions %e" %nrm2(x.value - x2.value))
