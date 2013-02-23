# The small LP of section 10.4 (Optimization problems).  

from cvxopt import matrix
from cvxopt.modeling import variable, op, dot

x = variable()  
y = variable()  
c1 = ( 2*x+y <= 3 )  
c2 = ( x+2*y <= 3 )  
c3 = ( x >= 0 )  
c4 = ( y >= 0 )  
lp1 = op(-4*x-5*y, [c1,c2,c3,c4])  
lp1.solve()  
print("\nstatus: %s" %lp1.status) 
print("optimal value: %f"  %lp1.objective.value()[0])
print("optimal x: %f" %x.value[0])
print("optimal y: %f" %y.value[0])  
print("optimal multiplier for 1st constraint: %f" %c1.multiplier.value[0])
print("optimal multiplier for 2nd constraint: %f" %c2.multiplier.value[0])
print("optimal multiplier for 3rd constraint: %f" %c3.multiplier.value[0])
print("optimal multiplier for 4th constraint: %f\n" %c4.multiplier.value[0])

x = variable(2)  
A = matrix([[2.,1.,-1.,0.], [1.,2.,0.,-1.]])  
b = matrix([3.,3.,0.,0.])  
c = matrix([-4.,-5.])  
ineq = ( A*x <= b )  
lp2 = op(dot(c,x), ineq)  
lp2.solve()  

print("\nstatus: %s" %lp2.status)  
print("optimal value: %f"  %lp2.objective.value()[0])
print("optimal x: \n") 
print(x.value) 
print("optimal multiplier: \n") 
print(ineq.multiplier.value)
