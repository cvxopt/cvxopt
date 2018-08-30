# The norm and penalty approximation problems of section 10.5 (Examples).

from cvxopt import normal, setseed
from cvxopt.modeling import variable, op, max, sum

setseed(0)
m, n = 500, 100
A = normal(m,n)
b = normal(m)

x1 = variable(n)
prob1=op(max(abs(A*x1+b)))
prob1.solve()

x2 = variable(n)
prob2=op(sum(abs(A*x2+b)))
prob2.solve()

x3 = variable(n)
prob3=op(sum(max(0, abs(A*x3+b)-0.75, 2*abs(A*x3+b)-2.25)))
prob3.solve()

try: import pylab
except ImportError: pass
else:
    pylab.subplot(311)
    pylab.hist(list(A*x1.value + b), m//5)
    pylab.subplot(312)
    pylab.hist(list(A*x2.value + b), m//5)
    pylab.subplot(313)
    pylab.hist(list(A*x3.value + b), m//5)
    pylab.show()
