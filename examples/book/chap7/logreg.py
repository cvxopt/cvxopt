# Figures 7.1, page 355.
# Logistic regression.

import pickle
from cvxopt import solvers, matrix, spdiag, log, exp, div
#solvers.options['show_progress'] = False

data = pickle.load(open("logreg.bin", 'rb'))
u, y = data['u'], data['y']

# minimize   sum_{y_k = 1} (a*uk + b) + sum log (1 + exp(a*u + b))
#
# two variables a, b. 

m = u.size[0]
A = matrix(1.0, (m,2))
A[:,0] = u
c = -matrix([sum( uk for uk, yk in zip(u,y) if yk ), sum(y) ])

# minimize  c'*x + sum log (1 + exp(A*x)) 
#
# variable x (2).

def F(x=None, z=None):
   if x is None: return 0, matrix(0.0, (2,1))
   w = exp(A*x)
   f = c.T*x + sum(log(1+w))
   grad = c + A.T * div(w, 1+w)  
   if z is None: return f, grad.T
   H = A.T * spdiag(div(w,(1+w)**2)) * A
   return f, grad.T, z[0]*H 
sol = solvers.cp(F)
a, b = sol['x'][0], sol['x'][1]

try: import pylab
except ImportError: pass
else:
    pylab.figure(facecolor='w')
    nopts = 200
    pts = -1.0 + 12.0/nopts * matrix(list(range(nopts))) 
    w = exp(a*pts + b)
    pylab.plot(u, y, 'o', pts, div(w, 1+w), '-')
    pylab.title('Logistic regression (fig. 7.1)')
    pylab.axis([-1, 11, -0.1, 1.1])
    pylab.xlabel('u')
    pylab.ylabel('Prob(y=1)')
    pylab.show()
