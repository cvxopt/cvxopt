# Poisson problem on arbitrary 2D domain with RHS f,
# Dirichlet boundary conditions, and an obstacle
# constraint.
#
# Dale Roberts <dale.o.roberts@gmail.com>

import numpy as np
import numpy.linalg as la

from matplotlib.delaunay.triangulate import Triangulation

def mesh(xs, ys, npoints):
    # randomly choose some points
    rng = np.random.RandomState(1234567890)
    rx = rng.uniform(xs[0], xs[1], size=npoints)
    ry = rng.uniform(ys[0], ys[1], size=npoints)
    # only take points in domain
    nx, ny = [], []
    for x,y in zip(rx,ry):
        if in_domain(x,y):
            nx.append(x)
            ny.append(y)
    # Delaunay triangulation
    tri = Triangulation(np.array(nx), np.array(ny))
    return tri
            
def A_e(v):
    # take vertices of element and return contribution to A
    Gi = np.matrix(np.vstack((np.ones((1,3)),v.T))).I
    G = Gi * np.matrix(np.vstack((np.zeros((1,2)),np.eye(2))))
    return la.det(np.vstack((np.ones((1,3)), v.T))) * G * G.T / 2

def b_e(v):
    # take vertices of element and return contribution to b
    vS = v.sum(axis=0)/3.0 # Centre of gravity
    return f(vS) * ((v[1,0]-v[0,0])*(v[2,1]-v[0,1])-(v[2,0]-v[0,0])*(v[1,1]-v[0,1])) / 6.0

def assemble(tri, boundary):
    # get elements and vertices from mesh
    elements = tri.triangle_nodes
    vertices = np.vstack((tri.x,tri.y)).T
    # number of vertices and elements
    N = vertices.shape[0]
    E = elements.shape[0]
    #Loop over elements and assemble LHS and RHS 
    A = np.zeros((N,N))
    b = np.zeros((N,1))
    g = np.zeros((N,1))
    for j in range(E):
        index = (elements[j,:]).tolist()
        A[np.ix_(index,index)] += A_e(vertices[index,:])
        b[index] += b_e(vertices[index,:])
    # find the 'free' vertices that we need to solve for    
    free = list(set(range(len(vertices))) - set(boundary))
    return A, b, free
   
def f(v):
    # the RHS f
    return 0.0

def g(v):
    # the obstacle constraint
    X, Y = v
    Z = max(np.sin(np.sqrt(16*X**2 + 16*Y**2)) - 0.5,0.0)
    return Z

def in_domain(x,y):
    # is a point in the domain?
    return np.sqrt(x**2 + y**2) <= 1

xs = (-1.,1.)
ys = (-1.,1.)
npoints = 1000

# generate mesh and determine boundary vertices
tri = mesh(xs, ys, npoints)
boundary = tri.hull

# Assemble problem
A, b, free  = assemble(tri, boundary)
N = A.shape[0]
G = -np.eye(N)

X, Y = tri.x, tri.y
g = np.sin(np.sqrt(16*X**2 + 16*Y**2)) - 0.5
g = np.where(g>0,g,0.0)

# convert problem to cvx datastructures

import cvxopt as cvx
import cvxopt.solvers as solvers

cvxA = cvx.sparse(cvx.matrix(A))
cvxb = cvx.matrix(-b)
cvxg = cvx.matrix(-g)
cvxG = cvx.sparse(cvx.matrix(G))

# solve unconstrained problem for 'free' vertices

u = np.zeros((N,1))
u[free] = solvers.qp(cvxA[free,free], cvxb[free])['x'][:,0]

# solve constrained problem for 'free' vertices

v = np.zeros((N,1))
v[free] = solvers.qp(cvxA[free,free], cvxb[free], cvxG[free,free], cvxg[free])['x'][:,0]

from matplotlib.pyplot import figure,show
from matplotlib.mlab import griddata

# remesh the data and solution
X, Y, Z, W = tri.x, tri.y, v.flatten(), g.flatten()
N = 40
xi = np.linspace(xs[0], xs[1], N)
yi = np.linspace(ys[0], ys[1], N)
zi = griddata(X, Y, Z, xi, yi)
wi = griddata(X, Y, W, xi, yi)

# filter solution
zi = np.where(np.isinf(zi), 0.0, zi)
zf = np.nan_to_num(zi)

wi = np.where(np.isinf(wi), 0.0, wi)
wf = np.nan_to_num(wi)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = figure()
ax = Axes3D(fig)
xim, yim = np.meshgrid(xi, yi)
ax.plot_wireframe(xim, yim, wf, rstride=1, cstride=1, color='k')
ax.plot_wireframe(xim, yim, zf, rstride=1, cstride=1, color='r')

# Save it for pgfplot

np.savetxt("obstacle2D-0.table",np.column_stack((xim.flat,yim.flat,np.array(wf).flat)),fmt="%.4f")
np.savetxt("obstacle2D-1.table",np.column_stack((xim.flat,yim.flat,np.array(zf).flat)),fmt="%.4f")
