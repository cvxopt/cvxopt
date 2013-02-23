"""
CVXOPT interface for MOSEK 5.0
"""

# Copyright 2004-2009 J. Dahl and L. Vandenberghe.
# 
# This file is part of CVXOPT version 1.1.2
#
# CVXOPT is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# CVXOPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from cvxopt.base import matrix, spmatrix, sparse

import pymosek as msk
from mosekarr import array, zeros, Float

# Define a stream printer to grab output from MOSEK 
def streamprinter(text): print text,
    
inf = 0.0  # numeric value doesn't matter

options = {}

def lp(c, G, h, A=None, b=None):
    """
    Solves a pair of primal and dual LPs 

        minimize    c'*x             maximize    -h'*z - b'*y 
        subject to  G*x + s = h      subject to  G'*z + A'*y + c = 0
                    A*x = b                      z >= 0.
                    s >= 0
                    
    using MOSEK 5.0.

    (solsta, x, z, y) = lp(c, G, h, A=None, b=None).

    Input arguments 

        G is m x n, h is m x 1, A is p x n, b is p x 1.  G and A must be 
        dense or sparse 'd' matrices.   h and b are dense 'd' matrices 
        with one column.  The default values for A and b are empty 
        matrices with zero rows.


    Return values

        solsta   the solution status.
                 If solsta is solsta.optimal, then (x, y, z) contains the 
                     primal-dual solution.
                 If solsta is solsta.prim_infeas_cer, then (x, y, z) is a 
                     certificate of primal infeasibility.
                 If solsta is solsta.dual_infeas_cer, then (x, y, z) is a 
                     certificate of dual infeasibility.
                 If solsta is solsta.unknown, then (x, y, z) are all None.

                 Other return values for solsta include:  
                     solsta.dual_feas  
                     solsta.near_dual_feas
                     solsta.near_optimal
                     solsta.near_prim_and_dual_feas
                     solsta.near_prim_feas
                     solsta.prim_and_dual_feas
                     solsta.prim_feas
                 in which case the (x,y,z) value may not be well-defined,
                 c.f., the MOSEK documentation.
        
        x, y, z  the primal-dual solution.                    

    Options are passed to MOSEK solvers via the mosek.options dictionary. 
    For example, the following turns off output from the MOSEK solvers
    
    >>> mosek.options = {iparam.log:0} 
    
    see chapter 14 of the MOSEK Python API manual.                    
    """
    if type(c) is not matrix or c.typecode != 'd' or c.size[1] != 1: 
        raise TypeError, "'c' must be a dense column matrix"
    n = c.size[0]
    if n < 1: raise ValueError, "number of variables must be at least 1"

    if (type(G) is not matrix and type(G) is not spmatrix) or \
        G.typecode != 'd' or G.size[1] != n:
        raise TypeError, "'G' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n 
    m = G.size[0]
    if m is 0: raise ValueError, "m cannot be 0"

    if type(h) is not matrix or h.typecode != 'd' or h.size != (m,1):
        raise TypeError, "'h' must be a 'd' matrix of size (%d,1)" %m

    if A is None:  A = spmatrix([], [], [], (0,n), 'd')
    if (type(A) is not matrix and type(A) is not spmatrix) or \
        A.typecode != 'd' or A.size[1] != n:
        raise TypeError, "'A' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n
    p = A.size[0]
    if b is None: b = matrix(0.0, (0,1))
    if type(b) is not matrix or b.typecode != 'd' or b.size != (p,1): 
        raise TypeError, "'b' must be a dense matrix of size (%d,1)" %p
 
    c   = array(c)        

    bkc = m*[ msk.boundkey.up ] + p*[ msk.boundkey.fx ]
    blc = m*[ -inf ] + [ bi for bi in b ]
    buc = matrix([h, b])

    bkx = n*[msk.boundkey.fr] 
    blx = n*[ -inf ] 
    bux = n*[ +inf ]

    colptr, asub, acof = sparse([G,A]).CCS
    aptrb, aptre = colptr[:-1], colptr[1:]

    mskhandle = msk.mosek ()
    env = mskhandle.Env ()
    env.set_Stream (msk.streamtype.log, streamprinter)
        
    env.init () 
    task = env.Task(0,0) 
    task.set_Stream (msk.streamtype.log, streamprinter) 

    # set MOSEK options 
    for (param, val) in options.items():
        if str(param)[:6] == "iparam":
            task.putintparam(param, val)
        elif str(param)[:6] == "dparam":
            task.putdouparam(param, val)
        elif str(param)[:6] == "sparam":
            task.putstrparam(param, val)
        else:
            raise ValueError, "invalid MOSEK parameter: "+str(param)

    task.inputdata (m+p, # number of constraints
                    n,   # number of variables
                    array(c), # linear objective coefficients  
                    0.0, # objective fixed value  
                    array(aptrb), 
                    array(aptre), 
                    array(asub),
                    array(acof), 
                    array(bkc),
                    array(blc),
                    array(buc), 
                    array(bkx),
                    array(blx),
                    array(bux)) 

    task.putobjsense(msk.objsense.minimize)

    task.optimize()

    task.solutionsummary (msk.streamtype.msg); 

    prosta, solsta = list(), list()
    task.getsolutionstatus(msk.soltype.bas, prosta, solsta)

    x, z = zeros(n, Float), zeros(m, Float)
    task.getsolutionslice(msk.soltype.bas, msk.solitem.xx, 0, n, x) 
    task.getsolutionslice(msk.soltype.bas, msk.solitem.suc, 0, m, z) 
    x, z = matrix(x), matrix(z)
    
    if p is not 0:
        yu, yl = zeros(p, Float), zeros(p, Float)
        task.getsolutionslice(msk.soltype.bas, msk.solitem.suc, m, m+p, yu) 
        task.getsolutionslice(msk.soltype.bas, msk.solitem.slc, m, m+p, yl) 
        y = matrix(yu) - matrix(yl)
    else:
        y = matrix(0.0, (0,1))

    if (solsta[0] is msk.solsta.unknown):
        return (solsta[0], None, None, None)
    else:
        return (solsta[0], x, z, y)


def conelp(c, G, h, dims = None):
    """
    Solves a pair of primal and dual SOCPs

        minimize    c'*x             
        subject to  Gl*x + sl = hl      
                    Gq[k]*x + sq[k] = hq[k],  k = 0, ..., N-1
                    sl >= 0,  
                    sq[k] >= 0, k = 0, ..., N-1

        maximize    -hl'*zl - sum_k hq[k]'*zq[k] 
        subject to  Gl'*zl + sum_k Gq[k]'*zq[k] + c = 0
                    zl >= 0,  zq[k] >= 0, k = 0, ..., N-1.
                    
    using MOSEK 5.0.

    (solsta, x, zl, zq) = conelp(c, G, h, dims = None)

    The formats of G and h are identical to that used in the
    solvers.conelp method, except that only carthesian and quadratic cones 
    are allow  (dims['s'] must be zero, if defined).
 
    Return values

        solsta   the solution status.
                 If solsta is solsta.optimal,
                   then (x, zl, zq) contains the primal-dual solution.
                 If solsta is solsta.prim_infeas_cer,
                   then (x, zl, zq) is a certificate of primal infeasibility.
                 If solsta is solsta.dual_infeas_cer,
                   then (x, zl, zq) is a certificate of dual infeasibility.
                 If solsta is solsta.unknown,
                   then (x, zl, zq) are all None

                 Other return values for solsta include:  
                   solsta.dual_feas  
                   solsta.near_dual_feas
                   solsta.near_optimal
                   solsta.near_prim_and_dual_feas
                   solsta.near_prim_feas
                   solsta.prim_and_dual_feas
                   solsta.prim_feas
                 in which case the (x,y,z) value may not be well-defined,
                 c.f., the MOSEK documentation.
        
        x, zl, zq  the primal-dual solution.


    Options are passed to MOSEK solvers via the mosek.options 
    dictionary, e.g., the following turns off output from 
    the MOSEK solvers
    
    >>> mosek.options = {iparam.log:0} 
    
    see chapter 14 of the MOSEK Python API manual.                    
    """

    if dims is None: 
        (solsta, x, y, z) = lp(c, G, h)
        return (solsta, x, z, None)

    try:
        if len(dims['s']) > 0: raise ValueError, "dims['s'] must be zero"
    except:
        pass

    N, n = G.size
    ml, mq = dims['l'], dims['q']
    cdim = ml + sum(mq)
    if cdim is 0: raise ValueError, "ml+mq cannot be 0"

    # Data for kth 'q' constraint are found in rows indq[k]:indq[k+1] of G.
    indq = [ dims['l'] ]  
    for k in dims['q']:  indq = indq + [ indq[-1] + k ] 

    if type(h) is not matrix or h.typecode != 'd' or h.size[1] != 1:
        raise TypeError, "'h' must be a 'd' matrix with 1 column" 
    if type(G) is matrix or type(G) is spmatrix:
        if G.typecode != 'd' or G.size[0] != cdim:
            raise TypeError, "'G' must be a 'd' matrix with %d rows " %cdim
        if h.size[0] != cdim:
            raise TypeError, "'h' must have %d rows" %cdim 
    else: 
        raise TypeError, "'G' must be a matrix"

    if min(dims['q'])<1: raise TypeError, \
            "dimensions of quadratic cones must be positive"

    bkc = n*[ msk.boundkey.fx ] 
    blc = array(-c)
    buc = array(-c)

    bkx = ml*[ msk.boundkey.lo ] + sum(mq)*[ msk.boundkey.fr ]
    blx = ml*[ 0.0 ] + sum(mq)*[ -inf ]
    bux = N*[ +inf ] 

    c   = -h        
    
    colptr, asub, acof = sparse([G.T]).CCS
    aptrb, aptre = colptr[:-1], colptr[1:]

    mskhandle = msk.mosek ()
    env = mskhandle.Env ()
    env.set_Stream (msk.streamtype.log, streamprinter)
        
    env.init () 
    task = env.Task(0,0) 
    task.set_Stream (msk.streamtype.log, streamprinter) 

    # set MOSEK options 
    for (param, val) in options.items():
        if str(param)[:6] == "iparam":
            task.putintparam(param, val)
        elif str(param)[:6] == "dparam":
            task.putdouparam(param, val)
        elif str(param)[:6] == "sparam":
            task.putstrparam(param, val)
        else:
            raise ValueError, "invalid MOSEK parameter: "+str(param)

    task.inputdata (n,   # number of constraints
                    N,   # number of variables
                    array(c), # linear objective coefficients  
                    0.0, # objective fixed value  
                    array(aptrb), 
                    array(aptre), 
                    array(asub),
                    array(acof), 
                    array(bkc),
                    array(blc),
                    array(buc), 
                    array(bkx),
                    array(blx),
                    array(bux)) 

    task.putobjsense(msk.objsense.maximize)

    for k in xrange(len(mq)):
        task.appendcone(msk.conetype.quad, 0.0, 
                        array(range(ml+sum(mq[:k]),ml+sum(mq[:k+1]))))
    task.optimize()

    task.solutionsummary (msk.streamtype.msg); 

    prosta, solsta = list(), list()
    task.getsolutionstatus(msk.soltype.itr, prosta, solsta)

    xu, xl, zq = zeros(n, Float), zeros(n, Float), zeros(sum(mq), Float)
    task.getsolutionslice(msk.soltype.itr, msk.solitem.slc, 0, n, xl) 
    task.getsolutionslice(msk.soltype.itr, msk.solitem.suc, 0, n, xu) 
    task.getsolutionslice(msk.soltype.itr, msk.solitem.xx, ml, N, zq) 
    x = matrix(xu-xl)

    if ml:
        zl = zeros(ml, Float)
        task.getsolutionslice(msk.soltype.itr, msk.solitem.xx, 0, ml, zl) 
        zl = matrix(zl)
    else:
        zl = matrix(0.0, (0,1))

    if (solsta[0] is msk.solsta.unknown):
        return (solsta[0], None, None, None)
    else:
        return (solsta[0], x, zl, zq)
    



def socp(c, Gl = None, hl = None, Gq = None, hq = None):
    """
    Solves a pair of primal and dual SOCPs

        minimize    c'*x             
        subject to  Gl*x + sl = hl      
                    Gq[k]*x + sq[k] = hq[k],  k = 0, ..., N-1
                    sl >= 0,  
                    sq[k] >= 0, k = 0, ..., N-1

        maximize    -hl'*zl - sum_k hq[k]'*zq[k] 
        subject to  Gl'*zl + sum_k Gq[k]'*zq[k] + c = 0
                    zl >= 0,  zq[k] >= 0, k = 0, ..., N-1.
                    
    using MOSEK 5.0.

    (solsta, x, zl, zq) = socp(c, Gl = None, hl = None, Gq = None, hq = None)

    Return values

        solsta   the solution status.
                 If solsta is solsta.optimal,
                   then (x, zl, zq) contains the primal-dual solution.
                 If solsta is solsta.prim_infeas_cer,
                   then (x, zl, zq) is a certificate of primal infeasibility.
                 If solsta is solsta.dual_infeas_cer,
                   then (x, zl, zq) is a certificate of dual infeasibility.
                 If solsta is solsta.unknown,
                   then (x, zl, zq) are all None

                 Other return values for solsta include:  
                   solsta.dual_feas  
                   solsta.near_dual_feas
                   solsta.near_optimal
                   solsta.near_prim_and_dual_feas
                   solsta.near_prim_feas
                   solsta.prim_and_dual_feas
                   solsta.prim_feas
                 in which case the (x,y,z) value may not be well-defined,
                 c.f., the MOSEK documentation.
        
        x, zl, zq  the primal-dual solution.


    Options are passed to MOSEK solvers via the mosek.options 
    dictionary, e.g., the following turns off output from 
    the MOSEK solvers
    
    >>> mosek.options = {iparam.log:0} 
    
    see chapter 14 of the MOSEK Python API manual.                    
    """
    if type(c) is not matrix or c.typecode != 'd' or c.size[1] != 1: 
        raise TypeError, "'c' must be a dense column matrix"
    n = c.size[0]
    if n < 1: raise ValueError, "number of variables must be at least 1"

    if Gl is None:  Gl = spmatrix([], [], [], (0,n), tc='d')
    if (type(Gl) is not matrix and type(Gl) is not spmatrix) or \
        Gl.typecode != 'd' or Gl.size[1] != n:
        raise TypeError, "'Gl' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n
    ml = Gl.size[0]
    if hl is None: hl = matrix(0.0, (0,1))
    if type(hl) is not matrix or hl.typecode != 'd' or \
        hl.size != (ml,1):
        raise TypeError, "'hl' must be a dense 'd' matrix of " \
            "size (%d,1)" %ml

    if Gq is None: Gq = []
    if type(Gq) is not list or [ G for G in Gq if (type(G) is not matrix 
        and type(G) is not spmatrix) or G.typecode != 'd' or 
        G.size[1] != n ]:
        raise TypeError, "'Gq' must be a list of sparse or dense 'd' "\
            "matrices with %d columns" %n 
    mq = [ G.size[0] for G in Gq ]
    a = [ k for k in xrange(len(mq)) if mq[k] == 0 ] 
    if a: raise TypeError, "the number of rows of Gq[%d] is zero" %a[0]
    if hq is None: hq = []
    if type(hq) is not list or len(hq) != len(mq) or [ h for h in hq if
        (type(h) is not matrix and type(h) is not spmatrix) or 
        h.typecode != 'd' ]: 
        raise TypeError, "'hq' must be a list of %d dense or sparse "\
            "'d' matrices" %len(mq)
    a = [ k for k in xrange(len(mq)) if hq[k].size != (mq[k], 1) ]
    if a:
        k = a[0]
        raise TypeError, "'hq[%d]' has size (%d,%d).  Expected size "\
            "is (%d,1)." %(k, hq[k].size[0], hq[k].size[1], mq[k]) 

    N = ml + sum(mq)
    h = matrix(0.0, (N,1))
    if type(Gl) is matrix or [ Gk for Gk in Gq if type(Gk) is matrix ]:
        G = matrix(0.0, (N, n))
    else:
        G = spmatrix([], [], [], (N, n), 'd')
    h[:ml] = hl
    G[:ml,:] = Gl
    ind = ml
    for k in xrange(len(mq)):
        h[ind : ind + mq[k]] = hq[k]
        G[ind : ind + mq[k], :] = Gq[k]
        ind += mq[k]

    bkc = n*[ msk.boundkey.fx ] 
    blc = array(-c)
    buc = array(-c)

    bkx = ml*[ msk.boundkey.lo ] + sum(mq)*[ msk.boundkey.fr ]
    blx = ml*[ 0.0 ] + sum(mq)*[ -inf ]
    bux = N*[ +inf ] 

    c   = -h        
    
    colptr, asub, acof = sparse([G.T]).CCS
    aptrb, aptre = colptr[:-1], colptr[1:]

    mskhandle = msk.mosek ()
    env = mskhandle.Env ()
    env.set_Stream (msk.streamtype.log, streamprinter)
        
    env.init () 
    task = env.Task(0,0) 
    task.set_Stream (msk.streamtype.log, streamprinter) 

    # set MOSEK options 
    for (param, val) in options.items():
        if str(param)[:6] == "iparam":
            task.putintparam(param, val)
        elif str(param)[:6] == "dparam":
            task.putdouparam(param, val)
        elif str(param)[:6] == "sparam":
            task.putstrparam(param, val)
        else:
            raise ValueError, "invalid MOSEK parameter: "+str(param)

    task.inputdata (n,   # number of constraints
                    N,   # number of variables
                    array(c), # linear objective coefficients  
                    0.0, # objective fixed value  
                    array(aptrb), 
                    array(aptre), 
                    array(asub),
                    array(acof), 
                    array(bkc),
                    array(blc),
                    array(buc), 
                    array(bkx),
                    array(blx),
                    array(bux)) 

    task.putobjsense(msk.objsense.maximize)

    for k in xrange(len(mq)):
        task.appendcone(msk.conetype.quad, 0.0, 
                        array(range(ml+sum(mq[:k]),ml+sum(mq[:k+1]))))
    task.optimize()

    task.solutionsummary (msk.streamtype.msg); 

    prosta, solsta = list(), list()
    task.getsolutionstatus(msk.soltype.itr, prosta, solsta)

    xu, xl, zq = zeros(n, Float), zeros(n, Float), zeros(sum(mq), Float)
    task.getsolutionslice(msk.soltype.itr, msk.solitem.slc, 0, n, xl) 
    task.getsolutionslice(msk.soltype.itr, msk.solitem.suc, 0, n, xu) 
    task.getsolutionslice(msk.soltype.itr, msk.solitem.xx, ml, N, zq) 
    x = matrix(xu-xl)

    zq = [ matrix(zq[sum(mq[:k]):sum(mq[:k+1])]) for k in xrange(len(mq)) ]
    
    if ml:
        zl = zeros(ml, Float)
        task.getsolutionslice(msk.soltype.itr, msk.solitem.xx, 0, ml, zl) 
        zl = matrix(zl)
    else:
        zl = matrix(0.0, (0,1))

    if (solsta[0] is msk.solsta.unknown):
        return (solsta[0], None, None, None)
    else:
        return (solsta[0], x, zl, zq)


def qp(P, q, G=None, h=None, A=None, b=None):
    """
    Solves a quadratic program

        minimize    (1/2)*x'*P*x + q'*x 
        subject to  G*x <= h      
                    A*x = b.                    
                    
    using MOSEK 5.0.

    (solsta, x, z, y) = qp(P, q, G=None, h=None, A=None, b=None)

    Return values

        solsta   the solution status.
                 If solsta is solsta.optimal,
                   then (x, y, z) contains the primal-dual solution.
                 If solsta is solsta.prim_infeas_cer,
                   then (x, y, z) is a certificate of primal infeasibility.
                 If solsta is solsta.dual_infeas_cer,
                   then (x, y, z) is a certificate of dual infeasibility.
                 If solsta is solsta.unknown,
                   then (x, y, z) are all None

                 Other return values for solsta include:  
                   solsta.dual_feas  
                   solsta.near_dual_feas
                   solsta.near_optimal
                   solsta.near_prim_and_dual_feas
                   solsta.near_prim_feas
                   solsta.prim_and_dual_feas
                   solsta.prim_feas
                 in which case the (x,y,z) value may not be well-defined,
                 c.f., the MOSEK documentation.
        
        x, z, y  the primal-dual solution.                    

    Options are passed to MOSEK solvers via the mosek.options 
    dictionary, e.g., the following turns off output from 
    the MOSEK solvers
    
    >>> mosek.options = {iparam.log:0} 
    
    see chapter 14 of the MOSEK Python API manual.                    
    """
    if (type(P) is not matrix and type(P) is not spmatrix) or \
        P.typecode != 'd' or P.size[0] != P.size[1]:
        raise TypeError, "'P' must be a square dense or sparse 'd' matrix "
    n = P.size[0]

    if n < 1: raise ValueError, "number of variables must be at least 1"

    if type(q) is not matrix or q.typecode != 'd' or q.size != (n,1):
        raise TypeError, "'q' must be a 'd' matrix of size (%d,1)" %n

    if G is None: G = spmatrix([], [], [], (0,n), 'd')
    if (type(G) is not matrix and type(G) is not spmatrix) or \
        G.typecode != 'd' or G.size[1] != n:
        raise TypeError, "'G' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n 

    m = G.size[0]
    if h is None: h = matrix(0.0, (0,1))
    if type(h) is not matrix or h.typecode != 'd' or h.size != (m,1):
        raise TypeError, "'h' must be a 'd' matrix of size (%d,1)" %m

    if A is None:  A = spmatrix([], [], [], (0,n), 'd')
    if (type(A) is not matrix and type(A) is not spmatrix) or \
        A.typecode != 'd' or A.size[1] != n:
        raise TypeError, "'A' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n
    p = A.size[0]
    if b is None: b = matrix(0.0, (0,1))
    if type(b) is not matrix or b.typecode != 'd' or b.size != (p,1): 
        raise TypeError, "'b' must be a dense matrix of size (%d,1)" %p
 
    if m+p is 0: raise ValueError, "m + p must be greater than 0"

    c   = array(q)        

    bkc = m*[ msk.boundkey.up ] + p*[ msk.boundkey.fx ]
    blc = m*[ -inf ] + [ bi for bi in b ]
    buc = matrix([h, b])

    bkx = n*[msk.boundkey.fr] 
    blx = n*[ -inf ] 
    bux = n*[ +inf ]

    colptr, asub, acof = sparse([G,A]).CCS
    aptrb, aptre = colptr[:-1], colptr[1:]

    mskhandle = msk.mosek ()
    env = mskhandle.Env ()
    env.set_Stream (msk.streamtype.log, streamprinter)
        
    env.init () 
    task = env.Task(0,0) 
    task.set_Stream (msk.streamtype.log, streamprinter) 

    # set MOSEK options 
    for (param, val) in options.items():
        if str(param)[:6] == "iparam":
            task.putintparam(param, val)
        elif str(param)[:6] == "dparam":
            task.putdouparam(param, val)
        elif str(param)[:6] == "sparam":
            task.putstrparam(param, val)
        else:
            raise ValueError, "invalid MOSEK parameter: "+str(param)

    task.inputdata (m+p, # number of constraints
                    n,   # number of variables
                    array(c), # linear objective coefficients  
                    0.0, # objective fixed value  
                    array(aptrb), 
                    array(aptre), 
                    array(asub),
                    array(acof), 
                    array(bkc),
                    array(blc),
                    array(buc), 
                    array(bkx),
                    array(blx),
                    array(bux)) 

    Ps = sparse(P)
    I, J = Ps.I, Ps.J
    tril = [ k for k in xrange(len(I)) if I[k] >= J[k] ]
    task.putqobj(array(I[tril]), array(J[tril]), array(Ps.V[tril]))
    
    task.putobjsense(msk.objsense.minimize)

    task.optimize()

    task.solutionsummary (msk.streamtype.msg); 

    prosta, solsta = list(), list()
    task.getsolutionstatus(msk.soltype.itr, prosta, solsta)

    x = zeros(n, Float)
    task.getsolutionslice(msk.soltype.itr, msk.solitem.xx, 0, n, x) 
    x = matrix(x)

    if m is not 0:
        z = zeros(m, Float)
        task.getsolutionslice(msk.soltype.itr, msk.solitem.suc, 0, m, z) 
        z = matrix(z)
    else:
        z = matrix(0.0, (0,1))

    if p is not 0:
        yu, yl = zeros(p, Float), zeros(p, Float)
        task.getsolutionslice(msk.soltype.itr, msk.solitem.suc, m, m+p, yu) 
        task.getsolutionslice(msk.soltype.itr, msk.solitem.slc, m, m+p, yl) 
        y = matrix(yu) - matrix(yl)
    else:
        y = matrix(0.0, (0,1))

    if (solsta[0] is msk.solsta.unknown):
        return (solsta[0], None, None, None)
    else:
        return (solsta[0], x, z, y)


def ilp(c, G, h, A=None, b=None, I=None):
    """
    Solves the mixed integer LP

        minimize    c'*x       
        subject to  G*x + s = h
                    A*x = b    
                    s >= 0
                    xi integer, forall i in I
                    
    using MOSEK 5.0.

    (solsta, x) = ilp(c, G, h, A=None, b=None, I=None).

    Input arguments 

        G is m x n, h is m x 1, A is p x n, b is p x 1.  G and A must be 
        dense or sparse 'd' matrices.   h and b are dense 'd' matrices 
        with one column.  The default values for A and b are empty 
        matrices with zero rows.

        I is a Python set with indices of integer elements of x.  By 
        default all elements in x are constrained to be integer, i.e.,
        the default value of I is I = set(range(n))

        Dual variables are not returned for MOSEK.

    Return values

        solsta   the solution status.
                 If solsta is solsta.integer_optimal,
                   then x contains the solution.
                 If solsta is solsta.unknown,
                   then x is None

                 Other return values for solsta include:  
                   mosek.solsta.near_integer_optimal
                 in which case the x value may not be well-defined,
                 c.f., the MOSEK documentation.
        
        x        the solution

    Options are passed to MOSEK solvers via the mosek.options
    dictionary, e.g., the following turns off output from 
    the MOSEK solvers
    
    >>> mosek.options = {iparam.log:0} 
    
    see chapter 14 of the MOSEK Python API manual.                    
    """
    if type(c) is not matrix or c.typecode != 'd' or c.size[1] != 1: 
        raise TypeError, "'c' must be a dense column matrix"
    n = c.size[0]
    if n < 1: raise ValueError, "number of variables must be at least 1"

    if (type(G) is not matrix and type(G) is not spmatrix) or \
        G.typecode != 'd' or G.size[1] != n:
        raise TypeError, "'G' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n 
    m = G.size[0]
    if m is 0: raise ValueError, "m cannot be 0"

    if type(h) is not matrix or h.typecode != 'd' or h.size != (m,1):
        raise TypeError, "'h' must be a 'd' matrix of size (%d,1)" %m

    if A is None:  A = spmatrix([], [], [], (0,n), 'd')
    if (type(A) is not matrix and type(A) is not spmatrix) or \
        A.typecode != 'd' or A.size[1] != n:
        raise TypeError, "'A' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n
    p = A.size[0]
    if b is None: b = matrix(0.0, (0,1))
    if type(b) is not matrix or b.typecode != 'd' or b.size != (p,1): 
        raise TypeError, "'b' must be a dense matrix of size (%d,1)" %p
 
    c   = array(c)        

    if I is None: I = set(range(n))

    if type(I) is not set: 
        raise TypeError, "invalid argument for integer index set"

    for i in I:
        if type(i) is not int: 
            raise TypeError, "invalid integer index set I"

    if len(I) > 0 and min(I) < 0: raise IndexError, \
            "negative element in integer index set I"
    if len(I) > 0 and max(I) > n-1: raise IndexError, \
            "maximum element in in integer index set I is larger than n-1"

    bkc = m*[ msk.boundkey.up ] + p*[ msk.boundkey.fx ]
    blc = m*[ -inf ] + [ bi for bi in b ]
    buc = matrix([h, b])

    bkx = n*[msk.boundkey.fr] 
    blx = n*[ -inf ] 
    bux = n*[ +inf ]

    colptr, asub, acof = sparse([G,A]).CCS
    aptrb, aptre = colptr[:-1], colptr[1:]

    mskhandle = msk.mosek ()
    env = mskhandle.Env ()
    env.set_Stream (msk.streamtype.log, streamprinter)
        
    env.init () 
    task = env.Task(0,0) 
    task.set_Stream (msk.streamtype.log, streamprinter) 

    # set MOSEK options 
    for (param, val) in options.items():
        if str(param)[:6] == "iparam":
            task.putintparam(param, val)
        elif str(param)[:6] == "dparam":
            task.putdouparam(param, val)
        elif str(param)[:6] == "sparam":
            task.putstrparam(param, val)
        else:
            raise ValueError, "invalid MOSEK parameter: "+str(param)
    
    task.inputdata (m+p, # number of constraints
                    n,   # number of variables
                    array(c), # linear objective coefficients  
                    0.0, # objective fixed value  
                    array(aptrb), 
                    array(aptre), 
                    array(asub),
                    array(acof), 
                    array(bkc),
                    array(blc),
                    array(buc), 
                    array(bkx),
                    array(blx),
                    array(bux)) 

    task.putobjsense(msk.objsense.minimize)

    # Define integer variables 
    if len(I) > 0:
        task.putvartypelist(array(I), 
                            array(len(I)*[ msk.variabletype.type_int ])) 

    task.putintparam (msk.iparam.mio_mode, msk.miomode.satisfied) 

    task.optimize()

    task.solutionsummary (msk.streamtype.msg); 

    prosta, solsta = list(), list()
    if len(I) > 0:
        task.getsolutionstatus(msk.soltype.itg, prosta, solsta)
    else:
        task.getsolutionstatus(msk.soltype.bas, prosta, solsta)
        
    x = zeros(n, Float)
    if len(I) > 0:
        task.getsolutionslice(msk.soltype.itg, msk.solitem.xx, 0, n, x) 
    else:
        task.getsolutionslice(msk.soltype.bas, msk.solitem.xx, 0, n, x) 
    x = matrix(x)

    if (solsta[0] is msk.solsta.unknown):
        return (solsta[0], None)
    else:
        return (solsta[0], x)


            
