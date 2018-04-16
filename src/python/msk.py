"""
CVXOPT interface for MOSEK 8
"""

# Copyright 2012-2018 M. Andersen and L. Vandenberghe.
# Copyright 2010-2011 L. Vandenberghe.
# Copyright 2004-2009 J. Dahl and L. Vandenberghe.
#
# This file is part of CVXOPT.
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


import mosek
from cvxopt import matrix, spmatrix, sparse

import sys

def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

inf = 0.0

options = {}

def lp(c, G, h, A=None, b=None, taskfile=None, **kwargs):
    """
    Solves a pair of primal and dual LPs

        minimize    c'*x             maximize    -h'*z - b'*y
        subject to  G*x + s = h      subject to  G'*z + A'*y + c = 0
                    A*x = b                      z >= 0.
                    s >= 0

    using MOSEK 8.

    (solsta, x, z, y) = lp(c, G, h, A=None, b=None).

    Input arguments

        c is n x 1, G is m x n, h is m x 1, A is p x n, b is p x 1.  G and
        A must be dense or sparse 'd' matrices.  c, h and b are dense 'd'
        matrices with one column.  The default values for A and b are
        empty matrices with zero rows.

        Optionally, the interface can write a .task file, required for
        support questions on the MOSEK solver.

    Return values

        solsta is a MOSEK solution status key.

            If solsta is mosek.solsta.optimal, then (x, y, z) contains the
                primal-dual solution.
            If solsta is mosek.solsta.prim_infeas_cer, then (x, y, z) is a
                certificate of primal infeasibility.
            If solsta is mosek.solsta.dual_infeas_cer, then (x, y, z) is a
                certificate of dual infeasibility.
            If solsta is mosek.solsta.unknown, then (x, y, z) are all None.

            Other return values for solsta include:
                mosek.solsta.dual_feas
                mosek.solsta.near_dual_feas
                mosek.solsta.near_optimal
                mosek.solsta.near_prim_and_dual_feas
                mosek.solsta.near_prim_feas
                mosek.solsta.prim_and_dual_feas
                mosek.solsta.prim_feas
             in which case the (x,y,z) value may not be well-defined.

        x, y, z  the primal-dual solution.

    Options are passed to MOSEK solvers via the msk.options dictionary.
    For example, the following turns off output from the MOSEK solvers

        >>> msk.options = {mosek.iparam.log: 0}

    see the MOSEK Python API manual.
    """

    with mosek.Env() as env:

        if type(c) is not matrix or c.typecode != 'd' or c.size[1] != 1:
            raise TypeError("'c' must be a dense column matrix")
        n = c.size[0]
        if n < 1: raise ValueError("number of variables must be at least 1")

        if (type(G) is not matrix and type(G) is not spmatrix) or \
            G.typecode != 'd' or G.size[1] != n:
            raise TypeError("'G' must be a dense or sparse 'd' matrix "\
                "with %d columns" %n)
        m = G.size[0]
        if m is 0: raise ValueError("m cannot be 0")

        if type(h) is not matrix or h.typecode != 'd' or h.size != (m,1):
            raise TypeError("'h' must be a 'd' matrix of size (%d,1)" %m)

        if A is None:  A = spmatrix([], [], [], (0,n), 'd')
        if (type(A) is not matrix and type(A) is not spmatrix) or \
            A.typecode != 'd' or A.size[1] != n:
            raise TypeError("'A' must be a dense or sparse 'd' matrix "\
                "with %d columns" %n)
        p = A.size[0]
        if b is None: b = matrix(0.0, (0,1))
        if type(b) is not matrix or b.typecode != 'd' or b.size != (p,1):
            raise TypeError("'b' must be a dense matrix of size (%d,1)" %p)

        bkc = m*[ mosek.boundkey.up ] + p*[ mosek.boundkey.fx ]
        blc = m*[ -inf ] + [ bi for bi in b ]
        buc = list(h) + list(b)

        bkx = n*[mosek.boundkey.fr]
        blx = n*[ -inf ]
        bux = n*[ +inf ]

        colptr, asub, acof = sparse([G,A]).CCS
        aptrb, aptre = colptr[:-1], colptr[1:]


        with env.Task(0,0) as task:
            task.set_Stream (mosek.streamtype.log, streamprinter)

            # set MOSEK options
            options = kwargs.get('options',globals()['options'])
            for (param, val) in options.items():
                if str(param)[:6] == "iparam":
                    task.putintparam(param, val)
                elif str(param)[:6] == "dparam":
                    task.putdouparam(param, val)
                elif str(param)[:6] == "sparam":
                    task.putstrparam(param, val)
                else:
                    raise ValueError("invalid MOSEK parameter: " + str(param))

            task.inputdata (m+p, # number of constraints
                            n,   # number of variables
                            list(c), # linear objective coefficients
                            0.0, # objective fixed value
                            list(aptrb),
                            list(aptre),
                            list(asub),
                            list(acof),
                            bkc,
                            blc,
                            buc,
                            bkx,
                            blx,
                            bux)

            task.putobjsense(mosek.objsense.minimize)

            if taskfile:
                task.writetask(taskfile)

            task.optimize()

            task.solutionsummary (mosek.streamtype.msg);

            solsta = task.getsolsta(mosek.soltype.bas)

            x, z = n*[ 0.0 ], m*[ 0.0 ]
            task.getsolutionslice(mosek.soltype.bas, mosek.solitem.xx, 0, n, x)
            task.getsolutionslice(mosek.soltype.bas, mosek.solitem.suc, 0, m, z)
            x, z = matrix(x), matrix(z)

            if p is not 0:
                yu, yl = p*[0.0], p*[0.0]
                task.getsolutionslice(mosek.soltype.bas, mosek.solitem.suc, m, m+p, yu)
                task.getsolutionslice(mosek.soltype.bas, mosek.solitem.slc, m, m+p, yl)
                y = matrix(yu) - matrix(yl)
            else:
                y = matrix(0.0, (0,1))

    if (solsta is mosek.solsta.unknown):
        return (solsta, None, None, None)
    else:
        return (solsta, x, z, y)


def conelp(c, G, h, dims=None, taskfile=None, **kwargs):
    """
    Solves a pair of primal and dual SOCPs

        minimize    c'*x
        subject to  G*x + s = h
                    s >= 0

        maximize    -h'*z
        subject to  G'*z + c = 0
                    z >= 0

    using MOSEK 8.

    The inequalities are with respect to a cone C defined as the Cartesian
    product of N + M + 1 cones:

        C = C_0 x C_1 x .... x C_N x C_{N+1} x ... x C_{N+M}.

    The first cone C_0 is the nonnegative orthant of dimension ml.
    The next N cones are second order cones of dimension mq[0], ...,
    mq[N-1].  The second order cone of dimension m is defined as

        { (u0, u1) in R x R^{m-1} | u0 >= ||u1||_2 }.

    The next M cones are positive semidefinite cones of order ms[0], ...,
    ms[M-1] >= 0.

    The formats of G and h are identical to that used in solvers.conelp().


    Input arguments.

        c is a dense 'd' matrix of size (n,1).

        dims is a dictionary with the dimensions of the components of C.
        It has three fields.
        - dims['l'] = ml, the dimension of the nonnegative orthant C_0.
          (ml >= 0.)
        - dims['q'] = mq = [ mq[0], mq[1], ..., mq[N-1] ], a list of N
          integers with the dimensions of the second order cones C_1, ...,
          C_N.  (N >= 0 and mq[k] >= 1.)
        - dims['s'] = ms = [ ms[0], ms[1], ..., ms[M-1] ], a list of M
          integers with the orders of the semidefinite cones C_{N+1}, ...,
          C_{N+M}.  (M >= 0 and ms[k] >= 0.)
        The default value of dims is {'l': G.size[0], 'q': [], 's': []}.

        G is a dense or sparse 'd' matrix of size (K,n), where

            K = ml + mq[0] + ... + mq[N-1] + ms[0]**2 + ... + ms[M-1]**2.

        Each column of G describes a vector

            v = ( v_0, v_1, ..., v_N, vec(v_{N+1}), ..., vec(v_{N+M}) )

        in V = R^ml x R^mq[0] x ... x R^mq[N-1] x S^ms[0] x ... x S^ms[M-1]
        stored as a column vector

            [ v_0; v_1; ...; v_N; vec(v_{N+1}); ...; vec(v_{N+M}) ].

        Here, if u is a symmetric matrix of order m, then vec(u) is the
        matrix u stored in column major order as a vector of length m**2.
        We use BLAS unpacked 'L' storage, i.e., the entries in vec(u)
        corresponding to the strictly upper triangular entries of u are
        not referenced.

        h is a dense 'd' matrix of size (K,1), representing a vector in V,
        in the same format as the columns of G.

        A is a dense or sparse 'd' matrix of size (p,n).  The default value
        is a sparse 'd' matrix of size (0,n).

        b is a dense 'd' matrix of size (p,1).   The default value is a
        dense 'd' matrix of size (0,1).

        Optionally, the interface can write a .task file, required for
        support questions on the MOSEK solver.

    Return values

        solsta is a MOSEK solution status key.

            If solsta is mosek.solsta.optimal,
                then (x, zl, zq, zs) contains the primal-dual solution.
            If solsta is moseksolsta.prim_infeas_cer,
                then (x, zl, zq, zs) is a certificate of dual infeasibility.
            If solsta is moseksolsta.dual_infeas_cer,
                then (x, zl, zq, zs) is a certificate of primal infeasibility.
            If solsta is mosek.solsta.unknown,
                then (x, zl, zq, zs) are all None

            Other return values for solsta include:
                mosek.solsta.dual_feas
                mosek.solsta.near_dual_feas
                mosek.solsta.near_optimal
                mosek.solsta.near_prim_and_dual_feas
                mosek.solsta.near_prim_feas
                mosek.solsta.prim_and_dual_feas
                mosek.solsta.prim_feas
            in which case the (x,y,z) value may not be well-defined.

        x, z the primal-dual solution.


    Options are passed to MOSEK solvers via the msk.options dictionary,
    e.g., the following turns off output from the MOSEK solvers

        >>> msk.options = {mosek.iparam.log:0}

    see the MOSEK Python API manual.
    """

    with mosek.Env() as env:

        if dims is None:
            (solsta, x, y, z) = lp(c, G, h)
            return (solsta, x, z, None)

        N, n = G.size

        ml, mq, ms = dims['l'], dims['q'], [ k*k for k in dims['s'] ]
        cdim = ml + sum(mq) + sum(ms)
        if cdim is 0: raise ValueError("ml+mq+ms cannot be 0")

        # Data for kth 'q' constraint are found in rows indq[k]:indq[k+1] of G.
        indq = [ dims['l'] ]
        for k in dims['q']:  indq = indq + [ indq[-1] + k ]

        # Data for the kth 's' constraint are found in rows indq[-1] + (inds[k]:inds[k+1]) of G.
        inds = [ 0 ]
        for k in dims['s']: inds = inds + [ inds[-1] + k*k ]

        if type(h) is not matrix or h.typecode != 'd' or h.size[1] != 1:
            raise TypeError("'h' must be a 'd' matrix with 1 column")
        if type(G) is matrix or type(G) is spmatrix:
            if G.typecode != 'd' or G.size[0] != cdim:
                raise TypeError("'G' must be a 'd' matrix with %d rows " %cdim)
            if h.size[0] != cdim:
                raise TypeError("'h' must have %d rows" %cdim)
        else:
            raise TypeError("'G' must be a matrix")

        if len(dims['q']) and min(dims['q'])<1: raise TypeError(
            "dimensions of quadratic cones must be positive")

        if len(dims['s']) and min(dims['s'])<1: raise TypeError(
            "dimensions of semidefinite cones must be positive")

        bkc = n*[ mosek.boundkey.fx ]
        blc = list(-c)
        buc = list(-c)

        dimx = ml + sum(mq)
        bkx  = ml*[ mosek.boundkey.lo ] + sum(mq)*[ mosek.boundkey.fr ]
        blx  = ml*[ 0.0 ] + sum(mq)*[ -inf ]
        bux  = dimx*[ +inf ]
        c    = list(-h)

        cl, cs = c[:dimx], sparse(c[dimx:])
        Gl, Gs = sparse(G[:dimx,:]), sparse(G[dimx:,:])
        colptr, asub, acof = Gl.T.CCS
        aptrb, aptre = colptr[:-1], colptr[1:]

        with env.Task(0,0) as task:
            task.set_Stream (mosek.streamtype.log, streamprinter)

            # set MOSEK options
            options = kwargs.get('options',globals()['options'])
            for (param, val) in options.items():
                if str(param)[:6] == "iparam":
                    task.putintparam(param, val)
                elif str(param)[:6] == "dparam":
                    task.putdouparam(param, val)
                elif str(param)[:6] == "sparam":
                    task.putstrparam(param, val)
                else:
                    raise ValueError("invalid MOSEK parameter: "+str(param))

            task.inputdata (n,    # number of constraints
                            dimx, # number of variables
                            cl,   # linear objective coefficients
                            0.0,  # objective fixed value
                            list(aptrb),
                            list(aptre),
                            list(asub),
                            list(acof),
                            bkc,
                            blc,
                            buc,
                            bkx,
                            blx,
                            bux)

            task.putobjsense(mosek.objsense.maximize)

            numbarvar = len(dims['s'])
            task.appendbarvars(dims['s'])

            barcsubj, barcsubk, barcsubl = (inds[-1])*[ 0 ], (inds[-1])*[ 0 ], (inds[-1])*[ 0 ]
            barcval = [ -h[indq[-1]+k] for k in range(inds[0], inds[-1])]
            for s in range(numbarvar):
                for (k,idx) in enumerate(range(inds[s],inds[s+1])):
                    barcsubk[idx] = k // dims['s'][s]
                    barcsubl[idx] = k % dims['s'][s]
                    barcsubj[idx] = s

            # filter out upper triangular part
            trilidx  = [ idx for idx in range(len(barcsubk)) if barcsubk[idx] >= barcsubl[idx] ]
            barcsubj = [ barcsubj[k] for k in trilidx ]
            barcsubk = [ barcsubk[k] for k in trilidx ]
            barcsubl = [ barcsubl[k] for k in trilidx ]
            barcval  = [ barcval[k]  for k in trilidx ]

            task.putbarcblocktriplet(len(trilidx), barcsubj, barcsubk, barcsubl, barcval)

            Gst = Gs.T
            barasubi = len(Gst)*[ 0 ]
            barasubj = len(Gst)*[ 0 ]
            barasubk = len(Gst)*[ 0 ]
            barasubl = len(Gst)*[ 0 ]
            baraval  = len(Gst)*[ 0.0 ]
            colptr, row, val = Gst.CCS

            for s in range(numbarvar):
                for j in range(ms[s]):
                    for idx in range(colptr[inds[s]+j], colptr[inds[s]+j+1]):
                        barasubi[idx] = row[idx]
                        barasubj[idx] = s
                        barasubk[idx] = j // dims['s'][s]
                        barasubl[idx] = j % dims['s'][s]
                        baraval[idx]  = val[idx]

            # filter out upper triangular part
            trilidx = [ idx for (idx, (k,l)) in enumerate(zip(barasubk,barasubl)) if k >= l ]
            barasubi = [ barasubi[k] for k in trilidx ]
            barasubj = [ barasubj[k] for k in trilidx ]
            barasubk = [ barasubk[k] for k in trilidx ]
            barasubl = [ barasubl[k] for k in trilidx ]
            baraval  = [ baraval[k]  for k in trilidx ]

            task.putbarablocktriplet(len(trilidx), barasubi, barasubj, barasubk, barasubl, baraval)

            for k in range(len(mq)):
                task.appendcone(mosek.conetype.quad, 0.0,
                                range(ml+sum(mq[:k]),ml+sum(mq[:k+1])))

            if taskfile:
                task.writetask(taskfile)

            task.optimize()

            task.solutionsummary (mosek.streamtype.msg);

            solsta = task.getsolsta(mosek.soltype.itr)

            xu, xl, zq = n*[ 0.0 ], n*[ 0.0 ], sum(mq)*[ 0.0 ]
            task.getsolutionslice(mosek.soltype.itr, mosek.solitem.slc, 0, n, xl)
            task.getsolutionslice(mosek.soltype.itr, mosek.solitem.suc, 0, n, xu)
            task.getsolutionslice(mosek.soltype.itr, mosek.solitem.xx, ml, dimx, zq)
            x = matrix(xu)-matrix(xl)
            zq = matrix(zq)

            for s in range(numbarvar):
                xx = (dims['s'][s]*(dims['s'][s] + 1) >> 1)*[0.0]
                task.getbarxj(mosek.soltype.itr, s, xx)

                xs = matrix(0.0, (dims['s'][s], dims['s'][s]))
                idx = 0
                for j in range(dims['s'][s]):
                    for i in range(j,dims['s'][s]):
                        xs[i,j] = xx[idx]
                        if i != j:
                            xs[j,i] = xx[idx]
                        idx += 1

                zq = matrix([zq, xs[:]])

            if ml:
                zl = ml*[ 0.0 ]
                task.getsolutionslice(mosek.soltype.itr, mosek.solitem.xx, 0, ml, zl)
                zl = matrix(zl)
            else:
                zl = matrix(0.0, (0,1))

    if (solsta is mosek.solsta.unknown):
        return (solsta, None, None)
    else:
        return (solsta, x, matrix([zl, zq]))


def socp(c, Gl=None, hl=None, Gq=None, hq=None, taskfile=None, **kwargs):
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

    using MOSEK 8.

    solsta, x, zl, zq = socp(c, Gl = None, hl = None, Gq = None, hq = None, taskfile=None)

    Return values

        solsta is a MOSEK solution status key.
            If solsta is mosek.solsta.optimal,
                then (x, zl, zq) contains the primal-dual solution.
            If solsta is mosek.solsta.prim_infeas_cer,
                then (x, zl, zq) is a certificate of dual infeasibility.
            If solsta is mosek.solsta.dual_infeas_cer,
                then (x, zl, zq) is a certificate of primal infeasibility.
            If solsta is mosek.solsta.unknown,
                then (x, zl, zq) are all None

            Other return values for solsta include:
                mosek.solsta.dual_feas
                mosek.solsta.near_dual_feas
                mosek.solsta.near_optimal
                mosek.solsta.near_prim_and_dual_feas
                mosek.solsta.near_prim_feas
                mosek.solsta.prim_and_dual_feas
                mosek.solsta.prim_feas
             in which case the (x,y,z) value may not be well-defined.

        x, zl, zq  the primal-dual solution.


    Options are passed to MOSEK solvers via the msk.options dictionary,
    e.g., the following turns off output from the MOSEK solvers

        >>> msk.options = {mosek.iparam.log: 0}

    see the MOSEK Python API manual.

    Optionally, the interface can write a .task file, required for
    support questions on the MOSEK solver.
    """

    with mosek.Env() as env:

        if type(c) is not matrix or c.typecode != 'd' or c.size[1] != 1:
            raise TypeError("'c' must be a dense column matrix")
        n = c.size[0]
        if n < 1: raise ValueError("number of variables must be at least 1")

        if Gl is None:  Gl = spmatrix([], [], [], (0,n), tc='d')
        if (type(Gl) is not matrix and type(Gl) is not spmatrix) or \
            Gl.typecode != 'd' or Gl.size[1] != n:
            raise TypeError("'Gl' must be a dense or sparse 'd' matrix "\
                "with %d columns" %n)
        ml = Gl.size[0]
        if hl is None: hl = matrix(0.0, (0,1))
        if type(hl) is not matrix or hl.typecode != 'd' or \
            hl.size != (ml,1):
            raise TypeError("'hl' must be a dense 'd' matrix of " \
                "size (%d,1)" %ml)

        if Gq is None: Gq = []
        if type(Gq) is not list or [ G for G in Gq if (type(G) is not matrix
            and type(G) is not spmatrix) or G.typecode != 'd' or
            G.size[1] != n ]:
            raise TypeError("'Gq' must be a list of sparse or dense 'd' "\
                "matrices with %d columns" %n)
        mq = [ G.size[0] for G in Gq ]
        a = [ k for k in range(len(mq)) if mq[k] == 0 ]
        if a: raise TypeError("the number of rows of Gq[%d] is zero" %a[0])
        if hq is None: hq = []
        if type(hq) is not list or len(hq) != len(mq) or [ h for h in hq if
            (type(h) is not matrix and type(h) is not spmatrix) or
            h.typecode != 'd' ]:
                raise TypeError("'hq' must be a list of %d dense or sparse "\
                    "'d' matrices" %len(mq))
        a = [ k for k in range(len(mq)) if hq[k].size != (mq[k], 1) ]
        if a:
            k = a[0]
            raise TypeError("'hq[%d]' has size (%d,%d).  Expected size "\
                "is (%d,1)." %(k, hq[k].size[0], hq[k].size[1], mq[k]))

        N = ml + sum(mq)
        h = matrix(0.0, (N,1))
        if type(Gl) is matrix or [ Gk for Gk in Gq if type(Gk) is matrix ]:
            G = matrix(0.0, (N, n))
        else:
            G = spmatrix([], [], [], (N, n), 'd')
        h[:ml] = hl
        G[:ml,:] = Gl
        ind = ml
        for k in range(len(mq)):
            h[ind : ind + mq[k]] = hq[k]
            G[ind : ind + mq[k], :] = Gq[k]
            ind += mq[k]

        bkc = n*[ mosek.boundkey.fx ]
        blc = list(-c)
        buc = list(-c)

        bkx = ml*[ mosek.boundkey.lo ] + sum(mq)*[ mosek.boundkey.fr ]
        blx = ml*[ 0.0 ] + sum(mq)*[ -inf ]
        bux = N*[ +inf ]

        c   = -h

        colptr, asub, acof = sparse([G.T]).CCS
        aptrb, aptre = colptr[:-1], colptr[1:]

        with env.Task(0,0) as task:
            task.set_Stream (mosek.streamtype.log, streamprinter)

            # set MOSEK options
            options = kwargs.get('options',globals()['options'])
            for (param, val) in options.items():
                if str(param)[:6] == "iparam":
                    task.putintparam(param, val)
                elif str(param)[:6] == "dparam":
                    task.putdouparam(param, val)
                elif str(param)[:6] == "sparam":
                    task.putstrparam(param, val)
                else:
                    raise ValueError("invalid MOSEK parameter: "+str(param))

            task.inputdata (n,   # number of constraints
                            N,   # number of variables
                            list(c), # linear objective coefficients
                            0.0, # objective fixed value
                            list(aptrb),
                            list(aptre),
                            list(asub),
                            list(acof),
                            bkc,
                            blc,
                            buc,
                            bkx,
                            blx,
                            bux)

            task.putobjsense(mosek.objsense.maximize)

            for k in range(len(mq)):
                task.appendcone(mosek.conetype.quad, 0.0,
                                list(range(ml+sum(mq[:k]),ml+sum(mq[:k+1]))))

            if taskfile:
                task.writetask(taskfile)

            task.optimize()

            task.solutionsummary (mosek.streamtype.msg);

            solsta = task.getsolsta(mosek.soltype.itr)

            xu, xl, zq = n*[0.0], n*[0.0], sum(mq)*[0.0]
            task.getsolutionslice(mosek.soltype.itr, mosek.solitem.slc, 0, n, xl)
            task.getsolutionslice(mosek.soltype.itr, mosek.solitem.suc, 0, n, xu)
            task.getsolutionslice(mosek.soltype.itr, mosek.solitem.xx, ml, N, zq)
            x = matrix(xu) - matrix(xl)

            zq = [ matrix(zq[sum(mq[:k]):sum(mq[:k+1])]) for k in range(len(mq)) ]

            if ml:
                zl = ml*[0.0]
                task.getsolutionslice(mosek.soltype.itr, mosek.solitem.xx, 0, ml,
                    zl)
                zl = matrix(zl)
            else:
                zl = matrix(0.0, (0,1))

    if (solsta is mosek.solsta.unknown):
        return (solsta, None, None, None)
    else:
        return (solsta, x, zl, zq)


def qp(P, q, G=None, h=None, A=None, b=None, taskfile=None, **kwargs):
    """
    Solves a quadratic program

        minimize    (1/2)*x'*P*x + q'*x
        subject to  G*x <= h
                    A*x = b.

    using MOSEK 8.

    solsta, x, z, y = qp(P, q, G=None, h=None, A=None, b=None, taskfile=None)

    Return values

        solsta is a MOSEK solution status key.

            If solsta is mosek.solsta.optimal,
                then (x, y, z) contains the primal-dual solution.
            If solsta is mosek.solsta.prim_infeas_cer,
                then (x, y, z) is a certificate of primal infeasibility.
            If solsta is mosek.solsta.dual_infeas_cer,
                then (x, y, z) is a certificate of dual infeasibility.
            If solsta is mosek.solsta.unknown, then (x, y, z) are all None.

            Other return values for solsta include:
                mosek.solsta.dual_feas
                mosek.solsta.near_dual_feas
                mosek.solsta.near_optimal
                mosek.solsta.near_prim_and_dual_feas
                mosek.solsta.near_prim_feas
                mosek.solsta.prim_and_dual_feas
                mosek.solsta.prim_feas
            in which case the (x,y,z) value may not be well-defined.

        x, z, y  the primal-dual solution.

    Options are passed to MOSEK solvers via the msk.options dictionary,
    e.g., the following turns off output from the MOSEK solvers

        >>> msk.options = {mosek.iparam.log: 0}

    see the MOSEK Python API manual.

    Optionally, the interface can write a .task file, required for
    support questions on the MOSEK solver.
    """

    with mosek.Env() as env:

        if (type(P) is not matrix and type(P) is not spmatrix) or \
            P.typecode != 'd' or P.size[0] != P.size[1]:
            raise TypeError("'P' must be a square dense or sparse 'd' matrix ")
        n = P.size[0]

        if n < 1: raise ValueError("number of variables must be at least 1")

        if type(q) is not matrix or q.typecode != 'd' or q.size != (n,1):
            raise TypeError("'q' must be a 'd' matrix of size (%d,1)" %n)

        if G is None: G = spmatrix([], [], [], (0,n), 'd')
        if (type(G) is not matrix and type(G) is not spmatrix) or \
            G.typecode != 'd' or G.size[1] != n:
            raise TypeError("'G' must be a dense or sparse 'd' matrix "\
                "with %d columns" %n)

        m = G.size[0]
        if h is None: h = matrix(0.0, (0,1))
        if type(h) is not matrix or h.typecode != 'd' or h.size != (m,1):
            raise TypeError("'h' must be a 'd' matrix of size (%d,1)" %m)

        if A is None:  A = spmatrix([], [], [], (0,n), 'd')
        if (type(A) is not matrix and type(A) is not spmatrix) or \
            A.typecode != 'd' or A.size[1] != n:
            raise TypeError("'A' must be a dense or sparse 'd' matrix "\
                "with %d columns" %n)
        p = A.size[0]
        if b is None: b = matrix(0.0, (0,1))
        if type(b) is not matrix or b.typecode != 'd' or b.size != (p,1):
            raise TypeError("'b' must be a dense matrix of size (%d,1)" %p)

        if m+p is 0: raise ValueError("m + p must be greater than 0")

        c = list(q)

        bkc = m*[ mosek.boundkey.up ] + p*[ mosek.boundkey.fx ]
        blc = m*[ -inf ] + [ bi for bi in b ]
        buc = list(h)+list(b)

        bkx = n*[mosek.boundkey.fr]
        blx = n*[ -inf ]
        bux = n*[ +inf ]

        colptr, asub, acof = sparse([G,A]).CCS
        aptrb, aptre = colptr[:-1], colptr[1:]

        with env.Task(0,0) as task:
            task.set_Stream (mosek.streamtype.log, streamprinter)

            # set MOSEK options
            options = kwargs.get('options',globals()['options'])
            for (param, val) in options.items():
                if str(param)[:6] == "iparam":
                    task.putintparam(param, val)
                elif str(param)[:6] == "dparam":
                    task.putdouparam(param, val)
                elif str(param)[:6] == "sparam":
                    task.putstrparam(param, val)
                else:
                    raise ValueError("invalid MOSEK parameter: "+str(param))

            task.inputdata (m+p, # number of constraints
                            n,   # number of variables
                            c, # linear objective coefficients
                            0.0, # objective fixed value
                            list(aptrb),
                            list(aptre),
                            list(asub),
                            list(acof),
                            bkc,
                            blc,
                            buc,
                            bkx,
                            blx,
                            bux)

            Ps = sparse(P)
            I, J = Ps.I, Ps.J
            tril = [ k for k in range(len(I)) if I[k] >= J[k] ]
            task.putqobj(list(I[tril]), list(J[tril]), list(Ps.V[tril]))

            task.putobjsense(mosek.objsense.minimize)

            if taskfile:
                task.writetask(taskfile)

            task.optimize()

            task.solutionsummary (mosek.streamtype.msg);

            solsta = task.getsolsta(mosek.soltype.itr)

            x = n*[ 0.0 ]
            task.getsolutionslice(mosek.soltype.itr, mosek.solitem.xx, 0, n, x)
            x = matrix(x)

            if m is not 0:
                z = m*[0.0]
                task.getsolutionslice(mosek.soltype.itr, mosek.solitem.suc, 0, m,
                    z)
                z = matrix(z)
            else:
                z = matrix(0.0, (0,1))

            if p is not 0:
                yu, yl = p*[0.0], p*[0.0]
                task.getsolutionslice(mosek.soltype.itr, mosek.solitem.suc, m, m+p,
                    yu)
                task.getsolutionslice(mosek.soltype.itr, mosek.solitem.slc, m, m+p,
                    yl)
                y = matrix(yu) - matrix(yl)
            else:
                y = matrix(0.0, (0,1))

    if (solsta is mosek.solsta.unknown):
        return (solsta, None, None, None)
    else:
        return (solsta, x, z, y)


def ilp(c, G, h, A=None, b=None, I=None, taskfile=None, **kwargs):
    """
    Solves the mixed integer LP

        minimize    c'*x
        subject to  G*x + s = h
                    A*x = b
                    s >= 0
                    xi integer, forall i in I

    using MOSEK 8.

    solsta, x = ilp(c, G, h, A=None, b=None, I=None, taskfile=None).

    Input arguments

        G is m x n, h is m x 1, A is p x n, b is p x 1.  G and A must be
        dense or sparse 'd' matrices.   h and b are dense 'd' matrices
        with one column.  The default values for A and b are empty
        matrices with zero rows.

        I is a Python set with indices of integer elements of x.  By
        default all elements in x are constrained to be integer, i.e.,
        the default value of I is I = set(range(n))

        Dual variables are not returned for MOSEK.

        Optionally, the interface can write a .task file, required for
        support questions on the MOSEK solver.

    Return values

        solsta is a MOSEK solution status key.

            If solsta is mosek.solsta.integer_optimal, then x contains
                the solution.
            If solsta is mosek.solsta.unknown, then x is None.

            Other return values for solsta include:
                mosek.solsta.near_integer_optimal
            in which case the x value may not be well-defined,
            c.f., section 17.48 of the MOSEK Python API manual.

        x is the solution

    Options are passed to MOSEK solvers via the msk.options dictionary,
    e.g., the following turns off output from the MOSEK solvers

    >>> msk.options = {mosek.iparam.log: 0}

    see the MOSEK Python API manual.
    """

    with mosek.Env() as env:

        if type(c) is not matrix or c.typecode != 'd' or c.size[1] != 1:
            raise TypeError("'c' must be a dense column matrix")
        n = c.size[0]
        if n < 1: raise ValueError("number of variables must be at least 1")

        if (type(G) is not matrix and type(G) is not spmatrix) or \
            G.typecode != 'd' or G.size[1] != n:
            raise TypeError("'G' must be a dense or sparse 'd' matrix "\
                "with %d columns" %n)
        m = G.size[0]
        if m is 0: raise ValueError("m cannot be 0")

        if type(h) is not matrix or h.typecode != 'd' or h.size != (m,1):
            raise TypeError("'h' must be a 'd' matrix of size (%d,1)" %m)

        if A is None:  A = spmatrix([], [], [], (0,n), 'd')
        if (type(A) is not matrix and type(A) is not spmatrix) or \
            A.typecode != 'd' or A.size[1] != n:
            raise TypeError("'A' must be a dense or sparse 'd' matrix "\
                "with %d columns" %n)
        p = A.size[0]
        if b is None: b = matrix(0.0, (0,1))
        if type(b) is not matrix or b.typecode != 'd' or b.size != (p,1):
            raise TypeError("'b' must be a dense matrix of size (%d,1)" %p)

        if I is None: I = set(range(n))

        if type(I) is not set:
            raise TypeError("invalid argument for integer index set")

        for i in I:
            if type(i) is not int:
                raise TypeError("invalid integer index set I")

        if len(I) > 0 and min(I) < 0: raise IndexError(
                "negative element in integer index set I")
        if len(I) > 0 and max(I) > n-1: raise IndexError(
                "maximum element in in integer index set I is larger than n-1")

        bkc = m*[ mosek.boundkey.up ] + p*[ mosek.boundkey.fx ]
        blc = m*[ -inf ] + [ bi for bi in b ]
        buc = list(h) + list(b)

        bkx = n*[mosek.boundkey.fr]
        blx = n*[ -inf ]
        bux = n*[ +inf ]

        colptr, asub, acof = sparse([G,A]).CCS
        aptrb, aptre = colptr[:-1], colptr[1:]

        with env.Task(0,0) as task:
            task.set_Stream (mosek.streamtype.log, streamprinter)

            # set MOSEK options
            options = kwargs.get('options',globals()['options'])
            for (param, val) in options.items():
                if str(param)[:6] == "iparam":
                    task.putintparam(param, val)
                elif str(param)[:6] == "dparam":
                    task.putdouparam(param, val)
                elif str(param)[:6] == "sparam":
                    task.putstrparam(param, val)
                else:
                    raise ValueError("invalid MOSEK parameter: "+str(param))

            task.inputdata (m+p, # number of constraints
                            n,   # number of variables
                            list(c), # linear objective coefficients
                            0.0, # objective fixed value
                            list(aptrb),
                            list(aptre),
                            list(asub),
                            list(acof),
                            bkc,
                            blc,
                            buc,
                            bkx,
                            blx,
                            bux)

            task.putobjsense(mosek.objsense.minimize)

            # Define integer variables
            if len(I) > 0:
                task.putvartypelist(list(I), len(I)*[ mosek.variabletype.type_int ])

            task.putintparam (mosek.iparam.mio_mode, mosek.miomode.satisfied)

            if taskfile:
                task.writetask(taskfile)

            task.optimize()

            task.solutionsummary (mosek.streamtype.msg);

            if len(I) > 0:
                solsta = task.getsolsta(mosek.soltype.itg)
            else:
                solsta = task.getsolsta(mosek.soltype.bas)

            x = n*[0.0]
            if len(I) > 0:
                task.getsolutionslice(mosek.soltype.itg, mosek.solitem.xx, 0, n, x)
            else:
                task.getsolutionslice(mosek.soltype.bas, mosek.solitem.xx, 0, n, x)
            x = matrix(x)

    if (solsta is mosek.solsta.unknown):
        return (solsta, None)
    else:
        return (solsta, x)
