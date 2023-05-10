# Copyright 2012-2023 M. Andersen and L. Vandenberghe.
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

import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
__all__ = []

use_C = True

if use_C:
  scale = misc_solvers.scale
else:
  def scale(x, W, trans = 'N', inverse = 'N'):  
    """
    Applies Nesterov-Todd scaling or its inverse.
    
    Computes 
    
         x := W*x        (trans is 'N', inverse = 'N')  
         x := W^T*x      (trans is 'T', inverse = 'N')  
         x := W^{-1}*x   (trans is 'N', inverse = 'I')  
         x := W^{-T}*x   (trans is 'T', inverse = 'I'). 
    
    x is a dense 'd' matrix.
    
    W is a dictionary with entries:
    
    - W['dnl']: positive vector
    - W['dnli']: componentwise inverse of W['dnl']
    - W['d']: positive vector
    - W['di']: componentwise inverse of W['d']
    - W['v']: lists of 2nd order cone vectors with unit hyperbolic norms
    - W['beta']: list of positive numbers
    - W['r']: list of square matrices 
    - W['rti']: list of square matrices.  rti[k] is the inverse transpose
      of r[k].
    
    The 'dnl' and 'dnli' entries are optional, and only present when the 
    function is called from the nonlinear solver.
    """

    ind = 0

    # Scaling for nonlinear component xk is xk := dnl .* xk; inverse 
    # scaling is xk ./ dnl = dnli .* xk, where dnl = W['dnl'], 
    # dnli = W['dnli'].

    if 'dnl' in W:
        if inverse == 'N': w = W['dnl']
        else: w = W['dnli']
        for k in range(x.size[1]):
            blas.tbmv(w, x, n = w.size[0], k = 0, ldA = 1, offsetx = 
                k*x.size[0])
        ind += w.size[0]


    # Scaling for linear 'l' component xk is xk := d .* xk; inverse 
    # scaling is xk ./ d = di .* xk, where d = W['d'], di = W['di'].

    if inverse == 'N': w = W['d']
    else: w = W['di']
    for k in range(x.size[1]):
        blas.tbmv(w, x, n = w.size[0], k = 0, ldA = 1, offsetx = 
            k*x.size[0] + ind)
    ind += w.size[0]
  

    # Scaling for 'q' component is 
    #
    #     xk := beta * (2*v*v' - J) * xk
    #         = beta * (2*v*(xk'*v)' - J*xk)
    #
    # where beta = W['beta'][k], v = W['v'][k], J = [1, 0; 0, -I].
    #
    # Inverse scaling is
    #
    #     xk := 1/beta * (2*J*v*v'*J - J) * xk
    #         = 1/beta * (-J) * (2*v*((-J*xk)'*v)' + xk). 

    w = matrix(0.0, (x.size[1], 1))
    for k in range(len(W['v'])):
        v = W['v'][k]
        m = v.size[0]
        if inverse == 'I':  
            blas.scal(-1.0, x, offset = ind, inc = x.size[0])
        blas.gemv(x, v, w, trans = 'T', m = m, n = x.size[1], offsetA = 
            ind, ldA = x.size[0])
        blas.scal(-1.0, x, offset = ind, inc = x.size[0])
        blas.ger(v, w, x, alpha = 2.0, m = m, n = x.size[1], ldA = 
            x.size[0], offsetA = ind)
        if inverse == 'I': 
            blas.scal(-1.0, x, offset = ind, inc = x.size[0])
            a = 1.0 / W['beta'][k] 
        else:
            a = W['beta'][k] 
        for i in range(x.size[1]):
            blas.scal(a, x, n = m, offset = ind + i*x.size[0])
        ind += m


    # Scaling for 's' component xk is
    #
    #     xk := vec( r' * mat(xk) * r )  if trans = 'N'
    #     xk := vec( r * mat(xk) * r' )  if trans = 'T'.
    #
    # r is kth element of W['r'].
    #
    # Inverse scaling is
    #
    #     xk := vec( rti * mat(xk) * rti' )  if trans = 'N'
    #     xk := vec( rti' * mat(xk) * rti )  if trans = 'T'.
    #
    # rti is kth element of W['rti'].

    maxn = max( [0] + [ r.size[0] for r in W['r'] ] )
    a = matrix(0.0, (maxn, maxn))
    for k in range(len(W['r'])):

        if inverse == 'N':
            r = W['r'][k]
            if trans == 'N': t = 'T'
            else: t = 'N'
        else:
            r = W['rti'][k]
            t = trans

        n = r.size[0]
        for i in range(x.size[1]):

            # scale diagonal of xk by 0.5
            blas.scal(0.5, x, offset = ind + i*x.size[0], inc = n+1, n = n)

            # a = r*tril(x) (t is 'N') or a = tril(x)*r  (t is 'T')
            blas.copy(r, a)
            if t == 'N':   
                blas.trmm(x, a, side = 'R', m = n, n = n, ldA = n, ldB = n,
                    offsetA = ind + i*x.size[0])
            else:    
                blas.trmm(x, a, side = 'L', m = n, n = n, ldA = n, ldB = n,
                    offsetA = ind + i*x.size[0])
 
            # x := (r*a' + a*r')  if t is 'N'
            # x := (r'*a + a'*r)  if t is 'T'
            blas.syr2k(r, a, x, trans = t, n = n, k = n, ldB = n, ldC = n,
                offsetC = ind + i*x.size[0])
 
        ind += n**2


if use_C:
  scale2 = misc_solvers.scale2
else:
  def scale2(lmbda, x, dims, mnl = 0, inverse = 'N'):
    """
    Evaluates

        x := H(lambda^{1/2}) * x   (inverse is 'N')
        x := H(lambda^{-1/2}) * x  (inverse is 'I').
    
    H is the Hessian of the logarithmic barrier.
    """
      

    # For the nonlinear and 'l' blocks, 
    #
    #     xk := xk ./ l   (inverse is 'N')
    #     xk := xk .* l   (inverse is 'I')
    #
    # where l is lmbda[:mnl+dims['l']].

    if inverse == 'N':
        blas.tbsv(lmbda, x, n = mnl + dims['l'], k = 0, ldA = 1)
    else:
        blas.tbmv(lmbda, x, n = mnl + dims['l'], k = 0, ldA = 1)
   
  
    # For 'q' blocks, if inverse is 'N',
    #
    #     xk := 1/a * [ l'*J*xk;  
    #         xk[1:] - (xk[0] + l'*J*xk) / (l[0] + 1) * l[1:] ].
    #
    # If inverse is 'I',
    #
    #     xk := a * [ l'*xk; 
    #         xk[1:] + (xk[0] + l'*xk) / (l[0] + 1) * l[1:] ].
    #
    # a = sqrt(lambda_k' * J * lambda_k), l = lambda_k / a.

    ind = mnl + dims['l']
    for m in dims['q']:
        a = jnrm2(lmbda, n = m, offset = ind)
        if inverse == 'N':
            lx = jdot(lmbda, x, n = m, offsetx = ind, offsety = ind)/a
        else:
            lx = blas.dot(lmbda, x, n = m, offsetx = ind, offsety = ind)/a
        x0 = x[ind]
        x[ind] = lx
        c = (lx + x0) / (lmbda[ind]/a + 1) / a 
        if inverse == 'N':  c *= -1.0
        blas.axpy(lmbda, x, alpha = c, n = m-1, offsetx = ind+1, offsety =
            ind+1)
        if inverse == 'N': a = 1.0/a 
        blas.scal(a, x, offset = ind, n = m)
        ind += m
        

    # For the 's' blocks, if inverse is 'N',
    #
    #     xk := vec( diag(l)^{-1/2} * mat(xk) * diag(l)^{-1/2}).
    #
    # If inverse is 'I',
    #
    #     xk := vec( diag(l)^{1/2} * mat(xk) * diag(l)^{1/2}).
    #
    # where l is kth block of lambda.
    # 
    # We scale upper and lower triangular part of mat(xk) because the
    # inverse operation will be applied to nonsymmetric matrices.

    ind2 = ind
    for k in range(len(dims['s'])):
        m = dims['s'][k]
        for j in range(m):
            c = math.sqrt(lmbda[ind2+j]) * base.sqrt(lmbda[ind2:ind2+m])
            if inverse == 'N':  
                blas.tbsv(c, x, n = m, k = 0, ldA = 1, offsetx = ind + j*m)
            else:
                blas.tbmv(c, x, n = m, k = 0, ldA = 1, offsetx = ind + j*m)
        ind += m*m
        ind2 += m


def compute_scaling(s, z, lmbda, dims, mnl = None):
    """
    Returns the Nesterov-Todd scaling W at points s and z, and stores the 
    scaled variable in lmbda. 
    
        W * z = W^{-T} * s = lmbda. 

    """
     
    W = {}

    # For the nonlinear block:
    #
    #     W['dnl'] = sqrt( s[:mnl] ./ z[:mnl] )
    #     W['dnli'] = sqrt( z[:mnl] ./ s[:mnl] )
    #     lambda[:mnl] = sqrt( s[:mnl] .* z[:mnl] )

    if mnl is None:
        mnl = 0
    else:
        W['dnl'] = base.sqrt( base.div( s[:mnl], z[:mnl] ))
        W['dnli'] = W['dnl']**-1
        lmbda[:mnl] = base.sqrt( base.mul( s[:mnl], z[:mnl] ) ) 
        

    # For the 'l' block: 
    #
    #     W['d'] = sqrt( sk ./ zk )
    #     W['di'] = sqrt( zk ./ sk )
    #     lambdak = sqrt( sk .* zk )
    #
    # where sk and zk are the first dims['l'] entries of s and z.
    # lambda_k is stored in the first dims['l'] positions of lmbda.
             
    m = dims['l']
    W['d'] = base.sqrt( base.div( s[mnl:mnl+m], z[mnl:mnl+m] ))
    W['di'] = W['d']**-1
    lmbda[mnl:mnl+m] = base.sqrt( base.mul( s[mnl:mnl+m], z[mnl:mnl+m] ) ) 


    # For the 'q' blocks, compute lists 'v', 'beta'.
    #
    # The vector v[k] has unit hyperbolic norm: 
    # 
    #     (sqrt( v[k]' * J * v[k] ) = 1 with J = [1, 0; 0, -I]).
    # 
    # beta[k] is a positive scalar.
    #
    # The hyperbolic Householder matrix H = 2*v[k]*v[k]' - J
    # defined by v[k] satisfies 
    # 
    #     (beta[k] * H) * zk  = (beta[k] * H) \ sk = lambda_k
    #
    # where sk = s[indq[k]:indq[k+1]], zk = z[indq[k]:indq[k+1]].
    #
    # lambda_k is stored in lmbda[indq[k]:indq[k+1]].
           
    ind = mnl + dims['l']
    W['v'] = [ matrix(0.0, (k,1)) for k in dims['q'] ]
    W['beta'] = len(dims['q']) * [ 0.0 ] 

    for k in range(len(dims['q'])):
        m = dims['q'][k]
        v = W['v'][k]

        # a = sqrt( sk' * J * sk )  where J = [1, 0; 0, -I]
        aa = jnrm2(s, offset = ind, n = m)

        # b = sqrt( zk' * J * zk )
        bb = jnrm2(z, offset = ind, n = m) 

        # beta[k] = ( a / b )**1/2
        W['beta'][k] = math.sqrt( aa / bb )

        # c = sqrt( (sk/a)' * (zk/b) + 1 ) / sqrt(2)    
        cc = math.sqrt( ( blas.dot(s, z, n = m, offsetx = ind, offsety = 
            ind) / aa / bb + 1.0 ) / 2.0 )

        # vk = 1/(2*c) * ( (sk/a) + J * (zk/b) )
        blas.copy(z, v, offsetx = ind, n = m)
        blas.scal(-1.0/bb, v)
        v[0] *= -1.0 
        blas.axpy(s, v, 1.0/aa, offsetx = ind, n = m)
        blas.scal(1.0/2.0/cc, v)

        # v[k] = 1/sqrt(2*(vk0 + 1)) * ( vk + e ),  e = [1; 0]
        v[0] += 1.0
        blas.scal(1.0/math.sqrt(2.0 * v[0]), v)
            
        # To get the scaled variable lambda_k
        # 
        #     d =  sk0/a + zk0/b + 2*c
        #     lambda_k = [ c; 
        #                  (c + zk0/b)/d * sk1/a + (c + sk0/a)/d * zk1/b ]
        #     lambda_k *= sqrt(a * b)

        lmbda[ind] = cc
        dd = 2*cc + s[ind]/aa + z[ind]/bb
        blas.copy(s, lmbda, offsetx = ind+1, offsety = ind+1, n = m-1) 
        blas.scal((cc + z[ind]/bb)/dd/aa, lmbda, n = m-1, offset = ind+1)
        blas.axpy(z, lmbda, (cc + s[ind]/aa)/dd/bb, n = m-1, offsetx = 
            ind+1, offsety = ind+1)
        blas.scal(math.sqrt(aa*bb), lmbda, offset = ind, n = m)

        ind += m


    # For the 's' blocks: compute two lists 'r' and 'rti'.
    #
    #     r[k]' * sk^{-1} * r[k] = diag(lambda_k)^{-1}
    #     r[k]' * zk * r[k] = diag(lambda_k)
    #
    # where sk and zk are the entries inds[k] : inds[k+1] of
    # s and z, reshaped into symmetric matrices.
    #
    # rti[k] is the inverse of r[k]', so 
    #
    #     rti[k]' * sk * rti[k] = diag(lambda_k)^{-1}
    #     rti[k]' * zk^{-1} * rti[k] = diag(lambda_k).
    #
    # The vectors lambda_k are stored in 
    # 
    #     lmbda[ dims['l'] + sum(dims['q']) : -1 ]
            
    W['r'] = [ matrix(0.0, (m,m)) for m in dims['s'] ]
    W['rti'] = [ matrix(0.0, (m,m)) for m in dims['s'] ]
    work = matrix(0.0, (max( [0] + dims['s'] )**2, 1))
    Ls = matrix(0.0, (max( [0] + dims['s'] )**2, 1))
    Lz = matrix(0.0, (max( [0] + dims['s'] )**2, 1))

    ind2 = ind
    for k in range(len(dims['s'])):
        m = dims['s'][k]
        r, rti = W['r'][k], W['rti'][k]

        # Factor sk = Ls*Ls'; store Ls in ds[inds[k]:inds[k+1]].
        blas.copy(s, Ls, offsetx = ind2, n = m**2) 
        lapack.potrf(Ls, n = m, ldA = m)

        # Factor zs[k] = Lz*Lz'; store Lz in dz[inds[k]:inds[k+1]].
        blas.copy(z, Lz, offsetx = ind2, n = m**2) 
        lapack.potrf(Lz, n = m, ldA = m)
	 
        # SVD Lz'*Ls = U*diag(lambda_k)*V'.  Keep U in work. 
        for i in range(m): 
            blas.scal(0.0, Ls, offset = i*m, n = i)
        blas.copy(Ls, work, n = m**2)
        blas.trmm(Lz, work, transA = 'T', ldA = m, ldB = m, n = m, m = m) 
        lapack.gesvd(work, lmbda, jobu = 'O', ldA = m, m = m, n = m, 
            offsetS = ind)
	       
        # r = Lz^{-T} * U 
        blas.copy(work, r, n = m*m)
        blas.trsm(Lz, r, transA = 'T', m = m, n = m, ldA = m)

        # rti = Lz * U 
        blas.copy(work, rti, n = m*m)
        blas.trmm(Lz, rti, m = m, n = m, ldA = m)

        # r := r * diag(sqrt(lambda_k))
        # rti := rti * diag(1 ./ sqrt(lambda_k))
        for i in range(m):
            a = math.sqrt( lmbda[ind+i] )
            blas.scal(a, r, offset = m*i, n = m)
            blas.scal(1.0/a, rti, offset = m*i, n = m)

        ind += m
        ind2 += m*m

    return W


def update_scaling(W, lmbda, s, z):
    """
    Updates the Nesterov-Todd scaling matrix W and the scaled variable 
    lmbda so that on exit
    
          W * zt = W^{-T} * st = lmbda.
     
    On entry, the nonlinear, 'l' and 'q' components of the arguments s 
    and z contain W^{-T}*st and W*zt, i.e, the new iterates in the current 
    scaling.
    
    The 's' components contain the factors Ls, Lz in a factorization of 
    the new iterates in the current scaling, W^{-T}*st = Ls*Ls',   
    W*zt = Lz*Lz'.
    """
  

    # Nonlinear and 'l' blocks
    #
    #    d :=  d .* sqrt( s ./ z )
    #    lmbda := lmbda .* sqrt(s) .* sqrt(z)

    if 'dnl' in W:
        mnl = len(W['dnl'])
    else:
        mnl = 0
    ml = len(W['d'])
    m = mnl + ml
    s[:m] = base.sqrt( s[:m] )
    z[:m] = base.sqrt( z[:m] )
 
    # d := d .* s .* z 
    if 'dnl' in W:
        blas.tbmv(s, W['dnl'], n = mnl, k = 0, ldA = 1)
        blas.tbsv(z, W['dnl'], n = mnl, k = 0, ldA = 1)
        W['dnli'][:] = W['dnl'][:] ** -1
    blas.tbmv(s, W['d'], n = ml, k = 0, ldA = 1, offsetA = mnl)
    blas.tbsv(z, W['d'], n = ml, k = 0, ldA = 1, offsetA = mnl)
    W['di'][:] = W['d'][:] ** -1
         
    # lmbda := s .* z
    blas.copy(s, lmbda, n = m)
    blas.tbmv(z, lmbda, n = m, k = 0, ldA = 1)


    # 'q' blocks.
    # 
    # Let st and zt be the new variables in the old scaling:
    #
    #     st = s_k,   zt = z_k
    #
    # and a = sqrt(st' * J * st),  b = sqrt(zt' * J * zt).
    #
    # 1. Compute the hyperbolic Householder transformation 2*q*q' - J 
    #    that maps st/a to zt/b.
    # 
    #        c = sqrt( (1 + st'*zt/(a*b)) / 2 ) 
    #        q = (st/a + J*zt/b) / (2*c). 
    #
    #    The new scaling point is 
    #
    #        wk := betak * sqrt(a/b) * (2*v[k]*v[k]' - J) * q 
    #
    #    with betak = W['beta'][k].
    # 
    # 3. The scaled variable:
    #
    #        lambda_k0 = sqrt(a*b) * c
    #        lambda_k1 = sqrt(a*b) * ( (2vk*vk' - J) * (-d*q + u/2) )_1
    #
    #    where 
    #
    #        u = st/a - J*zt/b 
    #        d = ( vk0 * (vk'*u) + u0/2 ) / (2*vk0 *(vk'*q) - q0 + 1).
    #
    # 4. Update scaling
    #   
    #        v[k] := wk^1/2 
    #              = 1 / sqrt(2*(wk0 + 1)) * (wk + e).
    #        beta[k] *=  sqrt(a/b)

    ind = m
    for k in range(len(W['v'])):

        v = W['v'][k]
        m = len(v)

        # ln = sqrt( lambda_k' * J * lambda_k )
        ln = jnrm2(lmbda, n = m, offset = ind) 

        # a = sqrt( sk' * J * sk ) = sqrt( st' * J * st ) 
        # s := s / a = st / a
        aa = jnrm2(s, offset = ind, n = m)
        blas.scal(1.0/aa, s, offset = ind, n = m)

        # b = sqrt( zk' * J * zk ) = sqrt( zt' * J * zt )
        # z := z / a = zt / b
        bb = jnrm2(z, offset = ind, n = m) 
        blas.scal(1.0/bb, z, offset = ind, n = m)

        # c = sqrt( ( 1 + (st'*zt) / (a*b) ) / 2 )
        cc = math.sqrt( ( 1.0 + blas.dot(s, z, offsetx = ind, offsety = 
            ind, n = m) ) / 2.0 )

        # vs = v' * st / a 
        vs = blas.dot(v, s, offsety = ind, n = m) 

        # vz = v' * J *zt / b
        vz = jdot(v, z, offsety = ind, n = m) 

        # vq = v' * q where q = (st/a + J * zt/b) / (2 * c)
        vq = (vs + vz ) / 2.0 / cc

        # vu = v' * u  where u =  st/a - J * zt/b 
        vu = vs - vz  

        # lambda_k0 = c
        lmbda[ind] = cc

        # wk0 = 2 * vk0 * (vk' * q) - q0 
        wk0 = 2 * v[0] * vq - ( s[ind] + z[ind] ) / 2.0 / cc 

        # d = (v[0] * (vk' * u) - u0/2) / (wk0 + 1)
        dd = (v[0] * vu - s[ind]/2.0 + z[ind]/2.0) / (wk0 + 1.0)

        # lambda_k1 = 2 * v_k1 * vk' * (-d*q + u/2) - d*q1 + u1/2
        blas.copy(v, lmbda, offsetx = 1, offsety = ind+1, n = m-1)
        blas.scal(2.0 * (-dd * vq + 0.5 * vu), lmbda, offset = ind+1, 
           n = m-1)
        blas.axpy(s, lmbda, 0.5 * (1.0 - dd/cc), offsetx = ind+1, offsety 
           = ind+1, n = m-1)
        blas.axpy(z, lmbda, 0.5 * (1.0 + dd/cc), offsetx = ind+1, offsety
           = ind+1, n = m-1)

        # Scale so that sqrt(lambda_k' * J * lambda_k) = sqrt(aa*bb).
        blas.scal(math.sqrt(aa*bb), lmbda, offset = ind, n = m)
            
        # v := (2*v*v' - J) * q 
        #    = 2 * (v'*q) * v' - (J* st/a + zt/b) / (2*c)
        blas.scal(2.0 * vq, v)
        v[0] -= s[ind] / 2.0 / cc
        blas.axpy(s, v,  0.5/cc, offsetx = ind+1, offsety = 1, n = m-1)
        blas.axpy(z, v, -0.5/cc, offsetx = ind, n = m)

        # v := v^{1/2} = 1/sqrt(2 * (v0 + 1)) * (v + e)
        v[0] += 1.0
        blas.scal(1.0 / math.sqrt(2.0 * v[0]), v)

        # beta[k] *= ( aa / bb )**1/2
        W['beta'][k] *= math.sqrt( aa / bb )
            
        ind += m


    # 's' blocks
    # 
    # Let st, zt be the updated variables in the old scaling:
    # 
    #     st = Ls * Ls', zt = Lz * Lz'.
    #
    # where Ls and Lz are the 's' components of s, z.
    #
    # 1.  SVD Lz'*Ls = Uk * lambda_k^+ * Vk'.
    #
    # 2.  New scaling is 
    #
    #         r[k] := r[k] * Ls * Vk * diag(lambda_k^+)^{-1/2}
    #         rti[k] := r[k] * Lz * Uk * diag(lambda_k^+)^{-1/2}.
    #

    work = matrix(0.0, (max( [0] + [r.size[0] for r in W['r']])**2, 1))
    ind = mnl + ml + sum([ len(v) for v in W['v'] ])
    ind2, ind3 = ind, 0
    for k in range(len(W['r'])):
        r, rti = W['r'][k], W['rti'][k]
        m = r.size[0]

        # r := r*sk = r*Ls
        blas.gemm(r, s, work, m = m, n = m, k = m, ldB = m, ldC = m,
            offsetB = ind2)
        blas.copy(work, r, n = m**2)

        # rti := rti*zk = rti*Lz
        blas.gemm(rti, z, work, m = m, n = m, k = m, ldB = m, ldC = m,
            offsetB = ind2)
        blas.copy(work, rti, n = m**2)

        # SVD Lz'*Ls = U * lmbds^+ * V'; store U in sk and V' in zk.
        blas.gemm(z, s, work, transA = 'T', m = m, n = m, k = m, ldA = m,
            ldB = m, ldC = m, offsetA = ind2, offsetB = ind2)
        lapack.gesvd(work, lmbda, jobu = 'A', jobvt = 'A', m = m, n = m, 
            ldA = m, U = s, Vt = z, ldU = m, ldVt = m, offsetS = ind, 
            offsetU = ind2, offsetVt = ind2)

        # r := r*V
        blas.gemm(r, z, work, transB = 'T', m = m, n = m, k = m, ldB = m,
            ldC = m, offsetB = ind2)
        blas.copy(work, r, n = m**2)

        # rti := rti*U
        blas.gemm(rti, s, work, n = m, m = m, k = m, ldB = m, ldC = m,
            offsetB = ind2)
        blas.copy(work, rti, n = m**2)

        # r := r*lambda^{-1/2}; rti := rti*lambda^{-1/2}
        for i in range(m):    
            a = 1.0 / math.sqrt(lmbda[ind+i])
            blas.scal(a, r, offset = m*i, n = m)
            blas.scal(a, rti, offset = m*i, n = m)

        ind += m
        ind2 += m*m
        ind3 += m


if use_C:
  pack = misc_solvers.pack
else:
  def pack(x, y, dims, mnl = 0, offsetx = 0, offsety = 0):
     """
     Copy x to y using packed storage.
    
     The vector x is an element of S, with the 's' components stored in 
     unpacked storage.  On return, x is copied to y with the 's' components
     stored in packed storage and the off-diagonal entries scaled by 
     sqrt(2).
     """

     nlq = mnl + dims['l'] + sum(dims['q'])
     blas.copy(x, y, n = nlq, offsetx = offsetx, offsety = offsety)
     iu, ip = offsetx + nlq, offsety + nlq
     for n in dims['s']:
         for k in range(n):
             blas.copy(x, y, n = n-k, offsetx = iu + k*(n+1), offsety = ip)
             y[ip] /= math.sqrt(2)
             ip += n-k
         iu += n**2 
     np = sum([ int(n*(n+1)/2) for n in dims['s'] ])
     blas.scal(math.sqrt(2.0), y, n = np, offset = offsety+nlq)
     

if use_C:
  pack2 = misc_solvers.pack2
else:
  def pack2(x, dims, mnl = 0):
     """
     In-place version of pack(), which also accepts matrix arguments x.  
     The columns of x are elements of S, with the 's' components stored in
     unpacked storage.  On return, the 's' components are stored in packed
     storage and the off-diagonal entries are scaled by sqrt(2).
     """

     if not dims['s']: return
     iu = mnl + dims['l'] + sum(dims['q'])
     ip = iu
     for n in dims['s']:
         for k in range(n):
             x[ip, :] = x[iu + (n+1)*k, :]
             x[ip + 1 : ip+n-k, :] = x[iu + (n+1)*k + 1: iu + n*(k+1), :] \
                 * math.sqrt(2.0)
             ip += n - k
         iu += n**2 
     np = sum([ int(n*(n+1)/2) for n in dims['s'] ])
     

if use_C:
  unpack = misc_solvers.unpack
else:
  def unpack(x, y, dims, mnl = 0, offsetx = 0, offsety = 0):
     """
     The vector x is an element of S, with the 's' components stored
     in unpacked storage and off-diagonal entries scaled by sqrt(2).
     On return, x is copied to y with the 's' components stored in 
     unpacked storage.
     """

     nlq = mnl + dims['l'] + sum(dims['q'])
     blas.copy(x, y, n = nlq, offsetx = offsetx, offsety = offsety)
     iu, ip = offsety+nlq, offsetx+nlq
     for n in dims['s']:
         for k in range(n):
             blas.copy(x, y, n = n-k, offsetx = ip, offsety = iu+k*(n+1))
             y[iu+k*(n+1)] *= math.sqrt(2)
             ip += n-k
         iu += n**2 
     nu = sum([ n**2 for n in dims['s'] ])
     blas.scal(1.0/math.sqrt(2.0), y, n = nu, offset = offsety+nlq)


if use_C:
  sdot = misc_solvers.sdot
else:
  def sdot(x, y, dims, mnl = 0):
    """
    Inner product of two vectors in S.
    """
    
    ind = mnl + dims['l'] + sum(dims['q'])
    a = blas.dot(x, y, n = ind)
    for m in dims['s']:
        a += blas.dot(x, y, offsetx = ind, offsety = ind, incx = m+1, 
            incy = m+1, n = m)
        for j in range(1, m):
            a += 2.0 * blas.dot(x, y, incx = m+1, incy = m+1, 
                offsetx = ind+j, offsety = ind+j, n = m-j)
        ind += m**2
    return a


def sdot2(x, y):
    """
    Inner product of two block-diagonal symmetric dense 'd' matrices.

    x and y are square dense 'd' matrices, or lists of N square dense 'd' 
    matrices.
    """

    a = 0.0
    if type(x) is matrix:
        n = x.size[0]
        a += blas.dot(x, y, incx=n+1, incy=n+1, n=n)
        for j in range(1,n):
            a += 2.0 * blas.dot(x, y, incx=n+1, incy=n+1, offsetx=j,
                offsety=j, n=n-j)

    else:
        for k in range(len(x)):
            n = x[k].size[0]
            a += blas.dot(x[k], y[k], incx=n+1, incy=n+1, n=n)
            for j in range(1,n):
                a += 2.0 * blas.dot(x[k], y[k], incx=n+1, incy=n+1, 
                    offsetx=j, offsety=j, n=n-j)
    return a


def snrm2(x, dims, mnl = 0): 
    """
    Returns the norm of a vector in S
    """

    return math.sqrt(sdot(x, x, dims, mnl))


if use_C:
  trisc = misc_solvers.trisc
else:
  def trisc(x, dims, offset = 0): 
    """
    Sets upper triangular part of the 's' components of x equal to zero
    and scales the strictly lower triangular part by 2.0.
    """ 

    m = dims['l'] + sum(dims['q']) + sum([ k**2 for k in dims['s'] ]) 
    ind = offset + dims['l'] + sum(dims['q'])
    for mk in dims['s']:
        for j in range(1, mk):  
            blas.scal(0.0, x, n = mk-j, inc = mk, offset = 
                    ind + j*(mk + 1) - 1) 
            blas.scal(2.0, x, offset = ind + mk*(j-1) + j, n = mk-j) 
        ind += mk**2


if use_C:
  triusc = misc_solvers.triusc
else:
  def triusc(x, dims, offset = 0):
    """
    Scales the strictly lower triangular part of the 's' components of x 
    by 0.5.
    """ 

    m = dims['l'] + sum(dims['q']) + sum([ k**2 for k in dims['s'] ]) 
    ind = offset + dims['l'] + sum(dims['q'])
    for mk in dims['s']:
        for j in range(1, mk):  
                blas.scal(0.5, x, offset = ind + mk*(j-1) + j, n = mk-j) 
        ind += mk**2


def sgemv(A, x, y, dims, trans = 'N', alpha = 1.0, beta = 0.0, n = None, 
    offsetA = 0, offsetx = 0, offsety = 0): 
    """
    Matrix-vector multiplication.

    A is a matrix or spmatrix of size (m, n) where 
    
        N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ) 

    representing a mapping from R^n to S.  
    
    If trans is 'N': 
    
        y := alpha*A*x + beta * y   (trans = 'N').
    
    x is a vector of length n.  y is a vector of length N.
    
    If trans is 'T':
    
        y := alpha*A'*x + beta * y  (trans = 'T').
    
    x is a vector of length N.  y is a vector of length n.
    
    The 's' components in S are stored in unpacked 'L' storage.
    """

    m = dims['l'] + sum(dims['q']) + sum([ k**2 for k in dims['s'] ]) 
    if n is None: n = A.size[1]
    if trans == 'T' and alpha:  trisc(x, dims, offsetx)
    base.gemv(A, x, y, trans = trans, alpha = alpha, beta = beta, m = m,
        n = n, offsetA = offsetA, offsetx = offsetx, offsety = offsety)
    if trans == 'T' and alpha: triusc(x, dims, offsetx)


def jdot(x, y, n = None, offsetx = 0, offsety = 0):
    """
    Returns x' * J * y, where J = [1, 0; 0, -I].
    """

    if n is None: 
        if len(x) != len(y): raise ValueError("x and y must have the "\
             "same length")
        n = len(x)
    return x[offsetx] * y[offsety] - blas.dot(x, y, n = n-1, 
        offsetx = offsetx + 1, offsety = offsety + 1) 


def jnrm2(x, n = None, offset = 0):
    """
    Returns sqrt(x' * J * x) where J = [1, 0; 0, -I], for a vector
    x in a second order cone. 
    """

    if n is None:  n = len(x)
    a = blas.nrm2(x, n = n-1, offset = offset+1)
    return math.sqrt(x[offset] - a) * math.sqrt(x[offset] + a)


if use_C:
  symm = misc_solvers.symm
else:
  def symm(x, n, offset = 0):
    """
    Converts lower triangular matrix to symmetric.  
    Fills in the upper triangular part of the symmetric matrix stored in 
    x[offset : offset+n*n] using 'L' storage.
    """

    if n <= 1:  return
    for i in range(n-1):
        blas.copy(x, x, offsetx = offset + i*(n+1) + 1, offsety = 
            offset + (i+1)*(n+1) - 1, incy = n, n = n-i-1)


if use_C:
  sprod = misc_solvers.sprod
else:
  def sprod(x, y, dims, mnl = 0, diag = 'N'):   
    """
    The product x := (y o x).  If diag is 'D', the 's' part of y is 
    diagonal and only the diagonal is stored.
    """


    # For the nonlinear and 'l' blocks:  
    #
    #     yk o xk = yk .* xk.

    blas.tbmv(y, x, n = mnl + dims['l'], k = 0, ldA = 1) 


    # For 'q' blocks: 
    #
    #               [ l0   l1'  ]
    #     yk o xk = [           ] * xk
    #               [ l1   l0*I ] 
    #
    # where yk = (l0, l1).
    
    ind = mnl + dims['l']
    for m in dims['q']:
        dd = blas.dot(x, y, offsetx = ind, offsety = ind, n = m)
        blas.scal(y[ind], x, offset = ind+1, n = m-1)
        blas.axpy(y, x, alpha = x[ind], n = m-1, offsetx = ind+1, offsety 
            = ind+1)
        x[ind] = dd
        ind += m


    # For the 's' blocks:
    #
    #    yk o sk = .5 * ( Yk * mat(xk) + mat(xk) * Yk )
    # 
    # where Yk = mat(yk) if diag is 'N' and Yk = diag(yk) if diag is 'D'.

    if diag == 'N':
        maxm = max([0] + dims['s'])
        A = matrix(0.0, (maxm, maxm))

        for m in dims['s']:
            blas.copy(x, A, offsetx = ind, n = m*m)

            # Write upper triangular part of A and yk.
            for i in range(m-1):
                symm(A, m)
                symm(y, m, offset = ind)

            # xk = 0.5 * (A*yk + yk*A)
            blas.syr2k(A, y, x, alpha = 0.5, n = m, k = m, ldA = m,  ldB = 
                m, ldC = m, offsetB = ind, offsetC = ind)

            ind += m*m

    else:
        ind2 = ind
        for m in dims['s']:
            for j in range(m):
                u = 0.5 * ( y[ind2+j:ind2+m] + y[ind2+j] )
                blas.tbmv(u, x, n = m-j, k = 0, ldA = 1, offsetx = 
                    ind + j*(m+1))  
            ind += m*m
            ind2 += m


def ssqr(x, y, dims, mnl = 0):
    """
    The product x := y o y.   The 's' components of y are diagonal and
    only the diagonals of x and y are stored.     
    """

    blas.copy(y, x)
    blas.tbmv(y, x, n = mnl + dims['l'], k = 0, ldA = 1) 
    ind = mnl + dims['l']
    for m in dims['q']:
        x[ind] = blas.nrm2(y, offset = ind, n = m)**2
        blas.scal(2.0*y[ind], x, n = m-1, offset = ind+1)
        ind += m 
    blas.tbmv(y, x, n = sum(dims['s']), k = 0, ldA = 1, offsetA = ind, 
        offsetx = ind) 


if use_C:
  sinv = misc_solvers.sinv
else:
  def sinv(x, y, dims, mnl = 0):   
    r"""
    The inverse product x := (y o\ x), when the 's' components of y are 
    diagonal.
    """
    
    # For the nonlinear and 'l' blocks:  
    # 
    #     yk o\ xk = yk .\ xk.

    blas.tbsv(y, x, n = mnl + dims['l'], k = 0, ldA = 1)


    # For the 'q' blocks: 
    #
    #                        [ l0   -l1'              ]  
    #     yk o\ xk = 1/a^2 * [                        ] * xk
    #                        [ -l1  (a*I + l1*l1')/l0 ]
    #
    # where yk = (l0, l1) and a = l0^2 - l1'*l1.

    ind = mnl + dims['l']
    for m in dims['q']:
        aa = jnrm2(y, n = m, offset = ind)**2
        cc = x[ind]
        dd = blas.dot(y, x, offsetx = ind+1, offsety = ind+1, n = m-1)
        x[ind] = cc * y[ind] - dd
        blas.scal(aa / y[ind], x, n = m-1, offset = ind+1)
        blas.axpy(y, x, alpha = dd/y[ind] - cc, n = m-1, offsetx = ind+1, 
            offsety = ind+1)
        blas.scal(1.0/aa, x, n = m, offset = ind)
        ind += m


    # For the 's' blocks:
    #
    #     yk o\ xk =  xk ./ gamma
    #
    # where gammaij = .5 * (yk_i + yk_j).

    ind2 = ind
    for m in dims['s']:
        for j in range(m):
            u = 0.5 * ( y[ind2+j:ind2+m] + y[ind2+j] )
            blas.tbsv(u, x, n = m-j, k = 0, ldA = 1, offsetx = ind + 
                j*(m+1))  
        ind += m*m
        ind2 += m


if use_C:
  max_step = misc_solvers.max_step
else:
  def max_step(x, dims, mnl = 0, sigma = None):
    """
    Returns min {t | x + t*e >= 0}, where e is defined as follows
    
    - For the nonlinear and 'l' blocks: e is the vector of ones.
    - For the 'q' blocks: e is the first unit vector.
    - For the 's' blocks: e is the identity matrix.
    
    When called with the argument sigma, also returns the eigenvalues 
    (in sigma) and the eigenvectors (in x) of the 's' components of x.
    """

    t = []
    ind = mnl + dims['l']
    if ind: t += [ -min(x[:ind]) ] 
    for m in dims['q']:
        if m: t += [ blas.nrm2(x, offset = ind + 1, n = m-1) - x[ind] ]
        ind += m
    if sigma is None and dims['s']:  
        Q = matrix(0.0, (max(dims['s']), max(dims['s'])))
        w = matrix(0.0, (max(dims['s']),1))
    ind2 = 0
    for m in dims['s']:
        if sigma is None:
            blas.copy(x, Q, offsetx = ind, n = m**2)
            lapack.syevr(Q, w, range = 'I', il = 1, iu = 1, n = m, ldA = m)
            if m:  t += [ -w[0] ]
        else:            
            lapack.syevd(x, sigma, jobz = 'V', n = m, ldA = m, offsetA = 
                ind, offsetW = ind2)
            if m:  t += [ -sigma[ind2] ] 
        ind += m*m
        ind2 += m
    if t: return max(t)
    else: return 0.0


def kkt_ldl(G, dims, A, mnl = 0, kktreg = None):
    """
    Solution of KKT equations by a dense LDL factorization of the 
    3 x 3 system.
    
    Returns a function that (1) computes the LDL factorization of
    
        [ H           A'   GG'*W^{-1} ] 
        [ A           0    0          ],
        [ W^{-T}*GG   0   -I          ] 
    
    given H, Df, W, where GG = [Df; G], and (2) returns a function for 
    solving 
    
        [ H     A'   GG'   ]   [ ux ]   [ bx ]
        [ A     0    0     ] * [ uy ] = [ by ].
        [ GG    0   -W'*W  ]   [ uz ]   [ bz ]
    
    H is n x n,  A is p x n, Df is mnl x n, G is N x n where
    N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ).
    """
    
    p, n = A.size
    ldK = n + p + mnl + dims['l'] + sum(dims['q']) + sum([ int(k*(k+1)/2)
        for k in dims['s'] ])
    K = matrix(0.0, (ldK, ldK))
    ipiv = matrix(0, (ldK, 1))
    u = matrix(0.0, (ldK, 1))
    g = matrix(0.0, (mnl + G.size[0], 1))

    def factor(W, H = None, Df = None):

        blas.scal(0.0, K)
        if H is not None: K[:n, :n] = H
        K[n:n+p, :n] = A
        for k in range(n):
            if mnl: g[:mnl] = Df[:,k]
            g[mnl:] = G[:,k]
            scale(g, W, trans = 'T', inverse = 'I')
            pack(g, K, dims, mnl, offsety = k*ldK + n + p)
        K[(ldK+1)*(p+n) :: ldK+1]  = -1.0
        if kktreg:
            K[0 : (ldK+1)*n : ldK+1]  += kktreg  # Reg. term, 1x1 block (positive)
            K[(ldK+1)*n :: ldK+1]  -= kktreg     # Reg. term, 2x2 block (negative)
        lapack.sytrf(K, ipiv)

        def solve(x, y, z):

            # Solve
            #
            #     [ H          A'   GG'*W^{-1} ]   [ ux   ]   [ bx        ]
            #     [ A          0    0          ] * [ uy   [ = [ by        ]
            #     [ W^{-T}*GG  0   -I          ]   [ W*uz ]   [ W^{-T}*bz ]
            #
            # and return ux, uy, W*uz.
            #
            # On entry, x, y, z contain bx, by, bz.  On exit, they contain
            # the solution ux, uy, W*uz.

            blas.copy(x, u)
            blas.copy(y, u, offsety = n)
            scale(z, W, trans = 'T', inverse = 'I') 
            pack(z, u, dims, mnl, offsety = n + p)
            lapack.sytrs(K, ipiv, u)
            blas.copy(u, x, n = n)
            blas.copy(u, y, offsetx = n, n = p)
            unpack(u, z, dims, mnl, offsetx = n + p)
    
        return solve

    return factor


def kkt_ldl2(G, dims, A, mnl = 0):
    """
    Solution of KKT equations by a dense LDL factorization of the 2 x 2 
    system.
    
    Returns a function that (1) computes the LDL factorization of
    
        [ H + GG' * W^{-1} * W^{-T} * GG   A' ]
        [                                     ]
        [ A                                0  ]
    
    given H, Df, W, where GG = [Df; G], and (2) returns a function for 
    solving 
    
        [ H    A'   GG'   ]   [ ux ]   [ bx ]
        [ A    0    0     ] * [ uy ] = [ by ].
        [ GG   0   -W'*W  ]   [ uz ]   [ bz ]
    
    H is n x n,  A is p x n, Df is mnl x n, G is N x n where
    N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ).
    """

    p, n = A.size
    ldK = n + p 
    K = matrix(0.0, (ldK, ldK))
    if p: ipiv = matrix(0, (ldK, 1))
    g = matrix(0.0, (mnl + G.size[0], 1))
    u = matrix(0.0, (ldK, 1))

    def factor(W, H = None, Df = None):

        blas.scal(0.0, K)
        if H is not None: K[:n, :n] = H
        K[n:,:n] = A
        for k in range(n):
            if mnl: g[:mnl] = Df[:,k]
            g[mnl:] = G[:,k]
            scale(g, W, trans = 'T', inverse = 'I')
            scale(g, W, inverse = 'I')
            if mnl: base.gemv(Df, g, K, trans = 'T', beta = 1.0, n = n-k, 
                offsetA = mnl*k, offsety = (ldK + 1)*k)
            sgemv(G, g, K, dims, trans = 'T', beta = 1.0, n = n-k,
                offsetA = G.size[0]*k, offsetx = mnl, offsety = 
                (ldK + 1)*k)
        if p: lapack.sytrf(K, ipiv)
        else: lapack.potrf(K)

        def solve(x, y, z):

            # Solve
            #
            #     [ H + GG' * W^{-1} * W^{-T} * GG    A' ]   [ ux ]   
            #     [                                      ] * [    ] 
            #     [ A                                 0  ]   [ uy ]   
            #
            #         [ bx + GG' * W^{-1} * W^{-T} * bz ]
            #     =   [                                 ]
            #         [ by                              ]
            #
            # and return x, y, W*z = W^{-T} * (GG*x - bz).

            blas.copy(z, g)
            scale(g, W, trans = 'T', inverse = 'I')
            scale(g, W, inverse = 'I')
            if mnl: 
                base.gemv(Df, g, u, trans = 'T')
                beta = 1.0
            else: 
                beta = 0.0
            sgemv(G, g, u, dims, trans = 'T', offsetx = mnl, beta = beta)
            blas.axpy(x, u)
            blas.copy(y, u, offsety = n)
            if p: lapack.sytrs(K, ipiv, u)
            else: lapack.potrs(K, u)
            blas.copy(u, x, n = n)
            blas.copy(u, y, offsetx = n, n = p)
            if mnl: base.gemv(Df, x, z, alpha = 1.0, beta = -1.0)
            sgemv(G, x, z, dims, alpha = 1.0, beta = -1.0, offsety = mnl)
            scale(z, W, trans = 'T', inverse = 'I')
	    
        return solve

    return factor


def kkt_chol(G, dims, A, mnl = 0):
    """
    Solution of KKT equations by reduction to a 2 x 2 system, a QR 
    factorization to eliminate the equality constraints, and a dense 
    Cholesky factorization of order n-p. 
    
    Computes the QR factorization
    
        A' = [Q1, Q2] * [R; 0]
    
    and returns a function that (1) computes the Cholesky factorization 
    
        Q_2^T * (H + GG^T * W^{-1} * W^{-T} * GG) * Q2 = L * L^T, 
    
    given H, Df, W, where GG = [Df; G], and (2) returns a function for 
    solving 
    
        [ H    A'   GG'    ]   [ ux ]   [ bx ]
        [ A    0    0      ] * [ uy ] = [ by ].
        [ GG   0    -W'*W  ]   [ uz ]   [ bz ]
    
    H is n x n,  A is p x n, Df is mnl x n, G is N x n where
    N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ).
    """

    p, n = A.size
    cdim = mnl + dims['l'] + sum(dims['q']) + sum([ k**2 for k in 
        dims['s'] ])
    cdim_pckd = mnl + dims['l'] + sum(dims['q']) + sum([ int(k*(k+1)/2)
        for k in dims['s'] ])

    # A' = [Q1, Q2] * [R; 0]  (Q1 is n x p, Q2 is n x n-p).
    if type(A) is matrix: 
        QA = A.T
    else: 
        QA = matrix(A.T)
    tauA = matrix(0.0, (p,1))
    lapack.geqrf(QA, tauA)

    Gs = matrix(0.0, (cdim, n))
    K = matrix(0.0, (n,n)) 
    bzp = matrix(0.0, (cdim_pckd, 1))
    yy = matrix(0.0, (p,1))

    def factor(W, H = None, Df = None):

        # Compute 
        #
        #     K = [Q1, Q2]' * (H + GG' * W^{-1} * W^{-T} * GG) * [Q1, Q2]
        #
        # and take the Cholesky factorization of the 2,2 block
        #
        #     Q_2' * (H + GG^T * W^{-1} * W^{-T} * GG) * Q2.

        # Gs = W^{-T} * GG in packed storage.
        if mnl: 
            Gs[:mnl, :] = Df
        Gs[mnl:, :] = G
        scale(Gs, W, trans = 'T', inverse = 'I')
        pack2(Gs, dims, mnl)

        # K = [Q1, Q2]' * (H + Gs' * Gs) * [Q1, Q2].
        blas.syrk(Gs, K, k = cdim_pckd, trans = 'T')
        if H is not None: K[:,:] += H
        symm(K, n)
        lapack.ormqr(QA, tauA, K, side = 'L', trans = 'T')
        lapack.ormqr(QA, tauA, K, side = 'R')

        # Cholesky factorization of 2,2 block of K.
        lapack.potrf(K, n = n-p, offsetA = p*(n+1))

        def solve(x, y, z):

            # Solve
            #
            #     [ 0          A'  GG'*W^{-1} ]   [ ux   ]   [ bx        ]
            #     [ A          0   0          ] * [ uy   ] = [ by        ]
            #     [ W^{-T}*GG  0   -I         ]   [ W*uz ]   [ W^{-T}*bz ]
            #
            # and return ux, uy, W*uz.
            #
            # On entry, x, y, z contain bx, by, bz.  On exit, they contain
            # the solution ux, uy, W*uz.
            #
            # If we change variables ux = Q1*v + Q2*w, the system becomes 
            # 
            #     [ K11 K12 R ]   [ v  ]   [Q1'*(bx+GG'*W^{-1}*W^{-T}*bz)]
            #     [ K21 K22 0 ] * [ w  ] = [Q2'*(bx+GG'*W^{-1}*W^{-T}*bz)]
            #     [ R^T 0   0 ]   [ uy ]   [by                           ]
            # 
            #     W*uz = W^{-T} * ( GG*ux - bz ).

            # bzp := W^{-T} * bz in packed storage 
            scale(z, W, trans = 'T', inverse = 'I')
            pack(z, bzp, dims, mnl)

            # x := [Q1, Q2]' * (x + Gs' * bzp)
            #    = [Q1, Q2]' * (bx + Gs' * W^{-T} * bz)
            blas.gemv(Gs, bzp, x, beta = 1.0, trans = 'T', m = cdim_pckd)
            lapack.ormqr(QA, tauA, x, side = 'L', trans = 'T')

            # y := x[:p] 
            #    = Q1' * (bx + Gs' * W^{-T} * bz)
            blas.copy(y, yy)
            blas.copy(x, y, n = p)

            # x[:p] := v = R^{-T} * by 
            blas.copy(yy, x)
            lapack.trtrs(QA, x, uplo = 'U', trans = 'T', n = p)

            # x[p:] := K22^{-1} * (x[p:] - K21*x[:p])
            #        = K22^{-1} * (Q2' * (bx + Gs' * W^{-T} * bz) - K21*v)
            blas.gemv(K, x, x, alpha = -1.0, beta = 1.0, m = n-p, n = p,
                offsetA = p, offsety = p)
            lapack.potrs(K, x, n = n-p, offsetA = p*(n+1), offsetB = p)

            # y := y - [K11, K12] * x
            #    = Q1' * (bx + Gs' * W^{-T} * bz) - K11*v - K12*w
            blas.gemv(K, x, y, alpha = -1.0, beta = 1.0, m = p, n = n)

            # y := R^{-1}*y
            #    = R^{-1} * (Q1' * (bx + Gs' * W^{-T} * bz) - K11*v 
            #      - K12*w)
            lapack.trtrs(QA, y, uplo = 'U', n = p)
           
            # x := [Q1, Q2] * x
            lapack.ormqr(QA, tauA, x, side = 'L')

            # bzp := Gs * x - bzp.
            #      = W^{-T} * ( GG*ux - bz ) in packed storage.
            # Unpack and copy to z.
            blas.gemv(Gs, x, bzp, alpha = 1.0, beta = -1.0, m = cdim_pckd)
            unpack(bzp, z, dims, mnl)

        return solve

    return factor


def kkt_chol2(G, dims, A, mnl = 0):
    """
    Solution of KKT equations by reduction to a 2 x 2 system, a sparse 
    or dense Cholesky factorization of order n to eliminate the 1,1 
    block, and a sparse or dense Cholesky factorization of order p.
    Implemented only for problems with no second-order or semidefinite
    cone constraints.
    
    Returns a function that (1) computes Cholesky factorizations of
    the matrices 
    
        S = H + GG' * W^{-1} * W^{-T} * GG,  
        K = A * S^{-1} *A'
    
    or (if K is singular in the first call to the function), the matrices
    
        S = H + GG' * W^{-1} * W^{-T} * GG + A' * A,  
        K = A * S^{-1} * A',
    
    given H, Df, W, where GG = [Df; G], and (2) returns a function for 
    solving 
    
        [ H     A'   GG'   ]   [ ux ]   [ bx ]
        [ A     0    0     ] * [ uy ] = [ by ].
        [ GG    0   -W'*W  ]   [ uz ]   [ bz ]
    
    H is n x n,  A is p x n, Df is mnl x n, G is dims['l'] x n.
    """

    if dims['q'] or dims['s']:
        raise ValueError("kktsolver option 'kkt_chol2' is implemented "\
            "only for problems with no second-order or semidefinite cone "\
            "constraints")
    p, n = A.size
    ml = dims['l']
    F = {'firstcall': True, 'singular': False}

    def factor(W, H = None, Df = None):

        if F['firstcall']:
            if type(G) is matrix: 
                F['Gs'] = matrix(0.0, G.size) 
            else:
                F['Gs'] = spmatrix(0.0, G.I, G.J, G.size) 
            if mnl:
                if type(Df) is matrix:
                    F['Dfs'] = matrix(0.0, Df.size) 
                else: 
                    F['Dfs'] = spmatrix(0.0, Df.I, Df.J, Df.size) 
            if (mnl and type(Df) is matrix) or type(G) is matrix or \
                type(H) is matrix:
                F['S'] = matrix(0.0, (n,n))
                F['K'] = matrix(0.0, (p,p))
            else:
                F['S'] = spmatrix([], [], [], (n,n), 'd')
                F['Sf'] = None
                if type(A) is matrix:
                    F['K'] = matrix(0.0, (p,p))
                else:
                    F['K'] = spmatrix([], [], [], (p,p), 'd')

        # Dfs = Wnl^{-1} * Df 
        if mnl: base.gemm(spmatrix(W['dnli'], list(range(mnl)), 
            list(range(mnl))), Df, F['Dfs'], partial = True)

        # Gs = Wl^{-1} * G.
        base.gemm(spmatrix(W['di'], list(range(ml)), list(range(ml))), 
            G, F['Gs'], partial = True)

        if F['firstcall']:
            base.syrk(F['Gs'], F['S'], trans = 'T') 
            if mnl: 
                base.syrk(F['Dfs'], F['S'], trans = 'T', beta = 1.0)
            if H is not None: 
                F['S'] += H
            try:
                if type(F['S']) is matrix: 
                    lapack.potrf(F['S']) 
                else:
                    F['Sf'] = cholmod.symbolic(F['S'])
                    cholmod.numeric(F['S'], F['Sf'])
            except ArithmeticError:
                F['singular'] = True 
                if type(A) is matrix and type(F['S']) is spmatrix:
                    F['S'] = matrix(0.0, (n,n))
                base.syrk(F['Gs'], F['S'], trans = 'T') 
                if mnl:
                    base.syrk(F['Dfs'], F['S'], trans = 'T', beta = 1.0)
                base.syrk(A, F['S'], trans = 'T', beta = 1.0) 
                if H is not None:
                    F['S'] += H
                if type(F['S']) is matrix: 
                    lapack.potrf(F['S']) 
                else:
                    F['Sf'] = cholmod.symbolic(F['S'])
                    cholmod.numeric(F['S'], F['Sf'])
            F['firstcall'] = False

        else:
            base.syrk(F['Gs'], F['S'], trans = 'T', partial = True)
            if mnl: base.syrk(F['Dfs'], F['S'], trans = 'T', beta = 1.0, 
                partial = True)
            if H is not None:
                F['S'] += H
            if F['singular']:
                base.syrk(A, F['S'], trans = 'T', beta = 1.0, partial = 
                    True) 
            if type(F['S']) is matrix: 
                lapack.potrf(F['S']) 
            else:
                cholmod.numeric(F['S'], F['Sf'])

        if type(F['S']) is matrix: 
            # Asct := L^{-1}*A'.  Factor K = Asct'*Asct.
            if type(A) is matrix: 
                Asct = A.T
            else: 
                Asct = matrix(A.T)
            blas.trsm(F['S'], Asct)
            blas.syrk(Asct, F['K'], trans = 'T')
            lapack.potrf(F['K'])

        else:
            # Asct := L^{-1}*P*A'.  Factor K = Asct'*Asct.
            if type(A) is matrix:
                Asct = A.T
                cholmod.solve(F['Sf'], Asct, sys = 7)
                cholmod.solve(F['Sf'], Asct, sys = 4)
                blas.syrk(Asct, F['K'], trans = 'T')
                lapack.potrf(F['K']) 
            else:
                Asct = cholmod.spsolve(F['Sf'], A.T, sys = 7)
                Asct = cholmod.spsolve(F['Sf'], Asct, sys = 4)
                base.syrk(Asct, F['K'], trans = 'T')
                Kf = cholmod.symbolic(F['K'])
                cholmod.numeric(F['K'], Kf)

        def solve(x, y, z):

            # Solve
            #
            #     [ H          A'  GG'*W^{-1} ]   [ ux   ]   [ bx        ]
            #     [ A          0   0          ] * [ uy   ] = [ by        ]
            #     [ W^{-T}*GG  0   -I         ]   [ W*uz ]   [ W^{-T}*bz ]
            #
            # and return ux, uy, W*uz.
            #
            # If not F['singular']:
            #
            #     K*uy = A * S^{-1} * ( bx + GG'*W^{-1}*W^{-T}*bz ) - by
            #     S*ux = bx + GG'*W^{-1}*W^{-T}*bz - A'*uy
            #     W*uz = W^{-T} * ( GG*ux - bz ).
            #    
            # If F['singular']:
            #
            #     K*uy = A * S^{-1} * ( bx + GG'*W^{-1}*W^{-T}*bz + A'*by )
            #            - by
            #     S*ux = bx + GG'*W^{-1}*W^{-T}*bz + A'*by - A'*y.
            #     W*uz = W^{-T} * ( GG*ux - bz ).

            # z := W^{-1} * z = W^{-1} * bz
            scale(z, W, trans = 'T', inverse = 'I') 

            # If not F['singular']:
            #     x := L^{-1} * P * (x + GGs'*z)
            #        = L^{-1} * P * (x + GG'*W^{-1}*W^{-T}*bz)
            #
            # If F['singular']:
            #     x := L^{-1} * P * (x + GGs'*z + A'*y))
            #        = L^{-1} * P * (x + GG'*W^{-1}*W^{-T}*bz + A'*y)

            if mnl: base.gemv(F['Dfs'], z, x, trans = 'T', beta = 1.0)
            base.gemv(F['Gs'], z, x, offsetx = mnl, trans = 'T', 
                beta = 1.0)
            if F['singular']:
                base.gemv(A, y, x, trans = 'T', beta = 1.0)
            if type(F['S']) is matrix:
                blas.trsv(F['S'], x)
            else:
                cholmod.solve(F['Sf'], x, sys = 7)
                cholmod.solve(F['Sf'], x, sys = 4)

            # y := K^{-1} * (Asc*x - y)
            #    = K^{-1} * (A * S^{-1} * (bx + GG'*W^{-1}*W^{-T}*bz) - by)
            #      (if not F['singular'])
            #    = K^{-1} * (A * S^{-1} * (bx + GG'*W^{-1}*W^{-T}*bz + 
            #      A'*by) - by)  
            #      (if F['singular']).

            base.gemv(Asct, x, y, trans = 'T', beta = -1.0)
            if type(F['K']) is matrix:
                lapack.potrs(F['K'], y)
            else:
                cholmod.solve(Kf, y)

            # x := P' * L^{-T} * (x - Asc'*y)
            #    = S^{-1} * (bx + GG'*W^{-1}*W^{-T}*bz - A'*y) 
            #      (if not F['singular'])  
            #    = S^{-1} * (bx + GG'*W^{-1}*W^{-T}*bz + A'*by - A'*y) 
            #      (if F['singular'])

            base.gemv(Asct, y, x, alpha = -1.0, beta = 1.0)
            if type(F['S']) is matrix:
                blas.trsv(F['S'], x, trans='T')
            else:
                cholmod.solve(F['Sf'], x, sys = 5)
                cholmod.solve(F['Sf'], x, sys = 8)

            # W*z := GGs*x - z = W^{-T} * (GG*x - bz)
            if mnl:
                base.gemv(F['Dfs'], x, z, beta = -1.0)
            base.gemv(F['Gs'], x, z, beta = -1.0, offsety = mnl)

        return solve

    return factor


def kkt_qr(G, dims, A):
    """
    Solution of KKT equations with zero 1,1 block, by eliminating the
    equality constraints via a QR factorization, and solving the
    reduced KKT system by another QR factorization.
    
    Computes the QR factorization
    
        A' = [Q1, Q2] * [R1; 0]
    
    and returns a function that (1) computes the QR factorization 
    
        W^{-T} * G * Q2 = Q3 * R3
    
    (with columns of W^{-T}*G in packed storage), and (2) returns a 
    function for solving 
    
        [ 0    A'   G'    ]   [ ux ]   [ bx ]
        [ A    0    0     ] * [ uy ] = [ by ].
        [ G    0   -W'*W  ]   [ uz ]   [ bz ]
    
    A is p x n and G is N x n where N = dims['l'] + sum(dims['q']) + 
    sum( k**2 for k in dims['s'] ).
    """
 
    p, n = A.size
    cdim = dims['l'] + sum(dims['q']) + sum([ k**2 for k in dims['s'] ])
    cdim_pckd = dims['l'] + sum(dims['q']) + sum([ int(k*(k+1)/2) for k in 
        dims['s'] ])

    # A' = [Q1, Q2] * [R1; 0]
    if type(A) is matrix:
        QA = +A.T
    else:
        QA = matrix(A.T)
    tauA = matrix(0.0, (p,1))
    lapack.geqrf(QA, tauA)

    Gs = matrix(0.0, (cdim, n))
    tauG = matrix(0.0, (n-p,1))
    u = matrix(0.0, (cdim_pckd, 1))
    vv = matrix(0.0, (n,1))
    w = matrix(0.0, (cdim_pckd, 1))

    def factor(W):

        # Gs = W^{-T}*G, in packed storage.
        Gs[:,:] = G
        scale(Gs, W, trans = 'T', inverse = 'I')
        pack2(Gs, dims)
 
        # Gs := [ Gs1, Gs2 ] 
        #     = Gs * [ Q1, Q2 ]
        lapack.ormqr(QA, tauA, Gs, side = 'R', m = cdim_pckd)

        # QR factorization Gs2 := [ Q3, Q4 ] * [ R3; 0 ] 
        lapack.geqrf(Gs, tauG, n = n-p, m = cdim_pckd, offsetA = 
            Gs.size[0]*p)

        def solve(x, y, z):

            # On entry, x, y, z contain bx, by, bz.  On exit, they 
            # contain the solution x, y, W*z of
            #
            #     [ 0         A'  G'*W^{-1} ]   [ x   ]   [bx       ]
            #     [ A         0   0         ] * [ y   ] = [by       ].
            #     [ W^{-T}*G  0   -I        ]   [ W*z ]   [W^{-T}*bz]
            #
            # The system is solved in five steps:
            #
            #       w := W^{-T}*bz - Gs1*R1^{-T}*by 
            #       u := R3^{-T}*Q2'*bx + Q3'*w
            #     W*z := Q3*u - w
            #       y := R1^{-1} * (Q1'*bx - Gs1'*(W*z))
            #       x := [ Q1, Q2 ] * [ R1^{-T}*by;  R3^{-1}*u ]

            # w := W^{-T} * bz in packed storage 
            scale(z, W, trans = 'T', inverse = 'I')
            pack(z, w, dims)

            # vv := [ Q1'*bx;  R3^{-T}*Q2'*bx ]
            blas.copy(x, vv)
            lapack.ormqr(QA, tauA, vv, trans='T') 
            lapack.trtrs(Gs, vv, uplo = 'U', trans = 'T', n = n-p, offsetA
                = Gs.size[0]*p, offsetB = p)

            # x[:p] := R1^{-T} * by 
            blas.copy(y, x)
            lapack.trtrs(QA, x, uplo = 'U', trans = 'T', n = p)

            # w := w - Gs1 * x[:p] 
            #    = W^{-T}*bz - Gs1*by 
            blas.gemv(Gs, x, w, alpha = -1.0, beta = 1.0, n = p, m = 
                cdim_pckd)

            # u := [ Q3'*w + v[p:];  0 ]
            #    = [ Q3'*w + R3^{-T}*Q2'*bx; 0 ]
            blas.copy(w, u)
            lapack.ormqr(Gs, tauG, u, trans = 'T', k = n-p, offsetA = 
                Gs.size[0]*p, m = cdim_pckd)
            blas.axpy(vv, u, offsetx = p, n = n-p)
            blas.scal(0.0, u, offset = n-p)

            # x[p:] := R3^{-1} * u[:n-p]  
            blas.copy(u, x, offsety = p, n = n-p)
            lapack.trtrs(Gs, x, uplo='U', n = n-p, offsetA = Gs.size[0]*p,
                offsetB = p)

            # x is now [ R1^{-T}*by;  R3^{-1}*u[:n-p] ]
            # x := [Q1 Q2]*x
            lapack.ormqr(QA, tauA, x) 
 
            # u := [Q3, Q4] * u - w 
            #    = Q3 * u[:n-p] - w
            lapack.ormqr(Gs, tauG, u, k = n-p, m = cdim_pckd, offsetA = 
                Gs.size[0]*p)
            blas.axpy(w, u, alpha = -1.0)  

            # y := R1^{-1} * ( v[:p] - Gs1'*u )
            #    = R1^{-1} * ( Q1'*bx - Gs1'*u )
            blas.copy(vv, y, n = p)
            blas.gemv(Gs, u, y, m = cdim_pckd, n = p, trans = 'T', alpha = 
                -1.0, beta = 1.0)
            lapack.trtrs(QA, y, uplo = 'U', n=p) 

            unpack(u, z, dims)

        return solve

    return factor
