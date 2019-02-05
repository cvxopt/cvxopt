from cvxopt import spmatrix, matrix, sparse, normal, mul, div, solvers, lapack, blas, base, misc, sqrt

def robsvm(X, d, gamma, P, e):
    """
    Solves the following robust SVM training problem:
    
       minimize    (1/2) w'*w + gamma*sum(v)
       subject to  diag(d)*(X*w + b*1) >= 1 - v + E*u
                   || S_j*w ||_2 <= u_j,  j = 1...t
                   v >= 0

    The variables are w, b, v, and u. The matrix E is a selector
    matrix with zeros and one '1' per row.  E_ij = 1 means that the
    i'th training vector is associated with the j'th uncertainty
    ellipsoid.

    A custom KKT solver that exploits low-rank structure is used, and
    a positive definite system of equations of order n is
    formed and solved at each iteration.

    ARGUMENTS
    
    X             m-by-n matrix with training vectors as rows

    d             m-vector with training labels (-1,+1)

    P             list of t symmetric matrices of order n

    e             m-vector where e[i] is the index of the uncertainty 
                  ellipsoid associated with the i'th training vector

    RETURNS

    w        n-vector
    
    b        scalar
    
    u        t-vector
    
    v        m-vector

    iters    number of interior-point iterations
    
    """

    m,n = X.size
    assert type(P) is list, "P must be a list of t symmtric positive definite matrices of order n."
    k = len(P)
    if k > 0:
        assert e.size == (m,1), "e must be an m-vector."
        assert max(e) < k and min(e) >= 0, "e[i] must be in {0,1,...,k-1}."
    
    E = spmatrix(1.,e,range(m),(k,m)).T
    d = matrix(d,tc='d')
    q = matrix(0.0, (n+k+1+m,1))
    q[n+k+1:] = gamma
    h = matrix(0.0,(2*m+k*(n+1),1))
    h[:m] = -1.0

    # linear operators Q and G
    def Q(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):
        y[:n] = alpha * x[:n] + beta * y[:n]

    def G(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):
        """
        Implements the linear operator

               [ -DX    E   -d   -I ]  
               [  0     0    0   -I ]  
               [  0   -e_1'  0    0 ]
          G =  [ -P_1'  0    0    0 ]     
               [  .     .    .    . ]    
               [  0   -e_k'  0    0 ]        
               [ -P_k'  0    0    0 ]       

        and its adjoint G'.

        """
        if trans == 'N':
            tmp = +y[:m]
            # y[:m] = alpha*(-DXw + Et - d*b - v) + beta*y[:m]
            base.gemv(E, x[n:n+k], tmp, alpha = alpha, beta = beta)
            blas.axpy(x[n+k+1:], tmp, alpha = -alpha)
            blas.axpy(d, tmp, alpha = -alpha*x[n+k])
            y[:m] = tmp

            base.gemv(X, x[:n], tmp, alpha = alpha, beta = 0.0)
            tmp = mul(d,tmp)
            y[:m] -= tmp
            
            # y[m:2*m] = -v
            y[m:2*m] = -alpha * x[n+k+1:] + beta * y[m:2*m]

            # SOC 1,...,k
            for i in range(k):
                l = 2*m+i*(n+1)
                y[l] = -alpha * x[n+i] + beta * y[l]
                y[l+1:l+1+n] = -alpha * P[i] * x[:n] + beta * y[l+1:l+1+n];

        else:
            tmp1 = mul(d,x[:m])
            tmp2 = y[:n]
            blas.gemv(X, tmp1, tmp2, trans = 'T', alpha = -alpha, beta = beta)
            for i in range(k):
                l = 2*m+1+i*(n+1)
                blas.gemv(P[i], x[l:l+n], tmp2, trans = 'T', alpha = -alpha, beta = 1.0)
            y[:n] = tmp2

            tmp2 = y[n:n+k]
            base.gemv(E, x[:m], tmp2, trans = 'T', alpha = alpha, beta = beta)
            blas.axpy(x[2*m:2*m+k*(1+n):n+1], tmp2, alpha = -alpha)
            y[n:n+k] = tmp2

            y[n+k] = -alpha * blas.dot(d,x[:m]) + beta * y[n+k]
            y[n+k+1:] = -alpha * (x[:m] + x[m:2*m]) + beta * y[n+k+1:]

    # precompute products Pi'*Pi
    Pt = []
    for p in P:
        y = matrix(0.0, (n,n))
        blas.syrk(p, y, trans = 'T')
        Pt.append(y)

    # scaled hyperbolic Householder transformations
    def qscal(u, beta, v, inv = False):
        """
        Transforms the vector u as
           u := beta * (2*v*v' - J) * u
        if 'inv' is False and as
           u := (1/beta) * (2*J*v*v'*J - J) * u
        if 'inv' is True.
        """
        if not inv:
            tmp = blas.dot(u,v)
            u[0] *= -1
            u += 2 * v * tmp
            u *= beta
        else:
            u[0] *= -1.0
            tmp = blas.dot(v,u)
            u[0] -= 2*v[0] * tmp 
            u[1:] += 2*v[1:] * tmp
            u /= beta

    # custom KKT solver
    def F(W): 
        """
        Custom solver for the system

        [  It  0   0    Xt'     0     At1' ...  Atk' ][ dwt  ]   [ rwt ]
        [  0   0   0    -d'     0      0   ...   0   ][ db   ]   [ rb  ]
        [  0   0   0    -I     -I      0   ...   0   ][ dv   ]   [ rv  ]
        [  Xt -d  -I  -Wl1^-2                        ][ dzl1 ]   [ rl1 ]
        [  0   0  -I         -Wl2^-2                 ][ dzl2 ] = [ rl2 ]
        [ At1  0   0                -W1^-2           ][ dz1  ]   [ r1  ] 
        [  |   |   |                       .         ][  |   ]   [  |  ]
        [ Atk  0   0                          -Wk^-2 ][ dzk  ]   [ rk  ]

        where

        It = [ I 0 ]  Xt = [ -D*X E ]  Ati = [ 0   -e_i' ]  
             [ 0 0 ]                         [ -Pi   0   ] 

        dwt = [ dw ]  rwt = [ rw ]
              [ dt ]        [ rt ].

        """

        # scalings and 'intermediate' vectors
        # db = inv(Wl1)^2 + inv(Wl2)^2
        db = W['di'][:m]**2 + W['di'][m:2*m]**2
        dbi = div(1.0,db)
        
        # dt = I - inv(Wl1)*Dbi*inv(Wl1)
        dt = 1.0 - mul(W['di'][:m]**2,dbi)
        dtsqrt = sqrt(dt)

        # lam = Dt*inv(Wl1)*d
        lam = mul(dt,mul(W['di'][:m],d))

        # lt = E'*inv(Wl1)*lam
        lt = matrix(0.0,(k,1))
        base.gemv(E, mul(W['di'][:m],lam), lt, trans = 'T')

        # Xs = sqrt(Dt)*inv(Wl1)*X
        tmp = mul(dtsqrt,W['di'][:m])
        Xs = spmatrix(tmp,range(m),range(m))*X

        # Es = D*sqrt(Dt)*inv(Wl1)*E
        Es = spmatrix(mul(d,tmp),range(m),range(m))*E

        # form Ab = I + sum((1/bi)^2*(Pi'*Pi + 4*(v'*v + 1)*Pi'*y*y'*Pi)) + Xs'*Xs
        #  and Bb = -sum((1/bi)^2*(4*ui*v'*v*Pi'*y*ei')) - Xs'*Es
        #  and D2 = Es'*Es + sum((1/bi)^2*(1+4*ui^2*(v'*v - 1))
        Ab = matrix(0.0,(n,n))
        Ab[::n+1] = 1.0
        base.syrk(Xs,Ab,trans = 'T', beta = 1.0)
        Bb = matrix(0.0,(n,k))
        Bb = -Xs.T*Es # inefficient!?
        D2 = spmatrix(0.0,range(k),range(k))
        base.syrk(Es,D2,trans = 'T', partial = True)
        d2 = +D2.V
        del D2
        py = matrix(0.0,(n,1))
        for i in range(k):
            binvsq = (1.0/W['beta'][i])**2
            Ab += binvsq*Pt[i]
            dvv = blas.dot(W['v'][i],W['v'][i])
            blas.gemv(P[i], W['v'][i][1:], py, trans = 'T', alpha = 1.0, beta = 0.0)
            blas.syrk(py, Ab, alpha = 4*binvsq*(dvv+1), beta = 1.0)
            Bb[:,i] -= 4*binvsq*W['v'][i][0]*dvv*py
            d2[i] += binvsq*(1+4*(W['v'][i][0]**2)*(dvv-1))
        
        d2i = div(1.0,d2)
        d2isqrt = sqrt(d2i)

        # compute a = alpha - lam'*inv(Wl1)*E*inv(D2)*E'*inv(Wl1)*lam
        alpha = blas.dot(lam,mul(W['di'][:m],d))
        tmp = matrix(0.0,(k,1))
        base.gemv(E,mul(W['di'][:m],lam), tmp, trans = 'T')
        tmp = mul(tmp, d2isqrt) #tmp = inv(D2)^(1/2)*E'*inv(Wl1)*lam
        a = alpha - blas.dot(tmp,tmp)

        # compute M12 = X'*D*inv(Wl1)*lam + Bb*inv(D2)*E'*inv(Wl1)*lam
        tmp = mul(tmp, d2isqrt)
        M12 = matrix(0.0,(n,1))
        blas.gemv(Bb,tmp,M12, alpha = 1.0)
        tmp = mul(d,mul(W['di'][:m],lam))
        blas.gemv(X,tmp,M12, trans = 'T', alpha = 1.0, beta = 1.0)

        # form and factor M
        sBb = Bb * spmatrix(d2isqrt,range(k), range(k)) 
        base.syrk(sBb, Ab, alpha = -1.0, beta = 1.0)
        M = matrix([[Ab, M12.T],[M12, a]])
        lapack.potrf(M)
        
        def f(x,y,z):
            
            # residuals
            rwt = x[:n+k]
            rb = x[n+k]
            rv = x[n+k+1:n+k+1+m]
            iw_rl1 = mul(W['di'][:m],z[:m])
            iw_rl2 = mul(W['di'][m:2*m],z[m:2*m])
            ri = [z[2*m+i*(n+1):2*m+(i+1)*(n+1)] for i in range(k)]
            
            # compute 'derived' residuals 
            # rbwt = rwt + sum(Ai'*inv(Wi)^2*ri) + [-X'*D; E']*inv(Wl1)^2*rl1
            rbwt = +rwt
            for i in range(k):
                tmp = +ri[i]
                qscal(tmp,W['beta'][i],W['v'][i],inv=True)
                qscal(tmp,W['beta'][i],W['v'][i],inv=True)
                rbwt[n+i] -= tmp[0]
                blas.gemv(P[i], tmp[1:], rbwt, trans = 'T', alpha = -1.0, beta = 1.0)
            tmp = mul(W['di'][:m],iw_rl1)
            tmp2 = matrix(0.0,(k,1))
            base.gemv(E,tmp,tmp2,trans='T')
            rbwt[n:] += tmp2
            tmp = mul(d,tmp) # tmp = D*inv(Wl1)^2*rl1
            blas.gemv(X,tmp,rbwt,trans='T', alpha = -1.0, beta = 1.0)
            
            # rbb = rb - d'*inv(Wl1)^2*rl1
            rbb = rb - sum(tmp)

            # rbv = rv - inv(Wl2)*rl2 - inv(Wl1)^2*rl1
            rbv = rv - mul(W['di'][m:2*m],iw_rl2) - mul(W['di'][:m],iw_rl1) 
            
            # [rtw;rtt] = rbwt + [-X'*D; E']*inv(Wl1)^2*inv(Db)*rbv 
            tmp = mul(W['di'][:m]**2, mul(dbi,rbv))
            rtt = +rbwt[n:] 
            base.gemv(E, tmp, rtt, trans = 'T', alpha = 1.0, beta = 1.0)
            rtw = +rbwt[:n]
            tmp = mul(d,tmp)
            blas.gemv(X, tmp, rtw, trans = 'T', alpha = -1.0, beta = 1.0)

            # rtb = rbb - d'*inv(Wl1)^2*inv(Db)*rbv
            rtb = rbb - sum(tmp)
            
            # solve M*[dw;db] = [rtw - Bb*inv(D2)*rtt; rtb + lt'*inv(D2)*rtt]
            tmp = mul(d2i,rtt)
            tmp2 = matrix(0.0,(n,1))
            blas.gemv(Bb,tmp,tmp2)
            dwdb = matrix([rtw - tmp2,rtb + blas.dot(mul(d2i,lt),rtt)]) 
            lapack.potrs(M,dwdb)

            # compute dt = inv(D2)*(rtt - Bb'*dw + lt*db)
            tmp2 = matrix(0.0,(k,1))
            blas.gemv(Bb, dwdb[:n], tmp2, trans='T')
            dt = mul(d2i, rtt - tmp2 + lt*dwdb[-1])

            # compute dv = inv(Db)*(rbv + inv(Wl1)^2*(E*dt - D*X*dw - d*db))
            dv = matrix(0.0,(m,1))
            blas.gemv(X,dwdb[:n],dv,alpha = -1.0)
            dv = mul(d,dv) - d*dwdb[-1]
            base.gemv(E, dt, dv, beta = 1.0)
            tmp = +dv  # tmp = E*dt - D*X*dw - d*db
            dv = mul(dbi, rbv + mul(W['di'][:m]**2,dv))

            # compute wdz1 = inv(Wl1)*(E*dt - D*X*dw - d*db - dv - rl1)
            wdz1 = mul(W['di'][:m], tmp - dv) - iw_rl1

            # compute wdz2 = - inv(Wl2)*(dv + rl2)
            wdz2 = - mul(W['di'][m:2*m],dv) - iw_rl2

            # compute wdzi = inv(Wi)*([-ei'*dt; -Pi*dw] - ri)
            wdzi = []
            tmp = matrix(0.0,(n,1))
            for i in range(k):
                blas.gemv(P[i],dwdb[:n],tmp, alpha = -1.0, beta = 0.0) 
                tmp1 = matrix([-dt[i],tmp])
                blas.axpy(ri[i],tmp1,alpha = -1.0)
                qscal(tmp1,W['beta'][i],W['v'][i],inv=True)
                wdzi.append(tmp1)

            # solution
            x[:n] = dwdb[:n]
            x[n:n+k] = dt
            x[n+k] = dwdb[-1]
            x[n+k+1:] = dv
            z[:m] = wdz1 
            z[m:2*m] = wdz2
            for i in range(k):
                z[2*m+i*(n+1):2*m+(i+1)*(n+1)] = wdzi[i]

        return f

    # solve cone QP and return solution
    sol = solvers.coneqp(Q, q, G, h, dims = {'l':2*m,'q':[n+1 for i in range(k)],'s':[]}, kktsolver = F)
    return sol['x'][:n], sol['x'][n+k], sol['x'][n:n+k], sol['x'][n+k+1:], sol['iterations']
