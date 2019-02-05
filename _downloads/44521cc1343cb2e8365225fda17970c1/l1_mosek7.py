from cvxopt import blas, lapack, solvers
from cvxopt import matrix, spdiag, mul, div, sparse 
from cvxopt import spmatrix, sqrt, base

try:
    import mosek
    import sys
    __MOSEK = True
except: __MOSEK = False

if __MOSEK:
    
    def l1mosek(P, q):
        """ 
        minimize    e'*v

        subject to  P*u - v <=  q
                   -P*u - v <= -q
        """

        from mosek.array import zeros

        m, n = P.size

        env = mosek.Env()
        task = env.Task(0,0)
        task.set_Stream(mosek.streamtype.log, lambda x: sys.stdout.write(x))

        task.appendvars( n + m)            # number of variables
        task.appendcons(  2*m )              # number of constraints
        task.putclist(range(n+m), n*[0.0] + m*[1.0])     # setup objective

        # input A matrix row by row
        for i in range(m):
            task.putarow( i, range(n) + [n+i] , list(P[i,:]) + [-1.0])
            task.putarow( i+m, range(n) + [n+i] , list(-P[i,:]) + [-1.0])

        # setup bounds on constraints
        task.putboundslice(mosek.accmode.con,
                           0, 2*m, 2*m*[mosek.boundkey.up], 2*m*[0.0], list(q)+list(-q))

        # setup variable bounds
        task.putboundslice(mosek.accmode.var,
                           0, n+m, (n+m)*[mosek.boundkey.fr], (n+m)*[0.0], (n+m)*[0.0])

        # optimize the task
        task.putobjsense(mosek.objsense.minimize)
        task.putintparam(mosek.iparam.optimizer,    mosek.optimizertype.intpnt)
        task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
        task.optimize()
        task.solutionsummary(mosek.streamtype.log)
        x = zeros(n, float)
        task.getsolutionslice(mosek.soltype.itr, mosek.solitem.xx, 0, n, x)
        return matrix(x)

    def l1mosek2(P, q):
        """ 
        minimize    e'*s + e'*t

        subject to  P*u - q = s - t
                    s, t >= 0
        """

        from mosek.array import zeros

        m, n = P.size

        env  = mosek.Env()
        task = env.Task(0,0)
        task.set_Stream(mosek.streamtype.log, lambda x: sys.stdout.write(x))

        task.appendvars( n + 2*m)          # number of variables
        task.appendcons( m)                # number of constraints
        task.putclist(range(n+2*m), n*[0.0] + 2*m*[1.0]) # setup objective

        # input A matrix row by row
        for i in range(m):
            task.putarow( i, range(n) + [n+i, n+m+i] , list(P[i,:]) + [-1.0, 1.0])

        # setup bounds on constraints
        task.putboundslice(mosek.accmode.con,
                           0, m, m*[mosek.boundkey.fx], list(q), list(q))

        # setup variable bounds
        task.putboundslice(mosek.accmode.var,
                           0, n, n*[mosek.boundkey.fr], n*[0.0], n*[0.0])

        task.putboundslice(mosek.accmode.var,
                           n, n+2*m, 2*m*[mosek.boundkey.lo], 2*m*[0.0], 2*m*[0.0])

        # optimize the task
        task.putobjsense(mosek.objsense.minimize)
        task.putintparam(mosek.iparam.optimizer,    mosek.optimizertype.intpnt)
        task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
        task.optimize()
        task.solutionsummary(mosek.streamtype.log)
        x = zeros(n, float)
        task.getsolutionslice(mosek.soltype.itr, mosek.solitem.xx, 0, n, x)
        return matrix(x)

def l1(P, q):

    """
    Returns the solution u of the ell-1 approximation problem

        (primal) minimize ||P*u - q||_1       
    
        (dual)   maximize    q'*w
                 subject to  P'*w = 0
                             ||w||_infty <= 1.
    """

    m, n = P.size

    # Solve equivalent LP 
    #
    #     minimize    [0; 1]' * [u; v]
    #     subject to  [P, -I; -P, -I] * [u; v] <= [q; -q]
    #
    #     maximize    -[q; -q]' * z 
    #     subject to  [P', -P']*z  = 0
    #                 [-I, -I]*z + 1 = 0 
    #                 z >= 0 
    
    c = matrix(n*[0.0] + m*[1.0])
    h = matrix([q, -q])

    def Fi(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):    
        if trans == 'N':
            # y := alpha * [P, -I; -P, -I] * x + beta*y
            u = P*x[:n]
            y[:m] = alpha * ( u - x[n:]) + beta*y[:m]
            y[m:] = alpha * (-u - x[n:]) + beta*y[m:]

        else:
            # y := alpha * [P', -P'; -I, -I] * x + beta*y
            y[:n] =  alpha * P.T * (x[:m] - x[m:]) + beta*y[:n]
            y[n:] = -alpha * (x[:m] + x[m:]) + beta*y[n:]


    def Fkkt(W): 

        # Returns a function f(x, y, z) that solves
        #
        # [ 0  0  P'      -P'      ] [ x[:n] ]   [ bx[:n] ]
        # [ 0  0 -I       -I       ] [ x[n:] ]   [ bx[n:] ]
        # [ P -I -W1^2     0       ] [ z[:m] ] = [ bz[:m] ]
        # [-P -I  0       -W2      ] [ z[m:] ]   [ bz[m:] ]
        #
        # On entry bx, bz are stored in x, z.
        # On exit x, z contain the solution, with z scaled (W['di'] .* z is
        # returned instead of z). 

        d1, d2 = W['d'][:m], W['d'][m:]
        D = 4*(d1**2 + d2**2)**-1
        A = P.T * spdiag(D) * P
        lapack.potrf(A)

        def f(x, y, z):

            x[:n] += P.T * ( mul( div(d2**2 - d1**2, d1**2 + d2**2), x[n:]) 
                + mul( .5*D, z[:m]-z[m:] ) )
            lapack.potrs(A, x)

            u = P*x[:n]
            x[n:] =  div( x[n:] - div(z[:m], d1**2) - div(z[m:], d2**2) + 
                mul(d1**-2 - d2**-2, u), d1**-2 + d2**-2 )

            z[:m] = div(u-x[n:]-z[:m], d1)
            z[m:] = div(-u-x[n:]-z[m:], d2)

        return f


    # Initial primal and dual points from least-squares solution.

    # uls minimizes ||P*u-q||_2; rls is the LS residual.
    uls =  +q
    lapack.gels(+P, uls)
    rls = P*uls[:n] - q 

    # x0 = [ uls;  1.1*abs(rls) ];   s0 = [q;-q] - [P,-I; -P,-I] * x0
    x0 = matrix( [uls[:n],  1.1*abs(rls)] ) 
    s0 = +h
    Fi(x0, s0, alpha=-1, beta=1) 

    # z0 = [ (1+w)/2; (1-w)/2 ] where w = (.9/||rls||_inf) * rls  
    # if rls is nonzero and w = 0 otherwise.
    if max(abs(rls)) > 1e-10:  
        w = .9/max(abs(rls)) * rls
    else: 
        w = matrix(0.0, (m,1))
    z0 = matrix([.5*(1+w), .5*(1-w)])

    dims = {'l': 2*m, 'q': [], 's': []}
    sol = solvers.conelp(c, Fi, h, dims, kktsolver = Fkkt,  
        primalstart={'x': x0, 's': s0}, dualstart={'z': z0})
    return sol['x'][:n]


def l1blas (P, q):

    """
    Returns the solution u of the ell-1 approximation problem

        (primal) minimize ||P*u - q||_1       
    
        (dual)   maximize    q'*w
                 subject to  P'*w = 0
                             ||w||_infty <= 1.
    """

    m, n = P.size

    # Solve equivalent LP 
    #
    #     minimize    [0; 1]' * [u; v]
    #     subject to  [P, -I; -P, -I] * [u; v] <= [q; -q]
    #
    #     maximize    -[q; -q]' * z 
    #     subject to  [P', -P']*z  = 0
    #                 [-I, -I]*z + 1 = 0 
    #                 z >= 0 
    
    c = matrix(n*[0.0] + m*[1.0])
    h = matrix([q, -q])

    u = matrix(0.0, (m,1)) 
    Ps = matrix(0.0, (m,n))
    A = matrix(0.0, (n,n))

    def Fi(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):    
        if trans == 'N':
            # y := alpha * [P, -I; -P, -I] * x + beta*y
            blas.gemv(P, x, u) 
            y[:m] = alpha * ( u - x[n:]) + beta*y[:m]
            y[m:] = alpha * (-u - x[n:]) + beta*y[m:]

        else:
            # y := alpha * [P', -P'; -I, -I] * x + beta*y
            blas.copy(x[:m] - x[m:], u)
            blas.gemv(P, u, y, alpha = alpha, beta = beta, trans = 'T')
            y[n:] = -alpha * (x[:m] + x[m:]) + beta*y[n:]


    def Fkkt(W): 

        # Returns a function f(x, y, z) that solves
        #
        # [ 0  0  P'      -P'      ] [ x[:n] ]   [ bx[:n] ]
        # [ 0  0 -I       -I       ] [ x[n:] ]   [ bx[n:] ]
        # [ P -I -D1^{-1}  0       ] [ z[:m] ] = [ bz[:m] ]
        # [-P -I  0       -D2^{-1} ] [ z[m:] ]   [ bz[m:] ]
        #
        # where D1 = diag(di[:m])^2, D2 = diag(di[m:])^2 and di = W['di'].
        #
        # On entry bx, bz are stored in x, z.
        # On exit x, z contain the solution, with z scaled (di .* z is
        # returned instead of z). 

        # Factor A = 4*P'*D*P where D = d1.*d2 ./(d1+d2) and
        # d1 = d[:m].^2, d2 = d[m:].^2.

        di = W['di']
        d1, d2 = di[:m]**2, di[m:]**2
        D = div( mul(d1,d2), d1+d2 )  
        Ds = spdiag(2 * sqrt(D))
        base.gemm(Ds, P, Ps)
        blas.syrk(Ps, A, trans = 'T')
        lapack.potrf(A)

        def f(x, y, z):

            # Solve for x[:n]:
            #
            #    A*x[:n] = bx[:n] + P' * ( ((D1-D2)*(D1+D2)^{-1})*bx[n:]
            #        + (2*D1*D2*(D1+D2)^{-1}) * (bz[:m] - bz[m:]) ).

            blas.copy(( mul( div(d1-d2, d1+d2), x[n:]) + 
                mul( 2*D, z[:m]-z[m:] ) ), u)
            blas.gemv(P, u, x, beta = 1.0, trans = 'T')
            lapack.potrs(A, x)

            # x[n:] := (D1+D2)^{-1} * (bx[n:] - D1*bz[:m] - D2*bz[m:]
            #     + (D1-D2)*P*x[:n])

            base.gemv(P, x, u)
            x[n:] =  div( x[n:] - mul(d1, z[:m]) - mul(d2, z[m:]) + 
                mul(d1-d2, u), d1+d2 )

            # z[:m] := d1[:m] .* ( P*x[:n] - x[n:] - bz[:m])
            # z[m:] := d2[m:] .* (-P*x[:n] - x[n:] - bz[m:]) 

            z[:m] = mul(di[:m],  u-x[n:]-z[:m])
            z[m:] = mul(di[m:], -u-x[n:]-z[m:])

        return f


    # Initial primal and dual points from least-squares solution.

    # uls minimizes ||P*u-q||_2; rls is the LS residual.
    uls =  +q
    lapack.gels(+P, uls)
    rls = P*uls[:n] - q 

    # x0 = [ uls;  1.1*abs(rls) ];   s0 = [q;-q] - [P,-I; -P,-I] * x0
    x0 = matrix( [uls[:n],  1.1*abs(rls)] ) 
    s0 = +h
    Fi(x0, s0, alpha=-1, beta=1) 

    # z0 = [ (1+w)/2; (1-w)/2 ] where w = (.9/||rls||_inf) * rls  
    # if rls is nonzero and w = 0 otherwise.
    if max(abs(rls)) > 1e-10:  
        w = .9/max(abs(rls)) * rls
    else: 
        w = matrix(0.0, (m,1))
    z0 = matrix([.5*(1+w), .5*(1-w)])

    dims = {'l': 2*m, 'q': [], 's': []}
    sol = solvers.conelp(c, Fi, h, dims, kktsolver = Fkkt,  
        primalstart={'x': x0, 's': s0}, dualstart={'z': z0})
    return sol['x'][:n]
