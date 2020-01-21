from cvxopt import matrix, normal, blas, setseed, mul, lapack, solvers
from cvxopt import exp, div, sqrt

from resource import getrusage, RUSAGE_SELF
def cputime():
    """
    Returns tuple (utime, stime) with CPU time spent since start. 
    """
    return (getrusage(RUSAGE_SELF).ru_utime,getrusage(RUSAGE_SELF).ru_stime)

def mcsvm(X, labels, gamma, kernel = 'linear', sigma = 1.0, degree = 1):
    """
    Solves the Crammer and Singer multiclass SVM training problem

        maximize    -(1/2) * tr(U' * Q * U) + tr(E' * U)  
        subject to  U <= gamma * E
                    U * 1_m = 0.

    The variable is an (N x m)-matrix U if N is the number of training
    examples and m the number of classes. 

    Q is a positive definite matrix of order N with Q[i,j] = K(xi, xj) 
    where K is a kernel function and xi is the ith row of X.

    The matrix E is an N x m matrix with E[i,j] = 1 if labels[i] = j
    and E[i,j] = 0 otherwise.

    Input arguments.

        X is a N x n matrix.  The rows are the training vectors.

        labels is a list of integers of length N with values 0, ..., m-1.
        labels[i] is the class of training example i.

        gamma is a positive parameter.

        kernel is a string with values 'linear' or 'poly'. 
        'linear':  K(u,v) = u'*v.
        'poly':    K(u,v) = (u'*v / sigma)**degree.

        sigma is a positive number.

        degree is a positive integer.


    Output.

        Returns a function classifier().  If Y is M x n then classifier(Y)
        returns a list with as its kth element

            argmax { j = 0, ..., m-1 | sum_{i=1}^N U[i,j] * K(xi, yk) }

        where yk' = Y[k, :], xi' = X[i, :], and U is the optimal solution
        of the QP.
    """

    N, n = X.size


    m = max(labels) + 1
    E = matrix(0.0, (N, m))
    E[matrix(range(N)) + N * matrix(labels)] = 1.0

    def G(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):
        """
        If trans is 'N', x is an N x m matrix, and y is an N*m-vector.

            y := alpha * x[:] + beta * y.

        If trans is 'T', x is an N*m vector, and y is an N x m matrix.

            y[:] := alpha * x + beta * y[:].

        """

        blas.scal(beta, y) 
        blas.axpy(x, y, alpha)

    h = matrix(gamma*E, (N*m, 1))
    
    ones = matrix(1.0, (m,1))
    def A(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):
        """
        If trans is 'N', x is an N x m matrix and y an N-vector.

            y := alpha * x * 1_m + beta y.

        If trans is 'T', x is an N vector and y an N x m matrix.

            y := alpha * x * 1_m' + beta y.
        """  

        if trans == 'N': 
            blas.gemv(x, ones, y, alpha = alpha, beta = beta)

        else: 
            blas.scal(beta, y)
            blas.ger(x, ones, y, alpha = alpha)
        
    b = matrix(0.0, (N,1))

    if kernel == 'linear' and N > n:

        def P(x, y, alpha = 1.0, beta = 0.0):
            """
            x and y are N x m matrices.   

                y =  alpha * X * X' * x + beta * y.

            """

            z = matrix(0.0, (n, m))
            blas.gemm(X, x, z, transA = 'T')
            blas.gemm(X, z, y, alpha = alpha, beta = beta)

    else:

        if kernel == 'linear':
            # Q = X * X'
            Q = matrix(0.0, (N,N))
            blas.syrk(X, Q)

        elif kernel == 'poly':
            # Q = (X * X' / sigma) ** degree
            Q = matrix(0.0, (N,N))
            blas.syrk(X, Q, alpha = 1.0/sigma )
            Q = Q**degree

        else:
            raise ValueError("invalid kernel type")


        def P(x, y, alpha = 1.0, beta = 0.0):
            """
            x and y are N x m matrices.   

                y =  alpha * Q * x + beta * y.

            """

            blas.symm(Q, x, y, alpha = alpha, beta = beta)
       

    if kernel == 'linear' and N > n:  # add separate code for n <= N <= m*n

        H = [ matrix(0.0, (n, n)) for k in range(m) ]
        S = matrix(0.0, (m*n, m*n))
        Xs = matrix(0.0, (N, n))
        wnm = matrix(0.0, (m*n, 1))
        wN = matrix(0.0, (N, 1))
        D = matrix(0.0, (N, 1))

        def kkt(W):
            """
            KKT solver for

                X*X' * ux  + uy * 1_m' + mat(uz) = bx
                                       ux * 1_m  = by
                            ux - d.^2 .* mat(uz) = mat(bz).

            ux and bx are N x m matrices.
            uy and by are N-vectors.
            uz and bz are N*m-vectors.  mat(uz) is the N x m matrix that 
                satisfies mat(uz)[:] = uz.
            d = mat(W['d']) a positive N x m matrix.

            If we eliminate uz from the last equation using 

                mat(uz) = (ux - mat(bz)) ./ d.^2
        
            we get two equations in ux, uy:

                X*X' * ux + ux ./ d.^2 + uy * 1_m' = bx + mat(bz) ./ d.^2
                                          ux * 1_m = by.

            From the 1st equation,

                uxk = (X*X' + Dk^-2)^-1 * (-uy + bxk + Dk^-2 * bzk)
                    = Dk * (I + Xk*Xk')^-1 * Dk * (-uy + bxk + Dk^-2 * bzk)

            for k = 1, ..., m, where Dk = diag(d[:,k]), Xk = Dk * X, 
            uxk is column k of ux, and bzk is column k of mat(bz).  

            We use the matrix inversion lemma

                ( I + Xk * Xk' )^-1 = I - Xk * (I + Xk' * Xk)^-1 * Xk'
                                    = I - Xk * Hk^-1 * Xk'
                                    = I - Xk * Lk^-T * Lk^-1 *  Xk'

            where Hk = I + Xk' * Xk = Lk * Lk' to write this as

                uxk = Dk * (I - Xk * Hk^-1 * Xk') * Dk *
                      (-uy + bxk + Dk^-2 * bzk)
                    = (Dk^2 - Dk^2 * X * Hk^-1 * X' * Dk^2) *
                      (-uy + bxk + Dk^-2 * bzk).

            Substituting this in the second equation gives an equation 
            for uy:

                sum_k (Dk^2 - Dk^2 * X * Hk^-1 * X' * Dk^2 ) * uy 
                    = -by + sum_k (Dk^2 - Dk^2 * X * Hk^-1 * X' * Dk^2) *
                      ( bxk + Dk^-2 * bzk ),

            i.e., with D = (sum_k Dk^2)^1/2,  Yk = D^-1 * Dk^2 * X * Lk^-T,

                D * ( I - sum_k Yk * Yk' ) * D * uy  
                    = -by + sum_k (Dk^2 - Dk^2 * X * Hk^-1 * X' * Dk^2) * 
                      ( bxk + Dk^-2 *bzk ).

            Another application of the matrix inversion lemma gives

                uy = D^-1 * (I + Y * S^-1 * Y') * D^-1 * 
                     ( -by + sum_k ( Dk^2 - Dk^2 * X * Hk^-1 * X' * Dk^2 ) *
                     ( bxk + Dk^-2 *bzk ) )

            with S = I - Y' * Y,  Y = [ Y1 ... Ym ].  


            Summary:

            1. Compute 

                   uy = D^-1 * (I + Y * S^-1 * Y') * D^-1 * 
                        ( -by + sum_k (Dk^2 - Dk^2 * X * Hk^-1 * X' * Dk^2)
                        * ( bxk + Dk^-2 *bzk ) )
 
            2. For k = 1, ..., m:

                   uxk = (Dk^2 - Dk^2 * X * Hk^-1 * X' * Dk^2) * 
                         (-uy + bxk + Dk^-2 * bzk)

            3. Solve for uz

                   d .* uz = ( ux - mat(bz) ) ./ d.
        
            Return ux, uy, d .* uz.

            """
###
            utime0, stime0 = cputime()      
###

            d = matrix(W['d'], (N, m))
            dsq = matrix(W['d']**2, (N, m))

            # Factor the matrices 
            #
            #     H[k] = I + Xk' * Xk 
            #          = I + X' * Dk^2 * X.
            #
            # Dk = diag(d[:,k]).
            
            for k in range(m):

                # H[k] = I
                blas.scal(0.0, H[k])
                H[k][::n+1] = 1.0

                # Xs = Dk * X 
                #    = diag(d[:,k]]) * X
                blas.copy(X, Xs)
                for j in range(n):
                    blas.tbmv(d, Xs, n = N, k = 0, ldA = 1, offsetA = k*N,
                        offsetx = j*N)

                # H[k] := H[k] + Xs' * Xs
                #       = I + Xk' * Xk
                blas.syrk(Xs, H[k], trans = 'T', beta = 1.0)

                # Factorization H[k] = Lk * Lk'
                lapack.potrf(H[k])

###
            utime, stime = cputime()      
            print("Factor Hk's: utime = %.2f, stime = %.2f" \
                %(utime-utime0, stime-stime0))
            utime0, stime0 = cputime()      
###


            # diag(D) = ( sum_k d[:,k]**2 ) ** 1/2
            #         = ( sum_k Dk^2) ** 1/2.

            blas.gemv(dsq, ones, D)
            D[:] = sqrt(D)

###
#            utime, stime = cputime()      
#            print("Compute D:  utime = %.2f, stime = %.2f" \
#                %(utime-utime0, stime-stime0))
            utime0, stime0 = cputime()      
###
            

            # S = I - Y'* Y is an m x m block matrix.  
            # The i,j block of Y' * Y is 
            # 
            #     Yi' * Yj = Li^-1 * X' * Di^2 * D^-1 * Dj^2 * X * Lj^-T.
            #
            # We compute only the lower triangular blocks in Y'*Y.

            blas.scal(0.0, S)
            for i in range(m):
                for j in range(i+1): 
                    
                    # Xs = Di * Dj * D^-1 * X
                    blas.copy(X, Xs)
                    blas.copy(d, wN, n = N, offsetx = i*N)
                    blas.tbmv(d, wN, n = N, k = 0, ldA = 1, offsetA = j*N)
                    blas.tbsv(D, wN, n = N, k = 0, ldA = 1)
                    for k in range(n):
                        blas.tbmv(wN, Xs, n = N, k = 0, ldA = 1, offsetx = k*N)

                    # block i, j of S is Xs' * Xs (as nonsymmetric matrix so we 
                    # get the correct multiple after scaling with Li, Lj)
                    blas.gemm(Xs, Xs, S, transA = 'T', ldC = m*n, 
                        offsetC = (j*n)*m*n + i*n)

###
            utime, stime = cputime()      
            print("Form S:      utime = %.2f, stime = %.2f" \
                %(utime-utime0, stime-stime0))
            utime0, stime0 = cputime()      
###

            for i in range(m):

                # multiply block row i of S on the left with Li^-1
                blas.trsm(H[i], S, m = n, n = (i+1)*n, ldB = m*n, 
                    offsetB = i*n)

                # multiply block column i of S on the right with Li^-T
                blas.trsm(H[i], S, side = 'R', transA = 'T', m = (m-i)*n, 
                    n = n, ldB = m*n, offsetB = i*n*(m*n + 1))

            blas.scal(-1.0, S)
            S[::(m*n+1)] += 1.0

###
            utime, stime = cputime()      
            print("Form S (2):  utime = %.2f, stime = %.2f" \
                %(utime-utime0, stime-stime0))
            utime0, stime0 = cputime()      
###

            # S = L*L'
            lapack.potrf(S)

###
            utime, stime = cputime()      
            print("Factor S:    utime = %.2f, stime = %.2f" \
                %(utime-utime0, stime-stime0))
            utime0, stime0 = cputime()      
###


            def f(x, y, z):
                """
                1. Compute 

                   uy = D^-1 * (I + Y * S^-1 * Y') * D^-1 * 
                        ( -by + sum_k (Dk^2 - Dk^2 * X * Hk^-1 * X' * Dk^2)
                        * ( bxk + Dk^-2 *bzk ) )
 
                2. For k = 1, ..., m:

                   uxk = (Dk^2 - Dk^2 * X * Hk^-1 * X' * Dk^2) * 
                         (-uy + bxk + Dk^-2 * bzk)

                3. Solve for uz

                   d .* uz = ( ux - mat(bz) ) ./ d.
        
                Return ux, uy, d .* uz.
                """

###
                utime0, stime0 = cputime()      
###
                
                # xk := Dk^2 * xk + zk 
                #     = Dk^2 * bxk + bzk 
                blas.tbmv(dsq, x, n = N*m, k = 0, ldA = 1)
                blas.axpy(z, x)


                # y := -y + sum_k ( I - Dk^2 * X * Hk^-1 * X' ) * xk
                #    = -y + x*ones - sum_k Dk^2 * X * Hk^-1 * X' * xk
                
                # y := -y + x*ones
                blas.gemv(x, ones, y, alpha = 1.0, beta = -1.0)

                # wnm = X' * x  (wnm interpreted as an n x m matrix)
                blas.gemm(X, x, wnm, m = n, k= N, n = m, transA = 'T', 
                    ldB = N, ldC = n)

                # wnm[:,k] = Hk \ wnm[:,k] (for wnm as an n x m matrix)
                for k in range(m):
                    lapack.potrs(H[k], wnm, offsetB = k*n)


                for k in range(m):

                    # wN = X * wnm[:,k]
                    blas.gemv(X, wnm, wN, offsetx = n*k)

                    # wN = Dk^2 * wN
                    blas.tbmv(dsq[:,k], wN, n = N, k = 0, ldA = 1)
           
                    # y := y - wN
                    blas.axpy(wN, y, -1.0)


                # y = D^-1 * (I + Y * S^-1 * Y') * D^-1 * y
                # 
                # Y = [Y1 ... Ym ], Yk = D^-1 * Dk^2 * X * Lk^-T.

                # y := D^-1 * y
                blas.tbsv(D, y, n = N, k = 0, ldA = 1)

                # wnm =  Y' * y  (interpreted as an Nm vector)
                #     = [ L1^-1 * X' * D1^2 * D^-1 * y;  
                #         L2^-1 * X' * D2^2 * D^-1 * y;  
                #         ...
                #         Lm^-1 * X' * Dm^2 * D^-1 * y ]  


                for k in range(m):

                    # wN = D^-1 * Dk^2 * y
                    blas.copy(y, wN)
                    blas.tbmv(dsq, wN, n = N, k = 0, ldA = 1, offsetA = k*N)
                    blas.tbsv(D, wN, n = N, k = 0, ldA = 1)

                    # wnm[:,k] = X' * wN  
                    blas.gemv(X, wN, wnm, trans = 'T', offsety = k*n)

                    # wnm[:,k] = Lk^-1 * wnm[:,k] 
                    blas.trsv(H[k], wnm, offsetx = k*n)


                # wnm := S^-1 * wnm  (an mn-vector)
                lapack.potrs(S, wnm)

                # y := y + Y * wnm
                #    = y + D^-1 * [ D1^2 * X * L1^-T ... D2^k * X * Lk^-T]
                #      * wnm

                for k in range(m):

                    # wnm[:,k] = Lk^-T * wnm[:,k]
                    blas.trsv(H[k], wnm, trans = 'T', offsetx = k*n)

                    # wN = X * wnm[:,k]
                    blas.gemv(X, wnm, wN, offsetx = k*n)

                    # wN = D^-1 * Dk^2 * wN
                    blas.tbmv(dsq, wN, n = N, k = 0, ldA = 1, offsetA = k*N)
                    blas.tbsv(D, wN, n = N, k = 0, ldA = 1)

                    # y += wN
                    blas.axpy(wN, y)

                
                # y := D^-1 *  y
                blas.tbsv(D, y, n = N, k = 0, ldA = 1)

                    
                # For k = 1, ..., m:
                #
                # xk = (I - Dk^2 * X * Hk^-1 * X') * (-Dk^2 * y + xk)
                 
                # x = x - [ D1^2 * y ... Dm^2 * y] (as an N x m matrix)
                for k in range(m):
                    blas.copy(y, wN)
                    blas.tbmv(dsq, wN, n = N, k = 0, ldA = 1, offsetA = k*N)
                    blas.axpy(wN, x, -1.0, offsety = k*N)

                # wnm  = X' * x (as an n x m matrix)
                blas.gemm(X, x, wnm, transA = 'T', m = n, n = m, k = N,
                    ldB = N, ldC = n)
                        
                # wnm[:,k] = Hk^-1 * wnm[:,k]
                for k in range(m):
                    lapack.potrs(H[k], wnm, offsetB = n*k)


                for k in range(m):
 
                    # wN = X * wnm[:,k]
                    blas.gemv(X, wnm, wN, offsetx = k*n)

                    # wN = Dk^2 * wN 
                    blas.tbmv(dsq, wN, n = N, k = 0, ldA = 1, offsetA = k*N)

                    # x[:,k] := x[:,k] - wN  
                    blas.axpy(wN, x, -1.0, n = N, offsety = k*N)


                # z := ( x - z ) ./ d
                blas.axpy(x, z, -1.0)
                blas.scal(-1.0, z)
                blas.tbsv(d, z, n = N*m, k = 0, ldA = 1)

###
                utime, stime = cputime()      
                print("Solve:       utime = %.2f, stime = %.2f" \
                    %(utime-utime0, stime-stime0))
###


            return f


    else:

        H = [ matrix(0.0, (N, N)) for k in range(m) ]
        S = matrix(0.0, (N, N))

        def kkt(W):
            """
            KKT solver for

                Q * ux  + uy * 1_m' + mat(uz) = bx
                                    ux * 1_m  = by
                         ux - d.^2 .* mat(uz) = mat(bz).

            ux and bx are N x m matrices.
            uy and by are N-vectors.
            uz and bz are N*m-vectors.  mat(uz) is the N x m matrix that 
                satisfies mat(uz)[:] = uz.
            d = mat(W['d']) a positive N x m matrix.

            If we eliminate uz from the last equation using 

                mat(uz) = (ux - mat(bz)) ./ d.^2
        
            we get two equations in ux, uy:

                Q * ux + ux ./ d.^2 + uy * 1_m' = bx + mat(bz) ./ d.^2
                                       ux * 1_m = by.

            From the 1st equation 

                uxk = -(Q + Dk)^-1 * uy + (Q + Dk)^-1 * (bxk + Dk * bzk)

            where uxk is column k of ux, Dk = diag(d[:,k].^-2), and bzk is 
            column k of mat(bz).  Substituting this in the second equation
            gives an equation for uy.

            1. Solve for uy

                   sum_k (Q + Dk)^-1 * uy = 
                       sum_k (Q + Dk)^-1 * (bxk + Dk * bzk) - by.
 
            2. Solve for ux (column by column)

                   Q * ux + ux ./ d.^2 = bx + mat(bz) ./ d.^2 - uy * 1_m'.

            3. Solve for uz

                   mat(uz) = ( ux - mat(bz) ) ./ d.^2.
        
            Return ux, uy, d .* uz.
            """

            # D = d.^-2
            D = matrix(W['di']**2, (N, m))

            blas.scal(0.0, S)
            for k in range(m):

                # Hk := Q + Dk
                blas.copy(Q, H[k])
                H[k][::N+1] += D[:, k]

                # Hk := Hk^-1 
                #     = (Q + Dk)^-1
                lapack.potrf(H[k])
                lapack.potri(H[k])

                # S := S + Hk 
                #    = S + (Q + Dk)^-1
                blas.axpy(H[k], S) 

            # Factor S = sum_k (Q + Dk)^-1
            lapack.potrf(S)

            def f(x, y, z):

                # z := mat(z) 
                #    = mat(bz)
                z.size = N, m

                # x := x + D .* z
                #    = bx + mat(bz) ./ d.^2
                x += mul(D, z)

                # y := y - sum_k (Q + Dk)^-1 * X[:,k] 
                #    = by - sum_k (Q + Dk)^-1 * (bxk + Dk * bzk) 
                for k in range(m):
                    blas.symv(H[k], x[:,k], y, alpha = -1.0, beta = 1.0)

                # y := H^-1 * y
                #    = -uy
                lapack.potrs(S, y)

                # x[:,k] := H[k] * (x[:,k] + y)
                #         = (Q + Dk)^-1 * (bxk + bzk ./ d.^2 + y)
                #         = ux[:,k]
                w = matrix(0.0, (N,1))
                for k in range(m):

                    # x[:,k] := x[:,k] + y
                    blas.axpy(y, x, offsety = N*k, n = N)
 
                    # w := H[k] * x[:,k]
                    #    = (Q + Dk)^-1 * (bxk + bzk ./ d.^2 + y)
                    blas.symv(H[k], x, w, offsetx = N*k) 

                    # x[:,k] := w
                    #         = ux[:,k]
                    blas.copy(w, x, offsety = N*k)
                
                # y := -y
                #    = uy
                blas.scal(-1.0, y)

                # z := (x - z) ./ d
                blas.axpy(x, z, -1.0)
                blas.tbsv(W['d'], z, n = m*N, k = 0, ldA = 1) 
                blas.scal(-1.0, z)
                z.size = N*m, 1

            return f


    utime0, stime0 = cputime()      
#    solvers.options['debug'] = True
#    solvers.options['maxiters'] = 1
    solvers.options['refinement'] = 1
    sol = solvers.coneqp(P, -E, G, h, A = A, b = b, kktsolver = kkt,
        xnewcopy = matrix, xdot = blas.dot, xaxpy = blas.axpy, 
        xscal = blas.scal) 
    utime, stime = cputime()
    utime -= utime0
    stime -= stime0
    print("utime = %.2f, stime = %.2f" % (utime, stime))
    U = sol['x']

    if kernel == 'linear':

        # W = X' * U 
        W = matrix(0.0, (n, m))
        blas.gemm(X, U, W, transA = 'T')

        def classifier(Y):
            # return [ argmax of Y[k,:] * W  for k in range(M) ]
            M = Y.size[0]
            S = Y * W
            c = []
            for i in range(M):
               a = zip(list(S[i,:]), range(m))
               a.sort(reverse = True)
               c += [ a[0][1] ]
            return c


    elif kernel == 'poly':

        def classifier(Y):
            M = Y.size[0]

            # K = Y * X' / sigma
            K = matrix(0.0, (M, N))
            blas.gemm(Y, X, K, transB = 'T', alpha = 1.0/sigma)

            S = K**degree * U

            c = []
            for i in range(M):
               a = zip(list(S[i,:]), range(m))
               a.sort(reverse = True)
               c += [ a[0][1] ]
            return c


    else:
        pass

    return classifier #, utime, sol['iterations']

if __name__ == '__main__':

    digits = range(10)
    for Ntrain in [ 50000, 60000 ]:
        X, labels = mnist.read(digits, Ntrain, bias = True)
        Ntrain = X.size[0]
        print("Ntrain = %i" %(Ntrain))
        X  = X / (256. * math.sqrt(X.size[1]))
        gamma = 100000.0 / Ntrain
    #    classifier = cs(X, labels, kernel = 'poly', gamma = gamma, degree = 2)
        classifier, utime, iters = cs(X, labels, gamma = gamma)
        s = classifier(X)
        err = [ k for k in range(Ntrain) if labels[k] != s[k] ]
        training_error = float(len(err)) / Ntrain
        print("training error = %0.3f" %(training_error))

        Ntest = None # ie all test examples for these digits 
        Y, labels = mnist.read(digits, Ntest, 'testing', bias = True)
        Ntest = Y.size[0]
        Y  = Y / (256. * math.sqrt(Y.size[1]))
        s = classifier(Y)

        err = [ k for k in range(Ntest) if labels[k] != s[k] ]
        testing_error = float(len(err)) / Ntrain
        print("testing error = %0.3f" %(testing_error))

        try:
            f = open("results.pic", "r")
            a = pickle.load(f)
            f.close()
        except:
            a = []
        a += [{
            'N': Ntrain, 
            'gamma': gamma, 
            'utime': utime,
            'iterations': iters,
            'training_error': training_error,
            'testing_error': testing_error
            }]
        f = open("results.pic", "w")
        pickle.dump(a, f)
        f.close()
    

#    if err:
#        mnist.show(Y[err,:], title = [s[k] for k in err])
#        pylab.show()
