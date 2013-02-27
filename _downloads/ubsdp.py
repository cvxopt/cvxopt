from cvxopt import matrix, blas, lapack, solvers, misc, normal
from cvxopt import sqrt, mul, spdiag
import cvxopt, math

def __cngrnc(r, x, alpha = 1.0, trans = 'N', offsetx = 0):
    """
    In-place congruence transformation

        x := alpha * r * x * r' if trans = 'N'.

        x := alpha * r' * x * r if trans = 'T'.

    r is a square n x n-matrix. 
    x is a square n x n-matrix or n^2-vector. 
    """

    n = r.size[0]

    # Scale diagonal of x by 1/2.  
    blas.scal(0.5, x, inc = n+1, offset = offsetx)
    
    # a := r*tril(x) if trans is 'N',  a := tril(x)*r if trans is 'T'.
    a = +r
    if trans == 'N':  
        blas.trmm(x, a, side = 'R', m = n, n = n, ldA = n, ldB = n, 
            offsetA = offsetx)
    else:  
        blas.trmm(x, a, side = 'L', m = n, n = n, ldA = n, ldB = n,
            offsetA = offsetx)

    # x := alpha * (a * r' + r * a')  
    #    = alpha * r * (tril(x) + tril(x)') * r' if trans is 'N'.
    # x := alpha * (a' * r  + r' * a)  
    #    = alpha * r' * (tril(x)' + tril(x)) * r if trans = 'T'.
    blas.syr2k(r, a, x, trans = trans, alpha = alpha, n = n, k = n, 
        ldB = n, ldC = n, offsetC = offsetx)



def ubsdp(c, A, B, pstart = None, dstart = None):
    """

        minimize  c'*x  + tr(X) 
        s.t.      sum_{i=1}^n xi * Ai - X <= B 
                  X >= 0

        maximize  -tr(B * Z0)
        s.t.      tr(Ai * Z0) + ci = 0,  i = 1, ..., n
                  -Z0 - Z1 + I = 0
                  Z0 >= 0,  Z1 >= 0.

    c is an n-vector.

    A is an m^2 x n-matrix.

    B is an m x m-matrix.
    """

    msq, n = A.size
    m = int(math.sqrt(msq))
    mpckd = int(m * (m+1) / 2)
    dims = {'l': 0, 'q': [], 's': [m, m]}

    # The primal variable is stored as a tuple (x, X).
    cc = (c, matrix(0.0, (m, m)))
    cc[1][::m+1] = 1.0

        
    def xnewcopy(u):

        return (+u[0], +u[1])


    def xdot(u, v):

        return blas.dot(u[0], v[0]) + misc.sdot2(u[1], v[1])


    def xscal(alpha, u):

        blas.scal(alpha, u[0]) 
        blas.scal(alpha, u[1]) 


    def xaxpy(u, v, alpha = 1.0):

        blas.axpy(u[0], v[0], alpha)
        blas.axpy(u[1], v[1], alpha)

    
    def G(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):
        """
        If trans is 'N':

            v[:msq] := alpha * (A*u[0] - u[1][:]) + beta * v[:msq]
            v[msq:] := -alpha * u[1][:] + beta * v[msq:].


        If trans is 'T':

            v[0] := alpha *  A' * u[:msq] + beta * v[0]
            v[1][:] := alpha * (-u[:msq] - u[msq:]) + beta * v[1][:].

        """
 
        if trans == 'N': 

            blas.gemv(A, u[0], v, alpha = alpha, beta = beta) 
            blas.axpy(u[1], v, alpha = -alpha)
            blas.scal(beta, v, offset = msq)
            blas.axpy(u[1], v, alpha = -alpha, offsety = msq)
            
        else:

            misc.sgemv(A, u, v[0], dims = {'l': 0, 'q': [], 's': [m]},
                alpha = alpha, beta = beta, trans = 'T')   
            blas.scal(beta, v[1])
            blas.axpy(u, v[1], alpha = -alpha, n = msq)
            blas.axpy(u, v[1], alpha = -alpha, n = msq, offsetx = msq)


    h = matrix(0.0, (2*msq, 1))
    blas.copy(B, h)
    
    L = matrix(0.0, (m, m))
    U = matrix(0.0, (m, m))
    Us = matrix(0.0, (m, m))
    Uti = matrix(0.0, (m, m))
    s = matrix(0.0, (m, 1))
    Asc = matrix(0.0, (msq, n))
    S = matrix(0.0, (m, m))
    tmp = matrix(0.0, (m, m))
    x1 = matrix(0.0, (m**2, 1))
    H = matrix(0.0, (n, n))

    def F(W):
        """
        Generate a solver for

                                             A'(uz0) = bx[0]
                                          -uz0 - uz1 = bx[1] 
            A(ux[0]) - ux[1] - r0*r0' * uz0 * r0*r0' = bz0 
                     - ux[1] - r1*r1' * uz1 * r1*r1' = bz1.

        uz0, uz1, bz0, bz1 are symmetric m x m-matrices.
        ux[0], bx[0] are n-vectors.
        ux[1], bx[1] are symmetric m x m-matrices.

        We first calculate a congruence that diagonalizes r0*r0' and r1*r1':
 
            U' * r0 * r0' * U = I,  U' * r1 * r1' * U = S.

        We then make a change of variables

            usx[0] = ux[0],  
            usx[1] = U' * ux[1] * U  
              usz0 = U^-1 * uz0 * U^-T  
              usz1 = U^-1 * uz1 * U^-T 

        and define 

              As() = U' * A() * U'  
            bsx[1] = U^-1 * bx[1] * U^-T
              bsz0 = U' * bz0 * U  
              bsz1 = U' * bz1 * U.  

        This gives

                             As'(usz0) = bx[0]
                          -usz0 - usz1 = bsx[1] 
            As(usx[0]) - usx[1] - usz0 = bsz0 
                -usx[1] - S * usz1 * S = bsz1.


        1. Eliminate usz0, usz1 using equations 3 and 4,

               usz0 = As(usx[0]) - usx[1] - bsz0
               usz1 = -S^-1 * (usx[1] + bsz1) * S^-1.

           This gives two equations in usx[0] an usx[1].

               As'(As(usx[0]) - usx[1]) = bx[0] + As'(bsz0)

               -As(usx[0]) + usx[1] + S^-1 * usx[1] * S^-1
                   = bsx[1] - bsz0 - S^-1 * bsz1 * S^-1.


        2. Eliminate usx[1] using equation 2:

               usx[1] + S * usx[1] * S 
                   = S * ( As(usx[0]) + bsx[1] - bsz0 ) * S - bsz1

           i.e., with Gamma[i,j] = 1.0 + S[i,i] * S[j,j],
 
               usx[1] = ( S * As(usx[0]) * S ) ./ Gamma 
                        + ( S * ( bsx[1] - bsz0 ) * S - bsz1 ) ./ Gamma.

           This gives an equation in usx[0].

               As'( As(usx[0]) ./ Gamma ) 
                   = bx0 + As'(bsz0) + 
                     As'( (S * ( bsx[1] - bsz0 ) * S - bsz1) ./ Gamma )
                   = bx0 + As'( ( bsz0 - bsz1 + S * bsx[1] * S ) ./ Gamma ).

        """

        # Calculate U s.t. 
        # 
        #     U' * r0*r0' * U = I,   U' * r1*r1' * U = diag(s).
 
        # Cholesky factorization r0 * r0' = L * L'
        blas.syrk(W['r'][0], L)
        lapack.potrf(L)

        # SVD L^-1 * r1 = U * diag(s) * V'  
        blas.copy(W['r'][1], U)
        blas.trsm(L, U) 
        lapack.gesvd(U, s, jobu = 'O')

        # s := s**2
        s[:] = s**2

        # Uti := U
        blas.copy(U, Uti)

        # U := L^-T * U
        blas.trsm(L, U, transA = 'T')

        # Uti := L * Uti = U^-T 
        blas.trmm(L, Uti)

        # Us := U * diag(s)^-1
        blas.copy(U, Us)
        for i in range(m):
            blas.tbsv(s, Us, n = m, k = 0, ldA = 1, incx = m, offsetx = i)

        # S is m x m with lower triangular entries s[i] * s[j] 
        # sqrtG is m x m with lower triangular entries sqrt(1.0 + s[i]*s[j])
        # Upper triangular entries are undefined but nonzero.

        blas.scal(0.0, S)
        blas.syrk(s, S)
        Gamma = 1.0 + S
        sqrtG = sqrt(Gamma)


        # Asc[i] = (U' * Ai * * U ) ./  sqrtG,  for i = 1, ..., n
        #        = Asi ./ sqrt(Gamma)
        blas.copy(A, Asc)
        misc.scale(Asc,   # only 'r' part of the dictionary is used   
            {'dnl': matrix(0.0, (0, 1)), 'dnli': matrix(0.0, (0, 1)),
             'd': matrix(0.0, (0, 1)), 'di': matrix(0.0, (0, 1)),
             'v': [], 'beta': [], 'r': [ U ], 'rti': [ U ]}) 
        for i in range(n):
            blas.tbsv(sqrtG, Asc, n = msq, k = 0, ldA = 1, offsetx = i*msq)

        # Convert columns of Asc to packed storage
        misc.pack2(Asc, {'l': 0, 'q': [], 's': [ m ]})

        # Cholesky factorization of Asc' * Asc.
        H = matrix(0.0, (n, n))
        blas.syrk(Asc, H, trans = 'T', k = mpckd)
        lapack.potrf(H)


        def solve(x, y, z):
            """

            1. Solve for usx[0]:

               Asc'(Asc(usx[0]))
                   = bx0 + Asc'( ( bsz0 - bsz1 + S * bsx[1] * S ) ./ sqrtG)
                   = bx0 + Asc'( ( bsz0 + S * ( bsx[1] - bssz1) S ) 
                     ./ sqrtG)

               where bsx[1] = U^-1 * bx[1] * U^-T, bsz0 = U' * bz0 * U, 
               bsz1 = U' * bz1 * U, bssz1 = S^-1 * bsz1 * S^-1 

            2. Solve for usx[1]:

               usx[1] + S * usx[1] * S 
                   = S * ( As(usx[0]) + bsx[1] - bsz0 ) * S - bsz1 

               usx[1] 
                   = ( S * (As(usx[0]) + bsx[1] - bsz0) * S - bsz1) ./ Gamma
                   = -bsz0 + (S * As(usx[0]) * S) ./ Gamma
                     + (bsz0 - bsz1 + S * bsx[1] * S ) . / Gamma
                   = -bsz0 + (S * As(usx[0]) * S) ./ Gamma
                     + (bsz0 + S * ( bsx[1] - bssz1 ) * S ) . / Gamma

               Unscale ux[1] = Uti * usx[1] * Uti'

            3. Compute usz0, usz1

               r0' * uz0 * r0 = r0^-1 * ( A(ux[0]) - ux[1] - bz0 ) * r0^-T
               r1' * uz1 * r1 = r1^-1 * ( -ux[1] - bz1 ) * r1^-T

            """

            # z0 := U' * z0 * U 
            #     = bsz0
            __cngrnc(U, z, trans = 'T')

            # z1 := Us' * bz1 * Us 
            #     = S^-1 * U' * bz1 * U * S^-1
            #     = S^-1 * bsz1 * S^-1
            __cngrnc(Us, z, trans = 'T', offsetx = msq)

            # x[1] := Uti' * x[1] * Uti 
            #       = bsx[1]
            __cngrnc(Uti, x[1], trans = 'T')
        
            # x[1] := x[1] - z[msq:] 
            #       = bsx[1] - S^-1 * bsz1 * S^-1
            blas.axpy(z, x[1], alpha = -1.0, offsetx = msq)


            # x1 = (S * x[1] * S + z[:msq] ) ./ sqrtG
            #    = (S * ( bsx[1] - S^-1 * bsz1 * S^-1) * S + bsz0 ) ./ sqrtG
            #    = (S * bsx[1] * S - bsz1 + bsz0 ) ./ sqrtG
            # in packed storage
            blas.copy(x[1], x1)
            blas.tbmv(S, x1, n = msq, k = 0, ldA = 1)
            blas.axpy(z, x1, n = msq)
            blas.tbsv(sqrtG, x1, n = msq, k = 0, ldA = 1)
            misc.pack2(x1, {'l': 0, 'q': [], 's': [m]})

            # x[0] := x[0] + Asc'*x1 
            #       = bx0 + Asc'( ( bsz0 - bsz1 + S * bsx[1] * S ) ./ sqrtG)
            #       = bx0 + As'( ( bz0 - bz1 + S * bx[1] * S ) ./ Gamma )
            blas.gemv(Asc, x1, x[0], m = mpckd, trans = 'T', beta = 1.0)

            # x[0] := H^-1 * x[0]
            #       = ux[0]
            lapack.potrs(H, x[0])


            # x1 = Asc(x[0]) .* sqrtG  (unpacked)
            #    = As(x[0])  
            blas.gemv(Asc, x[0], tmp, m = mpckd)
            misc.unpack(tmp, x1, {'l': 0, 'q': [], 's': [m]})
            blas.tbmv(sqrtG, x1, n = msq, k = 0, ldA = 1)


            # usx[1] = (x1 + (x[1] - z[:msq])) ./ sqrtG**2 
            #        = (As(ux[0]) + bsx[1] - bsz0 - S^-1 * bsz1 * S^-1) 
            #           ./ Gamma

            # x[1] := x[1] - z[:msq] 
            #       = bsx[1] - bsz0 - S^-1 * bsz1 * S^-1
            blas.axpy(z, x[1], -1.0, n = msq)

            # x[1] := x[1] + x1
            #       = As(ux) + bsx[1] - bsz0 - S^-1 * bsz1 * S^-1 
            blas.axpy(x1, x[1])

            # x[1] := x[1] / Gammma
            #       = (As(ux) + bsx[1] - bsz0 + S^-1 * bsz1 * S^-1 ) / Gamma
            #       = S^-1 * usx[1] * S^-1
            blas.tbsv(Gamma, x[1], n = msq, k = 0, ldA = 1)
            

            # z[msq:] := r1' * U * (-z[msq:] - x[1]) * U * r1
            #         := -r1' * U * S^-1 * (bsz1 + ux[1]) * S^-1 *  U * r1
            #         := -r1' * uz1 * r1
            blas.axpy(x[1], z, n = msq, offsety = msq)
            blas.scal(-1.0, z, offset = msq)
            __cngrnc(U, z, offsetx = msq)
            __cngrnc(W['r'][1], z, trans = 'T', offsetx = msq)

            # x[1] :=  S * x[1] * S
            #       =  usx1 
            blas.tbmv(S, x[1], n = msq, k = 0, ldA = 1)

            # z[:msq] = r0' * U' * ( x1 - x[1] - z[:msq] ) * U * r0
            #         = r0' * U' * ( As(ux) - usx1 - bsz0 ) * U * r0
            #         = r0' * U' *  usz0 * U * r0
            #         = r0' * uz0 * r0
            blas.axpy(x1, z, -1.0, n = msq)
            blas.scal(-1.0, z, n = msq)
            blas.axpy(x[1], z, -1.0, n = msq)
            __cngrnc(U, z)
            __cngrnc(W['r'][0], z, trans = 'T')

            # x[1] := Uti * x[1] * Uti'
            #       = ux[1]
            __cngrnc(Uti, x[1])


        return solve
    
    sol = solvers.conelp(cc, G, h, dims = {'l': 0, 's': [m, m], 'q': []},
        kktsolver = F, xnewcopy = xnewcopy, xdot = xdot, xaxpy = xaxpy, 
        xscal = xscal, primalstart = pstart, dualstart = dstart) 

    return matrix(sol['x'][1], (n,n))
