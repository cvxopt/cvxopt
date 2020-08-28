.. role:: raw-html(raw)
   :format: html

.. _c-lapack:

********************
The LAPACK Interface
********************

The module :mod:`kvxopt.lapack` includes functions for solving dense sets 
of linear equations, for the corresponding matrix factorizations (LU, 
Cholesky, :raw-html:`LDL<sup><small>T</small></sup>`),
for solving least-squares and least-norm problems, for 
QR factorization, for symmetric eigenvalue problems, singular value 
decomposition, and Schur factorization.  

In this chapter we briefly describe the Python calling sequences.  For 
further details on the underlying LAPACK functions we refer to the LAPACK 
Users' Guide and manual pages.  

The BLAS conventional storage scheme of the section :ref:`s-conventions` 
is used. As in the previous chapter, we omit from the function definitions
less important arguments that are useful for selecting submatrices.  The 
complete definitions are documented in the docstrings in the source code.

.. seealso:: 

    `LAPACK Users' Guide, Third Edition, SIAM, 1999
    <http://www.netlib.org/lapack/lug/lapack_lug.html>`_


General Linear Equations
========================

.. function:: kvxopt.lapack.gesv(A, B[, ipiv = None])

    Solves
    
    .. math::

        A X = B,

    where :math:`A` and :math:`B` are real or complex matrices, with 
    :math:`A` square and nonsingular.  

    The arguments ``A`` and ``B`` must have the same type (:const:`'d'` 
    or :const:`'z'`).  On entry, ``B``  contains the right-hand side 
    :math:`B`; on exit it contains the solution :math:`X`.  The optional 
    argument ``ipiv`` is an integer matrix of length at least :math:`n`.  
    If ``ipiv`` is provided, then :func:`gesv` solves the system, replaces
    ``A`` with the triangular factors in an LU factorization, and returns 
    the permutation matrix in ``ipiv``.  If ``ipiv`` is not specified, 
    then :func:`gesv` solves the system but does not return the LU 
    factorization and does not modify ``A``.  

    Raises an :exc:`ArithmeticError` if the matrix is singular.


.. function:: kvxopt.lapack.getrf(A, ipiv)

    LU factorization of a general, possibly rectangular, real or
    complex matrix,  
 
    .. math::
    
        A = PLU, 

    where :math:`A` is :math:`m` by :math:`n`.  

    The argument ``ipiv`` is an integer matrix of length at least 
    min{:math:`m`, :math:`n`}.  On exit, the lower triangular part of 
    ``A`` is replaced by :math:`L`, the upper triangular part by :math:`U`,
    and the permutation matrix is returned in ``ipiv``.

    Raises an :exc:`ArithmeticError` if the matrix is not full rank.


.. function:: kvxopt.lapack.getrs(A, ipiv, B[, trans = 'N'])

    Solves a general set of linear equations

    .. math::
     
        AX & = B  \quad (\mathrm{trans} = \mathrm{'N'}), \\ 
        A^TX & = B \quad (\mathrm{trans} = \mathrm{'T'}), \\
        A^HX & = B \quad (\mathrm{trans} = \mathrm{'C'}), 
    
    given the LU factorization computed by 
    :func:`gesv <kvxopt.lapack.gesv>` or 
    :func:`getrf <kvxopt.lapack.getrf>`.

    On entry, ``A`` and ``ipiv`` must contain the factorization as computed
    by :func:`gesv` or :func:`getrf`.  On entry, ``B`` contains the 
    right-hand side :math:`B`; on exit it contains the solution :math:`X`.
    ``B`` must have the same type as ``A``.


.. function:: kvxopt.lapack.getri(A, ipiv)

    Computes the inverse of a matrix.  

    On entry, ``A`` and ``ipiv`` must contain the factorization as computed
    by :func:`gesv <kvxopt.lapack.gesv>` or 
    :func:`getrf <kvxopt.lapack.getrf>`.  On exit, ``A`` contains the 
    matrix inverse.


In the following example we compute

.. math::

    x = (A^{-1} + A^{-T})b

for randomly generated problem data, factoring the coefficient matrix once.

>>> from kvxopt import matrix, normal
>>> from kvxopt.lapack import gesv, getrs
>>> n = 10
>>> A = normal(n,n)
>>> b = normal(n)
>>> ipiv = matrix(0, (n,1))
>>> x = +b
>>> gesv(A, x, ipiv)               # x = A^{-1}*b 
>>> x2 = +b
>>> getrs(A, ipiv, x2, trans='T')  # x2 = A^{-T}*b
>>> x += x2


Separate functions are provided for equations with band matrices.

.. function:: kvxopt.lapack.gbsv(A, kl, B[, ipiv = None])

    Solves

    .. math::

        A X = B,
    
    where :math:`A` and :math:`B` are real or complex matrices, with 
    :math:`A` :math:`n` by :math:`n` and banded with :math:`k_l` 
    subdiagonals.  

    The arguments ``A`` and ``B`` must have the same type (:const:`'d'` 
    or :const:`'z'`).  On entry, ``B`` contains the right-hand side 
    :math:`B`; on exit it contains the solution :math:`X`.  The optional 
    argument ``ipiv`` is an integer matrix of length at least :math:`n`.  
    If ``ipiv`` is provided, then ``A`` must have :math:`2k_l + k_u + 1` 
    rows.  On entry the diagonals of :math:`A` are stored in rows 
    :math:`k_l + 1` to :math:`2k_l + k_u + 1` of ``A``, using the BLAS 
    format for general band matrices (see the section 
    :ref:`s-conventions`).  On exit, the factorization is returned in 
    ``A`` and ``ipiv``.  If ``ipiv`` is not provided, then ``A`` must have
    :math:`k_l + k_u + 1` rows.  On entry the diagonals of :math:`A` are 
    stored in the rows of ``A``, following the standard BLAS format for 
    general band matrices.  In this case, :func:`gbsv` does not modify 
    ``A`` and does not return the factorization.

    Raises an :exc:`ArithmeticError` if the matrix is singular.


.. function:: kvxopt.lapack.gbtrf(A, m, kl, ipiv)

    LU factorization of a general :math:`m` by :math:`n` real or complex 
    band matrix with :math:`k_l` subdiagonals.

    The matrix is stored using the BLAS format for general band matrices 
    (see the section :ref:`s-conventions`), by providing the diagonals 
    (stored as rows of a :math:`k_u + k_l + 1` by :math:`n` matrix ``A``),
    the number of rows :math:`m`, and the number of subdiagonals 
    :math:`k_l`.  The argument ``ipiv`` is an integer matrix of length at 
    least min{:math:`m`, :math:`n`}.  On exit, ``A`` and ``ipiv`` contain 
    the details of the factorization.

    Raises an :exc:`ArithmeticError` if the matrix is not full rank.


.. function:: kvxopt.lapack.gbtrs({A, kl, ipiv, B[, trans = 'N'])

    Solves a set of linear equations 
   
    .. math::

        AX   & = B \quad (\mathrm{trans} = \mathrm{'N'}), \\
        A^TX & = B \quad (\mathrm{trans} = \mathrm{'T'}), \\
        A^HX & = B \quad (\mathrm{trans} = \mathrm{'C'}), 

    with :math:`A` a general band matrix with :math:`k_l` subdiagonals, 
    given the LU factorization computed by 
    :func:`gbsv <kvxopt.lapack.gbsv>` or 
    :func:`gbtrf <kvxopt.lapack.gbtrf>`.

    On entry, ``A`` and ``ipiv`` must contain the factorization as computed
    by :func:`gbsv` or :func:`gbtrf`.  On entry, ``B`` contains the 
    right-hand side :math:`B`; on exit it contains the solution :math:`X`.
    ``B`` must have the same type as ``A``.


As an example, we solve a linear equation with

.. math::

    A = \left[ \begin{array}{cccc}
        1 & 2 & 0 & 0 \\
        3 & 4 & 5 & 0 \\
        6 & 7 & 8 & 9 \\
        0 & 10 & 11 & 12 
        \end{array}\right], \qquad  
    B = \left[\begin{array}{c} 1 \\ 1 \\ 1 \\ 1 \end{array}\right].

>>> from kvxopt import matrix
>>> from kvxopt.lapack import gbsv, gbtrf, gbtrs
>>> n, kl, ku = 4, 2, 1
>>> A = matrix([[0., 1., 3., 6.], [2., 4., 7., 10.], [5., 8., 11., 0.], [9., 12., 0., 0.]])
>>> x = matrix(1.0, (n,1))
>>> gbsv(A, kl, x)
>>> print(x)
[ 7.14e-02]
[ 4.64e-01]
[-2.14e-01]
[-1.07e-01]

The code below illustrates how one can reuse the factorization returned
by :func:`gbsv <kvxopt.lapack.gbsv>`. 

>>> Ac = matrix(0.0, (2*kl+ku+1,n))
>>> Ac[kl:,:] = A
>>> ipiv = matrix(0, (n,1))
>>> x = matrix(1.0, (n,1))
>>> gbsv(Ac, kl, x, ipiv)                 # solves A*x = 1
>>> print(x)
[ 7.14e-02]
[ 4.64e-01]
[-2.14e-01]
[-1.07e-01]
>>> x = matrix(1.0, (n,1))
>>> gbtrs(Ac, kl, ipiv, x, trans='T')     # solve A^T*x = 1
>>> print(x)
[ 7.14e-02]
[ 2.38e-02]
[ 1.43e-01]
[-2.38e-02]

An alternative method uses :func:`gbtrf <kvxopt.lapack.gbtrf>` for the 
factorization.

>>> Ac[kl:,:] = A
>>> gbtrf(Ac, n, kl, ipiv)                 
>>> x = matrix(1.0, (n,1))
>>> gbtrs(Ac, kl, ipiv, x)                # solve A^T*x = 1
>>> print(x)                 
[ 7.14e-02]
[ 4.64e-01]
[-2.14e-01]
[-1.07e-01]
>>> x = matrix(1.0, (n,1))
>>> gbtrs(Ac, kl, ipiv, x, trans='T')     # solve A^T*x = 1
>>> print(x)
[ 7.14e-02]
[ 2.38e-02]
[ 1.43e-01]
[-2.38e-02]


The following functions can be used for tridiagonal matrices. They use a 
simpler matrix format, with the diagonals stored in three separate vectors.

.. function:: kvxopt.lapack.gtsv(dl, d, du, B))

    Solves

    .. math::

        A X = B,
    
    where :math:`A` is an :math:`n` by :math:`n` tridiagonal matrix. 

    The subdiagonal of :math:`A` is stored as a matrix ``dl`` of length 
    :math:`n-1`, the diagonal is stored as a matrix ``d`` of length 
    :math:`n`, and the superdiagonal is stored as a matrix ``du`` of 
    length :math:`n-1`.  The four arguments must have the same type 
    (:const:`'d'` or :const:`'z'`).  On exit ``dl``, ``d``, ``du`` are 
    overwritten with the details of the LU factorization of :math:`A`. 
    On entry, ``B`` contains the right-hand side :math:`B`; on exit it 
    contains the solution :math:`X`.

    Raises an :exc:`ArithmeticError` if the matrix is singular.


.. function:: kvxopt.lapack.gttrf(dl, d, du, du2, ipiv)

    LU factorization of an :math:`n` by :math:`n` tridiagonal matrix.

    The subdiagonal of :math:`A` is stored as a matrix ``dl`` of length 
    :math:`n-1`, the diagonal is stored as a matrix ``d`` of length 
    :math:`n`, and the superdiagonal is stored as a matrix ``du`` of length
    :math:`n-1`.  ``dl``, ``d`` and ``du`` must have the same type.  
    ``du2`` is a matrix of length :math:`n-2`, and of the same type as 
    ``dl``.  ``ipiv`` is an :const:`'i'` matrix of length :math:`n`.
    On exit, the five arguments contain the details of the factorization.

    Raises an :exc:`ArithmeticError` if the matrix is singular.


.. function:: kvxopt.lapack.gttrs(dl, d, du, du2, ipiv, B[, trans = 'N'])

    Solves a set of linear equations 

    .. math:: 

        AX   & = B \quad (\mathrm{trans} = \mathrm{'N'}), \\
        A^TX & = B \quad (\mathrm{trans} = \mathrm{'T'}), \\
        A^HX & = B \quad (\mathrm{trans} = \mathrm{'C'}), 
    
    where :math:`A` is an :math:`n` by :math:`n` tridiagonal matrix.

    The arguments ``dl``, ``d``, ``du``, ``du2``, and ``ipiv`` contain 
    the details of the LU factorization as returned by 
    :func:`gttrf <kvxopt.lapack.gttrf>`.
    On entry, ``B`` contains the right-hand side :math:`B`; on exit it 
    contains the solution :math:`X`.  ``B`` must have the same type as 
    the other arguments.


Positive Definite Linear Equations
==================================

.. function:: kvxopt.lapack.posv(A, B[, uplo = 'L'])

    Solves

    .. math:: 

        A X = B,

    where :math:`A` is a real symmetric or complex Hermitian positive 
    definite matrix.

    On exit, ``B`` is replaced by the solution, and ``A`` is overwritten 
    with the Cholesky factor.  The matrices ``A`` and ``B`` must have 
    the same type (:const:`'d'` or :const:`'z'`).

    Raises an :exc:`ArithmeticError` if the matrix is not positive 
    definite.


.. function:: kvxopt.lapack.potrf(A[, uplo = 'L'])

    Cholesky factorization 

    .. math:: 

        A = LL^T \qquad \mbox{or} \qquad A = LL^H

    of a positive definite real symmetric or complex Hermitian matrix 
    :math:`A`.  

    On exit, the lower triangular part of ``A`` (if ``uplo`` is 
    :const:`'L'`) or the upper triangular part (if ``uplo`` is 
    :const:`'U'`) is overwritten with the Cholesky factor or its 
    (conjugate) transpose.

    Raises an :exc:`ArithmeticError` if the matrix is not positive 
    definite.


.. function:: kvxopt.lapack.potrs(A, B[, uplo = 'L'])

    Solves a set of linear equations

    .. math::
     
        AX = B

    with a positive definite real symmetric or complex Hermitian matrix,
    given the Cholesky factorization computed by 
    :func:`posv <kvxopt.lapack.posv>` or 
    :func:`potrf <kvxopt.lapack.potrf>`.

    On entry, ``A`` contains the triangular factor, as computed by 
    :func:`posv` or :func:`potrf`.  On exit, ``B`` is replaced by the 
    solution.  ``B`` must have the same type as ``A``.


.. function:: kvxopt.lapack.potri(A[, uplo = 'L']) 

    Computes the inverse of a positive definite matrix.

    On entry, ``A`` contains the Cholesky factorization computed by 
    :func:`potrf <kvxopt.lapack.potri>` or 
    :func:`posv <kvxopt.lapack.posv>`.  On exit, it contains the matrix 
    inverse.


As an example, we use :func:`posv <kvxopt.lapack.posv>` to solve the 
linear system

.. math:: 
    :label: e-kkt-example

    \newcommand{\diag}{\mathop{\bf diag}}
    \left[ \begin{array}{cc} 
        -\diag(d)^2  & A \\ A^T  & 0 
    \end{array} \right]
    \left[ \begin{array}{c} x_1 \\ x_2 \end{array} \right]
    = 
    \left[ \begin{array}{c} b_1 \\ b_2 \end{array} \right]

by block-elimination.  We first pick a random problem.

>>> from kvxopt import matrix, div, normal, uniform
>>> from kvxopt.blas import syrk, gemv
>>> from kvxopt.lapack import posv
>>> m, n = 100, 50  
>>> A = normal(m,n)
>>> b1, b2 = normal(m), normal(n)
>>> d = uniform(m)

We then solve the equations 

.. math::

    \newcommand{\diag}{\mathop{\bf diag}}
    \begin{split}
    A^T \diag(d)^{-2}A x_2 & = b_2 + A^T \diag(d)^{-2} b_1 \\
    \diag(d)^2 x_1 & = Ax_2 - b_1.
    \end{split}


>>> Asc = div(A, d[:, n*[0]])                # Asc := diag(d)^{-1}*A
>>> B = matrix(0.0, (n,n))
>>> syrk(Asc, B, trans='T')                  # B := Asc^T * Asc = A^T * diag(d)^{-2} * A
>>> x1 = div(b1, d)                          # x1 := diag(d)^{-1}*b1
>>> x2 = +b2
>>> gemv(Asc, x1, x2, trans='T', beta=1.0)   # x2 := x2 + Asc^T*x1 = b2 + A^T*diag(d)^{-2}*b1 
>>> posv(B, x2)                              # x2 := B^{-1}*x2 = B^{-1}*(b2 + A^T*diag(d)^{-2}*b1)
>>> gemv(Asc, x2, x1, beta=-1.0)             # x1 := Asc*x2 - x1 = diag(d)^{-1} * (A*x2 - b1)
>>> x1 = div(x1, d)                          # x1 := diag(d)^{-1}*x1 = diag(d)^{-2} * (A*x2 - b1)


There are separate routines for equations with positive definite band 
matrices.

.. function:: kvxopt.lapack.pbsv(A, B[, uplo='L'])

    Solves

    .. math::

        AX = B

    where :math:`A` is a real symmetric or complex Hermitian positive 
    definite band matrix.  

    On entry, the diagonals of :math:`A` are stored in ``A``, using the 
    BLAS format for symmetric or Hermitian band matrices (see 
    section :ref:`s-conventions`).  On exit, ``B`` is replaced by the
    solution, and ``A`` is overwritten with the Cholesky factor (in the
    BLAS format for triangular band matrices).  The matrices ``A`` and 
    ``B`` must have the same type (:const:`'d'` or :const:`'z'`).

    Raises an :exc:`ArithmeticError` if the matrix is not positive 
    definite.


.. function:: kvxopt.lapack.pbtrf(A[, uplo = 'L'])

    Cholesky factorization 

    .. math::

        A = LL^T \qquad \mbox{or} \qquad A = LL^H

    of a positive definite real symmetric or complex Hermitian band matrix
    :math:`A`.  

    On entry, the diagonals of :math:`A` are stored in ``A``, using the 
    BLAS format for symmetric or Hermitian band matrices.  On exit, ``A`` 
    contains the Cholesky factor, in the BLAS format for triangular band 
    matrices.  

    Raises an :exc:`ArithmeticError` if the matrix is not positive 
    definite.


.. function:: kvxopt.lapack.pbtrs(A, B[, uplo = 'L'])

    Solves a set of linear equations

    .. math::

        AX=B

    with a positive definite real symmetric or complex Hermitian band 
    matrix, given the Cholesky factorization computed by 
    :func:`pbsv <kvxopt.lapack.pbsv>` or 
    :func:`pbtrf <kvxopt.lapack.pbtrf>`.  

    On entry, ``A`` contains the triangular factor, as computed by
    :func:`pbsv` or :func:`pbtrf`.  On exit, ``B`` is replaced by the 
    solution.  ``B`` must have the same type as ``A``.


The following functions are useful for tridiagonal systems.

.. function:: kvxopt.lapack.ptsv(d, e, B)

    Solves
    
    .. math::

       A X = B,

    where :math:`A` is an :math:`n` by :math:`n` positive definite real 
    symmetric or complex Hermitian tridiagonal matrix.  

    The diagonal of :math:`A` is stored as a :const:`'d'` matrix ``d`` of 
    length :math:`n` and its subdiagonal as a :const:`'d'` or :const:`'z'`
    matrix ``e`` of length :math:`n-1`.  The arguments ``e`` and ``B`` 
    must have the same type.  On exit ``d`` contains the diagonal elements
    of :math:`D` in the 
    :raw-html:`LDL<sup><small>T</small></sup>`
    or 
    :raw-html:`LDL<sup><small>H</small></sup>`
    factorization of :math:`A`, and 
    ``e`` contains the subdiagonal elements of the unit lower bidiagonal 
    matrix :math:`L`.  ``B`` is overwritten with the solution :math:`X`.  
    Raises an :exc:`ArithmeticError` if the matrix is singular.


.. function:: kvxopt.lapack.pttrf(d, e)

    :raw-html:`LDL<sup><small>T</small></sup>`
    or
    :raw-html:`LDL<sup><small>H</small></sup>`
    factorization of an :math:`n` by :math:`n` positive 
    definite real symmetric or complex Hermitian tridiagonal matrix 
    :math:`A`.

    On entry, the argument ``d`` is a :const:`'d'` matrix with the diagonal
    elements of :math:`A`.  The argument ``e`` is :const:`'d'` or 
    :const:`'z'` matrix containing the subdiagonal of :math:`A`.  On exit 
    ``d`` contains the diagonal elements of :math:`D`, and ``e`` contains 
    the subdiagonal elements of the unit lower bidiagonal matrix :math:`L`.
    
    Raises an :exc:`ArithmeticError` if the matrix is singular.


.. function:: kvxopt.lapack.pttrs(d, e, B[, uplo = 'L'])

    Solves a set of linear equations 

    .. math::

        AX = B 

    where :math:`A` is an :math:`n` by :math:`n` positive definite real 
    symmetric or complex Hermitian tridiagonal matrix, given its 
    :raw-html:`LDL<sup><small>T</small></sup>`
    or
    :raw-html:`LDL<sup><small>H</small></sup>`
    factorization.

    The argument ``d`` is the diagonal of the diagonal matrix :math:`D`.
    The argument ``uplo`` only matters for complex matrices.  If ``uplo`` 
    is :const:`'L'`, then on exit ``e`` contains the subdiagonal elements 
    of the unit bidiagonal matrix :math:`L`.  If ``uplo`` is :const:`'U'`,
    then ``e`` contains the complex conjugates of the elements of the unit
    bidiagonal matrix :math:`L`.  On exit, ``B`` is overwritten with the 
    solution :math:`X`.  ``B`` must have the same type as ``e``.


Symmetric and Hermitian Linear Equations
========================================

.. function:: kvxopt.lapack.sysv(A, B[, ipiv = None, uplo = 'L'])

    Solves

    .. math::
    
        AX = B
    
    where :math:`A` is a real or complex symmetric matrix  of order 
    :math:`n`.

    On exit, ``B`` is replaced by the solution.  The matrices ``A`` and 
    ``B`` must have the same type (:const:`'d'` or :const:`'z'`).  The 
    optional argument ``ipiv`` is an integer matrix of length at least 
    equal to :math:`n`.  If ``ipiv`` is provided, :func:`sysv` solves the 
    system and returns the factorization in ``A`` and ``ipiv``.  If 
    ``ipiv`` is not specified, :func:`sysv` solves the system but does not
    return the factorization and does not modify ``A``.

    Raises an :exc:`ArithmeticError` if the matrix is singular.


.. function:: kvxopt.lapack.sytrf(A, ipiv[, uplo = 'L'])

    :raw-html:`LDL<sup><small>T</small></sup>`
    factorization 

    .. math:: 

        PAP^T = LDL^T

    of a real or complex symmetric matrix :math:`A` of order :math:`n`.

    ``ipiv`` is an :const:`'i'` matrix of length at least :math:`n`.  On 
    exit, ``A`` and ``ipiv`` contain the factorization.

    Raises an :exc:`ArithmeticError` if the matrix is singular.


.. function:: kvxopt.lapack.sytrs(A, ipiv, B[, uplo = 'L'])

    Solves 

    .. math:: 

        A X = B

    given the 
    :raw-html:`LDL<sup><small>T</small></sup>`
    factorization computed by 
    :func:`sytrf <kvxopt.lapack.sytrf>` or 
    :func:`sysv <kvxopt.lapack.sysv>`. ``B`` must have the same type as 
    ``A``.


.. function:: kvxopt.lapack.sytri(A, ipiv[, uplo = 'L'])

    Computes the inverse of a real or complex symmetric matrix.

    On entry, ``A`` and ``ipiv`` contain the 
    :raw-html:`LDL<sup><small>T</small></sup>`
    factorization computed by :func:`sytrf <kvxopt.lapack.sytrf>` or 
    :func:`sysv <kvxopt.lapack.sysv>`.  
    On exit, ``A`` contains the inverse.


.. function:: kvxopt.lapack.hesv(A, B[, ipiv = None, uplo = 'L'])

    Solves

    .. math::

        A X = B

    where :math:`A` is a real symmetric or complex Hermitian of order 
    :math:`n`.

    On exit, ``B`` is replaced by the solution.  The matrices ``A`` and 
    ``B`` must have the same type (:const:`'d'` or :const:`'z'`).  The 
    optional argument ``ipiv`` is an integer matrix of length at least 
    :math:`n`.  If ``ipiv`` is provided, then :func:`hesv` solves the 
    system and returns the factorization in ``A`` and ``ipiv``.  If 
    ``ipiv`` is not specified, then :func:`hesv` solves the system but does
    not return the factorization and does not modify ``A``.

    Raises an :exc:`ArithmeticError` if the matrix is singular.


.. function:: kvxopt.lapack.hetrf(A, ipiv[, uplo = 'L'])

    :raw-html:`LDL<sup><small>H</small></sup>`
    factorization 

    .. math:: 

        PAP^T = LDL^H

    of a real symmetric or complex Hermitian matrix of order :math:`n`.
    ``ipiv`` is an :const:`'i'` matrix of length at least :math:`n`.  
    On exit, ``A`` and ``ipiv`` contain the factorization.

    Raises an :exc:`ArithmeticError` if the matrix is singular.


.. function:: kvxopt.lapack.hetrs(A, ipiv, B[, uplo = 'L'])

    Solves 

    .. math::

        A X = B
    
    given the 
    :raw-html:`LDL<sup><small>H</small></sup>`
    factorization computed by 
    :func:`hetrf <kvxopt.lapack.hetrf>` or 
    :func:`hesv <kvxopt.lapack.hesv>`.


.. function:: kvxopt.lapack.hetri(A, ipiv[, uplo = 'L'])

    Computes the inverse of a real symmetric or complex Hermitian  matrix.

    On entry, ``A`` and ``ipiv`` contain the 
    :raw-html:`LDL<sup><small>H</small></sup>`
    factorization computed 
    by :func:`hetrf <kvxopt.lapack.hetrf>` or 
    :func:`hesv <kvxopt.lapack.hesv>`.  On exit, ``A`` contains the 
    inverse.


As an example we solve the KKT system :eq:`e-kkt-example`.

>>> from kvxopt.lapack import sysv
>>> K = matrix(0.0, (m+n,m+n))
>>> K[: (m+n)*m : m+n+1] = -d**2
>>> K[:m, m:] = A
>>> x = matrix(0.0, (m+n,1))
>>> x[:m], x[m:] = b1, b2
>>> sysv(K, x, uplo='U')   


Triangular Linear Equations
===========================

.. function:: kvxopt.lapack.trtrs(A, B[, uplo = 'L', trans = 'N', diag = 'N'])

    Solves a triangular set of equations

    .. math::

        AX   & = B \quad (\mathrm{trans} = \mathrm{'N'}), \\
        A^TX & = B \quad (\mathrm{trans} = \mathrm{'T'}), \\
        A^HX & = B \quad (\mathrm{trans} = \mathrm{'C'}), 

    where :math:`A` is real or complex and triangular of order :math:`n`, 
    and :math:`B` is a matrix with :math:`n` rows.  

    ``A`` and ``B`` are matrices with the same type (:const:`'d'` or 
    :const:`'z'`).  :func:`trtrs` is similar to 
    :func:`blas.trsm <kvxopt.blas.trsm>`, except 
    that it raises an :exc:`ArithmeticError` if a diagonal element of ``A``
    is zero (whereas :func:`blas.trsm` returns :const:`inf` values).


.. function:: kvxopt.lapack.trtri(A[, uplo = 'L', diag = 'N'])

    Computes the inverse of a real or complex triangular matrix :math:`A`.  
    On exit, ``A`` contains the inverse.


.. function:: kvxopt.lapack.tbtrs(A, B[, uplo = 'L', trans = 'T', diag = 'N'])

    Solves a triangular set of equations

    .. math:: 

        AX   & = B \quad (\mathrm{trans} = \mathrm{'N'}), \\
        A^TX & = B \quad (\mathrm{trans} = \mathrm{'T'}), \\
        A^HX & = B \quad (\mathrm{trans} = \mathrm{'C'}), 

    where :math:`A` is real or complex triangular band matrix of order 
    :math:`n`, and :math:`B` is a matrix with :math:`n` rows.

    The diagonals of :math:`A` are stored in ``A`` using the BLAS 
    conventions for triangular band matrices.  ``A`` and ``B`` are 
    matrices with the same type (:const:`'d'` or :const:`'z'`).  On exit, 
    ``B`` is replaced by the solution :math:`X`.


Least-Squares and Least-Norm Problems
=====================================

.. function:: kvxopt.lapack.gels(A, B[, trans = 'N'])

    Solves least-squares and least-norm problems with a full rank :math:`m`
    by :math:`n` matrix :math:`A`.


    1. ``trans`` is :const:`'N'`.  If :math:`m` is greater than or equal
       to :math:`n`, :func:`gels` solves the least-squares problem

       .. math::

           \begin{array}{ll} 
           \mbox{minimize} & \|AX-B\|_F.
           \end{array} 
    
       If :math:`m` is less than or equal to :math:`n`, :func:`gels` solves
       the least-norm problem

       .. math::

           \begin{array}{ll} 
           \mbox{minimize} & \|X\|_F \\
           \mbox{subject to} & AX = B.
           \end{array}

    2. ``trans`` is :const:`'T'` or :const:`'C'` and ``A`` and ``B`` are 
       real.  If :math:`m` is greater than or equal to :math:`n`, 
       :func:`gels` solves the least-norm problem

       .. math:: 
    
           \begin{array}{ll} 
           \mbox{minimize} & \|X\|_F \\
           \mbox{subject to} & A^TX=B.
           \end{array}

       If :math:`m` is less than or equal to :math:`n`, :func:`gels` solves
       the least-squares problem

       .. math::

           \begin{array}{ll} 
           \mbox{minimize} & \|A^TX-B\|_F.
           \end{array}
    
    3. ``trans`` is :const:`'C'` and ``A`` and ``B`` are complex. If 
       :math:`m` is greater than or equal to :math:`n`, :func:`gels` solves
       the least-norm problem
   
       .. math::

           \begin{array}{ll} 
           \mbox{minimize} & \|X\|_F \\
           \mbox{subject to} & A^HX=B.
           \end{array}

       If :math:`m` is less than or equal to :math:`n`, :func:`gels` solves
       the least-squares problem

       .. math::

           \begin{array}{ll} 
           \mbox{minimize} & \|A^HX-B\|_F.
           \end{array}

    ``A`` and ``B`` must have the same typecode (:const:`'d'` or 
    :const:`'z'`).  ``trans`` = :const:`'T'` is not allowed if ``A`` is 
    complex.  On exit, the solution :math:`X` is stored as the leading 
    submatrix of ``B``.  The matrix ``A`` is overwritten with details of 
    the QR or the LQ factorization of :math:`A`.

    Note that :func:`gels` does not check whether :math:`A` is full rank.


The following functions compute QR and LQ factorizations. 

.. function:: kvxopt.lapack.geqrf(A, tau)

    QR factorization of a real or complex matrix ``A``:

    .. math:: 

        A = Q R.

    If :math:`A` is :math:`m` by :math:`n`, then :math:`Q` is :math:`m` by 
    :math:`m` and orthogonal/unitary, and :math:`R` is :math:`m` by  
    :math:`n` and upper triangular (if :math:`m` is greater than or equal 
    to :math:`n`), or upper trapezoidal (if :math:`m` is less than or 
    equal to :math:`n`).  

    ``tau``  is a matrix of the same type as ``A`` and of length 
    min{:math:`m`, :math:`n`}.  On exit, :math:`R` is stored in the upper 
    triangular/trapezoidal part of ``A``.  The matrix :math:`Q` is stored 
    as a product of min{:math:`m`, :math:`n`} elementary reflectors in 
    the first min{:math:`m`, :math:`n`} columns of ``A`` and in ``tau``.


.. function:: kvxopt.lapack.gelqf(A, tau)

    LQ factorization of a real or complex matrix ``A``:
  
    .. math::

        A = L Q.

    If :math:`A` is :math:`m` by :math:`n`, then :math:`Q` is :math:`n` by 
    :math:`n` and orthogonal/unitary, and :math:`L` is :math:`m` by 
    :math:`n` and lower triangular (if :math:`m` is less than or equal to 
    :math:`n`), or lower trapezoidal (if :math:`m` is greater than or equal
    to :math:`n`).  

    ``tau``  is a matrix of the same type as ``A`` and of length 
    min{:math:`m`, :math:`n`}.  On exit, :math:`L` is stored in the lower 
    triangular/trapezoidal part of ``A``.  The matrix :math:`Q` is stored 
    as a product of min{:math:`m`, :math:`n`} elementary reflectors in the
    first min{:math:`m`, :math:`n`} rows of ``A`` and in ``tau``.


.. function:: kvxopt.lapack.geqp3(A, jpvt, tau)

    QR factorization with column pivoting of a real or complex matrix 
    :math:`A`:

    .. math:: 

        A P = Q R.

    If :math:`A` is :math:`m` by :math:`n`, then :math:`Q` is :math:`m` 
    by :math:`m` and orthogonal/unitary, and :math:`R` is :math:`m` by 
    :math:`n` and upper triangular (if :math:`m` is greater than or equal 
    to :math:`n`), or upper trapezoidal (if :math:`m` is less than or equal
    to :math:`n`).  

    ``tau`` is a matrix of the same type as ``A`` and of length 
    min{:math:`m`, :math:`n`}.  ``jpvt`` is an integer matrix of 
    length :math:`n`.  On entry, if ``jpvt[k]`` is nonzero, then 
    column :math:`k` of :math:`A` is permuted to the front of :math:`AP`.
    Otherwise, column :math:`k` is a free column.

    On exit, ``jpvt`` contains the permutation :math:`P`:  the operation 
    :math:`AP` is equivalent to ``A[:, jpvt-1]``.  :math:`R` is stored
    in the upper triangular/trapezoidal part of ``A``.  The matrix 
    :math:`Q` is stored as a product of min{:math:`m`, :math:`n`} 
    elementary reflectors in the first min{:math:`m`,:math:`n`} columns 
    of ``A`` and in ``tau``.


In most applications, the matrix :math:`Q` is not needed explicitly, and
it is sufficient to be able to make products with :math:`Q` or its 
transpose.  The functions :func:`unmqr <kvxopt.lapack.unmqr>` and 
:func:`ormqr <kvxopt.lapack.ormqr>` multiply a matrix
with the orthogonal matrix computed by 
:func:`geqrf <kvxopt.lapack.geqrf>`.

.. function:: kvxopt.lapack.unmqr(A, tau, C[, side = 'L', trans = 'N'])

    Product with a real orthogonal or complex unitary matrix:

    .. math::

        \newcommand{\op}{\mathop{\mathrm{op}}}
        \begin{split}
        C & := \op(Q)C \quad (\mathrm{side} = \mathrm{'L'}), \\
        C & := C\op(Q) \quad (\mathrm{side} = \mathrm{'R'}), \\
        \end{split}

    where

    .. math::

        \newcommand{\op}{\mathop{\mathrm{op}}}
        \op(Q) =  \left\{ \begin{array}{ll}
            Q & \mathrm{trans} = \mathrm{'N'} \\
            Q^T & \mathrm{trans} = \mathrm{'T'} \\
            Q^H & \mathrm{trans} = \mathrm{'C'}.
        \end{array}\right.

    If ``A`` is :math:`m` by :math:`n`, then :math:`Q` is square of order 
    :math:`m` and orthogonal or unitary.  :math:`Q` is stored in the first
    min{:math:`m`, :math:`n`} columns of ``A`` and in ``tau`` as a 
    product of min{:math:`m`, :math:`n`} elementary reflectors, as 
    computed by :func:`geqrf <kvxopt.lapack.geqrf>`.  
    The matrices ``A``, ``tau``, and ``C`` 
    must have the same type.  ``trans`` = :const:`'T'` is only allowed if 
    the typecode is :const:`'d'`.


.. function:: kvxopt.lapack.ormqr(A, tau, C[, side = 'L', trans = 'N'])

    Identical to :func:`unmqr <kvxopt.lapack.unmqr>` but works only for 
    real matrices, and the 
    possible values of ``trans`` are :const:`'N'` and :const:`'T'`.


As an example, we solve a least-squares problem by a direct call to 
:func:`gels <kvxopt.lapack.gels>`, and by separate calls to 
:func:`geqrf <kvxopt.lapack.geqrf>`, 
:func:`ormqr <kvxopt.lapack.ormqr>`, and 
:func:`trtrs <kvxopt.lapack.trtrs>`.

>>> from kvxopt import blas, lapack, matrix, normal
>>> m, n = 10, 5
>>> A, b = normal(m,n), normal(m,1)
>>> x1 = +b
>>> lapack.gels(+A, x1)                  # x1[:n] minimizes || A*x - b ||_2
>>> tau = matrix(0.0, (n,1)) 
>>> lapack.geqrf(A, tau)                 # A = [Q1, Q2] * [R1; 0]
>>> x2 = +b
>>> lapack.ormqr(A, tau, x2, trans='T')  # x2 := [Q1, Q2]' * x2
>>> lapack.trtrs(A[:n,:], x2, uplo='U')  # x2[:n] := R1^{-1} * x2[:n]
>>> blas.nrm2(x1[:n] - x2[:n])
3.0050798580569307e-16


The next two functions make products with the orthogonal matrix computed 
by :func:`gelqf <kvxopt.lapack.gelqf>`.

.. function:: kvxopt.lapack.unmlq(A, tau, C[, side = 'L', trans = 'N'])

    Product with a real orthogonal or complex unitary matrix:

    .. math::

        \newcommand{\op}{\mathop{\mathrm{op}}}
        \begin{split}
        C & := \op(Q)C \quad (\mathrm{side} = \mathrm{'L'}), \\
        C & := C\op(Q) \quad (\mathrm{side} = \mathrm{'R'}), \\
        \end{split}

    where

    .. math::
        \newcommand{\op}{\mathop{\mathrm{op}}}
            \op(Q) =  \left\{ \begin{array}{ll}
                Q & \mathrm{trans} = \mathrm{'N'}, \\
                Q^T & \mathrm{trans} = \mathrm{'T'}, \\
                Q^H & \mathrm{trans} = \mathrm{'C'}.
            \end{array}\right.

    If ``A`` is :math:`m` by :math:`n`, then :math:`Q` is square of order 
    :math:`n` and orthogonal or unitary.  :math:`Q` is stored in the first
    min{:math:`m`, :math:`n`} rows of ``A`` and in ``tau`` as a product of
    min{:math:`m`, :math:`n`} elementary reflectors, as computed by  
    :func:`gelqf <kvxopt.lapack.gelqf>`.  
    The matrices ``A``, ``tau``, and ``C`` must have the 
    same type.  ``trans`` = :const:`'T'` is only allowed if the typecode 
    is :const:`'d'`.


.. function:: kvxopt.lapack.ormlq(A, tau, C[, side = 'L', trans = 'N'])

    Identical to :func:`unmlq <kvxopt.lapack.unmlq>` but works only for 
    real matrices, and the 
    possible values of ``trans`` or :const:`'N'` and :const:`'T'`.


As an example, we solve a least-norm problem by a direct call to 
:func:`gels <kvxopt.lapack.gels>`, and by separate calls to 
:func:`gelqf <kvxopt.lapack.gelqf>`, 
:func:`ormlq <kvxopt.lapack.ormlq>`, 
and :func:`trtrs <kvxopt.lapack.trtrs>`.

>>> from kvxopt import blas, lapack, matrix, normal
>>> m, n = 5, 10
>>> A, b = normal(m,n), normal(m,1)
>>> x1 = matrix(0.0, (n,1))
>>> x1[:m] = b
>>> lapack.gels(+A, x1)                  # x1 minimizes ||x||_2 subject to A*x = b
>>> tau = matrix(0.0, (m,1)) 
>>> lapack.gelqf(A, tau)                 # A = [L1, 0] * [Q1; Q2] 
>>> x2 = matrix(0.0, (n,1))
>>> x2[:m] = b                           # x2 = [b; 0]
>>> lapack.trtrs(A[:,:m], x2)            # x2[:m] := L1^{-1} * x2[:m]
>>> lapack.ormlq(A, tau, x2, trans='T')  # x2 := [Q1, Q2]' * x2
>>> blas.nrm2(x1 - x2)
0.0


Finally, if the matrix :math:`Q` is needed explicitly, it can be generated
from the output of :func:`geqrf <kvxopt.lapack.geqrf>` and 
:func:`gelqf <kvxopt.lapack.gelqf>` using one of the following functions.

.. function:: kvxopt.lapack.ungqr(A, tau)

    If ``A`` has size :math:`m` by :math:`n`, and ``tau`` has length 
    :math:`k`, then, on entry, the first ``k`` columns of the matrix ``A`` 
    and the entries of ``tau`` contai an unitary or orthogonal matrix
    :math:`Q` of order :math:`m`, as computed by 
    :func:`geqrf <kvxopt.lapack.geqrf>`.  On exit, 
    the first min{:math:`m`, :math:`n`} columns of :math:`Q` are contained
    in the leading columns of ``A``.


.. function:: kvxopt.lapack.orgqr(A, tau)

    Identical to :func:`ungqr <kvxopt.lapack.ungqr>` but works only for 
    real matrices.


.. function:: kvxopt.lapack.unglq(A, tau)

    If ``A`` has size :math:`m` by :math:`n`, and ``tau`` has length 
    :math:`k`, then, on entry, the first ``k`` rows of the matrix ``A`` 
    and the entries of ``tau`` contain a unitary or orthogonal matrix
    :math:`Q` of order :math:`n`, as computed by 
    :func:`gelqf <kvxopt.lapack.gelqf>`.  
    On exit, the first min{:math:`m`, :math:`n`} rows of :math:`Q` are 
    contained in the leading rows of ``A``.


.. function:: kvxopt.lapack.orglq(A, tau)

    Identical to :func:`unglq <kvxopt.lapack.unglq>` but works only for 
    real matrices.


We illustrate this with the QR factorization of the matrix

.. math::
    A = \left[\begin{array}{rrr}
        6 & -5 & 4 \\ 6 & 3 & -4 \\ 19 & -2 & 7 \\ 6 & -10 & -5 
        \end{array} \right]
      = \left[\begin{array}{cc}
        Q_1 & Q_2 \end{array}\right]
        \left[\begin{array}{c} R \\ 0 \end{array}\right]. 

>>> from kvxopt import matrix, lapack
>>> A = matrix([ [6., 6., 19., 6.], [-5., 3., -2., -10.], [4., -4., 7., -5] ])
>>> m, n = A.size
>>> tau = matrix(0.0, (n,1))
>>> lapack.geqrf(A, tau)
>>> print(A[:n, :])              # Upper triangular part is R.
[-2.17e+01  5.08e+00 -4.76e+00]
[ 2.17e-01 -1.06e+01 -2.66e+00]
[ 6.87e-01  3.12e-01 -8.74e+00]
>>> Q1 = +A
>>> lapack.orgqr(Q1, tau)
>>> print(Q1)
[-2.77e-01  3.39e-01 -4.10e-01]
[-2.77e-01 -4.16e-01  7.35e-01]
[-8.77e-01 -2.32e-01 -2.53e-01]
[-2.77e-01  8.11e-01  4.76e-01]
>>> Q = matrix(0.0, (m,m))
>>> Q[:, :n] = A
>>> lapack.orgqr(Q, tau)
>>> print(Q)                     # Q = [ Q1, Q2]
[-2.77e-01  3.39e-01 -4.10e-01 -8.00e-01]
[-2.77e-01 -4.16e-01  7.35e-01 -4.58e-01]
[-8.77e-01 -2.32e-01 -2.53e-01  3.35e-01]
[-2.77e-01  8.11e-01  4.76e-01  1.96e-01]


The orthogonal matrix in the factorization

.. math::

    A = \left[ \begin{array}{rrrr}
        3 & -16 & -10 & -1 \\
       -2 & -12 &  -3 &  4 \\
        9 &  19 &   6 & -6  
        \end{array}\right]
      = Q \left[\begin{array}{cc} R_1 & R_2 \end{array}\right]

can be generated as follows.

>>> A = matrix([ [3., -2., 9.], [-16., -12., 19.], [-10., -3., 6.], [-1., 4., -6.] ])
>>> m, n = A.size
>>> tau = matrix(0.0, (m,1))
>>> lapack.geqrf(A, tau)
>>> R = +A
>>> print(R)                     # Upper trapezoidal part is [R1, R2].
[-9.70e+00 -1.52e+01 -3.09e+00  6.70e+00]
[-1.58e-01  2.30e+01  1.14e+01 -1.92e+00]
[ 7.09e-01 -5.57e-01  2.26e+00  2.09e+00]
>>> lapack.orgqr(A, tau)
>>> print(A[:, :m])              # Q is in the first m columns of A.
[-3.09e-01 -8.98e-01 -3.13e-01]
[ 2.06e-01 -3.85e-01  9.00e-01]
[-9.28e-01  2.14e-01  3.04e-01]


Symmetric and Hermitian Eigenvalue Decomposition
================================================

The first four routines compute all or selected  eigenvalues and 
eigenvectors of a real symmetric matrix :math:`A`:

.. math::

    \newcommand{\diag}{\mathop{\bf diag}}
    A = V\diag(\lambda)V^T,\qquad  V^TV = I.

.. function:: kvxopt.lapack.syev(A, W[, jobz = 'N', uplo = 'L'])

    Eigenvalue decomposition of a real symmetric matrix of order :math:`n`.

    ``W`` is a real matrix of length at least :math:`n`.  On exit, ``W`` 
    contains the eigenvalues in ascending order.  If ``jobz`` is 
    :const:`'V'`, the eigenvectors are also computed and returned in ``A``.
    If ``jobz`` is :const:`'N'`, the eigenvectors are not returned and the 
    contents of ``A`` are destroyed.

    Raises an :exc:`ArithmeticError` if the eigenvalue decomposition fails.


.. function:: kvxopt.lapack.syevd(A, W[, jobz = 'N', uplo = 'L'])

    This is an alternative to :func:`syev <kvxopt.lapack.syev>`, based 
    on a different
    algorithm.  It is faster on large problems, but also uses more memory.


.. function:: kvxopt.lapack.syevx(A, W[, jobz = 'N', range = 'A', uplo = 'L', vl = 0.0, vu = 0.0, il = 1, iu = 1, Z = None])

    Computes selected eigenvalues and eigenvectors of a real symmetric 
    matrix of order :math:`n`.

    ``W`` is a real matrix of length at least :math:`n`.  On exit, ``W``  
    contains the eigenvalues in ascending order.  If ``range`` is 
    :const:`'A'`, all the eigenvalues are computed.  If ``range`` is 
    :const:`'I'`, eigenvalues :math:`i_l` through :math:`i_u` are 
    computed, where :math:`1 \leq i_l \leq i_u \leq n`.  If ``range`` is 
    :const:`'V'`, the eigenvalues in the interval :math:`(v_l, v_u]` are 
    computed. 

    If ``jobz`` is :const:`'V'`, the (normalized) eigenvectors are 
    computed, and returned in ``Z``.  If ``jobz`` is :const:`'N'`, the 
    eigenvectors are not computed.  In both cases, the contents of ``A`` 
    are destroyed on exit.

    ``Z`` is optional (and not referenced) if ``jobz`` is :const:`'N'`.
    It is required if ``jobz`` is :const:`'V'` and must have at least
    :math:`n` columns if ``range`` is :const:`'A'` or :const:`'V'` and  at
    least :math:`i_u - i_l + 1` columns if ``range`` is :const:`'I'`.

    :func:`syevx` returns the number of computed eigenvalues.


.. function:: kvxopt.lapack.syevr(A, W[, jobz = 'N', range = 'A', uplo = 'L', vl = 0.0, vu = 0.0, il = 1, iu = n, Z =  None])

    This is an alternative to :func:`syevx <kvxopt.lapack.syevr>`.  
    :func:`syevr` is the most 
    recent LAPACK routine for symmetric eigenvalue problems, and expected 
    to supersede the three other routines in future releases.


The next four routines can be used to compute eigenvalues and eigenvectors 
for complex Hermitian matrices:

.. math::

    \newcommand{\diag}{\mathop{\bf diag}}
    A = V\diag(\lambda)V^H,\qquad  V^HV = I.

For real symmetric matrices they are identical to the corresponding
:func:`syev*` routines.

.. function:: kvxopt.lapack.heev(A, W[, jobz = 'N', uplo = 'L'])

    Eigenvalue decomposition of a real symmetric or complex Hermitian
    matrix of order :math:`n`.

    The calling sequence is identical to 
    :func:`syev <kvxopt.lapack.syev>`,
    except that ``A`` can be real or complex.


.. function:: kvxopt.lapack.heevd(A, W[, jobz = 'N'[, uplo = 'L']])

    This is an alternative to :func:`heev <kvxopt.lapack.heevd>`. 


.. function:: kvxopt.lapack.heevx(A, W[, jobz = 'N', range = 'A', uplo = 'L', vl = 0.0, vu = 0.0, il = 1, iu = n, Z = None])

    Computes selected eigenvalues and eigenvectors of a real symmetric 
    or complex Hermitian matrix.

    The calling sequence is identical to 
    :func:`syevx <kvxopt.lapack.syevx>`, except that ``A`` 
    can be real or complex.  ``Z`` must have the same type as ``A``.


.. function:: kvxopt.lapack.heevr(A, W[, jobz = 'N', range = 'A', uplo = 'L', vl = 0.0, vu = 0.0, il = 1, iu = n, Z = None])

    This is an alternative to :func:`heevx <kvxopt.lapack.heevx>`. 


Generalized Symmetric Definite Eigenproblems
============================================

Three types of generalized eigenvalue problems can be solved:

.. math:: 
    :label: e-gevd

    \newcommand{\diag}{\mathop{\bf diag}}
    \begin{split}
        AZ  & = BZ\diag(\lambda)\quad \mbox{(type 1)}, \\
        ABZ & = Z\diag(\lambda) \quad \mbox{(type 2)}, \\
        BAZ & = Z\diag(\lambda) \quad \mbox{(type 3)}, 
    \end{split}

with :math:`A` and :math:`B` real symmetric or complex Hermitian, and 
:math:`B` is positive definite.  The matrix of eigenvectors is normalized 
as follows:

.. math::

    Z^H BZ = I \quad \mbox{(types 1 and 2)}, \qquad 
    Z^H B^{-1}Z = I \quad \mbox{(type 3)}.

.. function:: kvxopt.lapack.sygv(A, B, W[, itype = 1, jobz = 'N', uplo = 'L'])

    Solves the generalized eigenproblem :eq:`e-gevd` for real symmetric 
    matrices of order :math:`n`, stored in real matrices ``A`` and ``B``.
    ``itype`` is an integer with possible values 1, 2, 3, and specifies
    the type of eigenproblem.  ``W`` is a real matrix of length at least 
    :math:`n`.  On exit, it contains the eigenvalues in ascending order.
    On exit, ``B`` contains the Cholesky factor of :math:`B`.  If ``jobz``
    is :const:`'V'`, the eigenvectors are computed and returned in ``A``.
    If ``jobz`` is :const:`'N'`, the eigenvectors are not returned and the 
    contents of ``A`` are destroyed.


.. function:: kvxopt.lapack.hegv(A, B, W[, itype = 1, jobz = 'N', uplo = 'L'])

    Generalized eigenvalue problem :eq:`e-gevd` of real symmetric or 
    complex Hermitian matrix of order :math:`n`.  The calling sequence is 
    identical to :func:`sygv <kvxopt.lapack.sygv>`, except that 
    ``A`` and ``B`` can be real or complex.



Singular Value Decomposition
============================

.. function:: kvxopt.lapack.gesvd(A, S[, jobu = 'N', jobvt = 'N', U = None, Vt = None])

    Singular value decomposition 
    
    .. math::

        A = U \Sigma V^T, \qquad A = U \Sigma V^H

    of a real or complex :math:`m` by :math:`n` matrix :math:`A`.

    ``S`` is a real matrix of length at least min{:math:`m`, :math:`n`}.
    On exit, its first  min{:math:`m`, :math:`n`} elements are the 
    singular values in descending order.

    The argument ``jobu`` controls how many left singular vectors are
    computed.  The possible values are :const:`'N'`, :const:`'A'`, 
    :const:`'S'` and :const:`'O'`.  If ``jobu`` is :const:`'N'`, no left 
    singular vectors are computed.  If ``jobu`` is :const:`'A'`, all left  
    singular vectors are computed and returned as columns of ``U``.
    If ``jobu`` is :const:`'S'`, the first min{:math:`m`, :math:`n`} left 
    singular vectors are computed and returned as columns of ``U``.
    If ``jobu`` is :const:`'O'`, the first min{:math:`m`, :math:`n`} left 
    singular vectors are computed and returned as columns of ``A``.
    The argument ``U`` is \None\ (if ``jobu`` is :const:`'N'`
    or :const:`'A'`) or a matrix of the same type as ``A``.

    The argument ``jobvt`` controls how many right singular vectors are
    computed.  The possible values are :const:`'N'`, :const:`'A'`, 
    :const:`'S'` and :const:`'O'`.  If ``jobvt`` is :const:`'N'`, no right
    singular vectors are computed.  If ``jobvt`` is :const:`'A'`, all right
    singular vectors are computed and returned as rows of ``Vt``.
    If ``jobvt`` is :const:`'S'`, the first min{:math:`m`, :math:`n`} 
    right singular vectors are computed and their (conjugate) transposes 
    are returned as rows of ``Vt``.  If ``jobvt`` is :const:`'O'`, the 
    first min{:math:`m`, :math:`n`} right singular vectors are computed 
    and their (conjugate) transposes are returned as rows of ``A``.
    Note that the (conjugate) transposes of the right singular vectors 
    (i.e., the matrix :math:`V^H`) are returned in ``Vt`` or ``A``.
    The argument ``Vt`` can be :const:`None` (if ``jobvt`` is :const:`'N'`
    or :const:`'A'`) or a matrix of the same type as ``A``.

    On exit, the contents of ``A`` are destroyed.


.. function:: kvxopt.lapack.gesdd(A, S[, jobz = 'N', U = None, Vt = None])

    Singular value decomposition of a real or complex :math:`m` by 
    :math:`n` matrix..  This function is based on a divide-and-conquer 
    algorithm and is faster than :func:`gesvd <kvxopt.lapack.gesdd>`.

    ``S`` is a real matrix of length at least min{:math:`m`, :math:`n`}.
    On exit, its first min{:math:`m`, :math:`n`} elements are the 
    singular values in descending order.

    The argument ``jobz`` controls how many singular vectors are computed.
    The possible values are :const:`'N'`, :const:`'A'`, :const:`'S'` and 
    :const:`'O'`.  If ``jobz`` is :const:`'N'`, no singular vectors are 
    computed.  If ``jobz`` is :const:`'A'`, all :math:`m` left singular 
    vectors are computed and returned as columns of ``U`` and all 
    :math:`n` right singular vectors are computed and returned as rows of 
    ``Vt``.  If ``jobz`` is :const:`'S'`, the first 
    min{:math:`m`, :math:`n`} left and right singular vectors are computed
    and returned as columns of ``U`` and rows of ``Vt``.
    If ``jobz`` is :const:`'O'` and :math:`m` is greater than or equal
    to :math:`n`, the first :math:`n` left singular vectors are returned as
    columns of ``A`` and the :math:`n` right singular vectors are returned
    as rows of ``Vt``.  If ``jobz`` is :const:`'O'` and :math:`m` is less 
    than :math:`n`, the :math:`m` left singular vectors are returned as 
    columns of ``U`` and the first :math:`m` right singular vectors are 
    returned as rows of ``A``.  Note that the (conjugate) transposes of 
    the right singular vectors are returned in ``Vt`` or ``A``.

    The argument ``U`` can be :const:`None` (if ``jobz`` is :const:`'N'`
    or :const:`'A'` of ``jobz`` is :const:`'O'` and :math:`m` is greater 
    than or equal to  :math:`n`)  or a matrix of the same type as ``A``.
    The argument ``Vt`` can be \None\ (if ``jobz`` is :const:`'N'`
    or :const:`'A'` or ``jobz`` is :const:`'O'` and :math`m` is less than
    :math:`n`) or a matrix of the same type as ``A``.

    On exit, the contents of ``A`` are destroyed.


Schur and Generalized Schur Factorization
=========================================

.. function:: kvxopt.lapack.gees(A[, w = None, V = None, select = None])

    Computes the Schur factorization 

    .. math::

        A = V S V^T \quad \mbox{($A$ real)}, \qquad 
        A = V S V^H \quad \mbox{($A$ complex)}

    of a real or complex :math:`n` by :math:`n` matrix :math:`A`.  

    If :math:`A` is real, the matrix of Schur vectors :math:`V` is 
    orthogonal, and :math:`S` is a real upper quasi-triangular matrix with
    1 by 1 or 2 by 2 diagonal blocks.  The 2 by 2 blocks correspond to 
    complex conjugate pairs of eigenvalues of :math:`A`.
    If :math:`A` is complex, the matrix of Schur vectors :math:`V` is 
    unitary, and :math:`S` is a complex upper triangular matrix with the 
    eigenvalues of :math:`A` on the diagonal.

    The optional argument ``w`` is a complex matrix of length at least 
    :math:`n`.  If it is provided, the eigenvalues of ``A`` are returned 
    in ``w``.  The optional argument ``V`` is an :math:`n` by :math:`n` 
    matrix of the same type as ``A``.  If it is provided, then the Schur 
    vectors are returned in ``V``.

    The argument ``select`` is an optional ordering routine.  It must be a
    Python function that can be called as ``f(s)`` with a complex 
    argument ``s``, and returns :const:`True` or :const:`False`.  The 
    eigenvalues for which ``select`` returns :const:`True` will be selected
    to appear first along the diagonal.  (In the real Schur factorization,
    if either one of a complex conjugate pair of eigenvalues is selected, 
    then both are selected.)

    On exit, ``A`` is replaced with the matrix :math:`S`.  The function 
    :func:`gees` returns an integer equal to the number of eigenvalues 
    that were selected by the ordering routine.  If ``select`` is 
    :const:`None`, then :func:`gees` returns 0.


As an example we compute the complex Schur form of the matrix

.. math::

    A = \left[\begin{array}{rrrrr}
        -7 &  -11 & -6  & -4 &  11 \\
         5 &  -3  &  3  & -12 & 0 \\
        11 &  11  & -5  & -14 & 9 \\
        -4 &   8  &  0  &  8 &  6 \\
        13 & -19  & -12 & -8 & 10 
        \end{array}\right].


>>> A = matrix([[-7., 5., 11., -4., 13.], [-11., -3., 11., 8., -19.], [-6., 3., -5., 0., -12.], 
                [-4., -12., -14., 8., -8.], [11., 0., 9., 6., 10.]])
>>> S = matrix(A, tc='z')
>>> w = matrix(0.0, (5,1), 'z')
>>> lapack.gees(S, w)
0
>>> print(S)
[ 5.67e+00+j1.69e+01 -2.13e+01+j2.85e+00  1.40e+00+j5.88e+00 -4.19e+00+j2.05e-01  3.19e+00-j1.01e+01]
[ 0.00e+00-j0.00e+00  5.67e+00-j1.69e+01  1.09e+01+j5.93e-01 -3.29e+00-j1.26e+00 -1.26e+01+j7.80e+00]
[ 0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  1.27e+01+j3.43e-17 -6.83e+00+j2.18e+00  5.31e+00-j1.69e+00]
[ 0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  0.00e+00-j0.00e+00 -1.31e+01-j0.00e+00 -2.60e-01-j0.00e+00]
[ 0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  0.00e+00-j0.00e+00 -7.86e+00-j0.00e+00]
>>> print(w)
[ 5.67e+00+j1.69e+01]
[ 5.67e+00-j1.69e+01]
[ 1.27e+01+j3.43e-17]
[-1.31e+01-j0.00e+00]
[-7.86e+00-j0.00e+00]

An ordered Schur factorization with the eigenvalues in the left half of 
the complex plane ordered first, can be computed as follows. 

>>> S = matrix(A, tc='z')
>>> def F(x): return (x.real < 0.0)
...
>>> lapack.gees(S, w, select = F)
2
>>> print(S)
[-1.31e+01-j0.00e+00 -1.72e-01+j7.93e-02 -2.81e+00+j1.46e+00  3.79e+00-j2.67e-01  5.14e+00-j4.84e+00]
[ 0.00e+00-j0.00e+00 -7.86e+00-j0.00e+00 -1.43e+01+j8.31e+00  5.17e+00+j8.79e+00  2.35e+00-j7.86e-01]
[ 0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  5.67e+00+j1.69e+01 -1.71e+01-j1.41e+01  1.83e+00-j4.63e+00]
[ 0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  5.67e+00-j1.69e+01 -8.75e+00+j2.88e+00]
[ 0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  1.27e+01+j3.43e-17]
>>> print(w)
[-1.31e+01-j0.00e+00]
[-7.86e+00-j0.00e+00]
[ 5.67e+00+j1.69e+01]
[ 5.67e+00-j1.69e+01]
[ 1.27e+01+j3.43e-17]


.. function:: kvxopt.lapack.gges(A, B[, a = None, b = None, Vl = None, Vr = None, select = None])
     
    Computes the generalized Schur factorization 

    .. math::

        A = V_l S V_r^T, \quad B = V_l T V_r^T \quad 
            \mbox{($A$ and $B$ real)}, 
     
        A = V_l S V_r^H, \quad B = V_l T V_r^H, \quad 
            \mbox{($A$ and $B$ complex)}

    of a pair of real or complex :math:`n` by :math:`n` matrices 
    :math:`A`, :math:`B`.  

    If :math:`A` and :math:`B` are real, then the matrices of left and 
    right Schur vectors :math:`V_l` and :math:`V_r` are orthogonal, 
    :math:`S` is a real upper quasi-triangular matrix with 1 by 1 or 2 by 
    2 diagonal blocks, and :math:`T` is a real triangular matrix with 
    nonnegative diagonal.  The 2 by 2 blocks along the diagonal of 
    :math:`S` correspond to complex conjugate pairs of generalized 
    eigenvalues of :math:`A`, :math:`B`.  If :math:`A` and :math:`B` are 
    complex, the matrices of left and right Schur vectors :math:`V_l` and 
    :math:`V_r` are unitary, :math:`S` is complex upper triangular, and 
    :math:`T` is complex upper triangular with nonnegative real diagonal.  

    The optional arguments ``a`` and ``b`` are :const:`'z'` and 
    :const:`'d'` matrices of length at least :math:`n`.  If these are 
    provided, the generalized eigenvalues of ``A``, ``B`` are returned in 
    ``a`` and ``b``.  (The generalized eigenvalues are the ratios 
    ``a[k] / b[k]``.)  The optional arguments ``Vl`` and ``Vr`` are 
    :math:`n` by :math:`n` matrices of the same type as ``A`` and ``B``.   
    If they are provided, then the left Schur vectors are returned in 
    ``Vl`` and the right Schur vectors are returned in ``Vr``.

    The argument ``select`` is an optional ordering routine.  It must be 
    a Python function that can be called as ``f(x,y)`` with a complex 
    argument ``x`` and a real argument ``y``, and returns :const:`True` or
    :const:`False`.  The eigenvalues for which ``select`` returns 
    :const:`True` will be selected to appear first on the diagonal.
    (In the real Schur factorization, if either one of a complex conjugate
    pair of eigenvalues is selected, then both are selected.)

    On exit, ``A`` is replaced with the matrix :math:`S` and ``B`` is 
    replaced with the matrix :math:`T`.  The function :func:`gges` returns
    an integer equal to the number of eigenvalues that were selected by 
    the ordering routine.  If ``select`` is :const:`None`, then 
    :func:`gges` returns 0.


As an example, we compute the generalized complex Schur form of the 
matrix :math:`A` of the previous example, and 

.. math::

    B = \left[\begin{array}{ccccc}
        1 & 0 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 & 0 \\
        0 & 0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 0 & 0 
        \end{array}\right].


>>> A = matrix([[-7., 5., 11., -4., 13.], [-11., -3., 11., 8., -19.], [-6., 3., -5., 0., -12.], 
                [-4., -12., -14., 8., -8.], [11., 0., 9., 6., 10.]])
>>> B = matrix(0.0, (5,5))
>>> B[:19:6] = 1.0
>>> S = matrix(A, tc='z')
>>> T = matrix(B, tc='z')
>>> a = matrix(0.0, (5,1), 'z')
>>> b = matrix(0.0, (5,1))
>>> lapack.gges(S, T, a, b)
0
>>> print(S)
[ 6.64e+00-j8.87e+00 -7.81e+00-j7.53e+00  6.16e+00-j8.51e-01  1.18e+00+j9.17e+00  5.88e+00-j4.51e+00]
[ 0.00e+00-j0.00e+00  8.48e+00+j1.13e+01 -2.12e-01+j1.00e+01  5.68e+00+j2.40e+00 -2.47e+00+j9.38e+00]
[ 0.00e+00-j0.00e+00  0.00e+00-j0.00e+00 -1.39e+01-j0.00e+00  6.78e+00-j0.00e+00  1.09e+01-j0.00e+00]
[ 0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  0.00e+00-j0.00e+00 -6.62e+00-j0.00e+00 -2.28e-01-j0.00e+00]
[ 0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  0.00e+00-j0.00e+00 -2.89e+01-j0.00e+00]
>>> print(T)
[ 6.46e-01-j0.00e+00  4.29e-01-j4.79e-02  2.02e-01-j3.71e-01  1.08e-01-j1.98e-01 -1.95e-01+j3.58e-01]
[ 0.00e+00-j0.00e+00  8.25e-01-j0.00e+00 -2.17e-01+j3.11e-01 -1.16e-01+j1.67e-01  2.10e-01-j3.01e-01]
[ 0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  7.41e-01-j0.00e+00 -3.25e-01-j0.00e+00  5.87e-01-j0.00e+00]
[ 0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  8.75e-01-j0.00e+00  4.84e-01-j0.00e+00]
[ 0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  0.00e+00-j0.00e+00  0.00e+00-j0.00e+00]
>>> print(a)
[ 6.64e+00-j8.87e+00]
[ 8.48e+00+j1.13e+01]
[-1.39e+01-j0.00e+00]
[-6.62e+00-j0.00e+00]
[-2.89e+01-j0.00e+00]
>>> print(b)
[ 6.46e-01]
[ 8.25e-01]
[ 7.41e-01]
[ 8.75e-01]
[ 0.00e+00]



Example: Analytic Centering
===========================

The analytic centering problem is defined as

.. math::

    \begin{array}{ll}
        \mbox{minimize} & -\sum\limits_{i=1}^m \log(b_i-a_i^Tx).
    \end{array}


In the code below we solve the problem using Newton's method.  At each 
iteration the Newton direction is computed by solving a positive definite 
set of linear equations

.. math::

    \newcommand{\diag}{\mathop{\bf diag}}
    \newcommand{\ones}{\mathbf 1}
    A^T \diag(b-Ax)^{-2} A v = -\diag(b-Ax)^{-1}\ones

(where :math:`A` has rows :math:`a_i^T`), and a suitable step size is 
determined by a backtracking line search.

We use the level-3 BLAS function :func:`blas.syrk <kvxopt.blas.syrk>` to 
form the Hessian 
matrix and the LAPACK function :func:`posv <kvxopt.lapack.posv>` to 
solve the Newton system.  
The code can be further optimized by replacing the matrix-vector products 
with the level-2 BLAS function :func:`blas.gemv <kvxopt.blas.gemv>`.

::

    from kvxopt import matrix, log, mul, div, blas, lapack 
    from math import sqrt

    def acent(A,b):
        """  
        Returns the analytic center of A*x <= b.
        We assume that b > 0 and the feasible set is bounded.
        """

        MAXITERS = 100
        ALPHA = 0.01
        BETA = 0.5
        TOL = 1e-8

        m, n = A.size
        x = matrix(0.0, (n,1))
        H = matrix(0.0, (n,n))

        for iter in xrange(MAXITERS):
            
            # Gradient is g = A^T * (1./(b-A*x)).
            d = (b - A*x)**-1
            g = A.T * d

            # Hessian is H = A^T * diag(d)^2 * A.
            Asc = mul( d[:,n*[0]], A )
            blas.syrk(Asc, H, trans='T')

            # Newton step is v = -H^-1 * g.
            v = -g
            lapack.posv(H, v)

            # Terminate if Newton decrement is less than TOL.
            lam = blas.dot(g, v)
            if sqrt(-lam) < TOL: return x

            # Backtracking line search.
            y = mul(A*v, d)
            step = 1.0
            while 1-step*max(y) < 0: step *= BETA 
            while True:
                if -sum(log(1-step*y)) < ALPHA*step*lam: break
                step *= BETA
            x += step*v
