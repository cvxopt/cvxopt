.. _c-blas:

******************
The BLAS Interface
******************

The :mod:`kvxopt.blas` module provides an interface to the double-precision
real and complex Basic Linear Algebra Subprograms (BLAS).  The names and 
calling sequences of the Python functions in the interface closely match 
the corresponding Fortran BLAS routines (described in the references below)
and their functionality is exactly the same.  Many of the operations 
performed by the BLAS routines can be implemented in a more straightforward
way by using the matrix arithmetic of the section :ref:`s-arithmetic`, 
combined with the slicing and indexing of the section :ref:`s-indexing`.
As an example, ``C = A * B`` gives the same result as the BLAS call 
``gemm(A, B, C)``.  The BLAS interface offers two advantages.  First, 
some of the functions it includes are not easily implemented using the 
basic matrix arithmetic.  For example, BLAS includes functions that 
efficiently exploit symmetry or triangular matrix structure.  Second, there
is a performance difference that can be significant for large matrices.   
Although our implementation of the basic matrix arithmetic makes internal 
calls to BLAS, it also often requires creating temporary matrices to store 
intermediate results.  The BLAS functions on the other hand always operate 
directly on their matrix arguments and never require any copying to 
temporary matrices.  Thus they can be viewed as generalizations of the 
in-place matrix addition and scalar multiplication of the section 
:ref:`s-arithmetic` to more complicated operations.

.. seealso::

    * C. L. Lawson, R. J. Hanson, D. R. Kincaid, F. T. Krogh, 
      Basic Linear Algebra Subprograms for Fortran Use,
      ACM Transactions on Mathematical Software, 5(3), 309-323, 1975.

    * J. J. Dongarra, J. Du Croz, S. Hammarling, R. J. Hanson,
      An Extended Set of Fortran Basic Linear Algebra Subprograms,
      ACM Transactions on Mathematical Software, 14(1), 1-17, 1988.

    * J. J. Dongarra, J. Du Croz, S. Hammarling, I. Duff,
      A Set of Level 3 Basic Linear Algebra Subprograms,
      ACM Transactions on Mathematical Software, 16(1), 1-17, 1990.


.. _s-conventions:

Matrix Classes 
==============

The BLAS exploit several types of matrix structure: symmetric, Hermitian, 
triangular, and banded.   We represent all these matrix classes by dense 
real or complex :class:`matrix <kvxopt.matrix>` objects, with additional 
arguments that specify the structure.


**Vector** 
    A real or complex :math:`n`-vector is represented by a :class:`matrix`
    of type :const:`'d'` or :const:`'z'` and length :math:`n`, with the 
    entries of the vector stored in column-major order. 


**General matrix**
    A general real or complex :math:`m` by :math:`n` matrix is represented 
    by a real or complex :class:`matrix` of size (:math:`m`, :math:`n`).


**Symmetric matrix**
    A real or complex symmetric matrix of order :math:`n` is represented
    by a real or complex :class:`matrix` of size (:math:`n`, :math:`n`), 
    and a character argument ``uplo``  with two possible values:  
    :const:`'L'` and :const:`'U'`.  If ``uplo``  is :const:`'L'`, the lower
    triangular part of the symmetric matrix is stored; if ``uplo`` is 
    :const:`'U'`, the upper triangular part is stored.  A square 
    :class:`matrix` ``X`` of size (:math:`n`, :math:`n`) can therefore be 
    used to represent the symmetric matrices

    .. math::

        \left[\begin{array}{ccccc}
            X[0,0]   & X[1,0]   & X[2,0]   & \cdots & X[n-1,0] \\
            X[1,0]   & X[1,1]   & X[2,1]   & \cdots & X[n-1,1] \\
            X[2,0]   & X[2,1]   & X[2,2]   & \cdots & X[n-1,2] \\
            \vdots   & \vdots   & \vdots   & \ddots & \vdots \\
            X[n-1,0] & X[n-1,1] & X[n-1,2] & \cdots & X[n-1,n-1]
        \end{array}\right] \quad \mbox{(uplo = 'L')}, 

        \left[\begin{array}{ccccc}
            X[0,0]   & X[0,1]   & X[0,2]   & \cdots & X[0,n-1] \\
            X[0,1]   & X[1,1]   & X[1,2]   & \cdots & X[1,n-1] \\
            X[0,2]   & X[1,2]   & X[2,2]   & \cdots & X[2,n-1] \\
            \vdots   & \vdots   & \vdots   & \ddots & \vdots \\
            X[0,n-1] & X[1,n-1] & X[2,n-1] & \cdots & X[n-1,n-1]
        \end{array}\right] \quad \mbox{(uplo = U')}. 

    
**Complex Hermitian matrix**
    A complex Hermitian matrix of order :math:`n` is represented by a 
    :class:`matrix` of type :const:`'z'` and size (:math:`n`, :math:`n`), 
    and a character argument ``uplo``  with the same meaning as for 
    symmetric matrices.  A complex :class:`matrix` ``X`` of size 
    (:math:`n`, :math:`n`) can represent the Hermitian  matrices

    .. math::

        \left[\begin{array}{ccccc}
            \Re X[0,0]   & \bar X[1,0]   & \bar X[2,0] & \cdots & 
                \bar X[n-1,0] \\
            X[1,0]   & \Re X[1,1]   & \bar X[2,1]   & \cdots & 
                \bar X[n-1,1] \\
            X[2,0]   & X[2,1]   & \Re X[2,2]   & \cdots & \bar X[n-1,2] \\
                \vdots   & \vdots   & \vdots   & \ddots & \vdots \\
            X[n-1,0] & X[n-1,1] & X[n-1,2] & \cdots & \Re X[n-1,n-1]
        \end{array}\right] \quad \mbox{(uplo = 'L')},

        \left[\begin{array}{ccccc}
            \Re X[0,0]   & X[0,1]   & X[0,2]   & \cdots & X[0,n-1] \\
            \bar X[0,1]   & \Re X[1,1]   & X[1,2]   & \cdots & X[1,n-1] \\
            \bar X[0,2]   & \bar X[1,2]   & \Re X[2,2]   & \cdots & 
                X[2,n-1] \\
            \vdots   & \vdots   & \vdots   & \ddots & \vdots \\
            \bar X[0,n-1] & \bar X[1,n-1] & \bar X[2,n-1] & \cdots & 
                \Re X[n-1,n-1]
        \end{array}\right] \quad \mbox{(uplo = 'U')}.

    
**Triangular matrix**
    A real or complex triangular matrix of order :math:`n` is represented
    by a real or complex :class:`matrix` of size (:math:`n`, :math:`n`), 
    and two character arguments: an argument ``uplo``  with possible values
    :const:`'L'` and :const:`'U'` to distinguish between lower and upper 
    triangular matrices, and an argument ``diag``  with possible values 
    :const:`'U'` and :const:`'N'` to distinguish between unit and non-unit 
    triangular matrices.  A square :class:`matrix` ``X`` of size 
    (:math:`n`, :math:`n`) can represent the triangular matrices

    .. math::

        \left[\begin{array}{ccccc}
            X[0,0]   & 0        & 0        & \cdots & 0 \\
            X[1,0]   & X[1,1]   & 0        & \cdots & 0 \\
            X[2,0]   & X[2,1]   & X[2,2]   & \cdots & 0 \\
            \vdots   & \vdots   & \vdots   & \ddots & \vdots \\
            X[n-1,0] & X[n-1,1] & X[n-1,2] & \cdots & X[n-1,n-1]
        \end{array}\right] \quad \mbox{(uplo = 'L', diag = 'N')}, 

        \left[\begin{array}{ccccc}
            1   & 0   & 0   & \cdots & 0 \\
            X[1,0]   & 1   & 0   & \cdots & 0 \\
            X[2,0]   & X[2,1]   & 1   & \cdots & 0 \\
            \vdots   & \vdots   & \vdots   & \ddots & \vdots \\
            X[n-1,0] & X[n-1,1] & X[n-1,2] & \cdots & 1
        \end{array}\right] \quad \mbox{(uplo = 'L', diag = 'U')}, 

        \left[\begin{array}{ccccc}
            X[0,0]   & X[0,1]   & X[0,2]   & \cdots & X[0,n-1] \\
            0   & X[1,1]   & X[1,2]   & \cdots & X[1,n-1] \\
            0   & 0   & X[2,2]   & \cdots & X[2,n-1] \\
            \vdots   & \vdots   & \vdots   & \ddots & \vdots \\
            0 & 0 & 0 & \cdots & X[n-1,n-1]
        \end{array}\right] \quad \mbox{(uplo = 'U', diag = 'N')}, 

        \left[\begin{array}{ccccc}
            1   & X[0,1]   & X[0,2]   & \cdots & X[0,n-1] \\
            0   & 1   & X[1,2]   & \cdots & X[1,n-1] \\
            0   & 0   & 1   & \cdots & X[2,n-1] \\
            \vdots   & \vdots   & \vdots   & \ddots & \vdots \\
            0 & 0 & 0 & \cdots & 1
        \end{array}\right] \quad \mbox{(uplo = 'U', diag = 'U')}.

    
**General band matrix**
    A general real or complex :math:`m` by :math:`n` band matrix  with 
    :math:`k_l` subdiagonals and :math:`k_u` superdiagonals is represented 
    by a real or complex :class:`matrix` ``X`` of size 
    (:math:`k_l + k_u + 1`, :math:`n`), and the two integers :math:`m` and 
    :math:`k_l`.   The diagonals of the band matrix are stored in the rows 
    of ``X``, starting at the top diagonal, and shifted horizontally so that
    the entries of column :math:`k` of the band matrix are stored in column
    :math:`k` of ``X``.  A :class:`matrix` ``X`` of size 
    (:math:`k_l + k_u + 1`, :math:`n`) therefore represents the :math:`m` 
    by :math:`n` band matrix

    .. math::

        \left[ \begin{array}{ccccccc}
            X[k_u,0]     & X[k_u-1,1]   & X[k_u-2,2]     & \cdots & 
                X[0,k_u] & 0               & \cdots \\
            X[k_u+1,0]   & X[k_u,1]     & X[k_u-1,2]     & \cdots & 
                X[1,k_u] & X[0,k_u+1]   & \cdots \\
            X[k_u+2,0]   & X[k_u+1,1]     & X[k_u,2]       & \cdots & 
                X[2,k_u] & X[1,k_u+1] & \cdots \\ 
            \vdots      & \vdots         &  \vdots        & \ddots & 
                \vdots   & \vdots          & \ddots  \\
            X[k_u+k_l,0] & X[k_u+k_l-1,1] & X[k_u+k_l-2,2] & \cdots &  
                &  & \\
            0            & X[k_u+k_l,1]   & X[k_u+k_l-1,2] & \cdots &  
                &  & \\
            \vdots       & \vdots         & \vdots         & \ddots &  
                &  & 
        \end{array}\right].

    
**Symmetric band matrix**
    A real or complex symmetric band matrix of order :math:`n` with 
    :math:`k` subdiagonals, is represented by a real or complex matrix ``X``
    of size (:math:`k+1`, :math:`n`), and an argument ``uplo`` to indicate 
    whether the subdiagonals (``uplo`` is :const:`'L'`) or superdiagonals 
    (``uplo`` is :const:`'U'`) are stored.  The :math:`k+1` diagonals are 
    stored as rows of ``X``, starting at the top diagonal (i.e., the main 
    diagonal if ``uplo`` is :const:`'L'`,  or the :math:`k`-th superdiagonal
    if ``uplo`` is :const:`'U'`) and shifted horizontally so that the 
    entries of the :math:`k`-th column of the band matrix are stored in 
    column :math:`k` of ``X``.  A :class:`matrix` ``X`` of size 
    (:math:`k+1`, :math:`n`) can therefore represent the band matrices 

    .. math::
        
        \left[ \begin{array}{ccccccc}
            X[0,0] & X[1,0]   & X[2,0]   & \cdots & X[k,0]   & 0
                & \cdots \\
            X[1,0] & X[0,1]   & X[1,1]   & \cdots & X[k-1,1] & X[k,1]   
                & \cdots \\
            X[2,0] & X[1,1]   & X[0,2]   & \cdots & X[k-2,2] & X[k-1,2] 
                & \cdots \\
            \vdots & \vdots   &  \vdots  & \ddots & \vdots   & \vdots   
                & \ddots \\
            X[k,0] & X[k-1,1] & X[k-2,2] & \cdots &  &  & \\
            0      & X[k,1]   & X[k-1,2] & \cdots &  &  & \\
            \vdots & \vdots   & \vdots   & \ddots &  &  & 
        \end{array}\right] \quad \mbox{(uplo = 'L')}, 

        \left[ \begin{array}{ccccccc}
            X[k,0]   & X[k-1,1] & X[k-2,2] & \cdots & X[0,k] & 0        
                 & \cdots \\
            X[k-1,1] & X[k,1]   & X[k-1,2] & \cdots & X[1,k] & X[0,k+1] 
                 & \cdots \\
            X[k-2,2] & X[k-1,2] & X[k,2]   & \cdots & X[2,k] & X[1,k+1] 
                 & \cdots \\
            \vdots   & \vdots   & \vdots   & \ddots & \vdots & \vdots   
                 & \ddots \\
            X[0,k]   & X[1,k]   & X[2,k]   & \cdots &  &  & \\
            0        & X[0,k+1] & X[1,k+1] & \cdots &  &  & \\
            \vdots   & \vdots   & \vdots   & \ddots &  &  & 
        \end{array}\right] \quad \mbox{(uplo='U')}.

       
**Hermitian  band matrix**
    A complex Hermitian band matrix of order :math:`n` with :math:`k` 
    subdiagonals is represented by a complex matrix of size 
    (:math:`k+1`, :math:`n`) and an argument ``uplo``, with the same meaning
    as for symmetric band matrices.  A :class:`matrix` ``X`` of size 
    (:math:`k+1`, :math:`n`) can represent the band matrices 

    .. math::

        \left[ \begin{array}{ccccccc}
            \Re X[0,0] & \bar X[1,0]   & \bar X[2,0]   & \cdots & 
                \bar X[k,0]   & 0        & \cdots \\
            X[1,0] & \Re X[0,1]   & \bar X[1,1]   & \cdots & 
                \bar X[k-1,1] & \bar X[k,1]   & \cdots \\
            X[2,0] & X[1,1]   & \Re X[0,2]   & \cdots & 
                \bar X[k-2,2] & \bar X[k-1,2] & \cdots \\
            \vdots & \vdots   &  \vdots  & \ddots & \vdots   
                & \vdots   & \ddots \\
            X[k,0] & X[k-1,1] & X[k-2,2] & \cdots &  &  & \\
            0      & X[k,1]   & X[k-1,2] & \cdots &  &  & \\
            \vdots & \vdots   & \vdots   & \ddots &  &  & 
        \end{array}\right] \quad \mbox{(uplo = 'L')}, 

        \left[ \begin{array}{ccccccc}
            \Re X[k,0]   & X[k-1,1] & X[k-2,2] & \cdots & X[0,k] & 
                0        & \cdots \\
            \bar X[k-1,1] & \Re X[k,1]   & X[k-1,2] & \cdots & 
                X[1,k] & X[0,k+1] & \cdots \\
            \bar X[k-2,2] & \bar X[k-1,2] & \Re X[k,2]   & \cdots & 
                X[2,k] & X[1,k+1] & \cdots \\
            \vdots   & \vdots   & \vdots   & \ddots & \vdots & 
                \vdots   & \ddots \\
            \bar X[0,k]   & \bar X[1,k]   & \bar X[2,k]   & \cdots &  
                &  & \\
            0        & \bar X[0,k+1] & \bar X[1,k+1] & \cdots &  &  & \\
            \vdots   & \vdots   & \vdots   & \ddots &  &  & 
        \end{array}\right] \quad \mbox{(uplo='U')}.


**Triangular band matrix**
    A triangular band matrix of order :math:`n` with :math:`k` subdiagonals
    or superdiagonals is represented by a real complex matrix of size 
    (:math:`k+1`, :math:`n`) and two character arguments ``uplo``  and 
    ``diag``, with similar conventions as for symmetric band matrices. 
    A :class:`matrix` ``X`` of size (:math:`k+1`, :math:`n`) can represent 
    the band matrices 

    .. math::

        \left[ \begin{array}{cccc}
            X[0,0] & 0        & 0        & \cdots \\
            X[1,0] & X[0,1]   & 0        & \cdots  \\
            X[2,0] & X[1,1]   & X[0,2]   & \cdots \\
            \vdots & \vdots   & \vdots   & \ddots \\
            X[k,0] & X[k-1,1] & X[k-2,2] & \cdots \\
            0      & X[k,1]   & X[k-1,1] & \cdots \\
            \vdots & \vdots   & \vdots   & \ddots 
        \end{array}\right] \quad \mbox{(uplo = 'L', diag = 'N')}, 

        \left[ \begin{array}{cccc}
            1      & 0        & 0        & \cdots \\
            X[1,0] & 1        & 0        & \cdots  \\
            X[2,0] & X[1,1]   & 1        & \cdots \\
            \vdots & \vdots   & \vdots   & \ddots \\
            X[k,0] & X[k-1,1] & X[k-2,2] & \cdots \\
            0      & X[k,1]   & X[k-1,2] & \cdots \\
            \vdots & \vdots   & \vdots   & \ddots 
        \end{array}\right] \quad \mbox{(uplo = 'L', diag = 'U')},

        \left[ \begin{array}{ccccccc}
            X[k,0] & X[k-1,1] & X[k-2,3] & \cdots & X[0,k]  & 
                0        & \cdots\\
            0      & X[k,1]   & X[k-1,2] & \cdots & X[1,k]  & 
                X[0,k+1] & \cdots \\
            0      & 0        & X[k,2]   & \cdots & X[2,k]  & 
                X[1,k+1] & \cdots \\
            \vdots & \vdots   &  \vdots  & \ddots & \vdots  & 
                \vdots   & \ddots  
        \end{array}\right] \quad \mbox{(uplo = 'U', diag = 'N')},

        \left[ \begin{array}{ccccccc}
            1      & X[k-1,1] & X[k-2,3] & \cdots & X[0,k]  & 
                0        & \cdots\\
            0      & 1        & X[k-1,2] & \cdots & X[1,k]  & 
                X[0,k+1] & \cdots \\
            0      & 0        & 1        & \cdots & X[2,k]  & 
                X[1,k+1] & \cdots \\
            \vdots & \vdots   &  \vdots  & \ddots & \vdots  & 
                \vdots   & \ddots  
        \end{array}\right] \quad \mbox{(uplo = 'U', diag = 'U')}.


When discussing BLAS functions in the following sections we will omit 
several less important optional arguments that can be used to select 
submatrices for in-place operations.  The complete specification is  
documented in the docstrings of the source code, and can be viewed with the
:program:`pydoc` help program.


.. _s-blas-1:

Level 1 BLAS
============

The level 1 functions implement vector operations.  

.. function:: kvxopt.blas.scal(alpha, x)

    Scales a vector by a constant: 

    .. math::

        x := \alpha x.
    
    If ``x`` is a real :class:`matrix`, the scalar argument ``alpha`` must 
    be a Python integer or float.  If ``x`` is complex, ``alpha`` can be an 
    integer, float, or complex.


.. function:: kvxopt.blas.nrm2(x)

    Euclidean norm of a vector:  returns 

    .. math::

        \|x\|_2.


.. function:: kvxopt.blas.asum(x)

    1-Norm of a vector: returns 

    .. math::

        \|x\|_1 \quad \mbox{($x$ real)}, \qquad  
        \|\Re x\|_1 + \|\Im x\|_1 \quad \mbox{($x$ complex)}.


.. function:: kvxopt.blas.iamax(x)

    Returns 

    .. math::
 
        \mathop{\rm argmax}_{k=0,\ldots,n-1} |x_k| \quad \mbox{($x$ real)}, 
        \qquad
        \mathop{\rm argmax}_{k=0,\ldots,n-1} |\Re x_k| + |\Im x_k| \quad 
            \mbox{($x$ complex)}. 


    If more than one coefficient achieves the maximum, the index of the 
    first :math:`k` is returned.  


.. function:: kvxopt.blas.swap(x, y)

    Interchanges two vectors:

    .. math::

        x \leftrightarrow y.

    ``x``  and ``y`` are matrices of the same type (:const:`'d'` or 
    :const:`'z'`).
    

.. function:: kvxopt.blas.copy(x, y)

    Copies a vector to another vector:

    .. math::

        y := x.
    
    ``x`` and ``y`` are matrices of the same type (:const:`'d'` or 
    :const:`'z'`).


.. function:: kvxopt.blas.axpy(x, y[, alpha = 1.0])

    Constant times a vector plus a vector:  

    .. math::

        y := \alpha x + y.
    
    ``x`` and ``y`` are matrices of the same type (:const:`'d'` or 
    :const:`'z'`).  If ``x`` is real, the scalar argument ``alpha`` must be 
    a Python integer or float.  If ``x`` is complex, ``alpha`` can be an 
    integer, float, or complex.  


.. function:: kvxopt.blas.dot(x, y)

    Returns 

    .. math::

        x^Hy. 

    ``x`` and ``y`` are matrices of the same type (:const:`'d'` or 
    :const:`'z'`).  


.. function:: kvxopt.blas.dotu(x, y)

    Returns 

    .. math::

        x^Ty. 
    
    ``x`` and ``y`` are matrices of the same type (:const:`'d'` or 
    :const:`'z'`).



.. _s-blas-2:

Level 2 BLAS
============

The level 2 functions implement matrix-vector products and rank-1 and 
rank-2 matrix updates.  Different types of matrix structure can be exploited
using the conventions of the section :ref:`s-conventions`. 

.. function:: kvxopt.blas.gemv(A, x, y[, trans = 'N', alpha = 1.0, beta = 0.0])

    Matrix-vector product with a general matrix:  

    .. math::
        
        y & := \alpha Ax + \beta y \quad 
            (\mathrm{trans} = \mathrm{'N'}), \\
        y & := \alpha A^T x + \beta y \quad 
            (\mathrm{trans} = \mathrm{'T'}),  \\
        y & := \alpha A^H x + \beta y \quad 
            (\mathrm{trans} = \mathrm{'C'}). 

    The arguments ``A``, ``x``, and ``y`` must have the same type 
    (:const:`'d'` or :const:`'z'`).  Complex values of ``alpha`` and 
    ``beta`` are only allowed if ``A`` is complex. 


.. function:: kvxopt.blas.symv(A, x, y[, uplo = 'L', alpha = 1.0, beta = 0.0])

    Matrix-vector  product with a real symmetric matrix:  

    .. math::

        y := \alpha A x + \beta y,

    where :math:`A` is a real symmetric matrix.  The arguments ``A``, 
    ``x``, and ``y`` must have type :const:`'d'`, and ``alpha`` and 
    ``beta`` must be real.


.. function:: kvxopt.blas.hemv(A, x, y[, uplo = 'L', alpha = 1.0, beta = 0.0])

    Matrix-vector  product with a real symmetric or complex Hermitian 
    matrix: 

    .. math::

        y := \alpha A x + \beta y,

    where :math:`A` is real symmetric or complex Hermitian.  The arguments 
    ``A``, ``x``, ``y`` must have the same type (:const:`'d'` or 
    :const:`'z'`).  Complex values of ``alpha`` and ``beta`` are only
    allowed if ``A``  is complex. 


.. function:: kvxopt.blas.trmv(A, x[, uplo = 'L', trans = 'N', diag = 'N'])

    Matrix-vector  product with a triangular matrix: 

    .. math::

        x & := Ax \quad (\mathrm{trans} = \mathrm{'N'}), \\
        x & := A^T x \quad (\mathrm{trans} = \mathrm{'T'}), \\
        x & := A^H x \quad (\mathrm{trans} = \mathrm{'C'}), 

    where :math:`A` is square and triangular.  The arguments ``A`` and 
    ``x`` must have the same type (:const:`'d'` or :const:`'z'`).


.. function:: kvxopt.blas.trsv(A, x[, uplo = 'L', trans = 'N', diag = 'N'])

    Solution of a nonsingular triangular set of linear equations:

    .. math::
   
        x & := A^{-1}x \quad (\mathrm{trans} = \mathrm{'N'}), \\
        x & := A^{-T}x \quad (\mathrm{trans} = \mathrm{'T'}), \\
        x & := A^{-H}x \quad (\mathrm{trans} = \mathrm{'C'}), 

    where :math:`A` is square and triangular with nonzero diagonal elements.
    The arguments ``A``  and ``x`` must have the same type (:const:`'d'` or
    :const:`'z'`).


.. function:: kvxopt.blas.gbmv(A, m, kl, x, y[, trans = 'N', alpha = 1.0, beta = 0.0])
    
    Matrix-vector product with a general band matrix:

    .. math::

        y & := \alpha Ax + \beta y \quad 
            (\mathrm{trans} = \mathrm{'N'}), \\
        y & := \alpha A^T x + \beta y \quad
            (\mathrm{trans} = \mathrm{'T'}),  \\
        y & := \alpha A^H x + \beta y \quad 
            (\mathrm{trans} = \mathrm{'C'}),

    where  :math:`A` is a rectangular band matrix with :math:`m` rows and 
    :math:`k_l` subdiagonals.  The arguments ``A``, ``x``, ``y``  must have 
    the same type (:const:`'d'` or :const:`'z'`).  Complex values of 
    ``alpha``  and ``beta``  are only allowed if ``A`` is complex.


.. function:: kvxopt.blas.sbmv(A, x, y[, uplo = 'L', alpha = 1.0, beta = 0.0])

    Matrix-vector  product with a real symmetric band matrix:

    .. math::
 
        y := \alpha Ax + \beta y,

    where :math:`A`  is a real symmetric band matrix.  The arguments 
    ``A``, ``x``, ``y``  must have type :const:`'d'`, and ``alpha`` and 
    ``beta`` must be real.


.. function:: kvxopt.blas.hbmv(A, x, y[, uplo = 'L', alpha = 1.0, beta = 0.0])

    Matrix-vector  product with a real symmetric or complex Hermitian band 
    matrix:

    .. math::

        y := \alpha Ax + \beta y,

    where :math:`A` is a real symmetric or complex Hermitian band matrix.
    The arguments ``A``, ``x``,  ``y``  must have the same type
    (:const:`'d'` or :const:`'z'`).  Complex values of ``alpha`` and 
    ``beta``  are only allowed if ``A``  is complex. 


.. function:: kvxopt.blas.tbmv(A, x[, uplo = 'L', trans = 'N',  diag = 'N'])

    Matrix-vector  product with a triangular band matrix:

    .. math::

        x & := Ax \quad (\mathrm{trans} = \mathrm{'N'}), \\
        x & := A^T x \quad (\mathrm{trans} = \mathrm{'T'}), \\
        x & := A^H x \quad (\mathrm{trans} = \mathrm{'C'}). 

    The arguments ``A`` and ``x``  must have the same type (:const:`'d'` or
    :const:`'z'`).


.. function:: kvxopt.blas.tbsv(A, x[, uplo = 'L', trans = 'N', diag = 'N'])

    Solution of a triangular banded set of linear equations:

    .. math::

        x & := A^{-1}x \quad (\mathrm{trans} = \mathrm{'N'}), \\
        x & := A^{-T} x \quad (\mathrm{trans} = \mathrm{'T'}), \\
        x & := A^{-H} x \quad (\mathrm{trans} = \mathrm{'T'}), 

    where :math:`A` is a triangular band matrix of with nonzero diagonal 
    elements.  The arguments ``A``  and ``x``  must have the same type 
    (:const:`'d'` or :const:`'z'`).


.. function:: kvxopt.blas.ger(x, y, A[, alpha = 1.0])

    General rank-1 update:

    .. math::

        A := A + \alpha x y^H,

    where :math:`A` is a general matrix.  The arguments ``A``, ``x``, and 
    ``y``  must have the same type (:const:`'d'` or :const:`'z'`).  Complex
    values of ``alpha``  are only allowed if ``A``  is complex.


.. function:: kvxopt.blas.geru(x, y, A[, alpha = 1.0])

    General rank-1 update:

    .. math::

        A := A + \alpha x y^T, 

    where :math:`A` is a general matrix.  The arguments ``A``, ``x``,  and 
    ``y``  must have the same type (:const:`'d'` or :const:`'z'`).  Complex
    values of ``alpha``  are only allowed if ``A``  is complex.


.. function:: kvxopt.blas.syr(x, A[, uplo = 'L', alpha = 1.0])

    Symmetric rank-1 update:

    .. math::
 
        A := A + \alpha xx^T,

    where :math:`A` is a real symmetric matrix.  The arguments ``A``  and 
    ``x``  must have type :const:`'d'`.  ``alpha``  must be a real number.


.. function:: kvxopt.blas.her(x, A[, uplo = 'L', alpha = 1.0])

    Hermitian rank-1 update:

    .. math::

        A := A + \alpha xx^H, 

    where :math:`A` is a real symmetric or complex Hermitian matrix.  The 
    arguments ``A``  and ``x``  must have the same type (:const:`'d'` or 
    :const:`'z'`).  ``alpha``  must be a real number.


.. function:: kvxopt.blas.syr2(x, y, A[, uplo = 'L', alpha = 1.0])

    Symmetric rank-2  update:

    .. math::

        A := A + \alpha (xy^T + yx^T),

    where :math:`A` is a real symmetric matrix.  The arguments ``A``, ``x``,
    and ``y`` must have type :const:`'d'`.  ``alpha``  must be real.


.. function:: kvxopt.blas.her2(x, y, A[, uplo = 'L', alpha = 1.0])

    Symmetric rank-2  update:

    .. math::

        A := A + \alpha xy^H + \bar \alpha yx^H,

    where :math:`A` is a a real symmetric or complex Hermitian matrix.
    The arguments ``A``, ``x``, and ``y`` must have the same type  
    (:const:`'d'` or :const:`'z'`).  Complex values of ``alpha`` are only 
    allowed if ``A`` is complex.


As an example, the following code multiplies the tridiagonal matrix

.. math::

    A = \left[\begin{array}{rrrr}
          1 &  6 &  0 & 0 \\ 
          2 & -4 &  3 & 0 \\ 
          0 & -3 & -1 & 1 
    \end{array}\right]

with the vector :math:`x = (1,-1,2,-2)`.

>>> from kvxopt import matrix
>>> from kvxopt.blas import gbmv
>>> A = matrix([[0., 1., 2.],  [6., -4., -3.],  [3., -1., 0.],  [1., 0., 0.]])
>>> x = matrix([1., -1., 2., -2.])
>>> y = matrix(0., (3,1))
>>> gbmv(A, 3, 1, x, y)
>>> print(y)
[-5.00e+00]
[ 1.20e+01]
[-1.00e+00]


The following example illustrates the use of 
:func:`tbsv <kvxopt.blas.tbsv>`.

>>> from kvxopt import matrix
>>> from kvxopt.blas import tbsv
>>> A = matrix([-6., 5., -1., 2.], (1,4))
>>> x = matrix(1.0, (4,1))
>>> tbsv(A, x)  # x := diag(A)^{-1}*x
>>> print(x)
[-1.67e-01]
[ 2.00e-01]
[-1.00e+00]
[ 5.00e-01]


.. _s-blas-3:

Level 3 BLAS 
============

The level 3 BLAS include functions for matrix-matrix multiplication.

.. function:: kvxopt.blas.gemm(A, B, C[, transA = 'N', transB = 'N', alpha = 1.0, beta = 0.0])

    Matrix-matrix product of two general matrices:  

    .. math::

        \newcommand{\op}{\mathop{\mathrm{op}}}
        C := \alpha \op(A) \op(B) + \beta C 

    where

    .. math::

        \newcommand{\op}{\mathop{\mathrm{op}}}
        \op(A) =  \left\{ \begin{array}{ll}
            A & \mathrm{transA} = \mathrm{'N'} \\
            A^T & \mathrm{transA} = \mathrm{'T'} \\
            A^H & \mathrm{transA} = \mathrm{'C'} \end{array} \right.
        \qquad
        \op(B) =  \left\{ \begin{array}{ll}
            B & \mathrm{transB} = \mathrm{'N'} \\
            B^T & \mathrm{transB} = \mathrm{'T'} \\
            B^H & \mathrm{transB} = \mathrm{'C'}. \end{array} \right.

    The arguments ``A``, ``B``, and ``C`` must have the same type 
    (:const:`'d'` or :const:`'z'`).  Complex values of ``alpha`` and 
    ``beta`` are only allowed if ``A`` is complex.


.. function:: kvxopt.blas.symm(A, B, C[, side = 'L', uplo = 'L', alpha =1.0,  beta = 0.0])

    Product of a real or complex symmetric matrix :math:`A` and a general 
    matrix :math:`B`:

    .. math::

        C & := \alpha AB + \beta C \quad (\mathrm{side} = \mathrm{'L'}), \\
        C & := \alpha BA + \beta C \quad (\mathrm{side} = \mathrm{'R'}). 

    The arguments ``A``, ``B``, and ``C``  must have the same type 
    (:const:`'d'` or :const:`'z'`).  Complex values of ``alpha``  and 
    ``beta`` are only allowed if ``A`` is complex.


.. function:: kvxopt.blas.hemm(A, B, C[, side = 'L', uplo = 'L', alpha = 1.0,  beta = 0.0])

    Product of a real symmetric or complex Hermitian matrix :math:`A` and a 
    general matrix :math:`B`:

    .. math::
 
        C & := \alpha AB + \beta C \quad (\mathrm{side} = \mathrm{'L'}), \\
        C & := \alpha BA + \beta C \quad (\mathrm{side} = \mathrm{'R'}). 

    The arguments ``A``, ``B``,  and ``C`` must have the same type 
    (:const:`'d'` or :const:`'z'`).  Complex values of ``alpha`` and 
    ``beta``  are only allowed if ``A`` is complex.


.. function:: kvxopt.blas.trmm(A, B[, side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0])

    Product of a triangular matrix :math:`A` and a general matrix :math:`B`:

    .. math::

        \newcommand{\op}{\mathop{\mathrm{op}}} 
        \begin{split}
        B & := \alpha\op(A)B \quad (\mathrm{side} = \mathrm{'L'}), \\ 
        B & := \alpha B\op(A) \quad (\mathrm{side} = \mathrm{'R'}) 
        \end{split}

    where

    .. math::

        \newcommand{\op}{\mathop{\mathrm{op}}}
        \op(A) =  \left\{ \begin{array}{ll}
             A & \mathrm{transA} = \mathrm{'N'} \\
             A^T & \mathrm{transA} = \mathrm{'T'} \\
             A^H & \mathrm{transA} = \mathrm{'C'}. \end{array} \right.

    The arguments ``A`` and ``B`` must have the same type (:const:`'d'` or 
    :const:`'z'`).  Complex values of ``alpha`` are only allowed if ``A`` 
    is complex.


.. function:: kvxopt.blas.trsm(A, B[, side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = 1.0])

    Solution of a nonsingular triangular system of equations:

    .. math::
 
        \newcommand{\op}{\mathop{\mathrm{op}}}
        \begin{split}
        B & := \alpha \op(A)^{-1}B \quad (\mathrm{side} = \mathrm{'L'}), \\
        B & := \alpha B\op(A)^{-1} \quad (\mathrm{side} = \mathrm{'R'}), 
        \end{split}
        
    where

    .. math::

        \newcommand{\op}{\mathop{\mathrm{op}}}
        \op(A) =  \left\{ \begin{array}{ll}
            A & \mathrm{transA} = \mathrm{'N'} \\
            A^T & \mathrm{transA} = \mathrm{'T'} \\
            A^H & \mathrm{transA} = \mathrm{'C'}, \end{array} \right.

    :math:`A` is triangular and :math:`B` is a general matrix.  The 
    arguments ``A`` and ``B`` must have the same type (:const:`'d'` or 
    :const:`'z'`).  Complex values of ``alpha`` are only allowed if ``A`` 
    is complex.


.. function:: kvxopt.blas.syrk(A, C[, uplo = 'L', trans = 'N', alpha = 1.0, beta = 0.0])

    Rank-:math:`k` update of a real or complex symmetric matrix :math:`C`:

    .. math::

        C & := \alpha AA^T + \beta C \quad 
            (\mathrm{trans} = \mathrm{'N'}),  \\
        C & := \alpha A^TA + \beta C \quad 
            (\mathrm{trans} = \mathrm{'T'}), 

    where :math:`A` is a general matrix.  The arguments ``A`` and ``C`` 
    must have the same type (:const:`'d'` or :const:`'z'`).  Complex values
    of ``alpha``  and ``beta`` are only allowed if ``A`` is complex.


.. function:: kvxopt.blas.herk(A, C[, uplo = 'L', trans = 'N', alpha = 1.0, beta = 0.0])

    Rank-:math:`k` update of a real symmetric or complex Hermitian matrix 
    :math:`C`:

    .. math::

        C & := \alpha AA^H + \beta C \quad 
            (\mathrm{trans} = \mathrm{'N'}), \\
        C & := \alpha A^HA + \beta C \quad 
            (\mathrm{trans} = \mathrm{'C'}),

    where :math:`A` is a general matrix.  The arguments ``A`` and ``C`` 
    must have the same type (:const:`'d'` or :const:`'z'`).  ``alpha`` and 
    ``beta`` must be real.


.. function:: kvxopt.blas.syr2k(A, B, C[, uplo = 'L', trans = 'N', alpha = 1.0, beta = 0.0])

    Rank-:math:`2k` update of a real or complex symmetric matrix :math:`C`:

    .. math::

        C & := \alpha (AB^T + BA^T) + \beta C \quad 
            (\mathrm{trans} = \mathrm{'N'}), \\
        C & := \alpha (A^TB + B^TA) + \beta C \quad 
            (\mathrm{trans} = \mathrm{'T'}). 

    :math:`A` and :math:`B` are general real or complex matrices.  The 
    arguments ``A``, ``B``, and ``C`` must have the same type.  Complex 
    values of ``alpha``  and ``beta`` are only allowed if ``A`` is complex.


.. function:: kvxopt.blas.her2k(A, B, C[, uplo = 'L', trans = 'N', alpha = 1.0, beta = 0.0])

    Rank-:math:`2k` update of a real symmetric or complex Hermitian matrix 
    :math:`C`:

    .. math::

        C & := \alpha AB^H + \bar \alpha BA^H + \beta C \quad 
            (\mathrm{trans} = \mathrm{'N'}), \\
        C & := \alpha A^HB + \bar\alpha B^HA + \beta C \quad 
            (\mathrm{trans} = \mathrm{'C'}), 

    where :math:`A` and :math:`B` are general matrices.  The arguments 
    ``A``, ``B``, and ``C`` must have the same type (:const:`'d'` or 
    :const:`'z'`).   Complex values of ``alpha`` are only allowed if ``A`` 
    is complex.  ``beta`` must be real.
