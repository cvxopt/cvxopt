.. _c-fftw:

*******************
Discrete Transforms
*******************

The :mod:`kvxopt.fftw` module is an interface to the FFTW library and 
contains routines for discrete Fourier, cosine, and sine transforms.  
This module is optional, and only installed when the FFTW library is made 
available during the kvxopt installation.

.. seealso:: 

    `FFTW3 code, documentation, copyright and license 
    <http://www.fftw.org>`_


Discrete Fourier Transform 
==========================

.. function:: kvxopt.fftw.dft(X)

    Replaces the columns of a dense complex matrix with their discrete 
    Fourier transforms:  if ``X`` has :math:`n` rows,

    .. math::

        X[k,:] := \sum_{j=0}^{n-1} e^{-2\pi j k \sqrt{-1}/n} X[j,:],
            \qquad k=0,\ldots,n-1.

.. function:: kvxopt.fftw.idft(X)

    Replaces the columns of a dense complex matrix with their inverse 
    discrete Fourier transforms: if ``X`` has :math:`n` rows,

    .. math::
    
        X[k,:] := 
            \frac{1}{n} \sum_{j=0}^{n-1} e^{2\pi j k \sqrt{-1}/n} X[j,:],
            \qquad k=0,\ldots,n-1.
   

The module also includes a discrete *N*-dimensional Fourier transform.
The input matrix is interpreted as an *N*-dimensional matrix stored 
in column-major order.  The discrete *N*-dimensional Fourier transform
computes the corresponding one-dimensional transform along each dimension. 
For example, the two-dimensional transform applies a one-dimensional 
transform to all the columns of the matrix, followed by a one-dimensional 
transform to all the rows of the matrix. 

.. function:: kvxopt.fftw.dftn(X[, dims = X.size])

    Replaces a dense complex matrix with its *N*-dimensional discrete
    Fourier transform.  The dimensions of the *N*-dimensional matrix 
    are given by the *N*-tuple ``dims``.  The two-dimensional transform is 
    computed as ``dftn(X, X.size)``. 


.. function:: kvxopt.fftw.idftn(X[, dims = X.size])

    Replaces a dense complex *N*-dimensional matrix with its inverse 
    *N*-dimensional discrete Fourier transform.  The dimensions of the 
    matrix are given by the tuple ``dims``. The two-dimensional inverse 
    transform is computed as ``idftn(X, X.size)``.


Discrete Cosine Transform
=========================

.. function:: kvxopt.fftw.dct(X[, type = 2])

    Replaces the columns of a dense real matrix with their discrete
    cosine transforms.  The second argument, an integer between 1 and 4,
    denotes the type of transform (DCT-I, DCT-II, DCT-III, DCT-IV).
    The DCT-I transform requires that the row dimension of ``X`` is at 
    least 2.  These transforms are defined as follows (for a matrix with 
    :math:`n` rows).

    .. math::

        \mbox{DCT-I:} \qquad 
            X[k,:] & := X[0,:] + (-1)^k X[n-1,:] + 
                2 \sum_{j=1}^{n-2} X[j,:] \cos(\pi j k /(n-1)), 
                \qquad k=0,\ldots,n-1.\\
        \mbox{DCT-II:} \qquad
            X[k,:] & := 2 \sum_{j=0}^{n-1} X[j,:] \cos(\pi(j+1/2)k/n), 
               \qquad k=0,\ldots,n-1.\\
        \mbox{DCT-III:} \qquad
            X[k,:] & := 
                X[0,:] + 2 \sum_{j=1}^{n-1} X[j,:] \cos(\pi j(k+1/2)/n),
                \qquad k=0,\ldots,n-1.\\
        \mbox{DCT-IV:} \qquad
            X[k,:] & := 
                2 \sum_{j=0}^{n-1} X[j,:] \cos(\pi (j+1/2)(k+1/2)/n), 
                \qquad k=0,\ldots,n-1.


.. function:: kvxopt.fftw.idct(X[, type = 2])

    Replaces the columns of a dense real matrix with the inverses
    of the discrete cosine transforms defined above.  


The module also includes a discrete *N*-dimensional cosine transform.
The input matrix is interpreted as an *N*-dimensional matrix stored 
in column-major order.  The discrete *N*-dimensional cosine transform
computes the corresponding one-dimensional transform along each dimension. 
For example, the two-dimensional transform applies a one-dimensional 
transform to all the rows of the matrix, followed by a one-dimensional 
transform to all the columns of the matrix. 


.. function:: kvxopt.fftw.dctn(X[, dims = X.size, type = 2])

    Replaces a dense real matrix with its *N*-dimensional discrete cosine 
    transform. The dimensions of the *N*-dimensional matrix are given by  
    the *N*-tuple ``dims``.  The two-dimensional transform is computed as 
    ``dctn(X, X.size)``. 


.. function:: kvxopt.fftw.idctn(X[, dims = X.size, type = 2])

    Replaces a dense real *N*-dimensional matrix with its inverse 
    *N*-dimensional discrete cosine transform. The dimensions of the 
    matrix are given by the tuple ``dims``.  The two-dimensional inverse 
    transform is computed as ``idctn(X, X.size)``.


Discrete Sine Transform 
=======================

.. function:: kvxopt.fftw.dst(X, dims[, type = 1])

    Replaces the columns of a dense real matrix with their discrete
    sine transforms.  The second argument, an integer between 1 and 4,
    denotes the type of transform (DST-I, DST-II, DST-III, DST-IV).
    These transforms are defined as follows (for a matrix with :math:`n` 
    rows).

    .. math::
       
        \mbox{DST-I:} \qquad
            X[k,:] & := 
                2 \sum_{j=0}^{n-1} X[j,:] \sin(\pi(j+1)(k+1)/(n+1)), 
                \qquad k=0,\ldots,n-1.\\
        \mbox{DST-II:} \qquad
            X[k,:] & := 2 \sum_{j=0}^{n-1} X[j,:] \sin(\pi(j+1/2)(k+1)/n), 
                \qquad k=0,\ldots,n-1.\\
        \mbox{DST-III:} \qquad
            X[k,:] & := (-1)^k X[n-1,:] + 2 \sum_{j=0}^{n-2} 
                X[j,:] \sin(\pi(j+1)(k+1/2)/n), \qquad k=0,\ldots,n-1. \\
        \mbox{DST-IV:} \qquad
            X[k,:] & := 
                2 \sum_{j=0}^{n-1} X[j,:] \sin(\pi (j+1/2)(k+1/2)/n), 
                \qquad k=0,\ldots,n-1.


.. function:: kvxopt.fftw.idst(X, dims[, type = 1])

    Replaces the columns of a dense real matrix with the inverses of the 
    discrete sine transforms defined above.  


The module also includes a discrete *N*-dimensional sine transform.  The 
input matrix is interpreted as an *N*-dimensional matrix stored in 
column-major order.  The discrete *N*-dimensional sine transform computes 
the corresponding one-dimensional transform along each dimension.  For 
example, the two-dimensional transform applies a one-dimensional 
transform to all the rows of the matrix, followed by a one-dimensional 
transform to all the columns of the matrix. 

.. function:: kvxopt.fftw.dstn(X[, dims = X.size, type = 2])

    Replaces a dense real matrix with its *N*-dimensional discrete sine 
    transform. The dimensions of the *N*-dimensional matrix are given by 
    the *N*-tuple ``dims``.  The two-dimensional transform is computed as 
    ``dstn(X, X.size)``. 


.. function:: kvxopt.fftw.idstn(X[, dims = X.size, type = 2])

    Replaces a dense real *N*-dimensional matrix with its inverse 
    *N*-dimensional discrete sine transform.  The dimensions of the 
    matrix are given by the tuple ``dims``.  The two-dimensional inverse 
    transform is computed as ``idstn(X, X.size)``.
