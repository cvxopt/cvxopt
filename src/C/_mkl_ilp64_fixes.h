#ifndef CVXOPT_MKL_ILP64_FIXES_H
#define CVXOPT_MKL_ILP64_FIXES_H

#if defined(FIX_MKL_2025_ILP64_MISSING_SYMBOL) && defined(HAVE_BLAS_ILP64) && \
    !defined(OPENBLAS_ILP64_NAMING_SCHEME) && !defined(NO_APPEND_FORTRAN)
/*
 * MKL 2025.3 exports these ILP64 auxiliary LAPACK symbols without the
 * trailing Fortran underscore that BLAS_FUNC normally adds.
 */
#define dlarfx_64_ dlarfx_64
#define zlarfx_64_ zlarfx_64
#endif

#endif
