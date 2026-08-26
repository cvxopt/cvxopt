/*
 * Shared int*-to-CBLAS_INT* bridging shims for the BLAS routines called
 * from more than one extension module (misc_solvers.c, dsdp.c, fftw.c)
 * with plain C int arguments.  Not included from cvxopt.h itself: base.c
 * declares BLAS_FUNC(dscal)/BLAS_FUNC(dcopy) with void* arguments for its
 * generic type-dispatch tables, which would conflict with the double*
 * signatures used here.
 */
#ifndef __CVXOPT_INT_SHIMS__
#define __CVXOPT_INT_SHIMS__

extern void BLAS_FUNC(dcopy)(CBLAS_INT *n, double *x, CBLAS_INT *incx, double *y,
    CBLAS_INT *incy);
extern void BLAS_FUNC(dscal)(CBLAS_INT *n, double *alpha, double *x, CBLAS_INT *incx);

static void
cvxopt_int_dcopy(int *n, double *x, int *incx, double *y, int *incy)
{
  CBLAS_INT blas_n = *n, blas_incx = *incx, blas_incy = *incy;
  BLAS_FUNC(dcopy)(&blas_n, x, &blas_incx, y, &blas_incy);
}

static void
cvxopt_int_dscal(int *n, double *alpha, double *x, int *incx)
{
  CBLAS_INT blas_n = *n, blas_incx = *incx;
  BLAS_FUNC(dscal)(&blas_n, alpha, x, &blas_incx);
}

#endif  /* __CVXOPT_INT_SHIMS__ */
