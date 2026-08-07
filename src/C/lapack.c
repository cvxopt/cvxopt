/*
 * Copyright 2012-2026 M. Andersen and L. Vandenberghe.
 * Copyright 2010-2011 L. Vandenberghe.
 * Copyright 2004-2009 J. Dahl and L. Vandenberghe.
 *
 * This file is part of CVXOPT.
 *
 * CVXOPT is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * CVXOPT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "Python.h"
#include "cvxopt.h"
#include "_mkl_ilp64_fixes.h"
#include "misc.h"

#ifndef _MSC_VER
typedef complex double complex_t;
#else
typedef _Dcomplex complex_t;
#endif

#define err_lapack { PyErr_SetObject( (info < 0) ? PyExc_ValueError :\
    PyExc_ArithmeticError, Py_BuildValue("n", (Py_ssize_t) (info) )); \
    return NULL;}

PyDoc_STRVAR(lapack__doc__, "Interface to the LAPACK library.\n\n"
"Double-precision real and complex LAPACK routines for solving sets of\n"
"linear equations, linear least-squares and least-norm problems,\n"
"symmetric and Hermitian eigenvalue problems, singular value \n"
"decomposition, and Schur factorization.\n\n"
"For more details, see the LAPACK Users' Guide at \n"
"www.netlib.org/lapack/lug/lapack_lug.html.\n\n"
"Double and complex matrices and vectors are stored in CVXOPT matrices\n"
"using the conventional BLAS storage schemes, with the CVXOPT matrix\n"
"buffers interpreted as one-dimensional arrays.  For each matrix \n"
"argument X, an additional integer argument offsetX specifies the start\n"
"of the array, i.e., the pointer X->buffer + offsetX is passed to the\n"
"LAPACK function.  The other arguments (dimensions and options) have the\n"
"same meaning as in the LAPACK definition.  Default values of the\n"
"dimension arguments are derived from the CVXOPT matrix sizes.\n\n"
"If a routine from the LAPACK library returns with a positive 'info'\n"
"value, an ArithmeticError is raised.  If it returns with a negative\n"
"'info' value, a ValueError is raised.  In both cases the value of \n"
"'info' is returned as an argument to the exception.");

/* LAPACK prototypes */
extern CBLAS_INT BLAS_FUNC(ilaenv)(CBLAS_INT  *ispec, char **name, char **opts, CBLAS_INT *n1,
    CBLAS_INT *n2, CBLAS_INT *n3, CBLAS_INT *n4);

extern void BLAS_FUNC(dlarfg)(CBLAS_INT *n, double *alpha, double *x, CBLAS_INT *incx,
    double *tau);
extern void BLAS_FUNC(zlarfg)(CBLAS_INT *n, complex_t *alpha, complex_t *x,
    CBLAS_INT *incx, complex_t *tau);
extern void BLAS_FUNC(dlarfx)(char *side, CBLAS_INT *m, CBLAS_INT *n, double *V,
    double *tau, double *C, CBLAS_INT *ldc, double *work);
extern void BLAS_FUNC(zlarfx)(char *side, CBLAS_INT *m, CBLAS_INT *n,
    complex_t *V, complex_t *tau, complex_t *C, CBLAS_INT *ldc,
    complex_t *work);

extern void BLAS_FUNC(dlacpy)(char *uplo, CBLAS_INT *m, CBLAS_INT *n, double *A, CBLAS_INT *lda,
    double *B, CBLAS_INT *ldb);
extern void BLAS_FUNC(zlacpy)(char *uplo, CBLAS_INT *m, CBLAS_INT *n, complex_t *A,
    CBLAS_INT *lda, complex_t *B, CBLAS_INT *ldb);

extern void BLAS_FUNC(dgetrf)(CBLAS_INT *m, CBLAS_INT *n, double *A, CBLAS_INT *lda, CBLAS_INT *ipiv,
    CBLAS_INT *info);
extern void BLAS_FUNC(zgetrf)(CBLAS_INT *m, CBLAS_INT *n, complex_t *A, CBLAS_INT *lda, CBLAS_INT *ipiv,
    CBLAS_INT *info);
extern void BLAS_FUNC(dgetrs)(char *trans, CBLAS_INT *n, CBLAS_INT *nrhs, double *A, CBLAS_INT *lda,
    CBLAS_INT *ipiv, double *B, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(zgetrs)(char *trans, CBLAS_INT *n, CBLAS_INT *nrhs, complex_t *A,
    CBLAS_INT *lda, CBLAS_INT *ipiv, complex_t *B, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(dgetri)(CBLAS_INT *n, double *A, CBLAS_INT *lda, CBLAS_INT *ipiv, double *work,
    CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(zgetri)(CBLAS_INT *n, complex_t *A, CBLAS_INT *lda, CBLAS_INT *ipiv,
    complex_t *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(dgesv)(CBLAS_INT *n, CBLAS_INT *nrhs, double *A, CBLAS_INT *lda, CBLAS_INT *ipiv,
    double *B, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(zgesv)(CBLAS_INT *n, CBLAS_INT *nrhs, complex_t *A, CBLAS_INT *lda,
    CBLAS_INT *ipiv, complex_t *B, CBLAS_INT *ldb, CBLAS_INT *info);

extern void BLAS_FUNC(dgbtrf)(CBLAS_INT *m, CBLAS_INT *n, CBLAS_INT *kl, CBLAS_INT *ku, double *AB,
    CBLAS_INT *ldab, CBLAS_INT *ipiv, CBLAS_INT *info);
extern void BLAS_FUNC(zgbtrf)(CBLAS_INT *m, CBLAS_INT *n, CBLAS_INT *kl, CBLAS_INT *ku, complex_t *AB,
    CBLAS_INT *ldab, CBLAS_INT *ipiv, CBLAS_INT *info);
extern void BLAS_FUNC(dgbtrs)(char *trans, CBLAS_INT *n, CBLAS_INT *kl, CBLAS_INT *ku, CBLAS_INT *nrhs,
    double *AB, CBLAS_INT *ldab, CBLAS_INT *ipiv, double *B, CBLAS_INT *ldB, CBLAS_INT *info);
extern void BLAS_FUNC(zgbtrs)(char *trans, CBLAS_INT *n, CBLAS_INT *kl, CBLAS_INT *ku, CBLAS_INT *nrhs,
    complex_t *AB, CBLAS_INT *ldab, CBLAS_INT *ipiv, complex_t *B,
    CBLAS_INT *ldB, CBLAS_INT *info);
extern void BLAS_FUNC(dgbsv)(CBLAS_INT *n, CBLAS_INT *kl, CBLAS_INT *ku, CBLAS_INT *nrhs, double *ab,
    CBLAS_INT *ldab, CBLAS_INT *ipiv, double *b, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(zgbsv)(CBLAS_INT *n, CBLAS_INT *kl, CBLAS_INT *ku, CBLAS_INT *nrhs, complex_t *ab,
    CBLAS_INT *ldab, CBLAS_INT *ipiv, complex_t *b, CBLAS_INT *ldb, CBLAS_INT *info);

extern void BLAS_FUNC(dgttrf)(CBLAS_INT *n, double *dl, double *d, double *du,
    double *du2, CBLAS_INT *ipiv, CBLAS_INT *info);
extern void BLAS_FUNC(zgttrf)(CBLAS_INT *n, complex_t *dl, complex_t *d,
    complex_t *du, complex_t *du2, CBLAS_INT *ipiv, CBLAS_INT *info);
extern void BLAS_FUNC(dgttrs)(char *trans, CBLAS_INT *n, CBLAS_INT *nrhs, double *dl, double *d,
    double *du, double *du2, CBLAS_INT *ipiv, double *B, CBLAS_INT *ldB, CBLAS_INT *info);
extern void BLAS_FUNC(zgttrs)(char *trans, CBLAS_INT *n, CBLAS_INT *nrhs, complex_t *dl,
    complex_t *d, complex_t *du, complex_t *du2, 
    CBLAS_INT *ipiv, complex_t *B, CBLAS_INT *ldB, CBLAS_INT *info);
extern void BLAS_FUNC(dgtsv)(CBLAS_INT *n, CBLAS_INT *nrhs, double *dl, double *d, double *du,
    double *B, CBLAS_INT *ldB, CBLAS_INT *info);
extern void BLAS_FUNC(zgtsv)(CBLAS_INT *n, CBLAS_INT *nrhs, complex_t *dl,
    complex_t *d, complex_t *du, complex_t *B, CBLAS_INT *ldB,
    CBLAS_INT *info);

extern void BLAS_FUNC(dpotrf)(char *uplo, CBLAS_INT *n, double *A, CBLAS_INT *lda, CBLAS_INT *info);
extern void BLAS_FUNC(zpotrf)(char *uplo, CBLAS_INT *n, complex_t *A, CBLAS_INT *lda,
    CBLAS_INT *info);
extern void BLAS_FUNC(dpotrs)(char *uplo, CBLAS_INT *n, CBLAS_INT *nrhs, double *A, CBLAS_INT *lda,
    double *B, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(zpotrs)(char *uplo, CBLAS_INT *n, CBLAS_INT *nrhs, complex_t *A,
    CBLAS_INT *lda, complex_t *B, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(dpotri)(char *uplo, CBLAS_INT *n, double *A, CBLAS_INT *lda, CBLAS_INT *info);
extern void BLAS_FUNC(zpotri)(char *uplo, CBLAS_INT *n, complex_t *A, CBLAS_INT *lda,
    CBLAS_INT *info);
extern void BLAS_FUNC(dposv)(char *uplo, CBLAS_INT *n, CBLAS_INT *nrhs, double *A, CBLAS_INT *lda,
    double *B, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(zposv)(char *uplo, CBLAS_INT *n, CBLAS_INT *nrhs, complex_t *A,
    CBLAS_INT *lda, complex_t *B, CBLAS_INT *ldb, CBLAS_INT *info);

extern void BLAS_FUNC(dpbtrf)(char *uplo, CBLAS_INT *n, CBLAS_INT *kd, double *AB, CBLAS_INT *ldab,
    CBLAS_INT *info);
extern void BLAS_FUNC(zpbtrf)(char *uplo, CBLAS_INT *n, CBLAS_INT *kd, complex_t *AB,
    CBLAS_INT *ldab, CBLAS_INT *info);
extern void BLAS_FUNC(dpbtrs)(char *uplo, CBLAS_INT *n, CBLAS_INT *kd, CBLAS_INT *nrhs, double *AB,
    CBLAS_INT *ldab, double *B, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(zpbtrs)(char *uplo, CBLAS_INT *n, CBLAS_INT *kd, CBLAS_INT *nrhs,
    complex_t *AB, CBLAS_INT *ldab, complex_t *B, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(dpbsv)(char *uplo, CBLAS_INT *n, CBLAS_INT *kd, CBLAS_INT *nrhs, double *A,
    CBLAS_INT *lda, double *B, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(zpbsv)(char *uplo, CBLAS_INT *n, CBLAS_INT *kd, CBLAS_INT *nrhs,
    complex_t *A, CBLAS_INT *lda, complex_t *B, CBLAS_INT *ldb, CBLAS_INT *info);

extern void BLAS_FUNC(dpttrf)(CBLAS_INT *n, double *d, double *e, CBLAS_INT *info);
extern void BLAS_FUNC(zpttrf)(CBLAS_INT *n, double *d, complex_t *e, CBLAS_INT *info);
extern void BLAS_FUNC(dpttrs)(CBLAS_INT *n, CBLAS_INT *nrhs, double *d, double *e, double *B,
    CBLAS_INT *ldB, CBLAS_INT *info);
extern void BLAS_FUNC(zpttrs)(char *uplo, CBLAS_INT *n, CBLAS_INT *nrhs, double *d,
    complex_t *e, complex_t *B, CBLAS_INT *ldB, CBLAS_INT *info);
extern void BLAS_FUNC(dptsv)(CBLAS_INT *n, CBLAS_INT *nrhs, double *d, double *e, double *B,
    CBLAS_INT *ldB, CBLAS_INT *info);
extern void BLAS_FUNC(zptsv)(CBLAS_INT *n, CBLAS_INT *nrhs, double *d, complex_t *e,
    complex_t *B, CBLAS_INT *ldB, CBLAS_INT *info);

extern void BLAS_FUNC(dsytrf)(char *uplo, CBLAS_INT *n, double *A, CBLAS_INT *lda, CBLAS_INT *ipiv,
    double *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(zsytrf)(char *uplo, CBLAS_INT *n, complex_t *A, CBLAS_INT *lda,
    CBLAS_INT *ipiv, complex_t *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(zhetrf)(char *uplo, CBLAS_INT *n, complex_t *A, CBLAS_INT *lda,
    CBLAS_INT *ipiv, complex_t *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(dsytrs)(char *uplo, CBLAS_INT *n, CBLAS_INT *nrhs, double *A, CBLAS_INT *lda,
    CBLAS_INT *ipiv, double *B, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(zsytrs)(char *uplo, CBLAS_INT *n, CBLAS_INT *nrhs, complex_t *A,
    CBLAS_INT *lda, CBLAS_INT *ipiv, complex_t *B, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(zhetrs)(char *uplo, CBLAS_INT *n, CBLAS_INT *nrhs, complex_t *A,
    CBLAS_INT *lda, CBLAS_INT *ipiv, complex_t *B, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(dsytri)(char *uplo, CBLAS_INT *n, double *A, CBLAS_INT *lda, CBLAS_INT *ipiv,
    double *work, CBLAS_INT *info);
extern void BLAS_FUNC(zsytri)(char *uplo, CBLAS_INT *n, complex_t *A, CBLAS_INT *lda,
    CBLAS_INT *ipiv, complex_t *work, CBLAS_INT *info);
extern void BLAS_FUNC(zhetri)(char *uplo, CBLAS_INT *n, complex_t *A, CBLAS_INT *lda,
    CBLAS_INT *ipiv, complex_t *work, CBLAS_INT *info);
extern void BLAS_FUNC(dsysv)(char *uplo, CBLAS_INT *n, CBLAS_INT *nrhs, double *A, CBLAS_INT *lda,
    CBLAS_INT *ipiv, double *B, CBLAS_INT *ldb, double *work, CBLAS_INT *lwork,
    CBLAS_INT *info);
extern void BLAS_FUNC(zsysv)(char *uplo, CBLAS_INT *n, CBLAS_INT *nrhs, complex_t *A,
    CBLAS_INT *lda, CBLAS_INT *ipiv, complex_t *B, CBLAS_INT *ldb,
    complex_t *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(zhesv)(char *uplo, CBLAS_INT *n, CBLAS_INT *nrhs, complex_t *A,
    CBLAS_INT *lda, CBLAS_INT *ipiv, complex_t *B, CBLAS_INT *ldb,
    complex_t *work, CBLAS_INT *lwork, CBLAS_INT *info);

extern void BLAS_FUNC(dtrtrs)(char *uplo, char *trans, char *diag, CBLAS_INT *n, CBLAS_INT *nrhs,
    double  *a, CBLAS_INT *lda, double *b, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(ztrtrs)(char *uplo, char *trans, char *diag, CBLAS_INT *n, CBLAS_INT *nrhs,
    complex_t  *a, CBLAS_INT *lda, complex_t *b, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(dtrtri)(char *uplo, char *diag, CBLAS_INT *n, double  *a, CBLAS_INT *lda,
    CBLAS_INT *info);
extern void BLAS_FUNC(ztrtri)(char *uplo, char *diag, CBLAS_INT *n, complex_t  *a,
    CBLAS_INT *lda, CBLAS_INT *info);
extern void BLAS_FUNC(dtbtrs)(char *uplo, char *trans, char *diag, CBLAS_INT *n, CBLAS_INT *kd,
    CBLAS_INT *nrhs, double *ab, CBLAS_INT *ldab, double *b, CBLAS_INT *ldb, CBLAS_INT *info);
extern void BLAS_FUNC(ztbtrs)(char *uplo, char *trans, char *diag, CBLAS_INT *n, CBLAS_INT *kd,
    CBLAS_INT *nrhs, complex_t *ab, CBLAS_INT *ldab, complex_t *b,
    CBLAS_INT *ldb, CBLAS_INT *info);

extern void BLAS_FUNC(dgels)(char *trans, CBLAS_INT *m, CBLAS_INT *n, CBLAS_INT *nrhs, double *a,
    CBLAS_INT *lda, double *b, CBLAS_INT *ldb, double *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(zgels)(char *trans, CBLAS_INT *m, CBLAS_INT *n, CBLAS_INT *nrhs,
    complex_t *a, CBLAS_INT *lda, complex_t *b, CBLAS_INT *ldb,
    complex_t *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(dgeqrf)(CBLAS_INT *m, CBLAS_INT *n, double *a, CBLAS_INT *lda, double *tau,
    double *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(zgeqrf)(CBLAS_INT *m, CBLAS_INT *n, complex_t *a, CBLAS_INT *lda,
    complex_t *tau, complex_t *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(dormqr)(char *side, char *trans, CBLAS_INT *m, CBLAS_INT *n, CBLAS_INT *k,
    double *a, CBLAS_INT *lda, double *tau, double *c, CBLAS_INT *ldc, double *work,
    CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(zunmqr)(char *side, char *trans, CBLAS_INT *m, CBLAS_INT *n, CBLAS_INT *k,
    complex_t *a, CBLAS_INT *lda, complex_t *tau, complex_t *c,
    CBLAS_INT *ldc, complex_t *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(dorgqr)(CBLAS_INT *m, CBLAS_INT *n, CBLAS_INT *k, double *A, CBLAS_INT *lda,
    double *tau, double *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(zungqr)(CBLAS_INT *m, CBLAS_INT *n, CBLAS_INT *k, complex_t *A, CBLAS_INT *lda,
    complex_t *tau, complex_t *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(dorglq)(CBLAS_INT *m, CBLAS_INT *n, CBLAS_INT *k, double *A, CBLAS_INT *lda,
    double *tau, double *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(zunglq)(CBLAS_INT *m, CBLAS_INT *n, CBLAS_INT *k, complex_t *A, CBLAS_INT *lda,
    complex_t *tau, complex_t *work, CBLAS_INT *lwork, CBLAS_INT *info);

extern void BLAS_FUNC(dgelqf)(CBLAS_INT *m, CBLAS_INT *n, double *a, CBLAS_INT *lda, double *tau,
    double *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(zgelqf)(CBLAS_INT *m, CBLAS_INT *n, complex_t *a, CBLAS_INT *lda,
    complex_t *tau, complex_t *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(dormlq)(char *side, char *trans, CBLAS_INT *m, CBLAS_INT *n, CBLAS_INT *k,
    double *a, CBLAS_INT *lda, double *tau, double *c, CBLAS_INT *ldc, double *work,
    CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(zunmlq)(char *side, char *trans, CBLAS_INT *m, CBLAS_INT *n, CBLAS_INT *k,
    complex_t *a, CBLAS_INT *lda, complex_t *tau, complex_t *c,
    CBLAS_INT *ldc, complex_t *work, CBLAS_INT *lwork, CBLAS_INT *info);

extern void BLAS_FUNC(dgeqp3)(CBLAS_INT *m, CBLAS_INT *n, double *a, CBLAS_INT *lda, CBLAS_INT *jpvt,
    double *tau, double *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(zgeqp3)(CBLAS_INT *m, CBLAS_INT *n, complex_t *a, CBLAS_INT *lda, CBLAS_INT *jpvt,
    complex_t *tau, complex_t *work, CBLAS_INT *lwork, double *rwork,
    CBLAS_INT *info);

extern void BLAS_FUNC(dsyev)(char *jobz, char *uplo, CBLAS_INT *n, double *A, CBLAS_INT *lda,
    double *W, double *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(zheev)(char *jobz, char *uplo, CBLAS_INT *n, complex_t *A,
    CBLAS_INT *lda, double *W, complex_t *work, CBLAS_INT *lwork, double *rwork,
    CBLAS_INT *info);
extern void BLAS_FUNC(dsyevx)(char *jobz, char *range, char *uplo, CBLAS_INT *n, double *A,
    CBLAS_INT *lda, double *vl, double *vu, CBLAS_INT *il, CBLAS_INT *iu, double *abstol,
    CBLAS_INT *m, double *W, double *Z, CBLAS_INT *ldz, double *work, CBLAS_INT *lwork,
    CBLAS_INT *iwork, CBLAS_INT *ifail, CBLAS_INT *info);
extern void BLAS_FUNC(zheevx)(char *jobz, char *range, char *uplo, CBLAS_INT *n,
    complex_t *A, CBLAS_INT *lda, double *vl, double *vu, CBLAS_INT *il, CBLAS_INT *iu,
    double *abstol, CBLAS_INT *m, double *W, complex_t *Z, CBLAS_INT *ldz,
    complex_t *work, CBLAS_INT *lwork, double *rwork, CBLAS_INT *iwork,
    CBLAS_INT *ifail, CBLAS_INT *info);
extern void BLAS_FUNC(dsyevd)(char *jobz, char *uplo, CBLAS_INT *n, double *A, CBLAS_INT *ldA,
    double *W, double *work, CBLAS_INT *lwork, CBLAS_INT *iwork, CBLAS_INT *liwork,
    CBLAS_INT *info);
extern void BLAS_FUNC(zheevd)(char *jobz, char *uplo, CBLAS_INT *n, complex_t *A,
    CBLAS_INT *ldA, double *W, complex_t *work, CBLAS_INT *lwork, double *rwork,
    CBLAS_INT *lrwork, CBLAS_INT *iwork, CBLAS_INT *liwork, CBLAS_INT *info);
extern void BLAS_FUNC(dsyevr)(char *jobz, char *range, char *uplo, CBLAS_INT *n, double *A,
    CBLAS_INT *ldA, double *vl, double *vu, CBLAS_INT *il, CBLAS_INT *iu, double *abstol,
    CBLAS_INT *m, double *W, double *Z, CBLAS_INT *ldZ, CBLAS_INT *isuppz, double *work,
    CBLAS_INT *lwork, CBLAS_INT *iwork, CBLAS_INT *liwork, CBLAS_INT *info);
extern void BLAS_FUNC(zheevr)(char *jobz, char *range, char *uplo, CBLAS_INT *n,
    complex_t *A, CBLAS_INT *ldA, double *vl, double *vu, CBLAS_INT *il, CBLAS_INT *iu,
    double *abstol, CBLAS_INT *m, double *W, complex_t *Z, CBLAS_INT *ldZ,
    CBLAS_INT *isuppz, complex_t *work, CBLAS_INT *lwork, double *rwork,
    CBLAS_INT *lrwork, CBLAS_INT *iwork, CBLAS_INT *liwork, CBLAS_INT *info);

extern void BLAS_FUNC(dsygv)(CBLAS_INT *itype, char *jobz, char *uplo, CBLAS_INT *n, double *A,
    CBLAS_INT *lda, double *B, CBLAS_INT *ldb, double *W, double *work, CBLAS_INT *lwork,
    CBLAS_INT *info);
extern void BLAS_FUNC(zhegv)(CBLAS_INT *itype, char *jobz, char *uplo, CBLAS_INT *n,
    complex_t *A, CBLAS_INT *lda, complex_t *B, CBLAS_INT *ldb, double *W,
    complex_t *work, CBLAS_INT *lwork, double *rwork, CBLAS_INT *info);

extern void BLAS_FUNC(dgesvd)(char *jobu, char *jobvt, CBLAS_INT *m, CBLAS_INT *n, double *A,
    CBLAS_INT *ldA, double *S, double *U, CBLAS_INT *ldU, double *Vt, CBLAS_INT *ldVt,
    double *work, CBLAS_INT *lwork, CBLAS_INT *info);
extern void BLAS_FUNC(dgesdd)(char *jobz, CBLAS_INT *m, CBLAS_INT *n, double *A, CBLAS_INT *ldA,
    double *S, double *U, CBLAS_INT *ldU, double *Vt, CBLAS_INT *ldVt, double *work,
    CBLAS_INT *lwork, CBLAS_INT *iwork, CBLAS_INT *info);
extern void BLAS_FUNC(zgesvd)(char *jobu, char *jobvt, CBLAS_INT *m, CBLAS_INT *n,
    complex_t *A, CBLAS_INT *ldA, double *S, complex_t *U, CBLAS_INT *ldU,
    complex_t *Vt, CBLAS_INT *ldVt, complex_t *work, CBLAS_INT *lwork,
    double *rwork, CBLAS_INT *info);
extern void BLAS_FUNC(zgesdd)(char *jobz, CBLAS_INT *m, CBLAS_INT *n, complex_t *A,
    CBLAS_INT *ldA, double *S, complex_t *U, CBLAS_INT *ldU, complex_t *Vt,
    CBLAS_INT *ldVt, complex_t *work, CBLAS_INT *lwork, double *rwork,
    CBLAS_INT *iwork, CBLAS_INT *info);

extern void BLAS_FUNC(dgees)(char *jobvs, char *sort, CBLAS_INT (*select)(double *, double *), CBLAS_INT *n,
    double *A, CBLAS_INT *ldA, CBLAS_INT *sdim, double *wr, double *wi, double *vs,
    CBLAS_INT *ldvs, double *work, CBLAS_INT *lwork, CBLAS_INT *bwork, CBLAS_INT *info);
extern void BLAS_FUNC(zgees)(char *jobvs, char *sort, CBLAS_INT (*select)(complex_t *), CBLAS_INT *n,
    complex_t *A, CBLAS_INT *ldA, CBLAS_INT *sdim, complex_t *w,
    complex_t *vs, CBLAS_INT *ldvs, complex_t *work, CBLAS_INT *lwork,
    complex_t *rwork, CBLAS_INT *bwork, CBLAS_INT *info);
extern void BLAS_FUNC(dgges)(char *jobvsl, char *jobvsr, char *sort, CBLAS_INT (*delctg)(double *, double *, double *),
    CBLAS_INT *n, double *A, CBLAS_INT *ldA, double *B, CBLAS_INT *ldB, CBLAS_INT *sdim,
    double *alphar, double *alphai, double *beta, double *vsl, CBLAS_INT *ldvsl,
    double *vsr, CBLAS_INT *ldvsr, double *work, CBLAS_INT *lwork, CBLAS_INT *bwork,
    CBLAS_INT *info);
extern void BLAS_FUNC(zgges)(char *jobvsl, char *jobvsr, char *sort, CBLAS_INT (*delctg)(complex_t *, double *),
    CBLAS_INT *n, complex_t *A, CBLAS_INT *ldA, complex_t *B, CBLAS_INT *ldB,
    CBLAS_INT *sdim, complex_t *alpha, complex_t *beta,
    complex_t *vsl, CBLAS_INT *ldvsl, complex_t *vsr, CBLAS_INT *ldvsr,
    complex_t *work, CBLAS_INT *lwork, double *rwork, CBLAS_INT *bwork,
    CBLAS_INT *info);


static int number_from_pyobject(PyObject *o, number *a, int id)
{
    switch (id){
        case DOUBLE:
#if PY_MAJOR_VERSION >= 3
            if (!PyLong_Check(o) && !PyLong_Check(o) &&
                !PyFloat_Check(o)) return -1;
#else
            if (!PyInt_Check(o) && !PyLong_Check(o) &&
                !PyFloat_Check(o)) return -1;
#endif
            (*a).d = PyFloat_AsDouble(o);
            return 0;

        case COMPLEX:
#if PY_MAJOR_VERSION >= 3
            if (!PyLong_Check(o) && !PyLong_Check(o) &&
                !PyFloat_Check(o) && !PyComplex_Check(o)) return -1;
#else
            if (!PyInt_Check(o) && !PyLong_Check(o) &&
                !PyFloat_Check(o) && !PyComplex_Check(o)) return -1;
#endif
#ifndef _MSC_VER
            (*a).z = PyComplex_RealAsDouble(o) +
                I*PyComplex_ImagAsDouble(o);
#else
            (*a).z = _Cbuild(PyComplex_RealAsDouble(o),PyComplex_ImagAsDouble(o));
#endif
            return 0;
    }
    return -1;
}


static char doc_getrf[] =
    "LU factorization of a general real or complex m by n matrix.\n\n"
    "getrf(A, ipiv, m=A.size[0], n=A.size[1], ldA=max(1,A.size[0]),\n"
    "      offsetA=0)\n\n"
    "PURPOSE\n"
    "On exit, A is replaced with L, U in the factorization P*A = L*U\n"
    "and ipiv contains the permutation:\n"
    "P = P_min{m,n} * ... * P2 * P1 where Pi interchanges rows i and\n"
    "ipiv[i] of A (using the Fortran convention, i.e., the first row\n"
    "is numbered 1).\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "ipiv      'i' matrix of length at least min(m,n)\n\n"
    "m         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,m).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* getrf(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *ipiv;
    CBLAS_INT m=-1, n=-1, ldA=0, oA=0, info;
    char *kwlist[] = {"A", "ipiv", "m", "n", "ldA", "offsetA", NULL};
    Py_ssize_t _m = m, _n = n, _ldA = ldA, _oA = oA;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|nnnn", kwlist,
        &A, &ipiv, &_m, &_n, &_ldA, &_oA)) return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(ipiv) || ipiv ->id != INT) err_int_mtrx("ipiv");
    if (m < 0) m = A->nrows;
    if (n < 0) n = A->ncols;
    if (m == 0 || n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,m)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
    if (len(ipiv) < MIN(n,m)) err_buf_len("ipiv");

    CBLAS_INT *ipiv_ptr = cvxopt_cblas_int_acquire(ipiv, MIN(m,n), 0);
    if (!ipiv_ptr) return NULL;

    switch (MAT_ID(A)) {
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgetrf)(&m, &n, MAT_BUFD(A)+oA, &ldA, ipiv_ptr, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgetrf)(&m, &n, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr, &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            cvxopt_cblas_int_release(ipiv, ipiv_ptr, MIN(m,n), 0);
            err_invalid_id;
    }

    cvxopt_cblas_int_release(ipiv, ipiv_ptr, MIN(m,n), 1);

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_getrs[] =
    "Solves a general real or complex set of linear equations,\n"
    "given the LU factorization computed by getrf() or gesv().\n\n"
    "getrs(A, ipiv, B, trans='N', n=A.size[0], nrhs=B.size[1],\n"
    "      ldA = max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0,\n"
    "      offsetB=0)\n\n"
    "PURPOSE\n"
    "If trans is 'N', solves A*X = B.\n"
    "If trans is 'T', solves A^T*X = B.\n"
    "If trans is 'C', solves A^H*X = B.\n"
    "On entry, A and ipiv contain the LU factorization of an n by n\n"
    "matrix A as computed by getrf() or gesv().  On exit B is replaced\n"
    "by the solution X.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "ipiv      'i' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "trans     'N', 'T' or 'C'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "           used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* getrs(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B, *ipiv;
    CBLAS_INT n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
#if PY_MAJOR_VERSION >= 3
    int trans_ = 'N';
#endif
    char trans = 'N';
    char *kwlist[] = {"A", "ipiv", "B", "trans", "n", "nrhs", "ldA",
        "ldB", "offsetA", "offsetB", NULL};
    Py_ssize_t _n = n, _nrhs = nrhs, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|Cnnnnnn", kwlist,
        &A, &ipiv, &B, &trans_, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
    trans = trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cnnnnnn", kwlist,
        &A, &ipiv, &B, &trans, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
    if (len(ipiv) < n) err_buf_len("ipiv");

    CBLAS_INT *ipiv_ptr = cvxopt_cblas_int_acquire(ipiv, n, 1);
    if (!ipiv_ptr) return NULL;

    switch (MAT_ID(A)){
        case DOUBLE:
            if (trans == 'C') trans = 'T';
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgetrs)(&trans, &n, &nrhs, MAT_BUFD(A)+oA, &ldA, ipiv_ptr,
                MAT_BUFD(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgetrs)(&trans, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
                MAT_BUFZ(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

	default:
            cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
            err_invalid_id;
    }

    cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_getri[] =
    "Inverse of a real or complex matrix.\n\n"
    "getri(A, ipiv, n=A.size[0], ldA = max(1,A.size[0]), offsetA=0)\n\n"
    "PURPOSE\n"
    "Computes the inverse of real or complex matrix of order n.  On\n"
    "entry, A and ipiv contain the LU factorization, as returned by\n"
    "gesv() or getrf().  On exit A is replaced by the inverse.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "ipiv      'i' matrix\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* getri(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *ipiv;
    CBLAS_INT n=-1, ldA=0, oA=0, info, lwork;
    void *work;
    number wl;
    char *kwlist[] = {"A", "ipiv", "n", "ldA", "offsetA", NULL};
    Py_ssize_t _n = n, _ldA = ldA, _oA = oA;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|nnn", kwlist, &A,
        &ipiv, &_n, &_ldA, &_oA)) return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (len(ipiv) < n) err_buf_len("ipiv");

    CBLAS_INT *ipiv_ptr = cvxopt_cblas_int_acquire(ipiv, n, 1);
    if (!ipiv_ptr) return NULL;

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgetri)(&n, NULL, &ldA, NULL, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double)))) {
                cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgetri)(&n, MAT_BUFD(A)+oA, &ldA, ipiv_ptr, (double *) work,
                &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
            break;

        case COMPLEX:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgetri)(&n, NULL, &ldA, NULL, &wl.z, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
            if (!(work = (void *) calloc(lwork, sizeof(complex_t)))){
                cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgetri)(&n, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
                (complex_t *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
            break;

        default:
            cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
            err_invalid_id;
    }

    cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_gesv[] =
    "Solves a general real or complex set of linear equations.\n\n"
    "dgesv(A, B, ipiv=None, n=A.size[0], nrhs=B.size[1], \n"
    "      ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0, \n"
    "      offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves A*X=B with A n by n real or complex.\n"
    "If ipiv is provided, then on exit A is overwritten with the details\n"
    "of the LU factorization, and ipiv contains the permutation matrix.\n"
    "If ipiv is not provided, then gesv() does not return the \n"
    "factorization and does not modify A.  On exit B is replaced with\n"
    "the solution X.\n\n"
    "ARGUMENTS.\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "ipiv      'i' matrix of length at least n\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetA   nonnegative integer";

static PyObject* gesv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B, *ipiv=NULL;
    CBLAS_INT n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info, k;
    void *Ac=NULL;
    CBLAS_INT *ipivc=NULL;
    static char *kwlist[] = {"A", "B", "ipiv", "n", "nrhs", "ldA",
        "ldB", "offsetA", "offsetB", NULL};
    Py_ssize_t _n = n, _nrhs = nrhs, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Onnnnnn", kwlist,
        &A, &B, &ipiv, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB)) return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (ipiv && (!Matrix_Check(ipiv) || ipiv->id != INT))
        err_int_mtrx("ipiv");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
    if (ipiv && len(ipiv) < n) err_buf_len("ipiv");

    if (ipiv) {
        if (!(ipivc = cvxopt_cblas_int_acquire(ipiv, n, 0))) return NULL;
    }
    else if (!(ipivc = (CBLAS_INT *) calloc(n, sizeof(CBLAS_INT))))
        return PyErr_NoMemory();

    switch (MAT_ID(A)){
        case DOUBLE:
            if (ipiv)
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(dgesv)(&n, &nrhs, MAT_BUFD(A)+oA, &ldA, ipivc,
                    MAT_BUFD(B)+oB, &ldB, &info);
                Py_END_ALLOW_THREADS
            else {
                if (!(Ac = (void *) calloc(n*n, sizeof(double)))){
                    free(ipivc);
                    return PyErr_NoMemory();
                }
                for (k=0; k<n; k++) memcpy((double *) Ac + k*n,
                    MAT_BUFD(A)+oA+k*ldA, n*sizeof(double));
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(dgesv)(&n, &nrhs, (double *) Ac, &n, ipivc,
                    MAT_BUFD(B)+oB, &ldB, &info);
                Py_END_ALLOW_THREADS
                free(Ac);
            }
            break;

        case COMPLEX:
            if (ipiv)
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(zgesv)(&n, &nrhs, MAT_BUFZ(A)+oA, &ldA, ipivc,
                    MAT_BUFZ(B)+oB, &ldB, &info);
                Py_END_ALLOW_THREADS
            else {
                if (!(Ac = (void *) calloc(n*n, sizeof(complex_t)))){
                    free(ipivc);
                    return PyErr_NoMemory();
                }
                for (k=0; k<n; k++) memcpy((complex_t *) Ac + k*n,
                    MAT_BUFZ(A)+oA+k*ldA, n*sizeof(complex_t));
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(zgesv)(&n, &nrhs, (complex_t *) Ac, &n, ipivc,
                    MAT_BUFZ(B)+oB, &ldB, &info);
                Py_END_ALLOW_THREADS
                free(Ac);
            }
            break;

        default:
            if (ipiv) cvxopt_cblas_int_release(ipiv, ipivc, n, 0);
            else free(ipivc);
            err_invalid_id;
    }

    if (ipiv) cvxopt_cblas_int_release(ipiv, ipivc, n, 1);
    else free(ipivc);

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_gbtrf[] =
    "LU factorization of a real or complex m by n band matrix.\n\n"
    "gbtrf(A, m, kl, ipiv, n=A.size[1], ku=A.size[0]-2*kl-1,\n"
    "      ldA=max(1,A.size[0]), offsetA=0)\n\n"
    "PURPOSE\n"
    "Computes the LU factorization of an m by n band matrix with kl\n"
    "subdiagonals and ku superdiagonals.  On entry, the diagonals are\n"
    "stored in rows kl+1 to 2*kl+ku+1 of the array A, in the BLAS format\n"
    "for general band matrices.   On exit A and ipiv contains the\n"
    "factorization.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "m         nonnegative integer\n\n"
    "kl        nonnegative integer.\n\n"
    "ipiv      'i' matrix of length at least min(m,n)\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ku        nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= 2*kl+ku+1.  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* gbtrf(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *ipiv;
    CBLAS_INT m, kl, n=-1, ku=-1, ldA=0, oA=0, info;
    char *kwlist[] = {"A", "m", "kl", "ipiv", "n", "ku", "ldA", "offsetA",
        NULL};
    Py_ssize_t _m, _kl, _n = n, _ku = ku, _ldA = ldA, _oA = oA;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OnnO|nnnn", kwlist,
        &A, &_m, &_kl, &ipiv, &_n, &_ku, &_ldA, &_oA)) return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_kl, &kl) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ku, &ku) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (m < 0) err_nn_int("m");
    if (kl < 0) err_nn_int("kl");
    if (n < 0) n = A->ncols;
    if (m == 0 || n == 0) return Py_BuildValue("");
    if (ku < 0) ku = A->nrows - 2*kl - 1;
    if (ku < 0) err_nn_int("kl");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < 2*kl + ku + 1) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + 2*kl + ku + 1 > len(A)) err_buf_len("A");
    if (!Matrix_Check(ipiv) || ipiv ->id != INT) err_int_mtrx("ipiv");
    if (len(ipiv) < MIN(n,m)) err_buf_len("ipiv");

    CBLAS_INT *ipiv_ptr = cvxopt_cblas_int_acquire(ipiv, MIN(m,n), 0);
    if (!ipiv_ptr) return NULL;

    switch (MAT_ID(A)) {
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgbtrf)(&m, &n, &kl, &ku, MAT_BUFD(A)+oA, &ldA, ipiv_ptr,
                &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgbtrf)(&m, &n, &kl, &ku, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
                &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            cvxopt_cblas_int_release(ipiv, ipiv_ptr, MIN(m,n), 0);
            err_invalid_id;
    }

    cvxopt_cblas_int_release(ipiv, ipiv_ptr, MIN(m,n), 1);

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_gbtrs[] =
    "Solves a real or complex set of linear equations with a banded\n"
    "coefficient matrix, given the LU factorization computed by gbtrf()\n"
    "or gbsv().\n\n"
    "gbtrs(A, kl, ipiv, B, trans='N', n=A.size[1], ku=A.size[0]-2*kl-1,\n"
    "      nrhs=B.size[1], ldA=max(1,AB.size[0]), ldB=max(1,B.size[0]),\n"
    "      offsetA=0, offsetB=0)\n\n"
    "PURPOSE\n"
    "If trans is 'N', solves A*X = B.\n"
    "If trans is 'T', solves A^T*X = B.\n"
    "If trans is 'C', solves A^H*X = B.\n"
    "On entry, A and ipiv contain the LU factorization of an n by n\n"
    "band matrix A as computed by getrf() or gbsv().  On exit B is\n"
    "replaced by the solution X.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "kl        nonnegative integer\n\n"
    "ipiv      'i' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "trans     'N', 'T' or 'C'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ku        nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= 2*kl+ku+1.  If zero, the\n"
    "          default value is used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* gbtrs(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B, *ipiv;
    CBLAS_INT kl, n=-1, ku=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
#if PY_MAJOR_VERSION >= 3
    int trans_ = 'N';
#endif
    char trans = 'N';
    char *kwlist[] = {"A", "kl", "ipiv", "B", "trans", "n", "ku", "nrhs",
        "ldA", "ldB", "offsetA", "offsetB", NULL};
    Py_ssize_t _kl, _n = n, _ku = ku, _nrhs = nrhs, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OnOO|Cnnnnnnn", kwlist,
        &A, &_kl, &ipiv, &B, &trans_, &_n, &_ku, &_nrhs, &_ldA, &_ldB, &_oA,
        &_oB))
        return NULL;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OnOO|cnnnnnnn", kwlist,
        &A, &_kl, &ipiv, &B, &trans, &_n, &_ku, &_nrhs, &_ldA, &_ldB, &_oA,
        &_oB))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_kl, &kl) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ku, &ku) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (kl < 0) err_nn_int("kl");
    if (ku < 0) ku = A->nrows - 2*kl - 1;
    if (ku < 0) err_nn_int("kl");
    if (n < 0) n = A->ncols;
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < 2*kl+ku+1) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + 2*kl + ku + 1 > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
    if (len(ipiv) < n) err_buf_len("ipiv");

    CBLAS_INT *ipiv_ptr = cvxopt_cblas_int_acquire(ipiv, n, 1);
    if (!ipiv_ptr) return NULL;

    switch (MAT_ID(A)){
        case DOUBLE:
            if (trans == 'C') trans = 'T';
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgbtrs)(&trans, &n, &kl, &ku, &nrhs, MAT_BUFD(A)+oA, &ldA,
                ipiv_ptr, MAT_BUFD(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgbtrs)(&trans, &n, &kl, &ku, &nrhs, MAT_BUFZ(A)+oA, &ldA,
                ipiv_ptr, MAT_BUFZ(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

	default:
            cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
            err_invalid_id;
    }

    cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_gbsv[] =
    "Solves a real or complex set of linear equations with a banded\n"
    "coefficient matrix.\n\n"
    "gbsv(A, kl, B, ipiv=None, ku=None, n=A.size[1], nrhs=B.size[1],\n"
    "     ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0, \n"
    "     offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves A*X=B with A an n by n real or complex band matrix with kl\n"
    "subdiagonals and ku superdiagonals.\n"
    "If ipiv is provided, then on entry the kl+ku+1 diagonals of the\n"
    "matrix are stored in rows kl+1 to 2*kl+ku+1 of A, in the BLAS\n"
    "format for general band matrices.  On exit, A and ipiv contain the\n"
    "details of the factorization.  If ipiv is not provided, then on\n"
    "entry the diagonals of the matrix are stored in rows 1 to kl+ku+1 \n"
    "of A, and gbsv() does not return the factorization and does not\n"
    "modify A.  On exit B is replaced with solution X.\n\n"
    "ARGUMENTS.\n"
    "A         'd' or 'z' banded matrix\n\n"
    "kl        nonnegative integer\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "ipiv      'i' matrix of length at least n\n\n"
    "ku        nonnegative integer.  If negative, the default value is\n"
    "          used.  The default value is A.size[0]-kl-1 if ipiv is\n"
    "          not provided, and A.size[0]-2*kl-1 otherwise.\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= kl+ku+1 if ipiv is not provided\n"
    "          and ldA >= 2*kl+ku+1 if ipiv is provided.  If zero, the\n"
    "          default value is used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";


static PyObject* gbsv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B, *ipiv=NULL;
    void *Ac;
    CBLAS_INT kl, ku=-1, n=-1, nrhs=-1, ldA=0, oA=0, ldB=0, oB=0, info, k;
    CBLAS_INT *ipivc=NULL;
    static char *kwlist[] = {"A", "kl", "B", "ipiv", "ku", "n", "nrhs",
        "ldA", "ldB", "oA", "oB", NULL};
    Py_ssize_t _kl, _ku = ku, _n = n, _nrhs = nrhs, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OnO|Onnnnnnn", kwlist,
        &A, &_kl, &B, &ipiv, &_ku, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_kl, &kl) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ku, &ku) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (ipiv && (!Matrix_Check(ipiv) || ipiv->id != INT))
        err_int_mtrx("ipiv");
    if (n < 0) n = A->ncols;
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (kl < 0) err_nn_int("kl");
    if (ku < 0) ku = A->nrows - kl - 1 - (ipiv ? kl : 0);
    if (ku < 0) err_nn_int("ku");
    if (ldA == 0) ldA = MAX(1, A->nrows);
    if (ldA < ( ipiv ? 2*kl+ku+1 : kl+ku+1)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + (ipiv ? 2*kl+ku+1 : kl+ku+1) > len(A))
        err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
    if (ipiv && len(ipiv) < n) err_buf_len("ipiv");

    if (ipiv) {
        if (!(ipivc = cvxopt_cblas_int_acquire(ipiv, n, 0))) return NULL;
    }
    else if (!(ipivc = (CBLAS_INT *) calloc(n, sizeof(CBLAS_INT))))
        return PyErr_NoMemory();

    switch (MAT_ID(A)) {
        case DOUBLE:
            if (ipiv)
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(dgbsv)(&n, &kl, &ku, &nrhs, MAT_BUFD(A)+oA, &ldA, ipivc,
                    MAT_BUFD(B)+oB, &ldB, &info);
                Py_END_ALLOW_THREADS
            else {
                if (!(Ac = (void *) calloc((2*kl+ku+1)*n,
                    sizeof(double)))){
                    free(ipivc);
                    return PyErr_NoMemory();
                }
                for (k=0; k<n; k++)
                    memcpy((double *) Ac + kl + k*(2*kl+ku+1),
                        MAT_BUFD(A) + oA + k*ldA,
                        (kl+ku+1)*sizeof(double));
                ldA = 2*kl+ku+1;
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(dgbsv)(&n, &kl, &ku, &nrhs, (double *) Ac, &ldA, ipivc,
                    MAT_BUFD(B)+oB, &ldB, &info);
                Py_END_ALLOW_THREADS
                free(Ac);
            }
            break;

        case COMPLEX:
            if (ipiv)
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(zgbsv)(&n, &kl, &ku, &nrhs, MAT_BUFZ(A)+oA, &ldA, ipivc,
                    MAT_BUFZ(B)+oB, &ldB, &info);
                Py_END_ALLOW_THREADS
            else {
                if (!(Ac = (void *) calloc((2*kl+ku+1)*n,
                    sizeof(complex_t)))){
                    free(ipivc);
                    return PyErr_NoMemory();
                }
                for (k=0; k<n; k++)
                    memcpy((complex_t *) Ac + kl + k*(2*kl+ku+1),
                        MAT_BUFZ(A) + oA + k*ldA,
                        (kl+ku+1)*sizeof(complex_t));
                ldA = 2*kl+ku+1;
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(zgbsv)(&n, &kl, &ku, &nrhs, (complex_t *) Ac, &ldA,
                    ipivc, MAT_BUFZ(B)+oB, &ldB, &info);
                Py_END_ALLOW_THREADS
                free(Ac);
            }
            break;

        default:
            if (ipiv) cvxopt_cblas_int_release(ipiv, ipivc, n, 0);
            else free(ipivc);
            err_invalid_id;
    }

    if (ipiv) cvxopt_cblas_int_release(ipiv, ipivc, n, 1);
    else free(ipivc);

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_gttrf[] =
    "LU factorization of a real or complex tridiagonal matrix.\n\n"
    "gttrf(dl, d, du, du2, ipiv, n=len(d)-offsetd, offsetdl=0, offsetd=0,"
    "\n"
    "      offsetdu=0)\n\n"
    "PURPOSE\n"
    "Factors an n by n real or complex tridiagonal matrix A as A = P*L*U."
    "\n  A is specified by its lower diagonal dl, diagonal d, and upper\n"
    "diagonal du.  On exit dl, d, du, du2 and ipiv contain the details\n"
    "of the factorization.\n\n"
    "ARGUMENTS.\n"
    "dl        'd' or 'z' matrix\n\n"
    "d         'd' or 'z' matrix.  Must have the same type as dl.\n\n"
    "du        'd' or 'z' matrix.  Must have the same type as dl.\n\n"
    "du2       'd' or 'z' matrix of length at least n-2.  Must have the\n"
    "           same type as dl.\n\n"
    "ipiv      'i' matrix of length at least n\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "offsetdl  nonnegative integer\n\n"
    "offsetd   nonnegative integer\n\n"
    "offsetdu  nonnegative integer";

static PyObject* gttrf(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *dl, *d, *du, *du2, *ipiv;
    CBLAS_INT n=-1, odl=0, od=0, odu=0, info;
    static char *kwlist[] = {"dl", "d", "du", "du2", "ipiv", "n",
        "offsetdl", "offsetd", "offsetdu", NULL};
    Py_ssize_t _n = n, _odl = odl, _od = od, _odu = odu;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOOOO|nnnn", kwlist,
        &dl, &d, &du, &du2, &ipiv, &_n, &_odl, &_od, &_odu))
        return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_odl, &odl) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_od, &od) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_odu, &odu) < 0)
        return NULL;

    if (!Matrix_Check(dl)) err_mtrx("dl");
    if (!Matrix_Check(d)) err_mtrx("d");
    if (!Matrix_Check(du)) err_mtrx("du");
    if (!Matrix_Check(du2)) err_mtrx("du");
    if ((MAT_ID(dl) != MAT_ID(d)) || (MAT_ID(dl) != MAT_ID(du)) ||
        (MAT_ID(dl) != MAT_ID(du2))) err_conflicting_ids;
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (od < 0) err_nn_int("offsetd");
    if (n < 0) n = len(d) - od;
    if (n < 0) err_buf_len("d");
    if (n == 0) return Py_BuildValue("");
    if (odl < 0) err_nn_int("offsetdl");
    if (odl + n - 1  > len(dl)) err_buf_len("dl");
    if (od + n > len(d)) err_buf_len("d");
    if (odu < 0) err_nn_int("offsetdu");
    if (odu + n - 1  > len(du)) err_buf_len("du");
    if (n - 2  > len(du2)) err_buf_len("du2");
    if (len(ipiv) < n) err_buf_len("ipiv");
    if (n > len(ipiv)) err_buf_len("ipiv");

    CBLAS_INT *ipiv_ptr = cvxopt_cblas_int_acquire(ipiv, n, 0);
    if (!ipiv_ptr) return NULL;

    switch (MAT_ID(dl)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgttrf)(&n, MAT_BUFD(dl)+odl, MAT_BUFD(d)+od, MAT_BUFD(du)+odu,
                MAT_BUFD(du2), ipiv_ptr, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgttrf)(&n, MAT_BUFZ(dl)+odl, MAT_BUFZ(d)+od, MAT_BUFZ(du)+odu,
                MAT_BUFZ(du2), ipiv_ptr, &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
            err_invalid_id;
    }

    cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 1);

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_gttrs[] =
    "Solves a real or complex tridiagonal set of linear equations, \n"
    "given the LU factorization computed by gttrf().\n\n"
    "gttrs(dl, d, du, du2, ipiv, B, trans='N', n=len(d)-offsetd,\n"
    "      nrhs=B.size[1], ldB=max(1,B.size[0]), offsetdl=0, offsetd=0,\n"
    "      offsetdu=0, offsetB=0)\n\n"
    "PURPOSE\n"
    "If trans is 'N', solves A*X=B.\n"
    "If trans is 'T', solves A^T*X=B.\n"
    "If trans is 'C', solves A^H*X=B.\n"
    "On entry, dl, d, du, du2 and ipiv contain the LU factorization of \n"
    "an n by n tridiagonal matrix A as computed by gttrf().  On exit B\n"
    "is replaced by the solution X.\n\n"
    "ARGUMENTS.\n"
    "dl        'd' or 'z' matrix\n\n"
    "d         'd' or 'z' matrix.  Must have the same type as dl.\n\n"
    "du        'd' or 'z' matrix.  Must have the same type as dl.\n\n"
    "du2       'd' or 'z' matrix.  Must have the same type as dl.\n\n"
    "ipiv      'i' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type oas dl.\n\n"
    "trans     'N', 'T' or 'C'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetdl  nonnegative integer\n\n"
    "offsetd   nonnegative integer\n\n"
    "offsetdu  nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* gttrs(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *dl, *d, *du, *du2, *ipiv, *B;
#if PY_MAJOR_VERSION >= 3
    int trans_ = 'N';
#endif
    char trans = 'N';
    CBLAS_INT n=-1, nrhs=-1, ldB=0, odl=0, od=0, odu=0, oB=0, info;
    static char *kwlist[] = {"dl", "d", "du", "du2", "ipiv", "B", "trans",
        "n", "nrhs", "ldB", "offsetdl", "offsetd", "offsetdu", "offsetB",
        NULL};
    Py_ssize_t _n = n, _nrhs = nrhs, _ldB = ldB, _odl = odl, _od = od, _odu = odu, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOOOOO|cnnnnnnn",
        kwlist, &dl, &d, &du, &du2, &ipiv, &B, &trans, &_n, &_nrhs, &_ldB,
        &_odl, &_od, &_odu, &_oB)) return NULL;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOOOOO|cnnnnnnn",
        kwlist, &dl, &d, &du, &du2, &ipiv, &B, &trans, &_n, &_nrhs, &_ldB,
        &_odl, &_od, &_odu, &_oB)) return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_odl, &odl) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_od, &od) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_odu, &odu) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(dl)) err_mtrx("dl");
    if (!Matrix_Check(d)) err_mtrx("d");
    if (!Matrix_Check(du)) err_mtrx("du");
    if (!Matrix_Check(du2)) err_mtrx("du");
    if (!Matrix_Check(B)) err_mtrx("B");
    if ((MAT_ID(dl) != MAT_ID(d)) || (MAT_ID(dl) != MAT_ID(du)) ||
        (MAT_ID(dl) != MAT_ID(du2)) || (MAT_ID(dl) != MAT_ID(B)))
        err_conflicting_ids;
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (od < 0) err_nn_int("offsetd");
    if (n < 0) n = len(d) - od;
    if (n < 0) err_buf_len("d");
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (odl < 0) err_nn_int("offsetdl");
    if (odl + n - 1  > len(dl)) err_buf_len("dl");
    if (od + n > len(d)) err_buf_len("d");
    if (odu < 0) err_nn_int("offsetdu");
    if (odu + n - 1  > len(du)) err_buf_len("du");
    if (n - 2  > len(du2)) err_buf_len("du2");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
    if (n > len(ipiv)) err_buf_len("ipiv");

    CBLAS_INT *ipiv_ptr = cvxopt_cblas_int_acquire(ipiv, n, 1);
    if (!ipiv_ptr) return NULL;

    switch (MAT_ID(dl)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgttrs)(&trans, &n, &nrhs, MAT_BUFD(dl)+odl, MAT_BUFD(d)+od,
                MAT_BUFD(du)+odu, MAT_BUFD(du2), ipiv_ptr,
                MAT_BUFD(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgttrs)(&trans, &n, &nrhs, MAT_BUFZ(dl)+odl, MAT_BUFZ(d)+od,
                MAT_BUFZ(du)+odu, MAT_BUFZ(du2), ipiv_ptr,
                MAT_BUFZ(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
            err_invalid_id;
    }

    cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_gtsv[] =
    "Solves a real or complex set of linear equations with a tridiagonal\n"
    "coefficient matrix.\n\n"
    "gtsv(dl, d, du, B, n=len(d)-offsetd, nrhs=B.size[1], \n"
    "     ldB=max(1,B.size[0]), offsetdl=0, offsetd=0, offsetdu=0, \n"
    "     offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves A*X=B with A n by n real or complex and tridiagonal.\n"
    "A is specified by its lower diagonal dl, diagonal d and upper \n"
    "diagonal du.  On exit B is overwritten with the solution, and dl,\n"
    "d, du are overwritten with the elements of the upper triangular\n"
    "matrix in the LU factorization of A.\n\n"
    "ARGUMENTS.\n"
    "dl        'd' or 'z' matrix\n\n"
    "d         'd' or 'z' matrix.  Must have the same type as dl.\n\n"
    "du        'd' or 'z' matrix.  Must have the same type as dl.\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as dl.\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetdl  nonnegative integer\n\n"
    "offsetd   nonnegative integer\n\n"
    "offsetdu  nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* gtsv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *dl, *d, *du, *B;
    CBLAS_INT n=-1, nrhs=-1, ldB=0, odl=0, od=0, odu=0, oB=0, info;
    static char *kwlist[] = {"dl", "d", "du", "B", "n", "nrhs", "ldB",
        "offsetdl", "offsetd", "offsetdu", "offsetB", NULL};
    Py_ssize_t _n = n, _nrhs = nrhs, _ldB = ldB, _odl = odl, _od = od, _odu = odu, _oB = oB;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOOO|nnnnnnn", kwlist,
        &dl, &d, &du, &B, &_n, &_nrhs, &_ldB, &_odl, &_od, &_odu, &_oB))
        return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_odl, &odl) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_od, &od) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_odu, &odu) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(dl)) err_mtrx("dl");
    if (!Matrix_Check(d)) err_mtrx("d");
    if (!Matrix_Check(du)) err_mtrx("du");
    if (!Matrix_Check(B)) err_mtrx("B");
    if ((MAT_ID(dl) != MAT_ID(B)) || (MAT_ID(dl) != MAT_ID(d)) ||
        (MAT_ID(dl) != MAT_ID(du)) || (MAT_ID(dl) != MAT_ID(B)))
        err_conflicting_ids;
    if (od < 0) err_nn_int("offsetd");
    if (n < 0) n = len(d) - od;
    if (n < 0) err_buf_len("d");
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (odl < 0) err_nn_int("offsetdl");
    if (odl + n - 1  > len(dl)) err_buf_len("dl");
    if (od + n > len(d)) err_buf_len("d");
    if (odu < 0) err_nn_int("offsetdu");
    if (odu + n - 1  > len(du)) err_buf_len("du");
    if (oB < 0) err_nn_int("offsetB");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

    switch (MAT_ID(dl)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgtsv)(&n, &nrhs, MAT_BUFD(dl)+odl, MAT_BUFD(d)+od,
                MAT_BUFD(du)+odu, MAT_BUFD(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgtsv)(&n, &nrhs, MAT_BUFZ(dl)+odl, MAT_BUFZ(d)+od,
                MAT_BUFZ(du)+odu, MAT_BUFZ(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_potrf[] =
    "Cholesky factorization of a real symmetric or complex Hermitian\n"
    "positive definite matrix.\n\n"
    "potrf(A, uplo='L', n=A.size[0], ldA = max(1,A.size[0]), offsetA=0)"
    "\n\n"
    "PURPOSE\n"
    "Factors A as A=L*L^T or A = L*L^H, where A is n by n, real\n"
    "symmetric or complex Hermitian, and positive definite.\n"
    "On exit, if uplo='L', the lower triangular part of A is replaced\n"
    "by L.  If uplo='U', the upper triangular part is replaced by L^T\n"
    "or L^H.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* potrf(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A;
    CBLAS_INT n=-1, ldA=0, oA=0, info;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "uplo", "n", "ldA", "offsetA", NULL};
    Py_ssize_t _n = n, _ldA = ldA, _oA = oA;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|Cnnn", kwlist, &A,
        &uplo_, &_n, &_ldA, &_oA)) return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|cnnn", kwlist, &A,
        &uplo, &_n, &_ldA, &_oA)) return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A is not square");
            return NULL;
        }
    }
    if (uplo != 'U' && uplo != 'L') err_char("uplo", "'L', 'U'");
    if (n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1, A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");

    switch (MAT_ID(A)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dpotrf)(&uplo, &n, MAT_BUFD(A)+oA, &ldA, &info);
            Py_END_ALLOW_THREADS
	    break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zpotrf)(&uplo, &n, MAT_BUFZ(A)+oA, &ldA, &info);
            Py_END_ALLOW_THREADS
	    break;

	default:
	    err_invalid_id;
    }
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_potrs[] =
    "Solves a real symmetric or complex Hermitian positive definite set\n"
    "of linear equations, given the Cholesky factorization computed by\n"
    "potrf() or posv().\n\n"
    "potrs(A, B, uplo='L', n=A.size[0], nrhs=B.size[1],\n"
    "      ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0,\n"
    "      offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves A*X = B where A is n by n, real symmetric or complex\n"
    "Hermitian and positive definite, and B is n by nrhs.\n"
    "On entry, A contains the Cholesky factor, as returned by posv() or\n"
    "potrf().  On exit B is replaced by the solution X.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* potrs(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B;
    CBLAS_INT n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "B", "uplo", "n", "nrhs", "ldA", "ldB",
        "offsetA", "offsetB", NULL};
    Py_ssize_t _n = n, _nrhs = nrhs, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Cnnnnnn", kwlist,
        &A, &B, &uplo_, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB)) return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cnnnnnn", kwlist,
        &A, &B, &uplo, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB)) return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0) n = A->nrows;
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

    switch (MAT_ID(A)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dpotrs)(&uplo, &n, &nrhs, MAT_BUFD(A)+oA, &ldA, MAT_BUFD(B)+oB,
                &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zpotrs)(&uplo, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(B)+oB,
                &ldB, &info);
            Py_END_ALLOW_THREADS
	    break;

        default:
	    err_invalid_id;
    }
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_potri[] =
    "Inverse of a real symmetric or complex Hermitian positive definite\n"
    "matrix.\n\n"
    "potri(A, uplo='L', n=A.size[0], ldA=max(1,A.size[0]), offsetA=0)\n\n"
    "PURPOSE\n"
    "Computes the inverse of a real symmetric or complex Hermitian\n"
    "positive definite matrix of order n.  On entry, A contains the\n"
    "Cholesky factor, as returned by posv() or potrf().  On exit it is\n"
    "replaced by the inverse.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* potri(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A;
    CBLAS_INT n=-1, ldA=0, oA=0, info;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "uplo", "n", "ldA", "offsetA", NULL};
    Py_ssize_t _n = n, _ldA = ldA, _oA = oA;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|Cnnn", kwlist,
        &A, &uplo_, &_n, &_ldA, &_oA)) return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|cnnn", kwlist,
        &A, &uplo, &_n, &_ldA, &_oA)) return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0) n = A->nrows;
    if (n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");

    switch (MAT_ID(A)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dpotri)(&uplo, &n, MAT_BUFD(A)+oA, &ldA, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zpotri)(&uplo, &n, MAT_BUFZ(A)+oA, &ldA, &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_posv[] =
    "Solves a real symmetric or complex Hermitian positive definite set\n"
    "of linear equations.\n\n"
    "posv(A, B, uplo='L', n=A.size[0], nrhs=B.size[1], \n"
    "     ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0, \n"
    "     offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves A*X = B with A n by n, real symmetric or complex Hermitian,\n"
    "and positive definite, and B n by nrhs.\n"
    "On exit, if uplo is 'L',  the lower triangular part of A is\n"
    "replaced by L.  If uplo is 'U', the upper triangular part is\n"
    "replaced by L^H.  B is replaced by the solution.\n\n"
    "ARGUMENTS.\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* posv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B;
    CBLAS_INT n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "B", "uplo", "n", "nrhs", "ldA", "ldB",
        "offsetA", "offsetB", NULL};
    Py_ssize_t _n = n, _nrhs = nrhs, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Cnnnnnn", kwlist,
        &A, &B, &uplo_, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cnnnnnn", kwlist,
        &A, &B, &uplo, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0) n = A->nrows;
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

    switch (MAT_ID(A)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dposv)(&uplo, &n, &nrhs, MAT_BUFD(A)+oA, &ldA, MAT_BUFD(B)+oB,
                &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zposv)(&uplo, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(B)+oB,
                &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_pbtrf[] =
    "Cholesky factorization of a real symmetric or complex Hermitian\n"
    "positive definite band matrix.\n\n"
    "pbtrf(A, uplo='L', n=A.size[1], kd=A.size[0]-1, ldA=max(1,A.size[0]),"
    "\n"
    "      offsetA=0)\n\n"
    "PURPOSE\n"
    "Factors A as A=L*L^T or A = L*L^H, where A is an n by n real\n"
    "symmetric or complex Hermitian positive definite band matrix with\n"
    "kd subdiagonals and kd superdiagonals.  A is stored in the BLAS \n"
    "format for symmetric band matrices.  On exit, A contains the\n"
    "Cholesky factor in the BLAS format for triangular band matrices.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix.\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         nonnegative integer. If negative, the default value is\n"
    "          used.\n\n"
    "kd        nonnegative integer. If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= kd+1.  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* pbtrf(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A;
    CBLAS_INT n=-1, kd=-1, ldA=0, oA=0, info;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "uplo", "n", "kd", "ldA", "offsetA", NULL};
    Py_ssize_t _n = n, _kd = kd, _ldA = ldA, _oA = oA;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|Cnnnn", kwlist, &A,
        &uplo_, &_n, &_kd, &_ldA, &_oA))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|cnnnn", kwlist, &A,
        &uplo, &_n, &_kd, &_ldA, &_oA))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_kd, &kd) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (n < 0) n = A->ncols;
    if (n == 0) return Py_BuildValue("");
    if (uplo != 'U' && uplo != 'L') err_char("uplo", "'L', 'U'");
    if (kd < 0) kd = A->nrows - 1;
    if (kd < 0) err_nn_int("kd");
    if (ldA == 0) ldA = MAX(1, A->nrows);
    if (ldA < kd+1) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + kd + 1 > len(A)) err_buf_len("A");

    switch (MAT_ID(A)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dpbtrf)(&uplo, &n, &kd, MAT_BUFD(A)+oA, &ldA, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zpbtrf)(&uplo, &n, &kd, MAT_BUFZ(A)+oA, &ldA, &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_pbtrs[] =
    "Solves a real symmetric or complex Hermitian positive definite set\n"
    "of linear equations with a banded coefficient matrix, given the\n"
    "Cholesky factorization computed by pbtrf() or pbsv().\n\n"
    "pbtrs(A, B, uplo='L', n=A.size[1], kd=A.size[0]-1, nrhs=B.size[1],\n"
    "      ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0,\n"
    "      offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves A*X = B where A is an n by n real symmetric or complex \n"
    "Hermitian positive definite band matrix with kd subdiagonals and kd\n"
    "superdiagonals, and B is n by nrhs.  A contains the Cholesky factor\n"
    "of A, as returned by pbtrf() or pbtrs().  On exit, B is replaced by\n"
    "the solution X.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix.\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         nonnegative integer. If negative, the default value is\n"
    "          used.\n\n"
    "kd        nonnegative integer. If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer. If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= kd+1.  If zero, the default\n"
    "          value is used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* pbtrs(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B;
    CBLAS_INT n=-1, kd=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "B", "uplo", "n", "kd", "nrhs", "ldA", "ldB",
        "offsetA", "offsetB", NULL};
    Py_ssize_t _n = n, _kd = kd, _nrhs = nrhs, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Cnnnnnnn", kwlist,
        &A, &B, &uplo_, &_n, &_kd, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cnnnnnnn", kwlist,
        &A, &B, &uplo, &_n, &_kd, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_kd, &kd) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (uplo != 'U' && uplo != 'L') err_char("uplo", "'L', 'U'");
    if (n < 0) n = A->ncols;
    if (kd < 0) kd = A->nrows - 1;
    if (kd < 0) err_nn_int("kd");
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < kd+1) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + kd + 1 > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

    switch (MAT_ID(A)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dpbtrs)(&uplo, &n, &kd, &nrhs, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zpbtrs)(&uplo, &n, &kd, &nrhs, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_pbsv[] =
    "Solves a real symmetric or complex Hermitian positive definite set\n"
    "of linear equations with a banded coefficient matrix.\n\n"
    "pbsv(A, B, uplo='L', n=A.size[1], kd=A.size[0]-1, nrhs=B.size[1],\n"
    "     ldA=MAX(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0,\n"
    "     offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves A*X = B where A is an n by n real symmetric or complex\n"
    "Hermitian positive definite band matrix with kd subdiagonals and kd\n"
    "superdiagonals, and B is n by nrhs.\n"
    "On entry, A contains A in the BLAS format for symmetric band\n"
    "matrices.  On exit, A is replaced with the Cholesky factors, stored\n"
    "in the BLAS format for triangular band matrices.  B is replaced\n"
    "by the solution X.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix.\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         nonnegative integer. If negative, the default value is\n"
    "          used.\n\n"
    "kd        nonnegative integer. If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer. If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= kd+1.  If zero, the default\n"
    "          value is used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* pbsv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B;
    CBLAS_INT n=-1, kd=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "B", "uplo", "n", "kd", "nrhs", "ldA", "ldB",
        "offsetA", "offsetB", NULL};
    Py_ssize_t _n = n, _kd = kd, _nrhs = nrhs, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Cnnnnnnn", kwlist,
        &A, &B, &uplo_, &_n, &_kd, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cnnnnnnn", kwlist,
        &A, &B, &uplo, &_n, &_kd, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_kd, &kd) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (uplo != 'U' && uplo != 'L') err_char("uplo", "'L', 'U'");
    if (n < 0) n = A->ncols;
    if (kd < 0) kd = A->nrows - 1;
    if (kd < 0) err_nn_int("kd");
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < kd+1) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + kd + 1 > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

    switch (MAT_ID(A)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dpbsv)(&uplo, &n, &kd, &nrhs, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zpbsv)(&uplo, &n, &kd, &nrhs, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_pttrf[] =
    "Cholesky factorization of a real symmetric or complex Hermitian\n"
    "positive definite tridiagonal matrix.\n\n"
    "pttrf(d, e, n=len(d)-offsetd, offsetd=0, offsete=0)\n\n"
    "PURPOSE\n"
    "Factors A  as A = L*D*L^T or A = L*D*L^H where A is n by n, real\n"
    "symmetric or complex Hermitian, positive definite, and tridiagonal.\n"
    "On entry, d is the subdiagonal of A and e is the diagonal.  On \n"
    "exit, d contains the diagonal of D and e contains the subdiagonal\n"
    "of the unit bidiagonal matrix L.\n\n"
    "ARGUMENTS.\n"
    "d         'd' matrix\n\n"
    "e         'd' or 'z' matrix.\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "offsetd   nonnegative integer\n\n"
    "offsete   nonnegative integer";

static PyObject* pttrf(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *d, *e;
    CBLAS_INT n=-1, od=0, oe=0, info;
    static char *kwlist[] = {"d", "e", "n", "offsetd", "offsete", NULL};
    Py_ssize_t _n = n, _od = od, _oe = oe;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|nnn", kwlist, &d,
        &e, &_n, &_od, &_oe)) return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_od, &od) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oe, &oe) < 0)
        return NULL;

    if (!Matrix_Check(d)) err_mtrx("d");
    if (MAT_ID(d) != DOUBLE) err_type("d");
    if (!Matrix_Check(e)) err_mtrx("e");
    if (od < 0) err_nn_int("offsetd");
    if (n < 0) n = len(d) - od;
    if (n < 0) err_buf_len("d");
    if (od + n > len(d)) err_buf_len("d");
    if (n == 0) return Py_BuildValue("");
    if (oe < 0) err_nn_int("offsete");
    if (oe + n - 1  > len(e)) err_buf_len("e");

    switch (MAT_ID(e)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dpttrf)(&n, MAT_BUFD(d)+od, MAT_BUFD(e)+oe, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zpttrf)(&n, MAT_BUFD(d)+od, MAT_BUFZ(e)+oe, &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_pttrs[] =
    "Solves a real symmetric or complex Hermitian positive definite set\n"
    "of linear equations with a tridiagonal coefficient matrix, given \n"
    "the factorization computed by pttrf().\n\n"
    "pttrs(d, e, B, uplo='L', n=len(d)-offsetd, nrhs=B.size[1],\n"
    "      ldB=max(1,B.size[0], offsetd=0, offsete=0, offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves A*X=B with A n by n real or complex Hermitian positive\n"
    "definite and tridiagonal, and B n by nrhs.  On entry, d and e\n"
    "contain the Cholesky factorization L*D*L^T or L*D*L^H, for example,\n"
    "as returned by pttrf().  The argument d is the diagonal of the \n"
    "diagonal matrix D.  The argument uplo only matters in the complex\n"
    "case.  If uplo = 'L', then e is the subdiagonal of L.  If uplo='U',\n"
    "e is the superdiagonal of L^H.  On exit B is overwritten with the\n"
    "solution X. \n\n"
    "ARGUMENTS.\n"
    "d         'd' matrix\n\n"
    "e         'd' or 'z' matrix.\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as e.\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetd   nonnegative integer\n\n"
    "offsete   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* pttrs(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *d, *e, *B;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    CBLAS_INT n=-1, nrhs=-1, ldB=0, od=0, oe=0, oB=0, info;
    static char *kwlist[] = {"d", "e", "B", "uplo", "n", "nrhs", "ldB",
        "offsetd", "offsete", "offsetB", NULL};
    Py_ssize_t _n = n, _nrhs = nrhs, _ldB = ldB, _od = od, _oe = oe, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|Cnnnnnn", kwlist,
        &d, &e, &B, &uplo_, &_n, &_nrhs, &_ldB, &_od, &_oe, &_oB))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cnnnnnn", kwlist,
        &d, &e, &B, &uplo, &_n, &_nrhs, &_ldB, &_od, &_oe, &_oB))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_od, &od) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oe, &oe) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(d)) err_mtrx("d");
    if (MAT_ID(d) != DOUBLE) err_type("d");
    if (!Matrix_Check(e)) err_mtrx("e");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(e) != MAT_ID(B)) err_conflicting_ids;
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (od < 0) err_nn_int("offsetd");
    if (n < 0) n = len(d) - od;
    if (n < 0) err_buf_len("d");
    if (od + n > len(d)) err_buf_len("d");
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (oe < 0) err_nn_int("offsete");
    if (oe + n - 1  > len(e)) err_buf_len("e");
    if (oB < 0) err_nn_int("offsetB");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

    switch (MAT_ID(e)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dpttrs)(&n, &nrhs, MAT_BUFD(d)+od, MAT_BUFD(e)+oe,
                MAT_BUFD(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zpttrs)(&uplo, &n, &nrhs, MAT_BUFD(d)+od, MAT_BUFZ(e)+oe,
                MAT_BUFZ(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_ptsv[] =
    "Solves a real symmetric or complex Hermitian positive definite set\n"
    "of linear equations with a tridiagonal coefficient matrix.\n\n"
    "ptsv(d, e, B, n=len(d)-offsetd, nrhs=B.size[1], ldB=max(1,B.size[0],"
    "\n"
    "     offsetd=0, offsete=0, offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves A*X=B with A n by n real or complex Hermitian positive\n"
    "definite and tridiagonal.  A is specified by its diagonal d and\n"
    "subdiagonal e.  On exit B is overwritten with the solution, and d\n"
    "and e are overwritten with the elements of Cholesky factorization\n"
    "of A.\n\n"
    "ARGUMENTS.\n"
    "d         'd' matrix\n\n"
    "e         'd' or 'z' matrix.\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as e.\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetd   nonnegative integer\n\n"
    "offsete   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* ptsv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *d, *e, *B;
    CBLAS_INT n=-1, nrhs=-1, ldB=0, od=0, oe=0, oB=0, info;
    static char *kwlist[] = {"d", "e", "B", "n", "nrhs", "ldB", "offsetd",
        "offsete", "offsetB", NULL};
    Py_ssize_t _n = n, _nrhs = nrhs, _ldB = ldB, _od = od, _oe = oe, _oB = oB;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|nnnnnn", kwlist,
        &d, &e, &B, &_n, &_nrhs, &_ldB, &_od, &_oe, &_oB)) return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_od, &od) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oe, &oe) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(d)) err_mtrx("d");
    if (MAT_ID(d) != DOUBLE) err_type("d");
    if (!Matrix_Check(e)) err_mtrx("e");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(e) != MAT_ID(B)) err_conflicting_ids;
    if (od < 0) err_nn_int("offsetd");
    if (n < 0) n = len(d) - od;
    if (n < 0) err_buf_len("d");
    if (od + n > len(d)) err_buf_len("d");
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (oe < 0) err_nn_int("offsete");
    if (oe + n - 1  > len(e)) err_buf_len("e");
    if (oB < 0) err_nn_int("offsetB");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

    switch (MAT_ID(e)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dptsv)(&n, &nrhs, MAT_BUFD(d)+od, MAT_BUFD(e)+oe,
                MAT_BUFD(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zptsv)(&n, &nrhs, MAT_BUFD(d)+od, MAT_BUFZ(e)+oe,
                MAT_BUFZ(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_sytrf[] =
    "LDL^T factorization of a real or complex symmetric matrix.\n\n"
    "sytrf(A, ipiv, uplo='L', n=A.size[0], ldA=max(1,A.size[0]))\n\n"
    "PURPOSE\n"
    "Computes the LDL^T factorization of a real or complex symmetric\n"
    "n by n matrix  A.  On exit, A and ipiv contain the details of the\n"
    "factorization.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "ipiv      'i' matrix of length at least n\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* sytrf(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *ipiv;
    void *work;
    number wl;
    CBLAS_INT n=-1, ldA=0, oA=0, info, lwork;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "ipiv", "uplo", "n", "ldA", "offsetA", NULL};
    Py_ssize_t _n = n, _ldA = ldA, _oA = oA;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Cnnn", kwlist,
        &A, &ipiv, &uplo_, &_n, &_ldA, &_oA))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cnnn", kwlist,
        &A, &ipiv, &uplo, &_n, &_ldA, &_oA))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (len(ipiv) < n) err_buf_len("ipiv");

    CBLAS_INT *ipiv_ptr = cvxopt_cblas_int_acquire(ipiv, n, 0);
    if (!ipiv_ptr) return NULL;

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsytrf)(&uplo, &n, NULL, &ldA, NULL, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double)))){
                cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsytrf)(&uplo, &n, MAT_BUFD(A)+oA, &ldA, ipiv_ptr,
                (double *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
            break;

        case COMPLEX:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zsytrf)(&uplo, &n, NULL, &ldA, NULL, &wl.z, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
            if (!(work = (void *) calloc(lwork, sizeof(complex_t)))){
                cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zsytrf)(&uplo, &n, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
                (complex_t *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
            break;

        default:
            cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
            err_invalid_id;
    }

    cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 1);
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_hetrf[] =
    "LDL^H factorization of a real symmetric or complex Hermitian matrix."
    "\n\n"
    "hetrf(A, ipiv, uplo='L', n=A.size[0], ldA=max(1,A.size[0]))\n\n"
    "PURPOSE\n"
    "Computes the LDL^H factorization of a real symmetric or complex\n"
    "Hermitian n by n matrix  A.  On exit, A and ipiv contain the\n"
    "details of the factorization.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "ipiv      'i' matrix of length at least n\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* hetrf(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *ipiv;
    void *work;
    number wl;
    CBLAS_INT n=-1, ldA=0, oA=0, info, lwork;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "ipiv", "uplo", "n", "ldA", "offsetA", NULL};
    Py_ssize_t _n = n, _ldA = ldA, _oA = oA;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Cnnn", kwlist,
        &A, &ipiv, &uplo_, &_n, &_ldA, &_oA))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cnnn", kwlist,
        &A, &ipiv, &uplo, &_n, &_ldA, &_oA))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (len(ipiv) < n) err_buf_len("ipiv");

    CBLAS_INT *ipiv_ptr = cvxopt_cblas_int_acquire(ipiv, n, 0);
    if (!ipiv_ptr) return NULL;

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsytrf)(&uplo, &n, NULL, &ldA, NULL, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double)))){
                cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsytrf)(&uplo, &n, MAT_BUFD(A)+oA, &ldA, ipiv_ptr,
                (double *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
            break;

        case COMPLEX:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zhetrf)(&uplo, &n, NULL, &ldA, NULL, &wl.z, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
            if (!(work = (void *) calloc(lwork, sizeof(complex_t)))){
                cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zhetrf)(&uplo, &n, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
                (complex_t *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
            break;

        default:
            cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
            err_invalid_id;
    }

    cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 1);
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_sytrs[] =
    "Solves a real or complex symmetric set of linear equations,\n"
    "given the LDL^T factorization computed by sytrf() or sysv().\n\n"
    "sytrs(A, ipiv, B, uplo='L', n=A.size[0], nrhs=B.size[1],\n"
    "      ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0,\n"
    "      offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves A*X = B where A is real or complex symmetric and n by n,\n"
    "and B is n by nrhs.  On entry, A and ipiv contain the\n"
    "factorization of A as returned by sytrf() or sysv().  On exit, B is\n"
    "replaced by the solution.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "ipiv      'i' matrix \n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "ldB       nonnegative integer.  ldB >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* sytrs(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B, *ipiv;
    CBLAS_INT n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "ipiv", "B", "uplo", "n", "nrhs", "ldA", "ldB",
        "offsetA", "offsetB", NULL};
    Py_ssize_t _n = n, _nrhs = nrhs, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|Cnnnnnn", kwlist,
        &A, &ipiv, &B, &uplo_, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cnnnnnn", kwlist,
        &A, &ipiv, &B, &uplo, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
    if (len(ipiv) < n) err_buf_len("ipiv");

    CBLAS_INT *ipiv_ptr = cvxopt_cblas_int_acquire(ipiv, n, 1);
    if (!ipiv_ptr) return NULL;

    switch (MAT_ID(A)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsytrs)(&uplo, &n, &nrhs, MAT_BUFD(A)+oA, &ldA, ipiv_ptr,
                MAT_BUFD(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

	case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zsytrs)(&uplo, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
                MAT_BUFZ(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

	default:
            cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
	    err_invalid_id;
    }

    cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_hetrs[] =
    "Solves a real symmetric or complex Hermitian set of linear\n"
    "equations, given the LDL^H factorization computed by hetrf() or "
    "hesv().\n\n"
    "hetrs(A, ipiv, B, uplo='L', n=A.size[0], nrhs=B.size[1],\n"
    "      ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0,\n"
    "      offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves A*X = B where A is real symmetric or complex Hermitian\n"
    "and n by n, and B is n by nrhs.  On entry, A and ipiv contain\n"
    "the factorization of A as returned by hetrf or hesv.  On exit, B\n"
    "is replaced by the solution.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "ipiv      'i' matrix \n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'U' or 'L'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* hetrs(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B, *ipiv;
    CBLAS_INT n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
#if PY_MAJOR_VERSION >= 3
    int uplo_ ='L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "ipiv", "B", "uplo", "n", "nrhs", "ldA", "ldB",
        "offsetA", "offsetB", NULL};
    Py_ssize_t _n = n, _nrhs = nrhs, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|Cnnnnnn", kwlist,
        &A, &ipiv, &B, &uplo_, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cnnnnnn", kwlist,
        &A, &ipiv, &B, &uplo, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
    if (len(ipiv) < n) err_buf_len("ipiv");

    CBLAS_INT *ipiv_ptr = cvxopt_cblas_int_acquire(ipiv, n, 1);
    if (!ipiv_ptr) return NULL;

    switch (MAT_ID(A)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsytrs)(&uplo, &n, &nrhs, MAT_BUFD(A)+oA, &ldA,
                ipiv_ptr, MAT_BUFD(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

	case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zhetrs)(&uplo, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA,
                ipiv_ptr, MAT_BUFZ(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

	default:
            cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
	    err_invalid_id;
    }

    cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_sytri[] =
    "Inverse of a real or complex symmetric matrix.\n\n"
    "sytri(A, ipiv, uplo='L', n=A.size[0], ldA=max(1,A.size[0]),\n"
    "      offsetA=0)\n\n"
    "PURPOSE\n"
    "Computes the inverse of a real or complex symmetric matrix of\n"
    "order n.  On entry, A and ipiv contain the LDL^T factorization,\n"
    "as returned by sysv() or sytrf().  On exit A is replaced by the\n"
    "inverse.  \n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "ipiv      'i' matrix \n\n"
    "uplo      'L' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* sytri(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *ipiv;
    CBLAS_INT n=-1, ldA=0, oA=0, info;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    void *work;
    char *kwlist[] = {"A", "ipiv", "uplo", "n", "ldA", "offsetA", NULL};
    Py_ssize_t _n = n, _ldA = ldA, _oA = oA;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Cnnn", kwlist,
        &A, &ipiv, &uplo_, &_n, &_ldA, &_oA))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cnnn", kwlist,
        &A, &ipiv, &uplo, &_n, &_ldA, &_oA))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (len(ipiv) < n) err_buf_len("ipiv");

    CBLAS_INT *ipiv_ptr = cvxopt_cblas_int_acquire(ipiv, n, 1);
    if (!ipiv_ptr) return NULL;

    switch (MAT_ID(A)){
        case DOUBLE:
            if (!(work = (void *) calloc(n, sizeof(double)))) {
                cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsytri)(&uplo, &n, MAT_BUFD(A)+oA, &ldA, ipiv_ptr,
                (double *) work, &info);
            Py_END_ALLOW_THREADS
            free(work);
            break;

	case COMPLEX:
            if (!(work = (void *) calloc(2*n, sizeof(complex_t)))){
                cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zsytri)(&uplo, &n, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
                (complex_t *) work, &info);
            Py_END_ALLOW_THREADS
            free(work);
            break;

        default:
            cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
            err_invalid_id;
    }

    cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_hetri[] =
    "Inverse of a real symmetric or complex Hermitian matrix.\n\n"
    "hetri(A, ipiv, uplo='L', n=A.size[0], ldA=max(1,A.size[0]),\n"
    "      offsetA=0)\n\n"
    "PURPOSE\n"
    "Computes the inverse of a real symmetric or complex Hermitian\n"
    "matrix of order n.  On entry, A and ipiv contain the LDL^T\n"
    "factorization, as returned by hesv() or hetrf().  On exit A is\n"
    "replaced by the inverse. \n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "ipiv      'i' matrix \n\n"
    "uplo      'L' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* hetri(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *ipiv;
    CBLAS_INT n=-1, ldA=0, oA=0, info;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    void *work;
    char *kwlist[] = {"A", "ipiv", "uplo", "n", "ldA", "offsetA", NULL};
    Py_ssize_t _n = n, _ldA = ldA, _oA = oA;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Cnnn", kwlist,
        &A, &ipiv, &uplo_, &_n, &_ldA, &_oA))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cnnn", kwlist,
        &A, &ipiv, &uplo, &_n, &_ldA, &_oA))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(ipiv) || ipiv->id != INT) err_int_mtrx("ipiv");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (len(ipiv) < n) err_buf_len("ipiv");

    CBLAS_INT *ipiv_ptr = cvxopt_cblas_int_acquire(ipiv, n, 1);
    if (!ipiv_ptr) return NULL;

    switch (MAT_ID(A)){
        case DOUBLE:
            if (!(work = (void *) calloc(n, sizeof(double)))){
                cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsytri)(&uplo, &n, MAT_BUFD(A)+oA, &ldA, ipiv_ptr,
                (double *) work, &info);
            Py_END_ALLOW_THREADS
            free(work);
            break;

        case COMPLEX:
            if (!(work = (void *) calloc(n, sizeof(complex_t)))){
                cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zhetri)(&uplo, &n, MAT_BUFZ(A)+oA, &ldA, ipiv_ptr,
                (complex_t *) work, &info);
            Py_END_ALLOW_THREADS
            free(work);
            break;

        default:
            cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
            err_invalid_id;
    }

    cvxopt_cblas_int_release(ipiv, ipiv_ptr, n, 0);
    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_sysv[] =
    "Solves a real or complex symmetric set of linear equations.\n\n"
    "sysv(A, B, ipiv=None, uplo='L', n=A.size[0], nrhs=B.size[1],\n"
    "     ldA = max(1,A.size[0]), ldB = max(1,B.size[0]),\n"
    "     offsetA=0, offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves A*X = B where A is real or complex symmetric and n by n.\n"
    "If ipiv is provided, then on exit A and ipiv contain the details\n"
    "of the LDL^T factorization of A.  If ipiv is not provided, then\n"
    "the factorization is not returned and A is not modified.  On\n"
    "exit, B contains the solution.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "ipiv      'i' matrix of length at least n\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* sysv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B, *ipiv=NULL;
    CBLAS_INT n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info, lwork, k,
        *ipivc=NULL;
    void *work=NULL, *Ac=NULL;
    number wl;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "B", "ipiv", "uplo", "n", "nrhs", "ldA",
        "ldB", "offsetA", "offsetB", NULL};
    Py_ssize_t _n = n, _nrhs = nrhs, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|OCnnnnnn", kwlist,
        &A, &B, &ipiv, &uplo_, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Ocnnnnnn", kwlist,
        &A, &B, &ipiv, &uplo, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (ipiv && (!Matrix_Check(ipiv) || ipiv->id != INT))
        err_int_mtrx("ipiv");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
    if (ipiv && len(ipiv) < n) err_buf_len("ipiv");

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsytrf)(&uplo, &n, NULL, &ldA, NULL, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double))))
                return PyErr_NoMemory();
            if (ipiv) {
#if (CBLAS_INT_SIZE != SIZEOF_SIZE_T)
                if (!(ipivc = (CBLAS_INT *) calloc(n, sizeof(CBLAS_INT)))){
                    free(work);
                    return PyErr_NoMemory();
                }
                for (k=0; k<n; k++) ipivc[k] = MAT_BUFI(ipiv)[k];
#else
                ipivc = MAT_BUFI(ipiv);
#endif
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(dsysv)(&uplo, &n, &nrhs, MAT_BUFD(A)+oA, &ldA, ipivc,
                    MAT_BUFD(B)+oB, &ldB, (double *) work, &lwork,
                    &info);
                Py_END_ALLOW_THREADS
#if (CBLAS_INT_SIZE != SIZEOF_SIZE_T)
		for (k=0; k<n; k++) MAT_BUFI(ipiv)[k] = ipivc[k];
                free(ipivc);
#endif
	    }
            else {
                ipivc = (CBLAS_INT *) calloc(n, sizeof(CBLAS_INT));
                Ac = (void *) calloc(n*n, sizeof(double));
                if (!ipivc || !Ac){
                    free(work);  free(ipivc);  free(Ac);
                    return PyErr_NoMemory();
                }
                for (k=0; k<n; k++)
                    memcpy((double *) Ac + k*n, MAT_BUFD(A) + oA + k*ldA,
                        n*sizeof(double));
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(dsysv)(&uplo, &n, &nrhs, (double *) Ac, &n, ipivc,
                    MAT_BUFD(B)+oB, &ldB, work, &lwork, &info);
                Py_END_ALLOW_THREADS
                free(ipivc); free(Ac);
            }
            free(work);
            break;

        case COMPLEX:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zsytrf)(&uplo, &n, NULL, &ldA, NULL, &wl.z, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
            if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
                return PyErr_NoMemory();
            if (ipiv) {
#if (CBLAS_INT_SIZE != SIZEOF_SIZE_T)
                if (!(ipivc = (CBLAS_INT *) calloc(n, sizeof(CBLAS_INT)))){
                    free(work);
                    return PyErr_NoMemory();
                }
                for (k=0; k<n; k++) ipivc[k] = MAT_BUFI(ipiv)[k];
#else
                ipivc = MAT_BUFI(ipiv);
#endif
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(zsysv)(&uplo, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA, ipivc,
                    MAT_BUFZ(B)+oB, &ldB, (complex_t *) work, 
                    &lwork, &info);
                Py_END_ALLOW_THREADS
#if (CBLAS_INT_SIZE != SIZEOF_SIZE_T)
                for (k=0; k<n; k++) MAT_BUFI(ipiv)[k] = ipivc[k];
                free(ipivc);
#endif
            }
            else {
                ipivc = (CBLAS_INT *) calloc(n, sizeof(CBLAS_INT));
                Ac = (void *) calloc(n*n, sizeof(complex_t));
                if (!ipivc || !Ac){
                    free(work);  free(ipivc);  free(Ac);
                    return PyErr_NoMemory();
                }
                for (k=0; k<n; k++)
                    memcpy((complex_t *) Ac + k*n, 
                        MAT_BUFZ(A) + oA + k*ldA,
                        n*sizeof(complex_t));
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(zsysv)(&uplo, &n, &nrhs, (complex_t *) Ac, &n, ipivc,
                    MAT_BUFZ(B)+oB, &ldB, work, &lwork, &info);
                Py_END_ALLOW_THREADS
                free(ipivc);  free(Ac);
            }
            free(work);
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_hesv[] =
    "Solves a real symmetric or complex Hermitian set of linear\n"
    "equations.\n\n"
    "herv(A, B, ipiv=None, uplo='L', n=A.size[0], nrhs=B.size[1],\n"
    "     ldA = max(1,A.size[0]), ldB = max(1,B.size[0]), offsetA=0,\n"
    "     offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves A*X=B where A is real symmetric or complex Hermitian and\n"
    "n by n.  If ipiv is provided, then on exit A and ipiv contain\n"
    "the details of the LDL^H factorization of A.  If ipiv is not\n"
    "provided, then the factorization is not returned and A is not\n"
    "modified.  On exit, B contains the solution.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "ipiv      'i' matrix of length at least n\n\n"
    "uplo      'U' or 'L'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* hesv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B, *ipiv=NULL;
    CBLAS_INT n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info, lwork, k,
        *ipivc=NULL;
    void *work=NULL, *Ac=NULL;
    number wl;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "B", "ipiv", "uplo", "n", "nrhs", "ldA", "ldB",
        "offsetA", "offsetB", NULL};
    Py_ssize_t _n = n, _nrhs = nrhs, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|OCnnnnnn", kwlist,
        &A, &B, &ipiv, &uplo_, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Ocnnnnnn", kwlist,
        &A, &B, &ipiv, &uplo, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (ipiv && (!Matrix_Check(ipiv) || ipiv->id != INT))
        err_int_mtrx("ipiv");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");
    if (ipiv && len(ipiv) < n) err_buf_len("ipiv");

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsytrf)(&uplo, &n, NULL, &ldA, NULL, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double))))
                return PyErr_NoMemory();
            if (ipiv) {
#if (CBLAS_INT_SIZE != SIZEOF_SIZE_T)
                if (!(ipivc = (CBLAS_INT *) calloc(n,sizeof(CBLAS_INT)))){
                    free(work);
                    return PyErr_NoMemory();
                }
                int i; for (i=0; i<n; i++) ipivc[i] = MAT_BUFI(ipiv)[i];
#else
                ipivc = MAT_BUFI(ipiv);
#endif
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(dsysv)(&uplo, &n, &nrhs, MAT_BUFD(A)+oA, &ldA, ipivc,
                    MAT_BUFD(B)+oB, &ldB, (double *) work, &lwork,
                    &info);
                Py_END_ALLOW_THREADS
#if (CBLAS_INT_SIZE != SIZEOF_SIZE_T)
                for (i=0; i<n; i++) MAT_BUFI(ipiv)[i] = ipivc[i];
                free(ipivc);
#endif
            }
            else {
                ipivc = (CBLAS_INT *) calloc(n, sizeof(CBLAS_INT));
                Ac = (void *) calloc(n*n, sizeof(double));
                if (!ipivc || !Ac){
                    free(work);  free(ipivc);  free(Ac);
                    return PyErr_NoMemory();
                }
                for (k=0; k<n; k++)
                    memcpy((double *) Ac + k*n, MAT_BUFD(A) + oA + k*ldA,
                        n*sizeof(double));
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(dsysv)(&uplo, &n, &nrhs, (double *) Ac, &n, ipivc,
                    MAT_BUFD(B)+oB, &ldB, work, &lwork, &info);
                Py_END_ALLOW_THREADS
                free(ipivc);  free(Ac);
            }
            free(work);
            break;

        case COMPLEX:
            lwork = -1;
            BLAS_FUNC(zhetrf)(&uplo, &n, NULL, &ldA, NULL, &wl.z, &lwork, &info);
            lwork = (CBLAS_INT) creal(wl.z);
            if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
                return PyErr_NoMemory();
            if (ipiv) {
#if (CBLAS_INT_SIZE != SIZEOF_SIZE_T)
                if (!(ipivc = (CBLAS_INT *) calloc(n,sizeof(CBLAS_INT)))){
                    free(work);
                    return PyErr_NoMemory();
                }
                for (k=0; k<n; k++) ipivc[k] = MAT_BUFI(ipiv)[k];
#else
                ipivc = MAT_BUFI(ipiv);
#endif
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(zhesv)(&uplo, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA, ipivc,
                    MAT_BUFZ(B)+oB, &ldB, (complex_t *) work, 
                    &lwork, &info);
                Py_END_ALLOW_THREADS
#if (CBLAS_INT_SIZE != SIZEOF_SIZE_T)
                for (k=0; k<n; k++) MAT_BUFI(ipiv)[k] = ipivc[k];
                free(ipivc);
#endif
            }
            else {
                ipivc = (CBLAS_INT *) calloc(n, sizeof(CBLAS_INT));
                Ac = (void *) calloc(n*n, sizeof(complex_t));
                if (!ipivc || !Ac){
                    free(work);  free(ipivc);  free(Ac);
                    return PyErr_NoMemory();
                }
                for (k=0; k<n; k++)
                    memcpy((complex_t *) Ac + k*n, 
                        MAT_BUFZ(A) + oA + k*ldA,
                        n*sizeof(complex_t));
                Py_BEGIN_ALLOW_THREADS
                BLAS_FUNC(zhesv)(&uplo, &n, &nrhs, (complex_t *) Ac, &n, ipivc,
                    MAT_BUFZ(B)+oB, &ldB, work, &lwork, &info);
                Py_END_ALLOW_THREADS
                free(ipivc);  free(Ac);
            }
            free(work);
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_trtrs[] =
    "Solution of a triangular set of equations with multiple righthand\n"
    "sides.\n\n"
    "trtrs(A, B, uplo='L', trans='N', diag='N', n=A.size[0],\n"
    "      nrhs=B.size[1], ldA=max(1,A.size[0]), ldB=max(1,B.size[0]),\n"
    "      offsetA=0, offsetB=0)\n\n"
    "PURPOSE\n"
    "If trans is 'N', solves A*X = B.\n"
    "If trans is 'T', solves A^T*X = B.\n"
    "If trans is 'C', solves A^H*X = B.\n"
    "B is n by nrhs and A is triangular of order n.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "trans     'N', 'T' or 'C'\n\n"
    "diag      'N' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* trtrs(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B;
    CBLAS_INT n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', trans_ = 'N', diag_ = 'N';
#endif
    char uplo = 'L', trans = 'N', diag = 'N';
    char *kwlist[] = {"A", "B", "uplo", "trans", "diag", "n", "nrhs",
        "ldA", "ldB", "offsetA", "offsetB", NULL};
    Py_ssize_t _n = n, _nrhs = nrhs, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCnnnnnn", kwlist,
        &A, &B, &uplo_, &trans_, &diag_, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
    uplo = (char) uplo_;
    trans = (char) trans_;
    diag = (char) diag_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cccnnnnnn", kwlist,
        &A, &B, &uplo, &trans, &diag, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'N', 'U'");
    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (n < 0){
        n = A->nrows;
        if (A->nrows != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (trans == 'C') trans = 'T';
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dtrtrs)(&uplo, &trans, &diag, &n, &nrhs, MAT_BUFD(A)+oA,
                &ldA, MAT_BUFD(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(ztrtrs)(&uplo, &trans, &diag, &n, &nrhs, MAT_BUFZ(A)+oA,
                &ldA, MAT_BUFZ(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_trtri[] =
    "Inverse of a triangular matrix.\n\n"
    "trtri(A, uplo='L', diag='N', n=A.size[0], ldA=max(1,A.size[0]),\n"
    "      offsetA=0)\n\n"
    "PURPOSE\n"
    "Computes the inverse of a triangular matrix of order n.\n"
    "On exit, A is replaced with its inverse.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "uplo      'L' or 'U'\n\n"
    "diag      'N' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* trtri(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A;
    CBLAS_INT n=-1, ldA=0, oA=0, info;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', diag_ = 'N';
#endif
    char uplo = 'L', diag = 'N';
    char *kwlist[] = {"A", "uplo", "diag", "n", "ldA", "offsetA", NULL};
    Py_ssize_t _n = n, _ldA = ldA, _oA = oA;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|CCnnn", kwlist,
        &A, &uplo_, &diag_, &_n, &_ldA, &_oA)) return NULL;
    uplo = (char) uplo_;
    diag = (char) diag_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|ccnnn", kwlist,
        &A, &uplo, &diag, &_n, &_ldA, &_oA)) return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'N', 'U'");
    if (n < 0){
        n = A->nrows;
        if (A->nrows != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");

    switch (MAT_ID(A)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dtrtri)(&uplo, &diag, &n, MAT_BUFD(A)+oA, &ldA, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(ztrtri)(&uplo, &diag, &n, MAT_BUFZ(A)+oA, &ldA, &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_tbtrs[] =
    "Solution of a triangular set of equations with banded coefficient\n"
    "matrix.\n\n"
    "tbtrs(A, B, uplo='L', trans='N', diag='N', n=A.size[1], \n"
    "      kd=A.size[0]-1, nrhs=B.size[1], ldA=max(1,A.size[0]),\n"
    "      ldB=max(1,B.size[0]), offsetA=0, offsetB=0)\n\n"
    "PURPOSE\n"
    "If trans is 'N', solves A*X = B.\n"
    "If trans is 'T', solves A^T*X = B.\n"
    "If trans is 'C', solves A^H*X = B.\n"
    "B is n by nrhs and A is a triangular band matrix of order n with kd\n"
    "subdiagonals (uplo is 'L') or superdiagonals (uplo is 'U').\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "trans     'N', 'T' or 'C'\n\n"
    "diag      'N' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "kd        nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "nrhs      nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= kd+1.  If zero, the default\n"
    "          value is used.\n\n"
    "ldB       positive integer.  ldB >= max(1,n).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* tbtrs(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', trans_ = 'N', diag_ = 'N';
#endif
    char uplo = 'L', trans = 'N', diag = 'N';
    CBLAS_INT n=-1, kd=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info;
    char *kwlist[] = {"A", "B", "uplo", "trans", "diag", "n", "kd", "nrhs",
        "ldA", "ldB", "offsetA", "offsetB", NULL};
    Py_ssize_t _n = n, _kd = kd, _nrhs = nrhs, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCnnnnnnn", kwlist,
        &A, &B, &uplo_, &trans_, &diag_, &_n, &_kd, &_nrhs, &_ldA, &_ldB, &_oA,
        &_oB))
        return NULL;
    uplo = (char) uplo_;
    trans = (char) trans_;
    diag = (char) diag_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cccnnnnnnn", kwlist,
        &A, &B, &uplo, &trans, &diag, &_n, &_kd, &_nrhs, &_ldA, &_ldB, &_oA,
        &_oB))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_kd, &kd) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'N', 'U'");
    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (n < 0) n = A->ncols;
    if (kd < 0) kd = A->nrows - 1;
    if (kd < 0) err_nn_int("kd");
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < kd+1) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + kd + 1 > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > len(B)) err_buf_len("B");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (trans == 'C') trans = 'T';
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dtbtrs)(&uplo, &trans, &diag, &n, &kd, &nrhs, MAT_BUFD(A)+oA,
                &ldA, MAT_BUFD(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(ztbtrs)(&uplo, &trans, &diag, &n, &kd, &nrhs, MAT_BUFZ(A)+oA,
                &ldA, MAT_BUFZ(B)+oB, &ldB, &info);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_gels[] =
    "Solves least-squares and least-norm problems with full rank\n"
    "matrices.\n\n"
    "gels(A, B, trans='N', m=A.size[0], n=A.size[1], nrhs=B.size[1],\n"
    "     ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0,\n"
    "     offsetB=0)\n\n"
    "PURPOSE\n"
    "1. If trans is 'N' and A and B are real/complex:\n"
    "- if m >= n: minimizes ||A*X - B||_F.\n"
    "- if m < n: minimizes ||X||_F subject to A*X = B.\n\n"
    "2. If trans is 'N' or 'C' and A and B are real:\n"
    "- if m >= n: minimizes ||X||_F subject to A^T*X = B.\n"
    "- if m < n: minimizes ||X||_F subject to A^T*X = B.\n\n"
    "3. If trans is 'C' and A and B are complex:\n"
    "- if m >= n: minimizes ||X||_F subject to A^H*X = B.\n"
    "- if m < n: minimizes ||X||_F subject to A^H*X = B.\n\n"
    "A is an m by n matrix.  B has nrhs columns.  On exit, B is\n"
    "replaced with the solution, and A is replaced with the details\n"
    "of its QR or LQ factorization.\n\n"
    "Note that gels does not check whether A has full rank.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "trans     'N', 'T' or 'C' if A is real.  'N' or 'C' if A is\n"
    "          complex.\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "nrhs      integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the\n"
    "          default value is used.\n\n"
    "ldB       nonnegative integer.  ldB >= max(1,m,n).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* gels(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B;
    CBLAS_INT m=-1, n=-1, nrhs=-1, ldA=0, ldB=0, oA=0, oB=0, info, lwork;
    void *work;
    number wl;
#if PY_MAJOR_VERSION >= 3
    int trans_ = 'N';
#endif
    char trans = 'N';
    char *kwlist[] = {"A", "B", "trans", "m", "n", "nrhs", "ldA", "ldB",
        "offsetA", "offsetB", NULL};
    Py_ssize_t _m = m, _n = n, _nrhs = nrhs, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Cnnnnnnn",
        kwlist, &A, &B, &trans_, &_m, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cnnnnnnn",
        kwlist, &A, &B, &trans, &_m, &_n, &_nrhs, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_nrhs, &nrhs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (m < 0) m = A->nrows;
    if (n < 0) n = A->ncols;
    if (nrhs < 0) nrhs = B->ncols;
    if (m == 0 || n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,m)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(MAX(1,n),m)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + ((trans == 'N') ? n : m) > len(B))
        err_buf_len("B");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (trans == 'C') trans = 'T';
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgels)(&trans, &m, &n, &nrhs, NULL, &ldA, NULL, &ldB, &wl.d,
                &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgels)(&trans, &m, &n, &nrhs, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(B)+oB, &ldB, (double *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        case COMPLEX:
            if (trans == 'T') err_char("trans", "'N', 'C'");
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgels)(&trans, &m, &n, &nrhs, NULL, &ldA, NULL, &ldB, &wl.z,
                &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
            if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgels)(&trans, &m, &n, &nrhs, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(B)+oB, &ldB, (complex_t *) work, &lwork, 
                &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_geqrf[] =
    "QR factorization.\n\n"
    "geqrf(A, tau, m=A.size[0], n=A.size[1], ldA=max(1,A.size[0]),\n"
    "      offsetA=0)\n\n"
    "PURPOSE\n"
    "QR factorization of an m by n real or complex matrix A:\n\n"
    "A = Q*R = [Q1 Q2] * [R1; 0] if m >= n\n"
    "A = Q*R = Q * [R1 R2] if m <= n,\n\n"
    "where Q is m by m and orthogonal/unitary and R is m by n with R1\n"
    "upper triangular.  On exit, R is stored in the upper triangular\n"
    "part of A.  Q is stored as a product of k=min(m,n) elementary\n"
    "reflectors.  The parameters of the reflectors are stored in the\n"
    "first k entries of tau and in the lower triangular part of the\n"
    "first k columns of A.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "tau       'd' or 'z' matrix of length at least min(m,n).  Must\n"
    "          have the same type as A.\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* geqrf(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *tau;
    CBLAS_INT m=-1, n=-1, ldA=0, oA=0, info, lwork;
    void *work;
    number wl;
    char *kwlist[] = {"A", "tau", "m", "n", "ldA", "offsetA", NULL};
    Py_ssize_t _m = m, _n = n, _ldA = ldA, _oA = oA;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|nnnn", kwlist,
        &A, &tau, &_m, &_n, &_ldA, &_oA))
        return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(tau)) err_mtrx("tau");
    if (MAT_ID(A) != MAT_ID(tau)) err_conflicting_ids;
    if (m < 0) m = A->nrows;
    if (n < 0) n = A->ncols;
    if (m == 0 || n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,m)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
    if (len(tau) < MIN(m,n)) err_buf_len("tau");

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgeqrf)(&m, &n, NULL, &ldA, NULL, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgeqrf)(&m, &n, MAT_BUFD(A)+oA, &ldA, MAT_BUFD(tau),
                (double *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        case COMPLEX:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgeqrf)(&m, &n, NULL, &ldA, NULL, &wl.z, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
            if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgeqrf)(&m, &n, MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(tau),
                (complex_t *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_ormqr[] =
    "Product with a real orthogonal matrix.\n\n"
    "ormqr(A, tau, C, side='L', trans='N', m=C.size[0], n=C.size[1],\n"
    "      k=len(tau), ldA=max(1,A.size[0]), ldC=max(1,C.size[0]),\n"
    "      offsetA=0, offsetC=0)\n\n"
    "PURPOSE\n"
    "Computes\n"
    "C := Q*C   if side = 'L' and trans = 'N'.\n"
    "C := Q^T*C if side = 'L' and trans = 'T'.\n"
    "C := C*Q   if side = 'R' and trans = 'N'.\n"
    "C := C*Q^T if side = 'R' and trans = 'T'.\n"
    "C is m by n and Q is a square orthogonal matrix computed by geqrf."
    "\n"
    "Q is defined as a product of k elementary reflectors, stored as\n"
    "the first k columns of A and the first k entries of tau.\n\n"
    "ARGUMENTS\n"
    "A         'd' matrix\n\n"
    "tau       'd' matrix of length at least k\n\n"
    "C         'd' matrix\n\n"
    "side      'L' or 'R'\n\n"
    "trans     'N' or 'T'\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "k         integer.  k <= m if side = 'R' and k <= n if side = 'L'.\n"
    "          If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,m) if side = 'L'\n"
    "          and ldA >= max(1,n) if side = 'R'.  If zero, the\n"
    "          default value is used.\n\n"
    "ldC       nonnegative integer.  ldC >= max(1,m).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* ormqr(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *tau, *C;
    CBLAS_INT m=-1, n=-1, k=-1, ldA=0, ldC=0, oA=0, oC=0, info, lwork;
    void *work;
    number wl;
#if PY_MAJOR_VERSION >= 3
    int side_ = 'L', trans_ = 'N';
#endif
    char side = 'L', trans = 'N';
    char *kwlist[] = {"A", "tau", "C", "side", "trans", "m", "n", "k",
        "ldA", "ldC", "offsetA", "offsetC", NULL};
    Py_ssize_t _m = m, _n = n, _k = k, _ldA = ldA, _ldC = ldC, _oA = oA, _oC = oC;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|CCnnnnnnn",
        kwlist, &A, &tau, &C, &side_, &trans_, &_m, &_n, &_k, &_ldA, &_ldC,
        &_oA, &_oC))
        return NULL;
    side = (char) side_;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ccnnnnnnn",
        kwlist, &A, &tau, &C, &side, &trans, &_m, &_n, &_k, &_ldA, &_ldC,
        &_oA, &_oC))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_k, &k) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldC, &ldC) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oC, &oC) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(tau)) err_mtrx("tau");
    if (!Matrix_Check(C)) err_mtrx("C");
    if (MAT_ID(A) != MAT_ID(tau) || MAT_ID(A) != MAT_ID(C))
        err_conflicting_ids;
    if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");
    if (trans != 'N' && trans != 'T') err_char("trans", "'N', 'T'");
    if (m < 0) m = C->nrows;
    if (n < 0) n = C->ncols;
    if (k < 0) k = len(tau);
    if (m == 0 || n == 0 || k == 0) return Py_BuildValue("");
    if (k > ((side == 'L') ? m : n)) err_ld("k");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < ((side == 'L') ? MAX(1,m) : MAX(1,n))) err_ld("ldA");
    if (ldC == 0) ldC = MAX(1,C->nrows);
    if (ldC < MAX(1,m)) err_ld("ldC");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + k*ldA  > len(A)) err_buf_len("A");
    if (oC < 0) err_nn_int("offsetC");
    if (oC + (n-1)*ldC + m > len(C)) err_buf_len("C");
    if (len(tau) < k) err_buf_len("tau");

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dormqr)(&side, &trans, &m, &n, &k, NULL, &ldA, NULL, NULL,
                &ldC, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dormqr)(&side, &trans, &m, &n, &k, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(tau), MAT_BUFD(C)+oC, &ldC, (double *) work,
                &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_unmqr[] =
    "Product with a real or complex orthogonal matrix.\n\n"
    "unmqr(A, tau, C, side='L', trans='N', m=C.size[0], n=C.size[1],\n"
    "      k=len(tau), ldA=max(1,A.size[0]), ldC=max(1,C.size[0]),\n"
    "      offsetA=0, offsetC=0)\n\n"
    "PURPOSE\n"
    "Computes\n"
    "C := Q*C   if side = 'L' and trans = 'N'.\n"
    "C := Q^T*C if side = 'L' and trans = 'T'.\n"
    "C := Q^H*C if side = 'L' and trans = 'C'.\n"
    "C := C*Q   if side = 'R' and trans = 'N'.\n"
    "C := C*Q^T if side = 'R' and trans = 'T'.\n"
    "C := C*Q^H if side = 'R' and trans = 'C'.\n"
    "C is m by n and Q is a square orthogonal/unitary matrix computed\n"
    "by geqrf.  Q is defined as a product of k elementary reflectors,\n"
    "stored as the first k columns of A and the first k entries of tau."
    "\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "tau       'd' or 'z' matrix of length at least k.  Must have the\n"
    "          same type as A.\n\n"
    "C         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "side      'L' or 'R'\n\n"
    "trans     'N', 'T', or 'C'n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "k         integer.  k <= m if side = 'R' and k <= n if side = 'L'.\n"
    "          If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,m) if side = 'L'\n"
    "          and ldA >= max(1,n) if side = 'R'.  If zero, the\n"
    "          default value is used.\n\n"
    "ldC       nonnegative integer.  ldC >= max(1,m).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* unmqr(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *tau, *C;
    CBLAS_INT m=-1, n=-1, k=-1, ldA=0, ldC=0, oA=0, oC=0, info, lwork;
    void *work;
    number wl;
#if PY_MAJOR_VERSION >= 3
    int side_ = 'L', trans_ = 'N';
#endif
    char side = 'L', trans = 'N';
    char *kwlist[] = {"A", "tau", "C", "side", "trans", "m", "n", "k",
        "ldA", "ldC", "offsetA", "offsetC", NULL};
    Py_ssize_t _m = m, _n = n, _k = k, _ldA = ldA, _ldC = ldC, _oA = oA, _oC = oC;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|CCnnnnnnn",
        kwlist, &A, &tau, &C, &side_, &trans_, &_m, &_n, &_k, &_ldA, &_ldC,
        &_oA, &_oC))
        return NULL;
    side = (char) side_;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ccnnnnnnn",
        kwlist, &A, &tau, &C, &side, &trans, &_m, &_n, &_k, &_ldA, &_ldC,
        &_oA, &_oC))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_k, &k) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldC, &ldC) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oC, &oC) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(tau)) err_mtrx("tau");
    if (!Matrix_Check(C)) err_mtrx("C");
    if (MAT_ID(A) != MAT_ID(tau) || MAT_ID(A) != MAT_ID(C))
        err_conflicting_ids;
    if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");
    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (m < 0) m = C->nrows;
    if (n < 0) n = C->ncols;
    if (k < 0) k = len(tau);
    if (m == 0 || n == 0 || k == 0) return Py_BuildValue("");
    if (k > ((side == 'L') ? m : n)) err_ld("k");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < ((side == 'L') ? MAX(1,m) : MAX(1,n))) err_ld("ldA");
    if (ldC == 0) ldC = MAX(1,C->nrows);
    if (ldC < MAX(1,m)) err_ld("ldC");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + k*ldA > len(A)) err_buf_len("A");
    if (oC < 0) err_nn_int("offsetC");
    if (oC + (n-1)*ldC + m > len(C)) err_buf_len("C");
    if (len(tau) < k) err_buf_len("tau");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (trans == 'C') trans = 'T';
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dormqr)(&side, &trans, &m, &n, &k, NULL, &ldA, NULL, NULL,
                &ldC, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dormqr)(&side, &trans, &m, &n, &k, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(tau), MAT_BUFD(C)+oC, &ldC, (double *) work,
                &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        case COMPLEX:
            if (trans == 'T') err_char("trans", "'N', 'C'");
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zunmqr)(&side, &trans, &m, &n, &k, NULL, &ldA, NULL, NULL,
                &ldC, &wl.z, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
            if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zunmqr)(&side, &trans, &m, &n, &k, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(tau), MAT_BUFZ(C)+oC, &ldC, 
                (complex_t *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_orgqr[] =
    "Generate the orthogonal matrix in a QR factorization.\n\n"
    "ormqr(A, tau, m=A.size[0], n=min(A.size), k=len(tau), \n"
    "      ldA=max(1,A.size[0]), offsetA=0)\n\n"
    "PURPOSE\n"
    "On entry, A and tau contain an m by m orthogonal matrix Q.\n"
    "Q is defined as a product of k elementary reflectors, stored in the\n"
    "first k columns of A and in tau, as computed by geqrf().  On exit,\n"
    "the first n columns of Q are stored in the leading columns of A.\n\n"
    "ARGUMENTS\n"
    "A         'd' matrix\n\n"
    "tau       'd' matrix of length at least k\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  n <= m.  If negative, the default value is used."
    "\n\n"
    "k         integer.  k <= n.  If negative, the default value is \n"
    "          used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* orgqr(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *tau;
    CBLAS_INT m=-1, n=-1, k=-1, ldA=0, oA=0, info, lwork;
    void *work;
    number wl;
    char *kwlist[] = {"A", "tau", "m", "n", "k", "ldA", "offsetA", NULL};
    Py_ssize_t _m = m, _n = n, _k = k, _ldA = ldA, _oA = oA;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|nnnnn", kwlist, &A,
        &tau, &_m, &_n, &_k, &_ldA, &_oA)) return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_k, &k) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(tau)) err_mtrx("tau");
    if (MAT_ID(A) != MAT_ID(tau)) err_conflicting_ids;
    if (m < 0) m = A->nrows;
    if (n < 0) n = MIN(A->nrows, A->ncols);
    if (n > m) err_ld("n");
    if (k < 0) k = len(tau);
    if (k > n) err_ld("k");
    if (m == 0 || n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1, A->nrows);
    if (ldA <  MAX(1, m)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + n*ldA  > len(A)) err_buf_len("A");
    if (len(tau) < k) err_buf_len("tau");

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dorgqr)(&m, &n, &k, NULL, &ldA, NULL, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dorgqr)(&m, &n, &k, MAT_BUFD(A) + oA, &ldA, MAT_BUFD(tau),
                (double *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_ungqr[] =
    "Generate the orthogonal or unitary matrix in a QR factorization.\n\n"
    "ungqr(A, tau, m=A.size[0], n=min(A.size), k=len(tau), \n"
    "      ldA=max(1,A.size[0]), offsetA=0)\n\n"
    "PURPOSE\n"
    "On entry, A and tau contain an m by m orthogonal/unitary matrix Q.\n"
    "Q is defined as a product of k elementary reflectors, stored in the\n"
    "first k columns of A and in tau, as computed by geqrf().  On exit,\n"
    "the first n columns of Q are stored in the leading columns of A.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "tau       'd' or 'z' matrix of length at least k.  Must have the\n"
    "          same type as A.\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  n <= m.  If negative, the default value is used."
    "\n\n"
    "k         integer.  k <= n.  If negative, the default value is \n"
    "          used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* ungqr(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *tau;
    CBLAS_INT m=-1, n=-1, k=-1, ldA=0, oA=0, info, lwork;
    void *work;
    number wl;
    char *kwlist[] = {"A", "tau", "m", "n", "k", "ldA", "offsetA", NULL};
    Py_ssize_t _m = m, _n = n, _k = k, _ldA = ldA, _oA = oA;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|nnnnn",
        kwlist, &A, &tau, &_m, &_n, &_k, &_ldA, &_oA)) return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_k, &k) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(tau)) err_mtrx("tau");
    if (MAT_ID(A) != MAT_ID(tau)) err_conflicting_ids;
    if (m < 0) m = A->nrows;
    if (n < 0) n = MIN(A->nrows, A->ncols);
    if (n > m) err_ld("n");
    if (k < 0) k = len(tau);
    if (k > n) err_ld("k");
    if (m == 0 || n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1, A->nrows);
    if (ldA <  MAX(1, m)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + n*ldA  > len(A)) err_buf_len("A");
    if (len(tau) < k) err_buf_len("tau");

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dorgqr)(&m, &n, &k, NULL, &ldA, NULL, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dorgqr)(&m, &n, &k, MAT_BUFD(A) + oA, &ldA, MAT_BUFD(tau),
                (double *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        case COMPLEX:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zungqr)(&m, &n, &k, NULL, &ldA, NULL, &wl.z, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
            if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zungqr)(&m, &n, &k, MAT_BUFZ(A) + oA, &ldA, MAT_BUFZ(tau),
                (complex_t *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_gelqf[] =
    "LQ factorization.\n\n"
    "gelqf(A, tau, m=A.size[0], n=A.size[1], ldA=max(1,A.size[0]),\n"
    "      offsetA=0)\n\n"
    "PURPOSE\n"
    "LQ factorization of an m by n real or complex matrix A:\n\n"
    "A = L*Q = [L1; 0] * [Q1; Q2] if m <= n\n"
    "A = L*Q = [L1; L2] * Q if m >= n,\n\n"
    "where Q is n by n and orthogonal/unitary and L is m by n with L1\n"
    "lower triangular.  On exit, L is stored in the lower triangular\n"
    "part of A.  Q is stored as a product of k=min(m,n) elementary\n"
    "reflectors.  The parameters of the reflectors are stored in the\n"
    "first k entries of tau and in the upper  triangular part of the\n"
    "first k rows of A.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "tau       'd' or 'z' matrix of length at least min(m,n).  Must\n"
    "          have the same type as A.\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* gelqf(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *tau;
    CBLAS_INT m=-1, n=-1, ldA=0, oA=0, info, lwork;
    void *work;
    number wl;
    char *kwlist[] = {"A", "tau", "m", "n", "ldA", "offsetA", NULL};
    Py_ssize_t _m = m, _n = n, _ldA = ldA, _oA = oA;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|nnnn", kwlist,
        &A, &tau, &_m, &_n, &_ldA, &_oA))
        return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(tau)) err_mtrx("tau");
    if (MAT_ID(A) != MAT_ID(tau)) err_conflicting_ids;
    if (m < 0) m = A->nrows;
    if (n < 0) n = A->ncols;
    if (m == 0 || n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,m)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
    if (len(tau) < MIN(m,n)) err_buf_len("tau");

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgelqf)(&m, &n, NULL, &ldA, NULL, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgelqf)(&m, &n, MAT_BUFD(A)+oA, &ldA, MAT_BUFD(tau),
                (double *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        case COMPLEX:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgelqf)(&m, &n, NULL, &ldA, NULL, &wl.z, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
            if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgelqf)(&m, &n, MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(tau),
                (complex_t *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_ormlq[] =
    "Product with a real orthogonal matrix.\n\n"
    "ormlq(A, tau, C, side='L', trans='N', m=C.size[0], n=C.size[1],\n"
    "      k=min(A.size), ldA=max(1,A.size[0]), ldC=max(1,C.size[0]),\n"
    "      offsetA=0, offsetC=0)\n\n"
    "PURPOSE\n"
    "Computes\n"
    "C := Q*C   if side = 'L' and trans = 'N'.\n"
    "C := Q^T*C if side = 'L' and trans = 'T'.\n"
    "C := C*Q   if side = 'R' and trans = 'N'.\n"
    "C := C*Q^T if side = 'R' and trans = 'T'.\n"
    "C is m by n and Q is a square orthogonal matrix computed by gelqf."
    "\n"
    "Q is defined as a product of k elementary reflectors, stored as\n"
    "the first k rows of A and the first k entries of tau.\n\n"
    "ARGUMENTS\n"
    "A         'd' matrix\n\n"
    "tau       'd' matrix of length at least k\n\n"
    "C         'd' matrix\n\n"
    "side      'L' or 'R'\n\n"
    "trans     'N' or 'T'\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "k         integer.  k <= m if side = 'L' and k <= n if side = 'R'.\n"
    "          If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,k).  If zero, the\n"
    "          default value is used.\n\n"
    "ldC       nonnegative integer.  ldC >= max(1,m).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* ormlq(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *tau, *C;
    CBLAS_INT m=-1, n=-1, k=-1, ldA=0, ldC=0, oA=0, oC=0, info, lwork;
    void *work;
    number wl;
#if PY_MAJOR_VERSION >= 3
    int side_ = 'L', trans_ = 'N';
#endif
    char side = 'L', trans = 'N';
    char *kwlist[] = {"A", "tau", "C", "side", "trans", "m", "n", "k",
        "ldA", "ldC", "offsetA", "offsetC", NULL};
    Py_ssize_t _m = m, _n = n, _k = k, _ldA = ldA, _ldC = ldC, _oA = oA, _oC = oC;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|CCnnnnnnn",
        kwlist, &A, &tau, &C, &side_, &trans_, &_m, &_n, &_k, &_ldA, &_ldC,
        &_oA, &_oC))
        return NULL;
    side = (char) side_;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ccnnnnnnn",
        kwlist, &A, &tau, &C, &side, &trans, &_m, &_n, &_k, &_ldA, &_ldC,
        &_oA, &_oC))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_k, &k) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldC, &ldC) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oC, &oC) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(tau)) err_mtrx("tau");
    if (!Matrix_Check(C)) err_mtrx("C");
    if (MAT_ID(A) != MAT_ID(tau) || MAT_ID(A) != MAT_ID(C))
        err_conflicting_ids;
    if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");
    if (trans != 'N' && trans != 'T') err_char("trans", "'N', 'T'");
    if (m < 0) m = C->nrows;
    if (n < 0) n = C->ncols;
    if (k < 0) k = MIN(A->nrows, A->ncols);
    if (m == 0 || n == 0 || k == 0) return Py_BuildValue("");
    if (k > ((side == 'L') ? m : n)) err_ld("k");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,k)) err_ld("ldA");
    if (ldC == 0) ldC = MAX(1,C->nrows);
    if (ldC < MAX(1,m)) err_ld("ldC");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + ldA * ((side == 'L') ? m : n) > len(A)) err_buf_len("A");
    if (oC < 0) err_nn_int("offsetC");
    if (oC + (n-1)*ldC + m > len(C)) err_buf_len("C");
    if (len(tau) < k) err_buf_len("tau");

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dormlq)(&side, &trans, &m, &n, &k, NULL, &ldA, NULL, NULL,
                &ldC, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dormlq)(&side, &trans, &m, &n, &k, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(tau), MAT_BUFD(C)+oC, &ldC, (double *) work,
                &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_unmlq[] =
    "Product with a real or complex orthogonal matrix.\n\n"
    "unmlq(A, tau, C, side='L', trans='N', m=C.size[0], n=C.size[1],\n"
    "      k=min(A.size), ldA=max(1,A.size[0]), ldC=max(1,C.size[0]),\n"
    "      offsetA=0, offsetC=0)\n\n"
    "PURPOSE\n"
    "Computes\n"
    "C := Q*C   if side = 'L' and trans = 'N'.\n"
    "C := Q^T*C if side = 'L' and trans = 'T'.\n"
    "C := Q^H*C if side = 'L' and trans = 'C'.\n"
    "C := C*Q   if side = 'R' and trans = 'N'.\n"
    "C := C*Q^T if side = 'R' and trans = 'T'.\n"
    "C := C*Q^H if side = 'R' and trans = 'C'.\n"
    "C is m by n and Q is a square orthogonal/unitary matrix computed\n"
    "by gelqf.  Q is defined as a product of k elementary reflectors,\n"
    "stored as the first k rows of A and the first k entries of tau."
    "\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "tau       'd' or 'z' matrix of length at least k.  Must have the\n"
    "          same type as A.\n\n"
    "C         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "side      'L' or 'R'\n\n"
    "trans     'N', 'T', or 'C'n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "k         integer.  k <= m if side = 'R' and k <= n if side = 'L'.\n"
    "          If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,k).  If zero, the\n"
    "          default value is used.\n\n"
    "ldC       nonnegative integer.  ldC >= max(1,m).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* unmlq(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *tau, *C;
    CBLAS_INT m=-1, n=-1, k=-1, ldA=0, ldC=0, oA=0, oC=0, info, lwork;
    void *work;
    number wl;
#if PY_MAJOR_VERSION >= 3
    int side_ = 'L', trans_ = 'N';
#endif
    char side = 'L', trans = 'N';
    char *kwlist[] = {"A", "tau", "C", "side", "trans", "m", "n", "k",
        "ldA", "ldC", "offsetA", "offsetC", NULL};
    Py_ssize_t _m = m, _n = n, _k = k, _ldA = ldA, _ldC = ldC, _oA = oA, _oC = oC;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|CCnnnnnnn",
        kwlist, &A, &tau, &C, &side_, &trans_, &_m, &_n, &_k, &_ldA, &_ldC,
        &_oA, &_oC))
        return NULL;
    side = (char) side_;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ccnnnnnnn",
        kwlist, &A, &tau, &C, &side, &trans, &_m, &_n, &_k, &_ldA, &_ldC,
        &_oA, &_oC))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_k, &k) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldC, &ldC) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oC, &oC) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(tau)) err_mtrx("tau");
    if (!Matrix_Check(C)) err_mtrx("C");
    if (MAT_ID(A) != MAT_ID(tau) || MAT_ID(A) != MAT_ID(C))
        err_conflicting_ids;
    if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");
    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (m < 0) m = C->nrows;
    if (n < 0) n = C->ncols;
    if (k < 0) k = MIN(A->nrows, A->ncols);
    if (m == 0 || n == 0 || k == 0) return Py_BuildValue("");
    if (k > ((side == 'L') ? m : n)) err_ld("k");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,k)) err_ld("ldA");
    if (ldC == 0) ldC = MAX(1,C->nrows);
    if (ldC < MAX(1,m)) err_ld("ldC");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + ldA * ((side == 'L') ? m : n) > len(A)) err_buf_len("A");
    if (oC < 0) err_nn_int("offsetC");
    if (oC + (n-1)*ldC + m > len(C)) err_buf_len("C");
    if (len(tau) < k) err_buf_len("tau");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (trans == 'C') trans = 'T';
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dormlq)(&side, &trans, &m, &n, &k, NULL, &ldA, NULL, NULL,
                &ldC, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dormlq)(&side, &trans, &m, &n, &k, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(tau), MAT_BUFD(C)+oC, &ldC, (double *) work,
                &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        case COMPLEX:
            if (trans == 'T') err_char("trans", "'N', 'C'");
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zunmlq)(&side, &trans, &m, &n, &k, NULL, &ldA, NULL, NULL,
                &ldC, &wl.z, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
            if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zunmlq)(&side, &trans, &m, &n, &k, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(tau), MAT_BUFZ(C)+oC, &ldC, 
                (complex_t *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_orglq[] =
    "Generate the orthogonal matrix in an LQ factorization.\n\n"
    "orglq(A, tau, m=min(A.size), n=A.size[1], k=len(tau), \n"
    "      ldA=max(1,A.size[0]), offsetA=0)\n\n"
    "PURPOSE\n"
    "On entry, A and tau contain an n by n orthogonal matrix Q.\n"
    "Q is defined as a product of k elementary reflectors, stored in the\n"
    "first k rows of A and in tau, as computed by gelqf().  On exit,\n"
    "the first m rows of Q are stored in the leading rows of A.\n\n"
    "ARGUMENTS\n"
    "A         'd' matrix\n\n"
    "tau       'd' matrix of length at least k\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  n >= m.  If negative, the default value is used."
    "\n\n"
    "k         integer.  k <= m.  If negative, the default value is \n"
    "          used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* orglq(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *tau;
    CBLAS_INT m=-1, n=-1, k=-1, ldA=0, oA=0, info, lwork;
    void *work;
    number wl;
    char *kwlist[] = {"A", "tau", "m", "n", "k", "ldA", "offsetA", NULL};
    Py_ssize_t _m = m, _n = n, _k = k, _ldA = ldA, _oA = oA;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|nnnnn", kwlist, &A,
        &tau, &_m, &_n, &_k, &_ldA, &_oA)) return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_k, &k) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(tau)) err_mtrx("tau");
    if (MAT_ID(A) != MAT_ID(tau)) err_conflicting_ids;
    if (m < 0) m = MIN(A->nrows, A->ncols);
    if (n < 0) n = A->ncols;
    if (m > n) err_ld("n");
    if (k < 0) k = len(tau);
    if (k > m) err_ld("k");
    if (m == 0 || n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1, A->nrows);
    if (ldA <  MAX(1, m)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + n*ldA  > len(A)) err_buf_len("A");
    if (len(tau) < k) err_buf_len("tau");

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dorglq)(&m, &n, &k, NULL, &ldA, NULL, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dorglq)(&m, &n, &k, MAT_BUFD(A) + oA, &ldA, MAT_BUFD(tau),
                (double *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_unglq[] =
    "Generate the orthogonal or unitary matrix in an LQ factorization.\n\n"
    "unglq(A, tau, m=min(A.size), n=A.size[1], k=len(tau), \n"
    "      ldA=max(1,A.size[0]), offsetA=0)\n\n"
    "PURPOSE\n"
    "On entry, A and tau contain an n by n orthogonal/unitary matrix Q.\n"
    "Q is defined as a product of k elementary reflectors, stored in the\n"
    "first k rows of A and in tau, as computed by gelqf().  On exit,\n"
    "the first m rows of Q are stored in the leading rows of A.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "tau       'd' or 'z' matrix of length at least k.  Must have the\n"
    "          same type as A.\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  n >= m.  If negative, the default value is used."
    "\n\n"
    "k         integer.  k <= m.  If negative, the default value is \n"
    "          used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* unglq(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *tau;
    CBLAS_INT m=-1, n=-1, k=-1, ldA=0, oA=0, info, lwork;
    void *work;
    number wl;
    char *kwlist[] = {"A", "tau", "m", "n", "k", "ldA", "offsetA", NULL};
    Py_ssize_t _m = m, _n = n, _k = k, _ldA = ldA, _oA = oA;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|nnnnn",
        kwlist, &A, &tau, &_m, &_n, &_k, &_ldA, &_oA)) return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_k, &k) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(tau)) err_mtrx("tau");
    if (MAT_ID(A) != MAT_ID(tau)) err_conflicting_ids;
    if (m < 0) m = MIN(A->nrows, A->ncols);
    if (n < 0) n = A->ncols;
    if (m > n) err_ld("n");
    if (k < 0) k = len(tau);
    if (k > m) err_ld("k");
    if (m == 0 || n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1, A->nrows);
    if (ldA <  MAX(1, m)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + n*ldA  > len(A)) err_buf_len("A");
    if (len(tau) < k) err_buf_len("tau");

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dorglq)(&m, &n, &k, NULL, &ldA, NULL, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dorglq)(&m, &n, &k, MAT_BUFD(A) + oA, &ldA, MAT_BUFD(tau),
                (double *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        case COMPLEX:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zunglq)(&m, &n, &k, NULL, &ldA, NULL, &wl.z, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
            if (!(work = (void *) calloc(lwork, sizeof(complex_t))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zunglq)(&m, &n, &k, MAT_BUFZ(A) + oA, &ldA, MAT_BUFZ(tau),
                (complex_t *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_geqp3[] =
    "QR factorization with column pivoting.\n\n"
    "geqp3(A, jpvt, tau, m=A.size[0], n=A.size[1], ldA=max(1,A.size[0]),\n"
    "      offsetA=0)\n\n"
    "PURPOSE\n"
    "QR factorization with column pivoting of an m by n real or complex\n"
    "matrix A:\n\n"
    "A*P = Q*R = [Q1 Q2] * [R1; 0] if m >= n\n"
    "A*P = Q*R = Q * [R1 R2] if m <= n,\n\n"
    "where P is a permutation matrix, Q is m by m and orthogonal/unitary\n"
    "and R is m by n with R1 upper triangular.  On exit, R is stored in\n"
    "the upper triangular part of A.  Q is stored as a product of\n"
    "k=min(m,n) elementary reflectors.  The parameters of the\n"
    "reflectors are stored in the first k entries of tau and in the\n"
    "lower triangular part of the first k columns of A.  On entry, if\n"
    "jpvt[j] is nonzero, the jth column of A is permuted to the front of\n"
    "A*P.  If jpvt[j] is zero, the jth column is a free column.  On exit\n"
    "A*P = A[:, jpvt - 1].\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "jpvt      'i' matrix of length n\n\n"
    "tau       'd' or 'z' matrix of length min(m,n).  Must have the same\n"
    "          type as A.\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,m).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer";

static PyObject* geqp3(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *tau, *jpvt;
    CBLAS_INT m=-1, n=-1, ldA=0, oA=0, info, lwork;
    double *rwork = NULL;
    void *work = NULL;
    number wl;
    char *kwlist[] = {"A", "jpvt", "tau", "m", "n", "ldA", "offsetA",
        NULL};
    Py_ssize_t _m = m, _n = n, _ldA = ldA, _oA = oA;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|nnnn", kwlist,
        &A, &jpvt, &tau, &_m, &_n, &_ldA, &_oA))
        return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(jpvt) || jpvt ->id != INT) err_int_mtrx("jpvt");
    if (!Matrix_Check(tau)) err_mtrx("tau");
    if (MAT_ID(A) != MAT_ID(tau)) err_conflicting_ids;
    if (m < 0) m = A->nrows;
    if (n < 0) n = A->ncols;
    if (m == 0 || n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1, A->nrows);
    if (ldA < MAX(1,m)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
    if (len(jpvt) < n) err_buf_len("jpvt");
    if (len(tau) < MIN(m,n)) err_buf_len("tau");

    CBLAS_INT *jpvt_ptr = cvxopt_cblas_int_acquire(jpvt, n, 1);
    if (!jpvt_ptr) return NULL;

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgeqp3)(&m, &n, NULL, &ldA, NULL, NULL, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double)))) {
                cvxopt_cblas_int_release(jpvt, jpvt_ptr, n, 0);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgeqp3)(&m, &n, MAT_BUFD(A)+oA, &ldA, jpvt_ptr, MAT_BUFD(tau),
                (double *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
	    break;

        case COMPLEX:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgeqp3)(&m, &n, NULL, &ldA, NULL, NULL, &wl.z, &lwork, NULL,
                &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
            if (!(work = (void *) calloc(lwork, sizeof(complex_t))) ||
                !(rwork = (double *) calloc(2*n, sizeof(double)))) {
                free(work);
                free(rwork);
                cvxopt_cblas_int_release(jpvt, jpvt_ptr, n, 0);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgeqp3)(&m, &n, MAT_BUFZ(A)+oA, &ldA, jpvt_ptr, MAT_BUFZ(tau),
                (complex_t *) work, &lwork, rwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
            free(rwork);
	    break;

        default:
            cvxopt_cblas_int_release(jpvt, jpvt_ptr, n, 0);
            err_invalid_id;
    }

    cvxopt_cblas_int_release(jpvt, jpvt_ptr, n, 1);

    if (info) err_lapack
    else return Py_BuildValue("");
}



static char doc_syev[] =
    "Eigenvalue decomposition of a real symmetric matrix.\n\n"
    "syev(A, W, jobz='N', uplo='L', n=A.size[0], "
    "ldA = max(1,A.size[0]),\n"
    "     offsetA=0, offsetW=0)\n\n"
    "PURPOSE\n"
    "Returns eigenvalues/vectors of a real symmetric nxn matrix A.\n"
    "On exit, W contains the eigenvalues in ascending order.  If jobz\n"
    "is 'V', the (normalized) eigenvectors are also computed and\n"
    "returned in A.  If jobz is 'N', only the eigenvalues are\n"
    "computed, and the content of A is destroyed.\n\n"
    "ARGUMENTS\n"
    "A         'd' matrix\n\n"
    "W         'd' matrix of length at least n\n\n"
    "jobz      'N' or 'V'\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* syev(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *W;
    CBLAS_INT n=-1, ldA=0, oA=0, oW=0, info, lwork;
    double *work;
    number wl;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', jobz_ = 'N';
#endif
    char uplo = 'L', jobz = 'N';
    char *kwlist[] = {"A", "W", "jobz", "uplo", "n", "ldA", "offsetA",
        "offsetW", NULL};
    Py_ssize_t _n = n, _ldA = ldA, _oA = oA, _oW = oW;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCnnnn", kwlist,
        &A, &W, &jobz_, &uplo_, &_n, &_ldA, &_oA, &_oW))
        return NULL;
    jobz = (char) jobz_;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ccnnnn", kwlist,
        &A, &W, &jobz, &uplo, &_n, &_ldA, &_oA, &_oW))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oW, &oW) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
    if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("n", (Py_ssize_t) (0));
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oW < 0) err_nn_int("offsetW");
    if (oW + n > len(W)) err_buf_len("W");

    switch (MAT_ID(A)){
	case DOUBLE:
	    lwork=-1;
            Py_BEGIN_ALLOW_THREADS
	    BLAS_FUNC(dsyev)(&jobz, &uplo, &n, NULL, &ldA, NULL, &wl.d, &lwork,
                &info);
            Py_END_ALLOW_THREADS
	    lwork = (CBLAS_INT) wl.d;
	    if (!(work = calloc(lwork, sizeof(double))))
		return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
	    BLAS_FUNC(dsyev)(&jobz, &uplo, &n, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(W)+oW, work, &lwork, &info);
            Py_END_ALLOW_THREADS
	    free(work);
	    break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_heev[] =
    "Eigenvalue decomposition of a real symmetric or complex Hermitian"
    "\nmatrix.\n\n"
    "heev(A, W, jobz='N', uplo='L', n=A.size[0], "
    "ldA = max(1,A.size[0]),\n"
    "     offsetA=0, offsetW=0)\n\n"
    "PURPOSE\n"
    "Returns eigenvalues/vectors of a real symmetric or complex\n"
    "Hermitian nxn matrix A.  On exit, W contains the eigenvalues in\n"
    "ascending order.  If jobz is 'V', the (normalized) eigenvectors\n"
    "are also computed and returned in A.  If jobz is 'N', only the\n"
    "eigenvalues are computed, and the content of A is destroyed.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "W         'd' matrix of length at least n\n\n"
    "jobz      'N' or 'V'\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* heev(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *W;
    CBLAS_INT n=-1, ldA=0, oA=0, oW=0, info, lwork;
    double *rwork=NULL;
    void *work=NULL;
    number wl;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', jobz_ = 'N';
#endif
    char uplo = 'L', jobz = 'N';
    char *kwlist[] = {"A", "W", "jobz", "uplo", "n", "ldA", "offsetA",
        "offsetW", NULL};
    Py_ssize_t _n = n, _ldA = ldA, _oA = oA, _oW = oW;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCnnnn", kwlist,
        &A, &W, &jobz_, &uplo_, &_n, &_ldA, &_oA, &_oW))
        return NULL;
    jobz = (char) jobz_;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ccnnnn", kwlist,
        &A, &W, &jobz, &uplo, &_n, &_ldA, &_oA, &_oW))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oW, &oW) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
    if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oW < 0) err_nn_int("offsetW");
    if (oW + n > len(W)) err_buf_len("W");
    switch (MAT_ID(A)){
	case DOUBLE:
	    lwork=-1;
            Py_BEGIN_ALLOW_THREADS
	    BLAS_FUNC(dsyev)(&jobz, &uplo, &n, NULL, &ldA, NULL, &wl.d, &lwork,
                &info);
            Py_END_ALLOW_THREADS
	    lwork = (CBLAS_INT) wl.d;
	    if (!(work = (void *) calloc(lwork, sizeof(double))))
		return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
	    BLAS_FUNC(dsyev)(&jobz, &uplo, &n, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(W)+oW, (double *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
	    free(work);
	    break;

        case COMPLEX:
	    lwork=-1;
            Py_BEGIN_ALLOW_THREADS
	    BLAS_FUNC(zheev)(&jobz, &uplo, &n, NULL, &ldA, NULL, &wl.z, &lwork,
                NULL, &info);
            Py_END_ALLOW_THREADS
	    lwork = (CBLAS_INT) creal(wl.z);
	    work = (void *) calloc(lwork, sizeof(complex_t));
	    rwork = (double *) calloc(3*n-2, sizeof(double));
	    if (!work || !rwork){
		free(work);  free(rwork);
		return PyErr_NoMemory();
	    }
            Py_BEGIN_ALLOW_THREADS
	    BLAS_FUNC(zheev)(&jobz, &uplo, &n, MAT_BUFZ(A)+oA, &ldA,
		MAT_BUFD(W)+oW,  (complex_t *) work, &lwork, rwork,
		&info);
            Py_END_ALLOW_THREADS
	    free(work);  free(rwork);
	    break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_syevx[] =
    "Computes selected eigenvalues and eigenvectors of a real symmetric"
    "\nmatrix (expert driver).\n\n"
    "m = syevx(A, W, jobz='N', range='A', uplo='L', vl=0.0, vu=0.0, \n"
    "          il=1, iu=1, Z=None, n=A.size[0], ldA=max(1,A.size[0]),\n"
    "          ldZ=None, abstol=0.0, offsetA=0, offsetW=0,\n"
    "          offsetZ=0)\n\n"
    "PURPOSE\n"
    "Computes selected eigenvalues/vectors of a real symmetric n by n\n"
    "matrix A.\n"
    "If range is 'A', all eigenvalues are computed.\n"
    "If range is 'V', all eigenvalues in the interval (vl,vu] are\n"
    "computed.\n"
    "If range is 'I', all eigenvalues il through iu are computed\n"
    "(sorted in ascending order with 1 <= il <= iu <= n).\n"
    "If jobz is 'N', only the eigenvalues are returned in W.\n"
    "If jobz is 'V', the eigenvectors are also returned in Z.\n"
    "On exit, the content of A is destroyed.\n\n"
    "ARGUMENTS\n"
    "A         'd' matrix\n\n"
    "W         'd' matrix of length at least n.  On exit, contains\n"
    "          the computed eigenvalues in ascending order.\n\n"
    "jobz      'N' or 'V'\n\n"
    "range     'A', 'V' or 'I'\n\n"
    "uplo      'L' or 'U'\n\n"
    "vl,vu     doubles.  Only required when range is 'V'.\n\n"
    "il,iu     integers.  Only required when range is 'I'.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "Z         'd' matrix.  Only required when jobz is 'V'.  If range\n"
    "          is 'A' or 'V', Z must have at least n columns.  If\n"
    "          range is 'I', Z must have at least iu-il+1 columns.\n"
    "          On exit the first m columns of Z contain the computed\n"
    "          (normalized) eigenvectors.\n\n"
    "abstol    double.  Absolute error tolerance for eigenvalues.\n"
    "          If nonpositive, the LAPACK default value is used.\n\n"
    "ldZ       nonnegative integer.  ldZ >= 1 if jobz is 'N' and\n"
    "          ldZ >= max(1,n) if jobz is 'V'.  The default value\n"
    "          is 1 if jobz is 'N' and max(1,Z.size[0]) if jobz ='V'.\n"
    "          If zero, the default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetW   nonnegative integer\n\n"
    "offsetZ   nonnegative integer\n\n"
    "m         the number of eigenvalues computed";

static PyObject* syevx(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *W, *Z=NULL;
    CBLAS_INT n=-1, ldA=0, ldZ=0, il=1, iu=1, oA=0, oW=0, oZ=0, info, lwork,
        *iwork, m, *ifail=NULL;
    double *work, vl=0.0, vu=0.0, abstol=0.0;
    double wl;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', jobz_ = 'N', range_ = 'A';
#endif
    char uplo = 'L', jobz = 'N', range = 'A';
    char *kwlist[] = {"A", "W", "jobz", "range", "uplo", "vl", "vu",
	"il", "iu", "Z", "n", "ldA", "ldZ", "abstol", "offsetA",
        "offsetW", "offsetZ", NULL};
    Py_ssize_t _il = il, _iu = iu, _n = n, _ldA = ldA, _ldZ = ldZ, _oA = oA, _oW = oW, _oZ = oZ;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCddnnOnnndnnn",
        kwlist, &A, &W, &jobz_, &range_, &uplo_, &vl, &vu, &_il, &_iu, &Z,
	&_n, &_ldA, &_ldZ, &abstol, &_oA, &_oW, &_oZ))
        return NULL;
    jobz = (char) jobz_;
    range = (char) range_;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cccddnnOnnndnnn",
        kwlist, &A, &W, &jobz, &range, &uplo, &vl, &vu, &_il, &_iu, &Z,
	&_n, &_ldA, &_ldZ, &abstol, &_oA, &_oW, &_oZ))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_il, &il) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_iu, &iu) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldZ, &ldZ) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oW, &oW) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oZ, &oZ) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
    if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
    if (range != 'A' && range != 'V' && range != 'I')
	err_char("range", "'A', 'V', 'I'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("n", (Py_ssize_t) (0));
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (range == 'V' && vl >= vu){
        PyErr_SetString(PyExc_ValueError, "vl must be less than vu");
        return NULL;
    }
    if (range == 'I' && (il < 1 || il > iu || iu > n)){
        PyErr_SetString(PyExc_ValueError, "il and iu must satisfy "
            "1 <= il <= iu <= n");
        return NULL;
    }
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oW < 0) err_nn_int("offsetW");
    if (oW + n > len(W)) err_buf_len("W");
    if (jobz == 'V'){
        if (!Z || !Matrix_Check(Z) || MAT_ID(Z) != DOUBLE)
            err_dbl_mtrx("Z");
        if (ldZ == 0) ldZ = MAX(1,Z->nrows);
        if (ldZ < MAX(1,n)) err_ld("ldZ");
        if (oZ < 0) err_nn_int("offsetZ");
        if (oZ + ((range == 'I') ? iu-il : n-1)*ldZ + n > len(Z))
	    err_buf_len("Z");
    } else {
        if (ldZ == 0) ldZ = 1;
        if (ldZ < 1) err_ld("ldZ");
    }

    switch (MAT_ID(A)){
        case DOUBLE:
	    lwork = -1;
            Py_BEGIN_ALLOW_THREADS
	    BLAS_FUNC(dsyevx)(&jobz, &range, &uplo, &n, NULL, &ldA, &vl, &vu, &il,
                &iu, &abstol, &m, NULL, NULL, &ldZ, &wl, &lwork, NULL,
	       	NULL, &info);
            Py_END_ALLOW_THREADS
	    lwork = (CBLAS_INT) wl;
	    work = (double *) calloc(lwork, sizeof(double));
	    iwork = (CBLAS_INT *) calloc(5*n, sizeof(CBLAS_INT));
	    if (jobz == 'V') ifail = (CBLAS_INT *) calloc(n, sizeof(CBLAS_INT));
	    if (!work || !iwork || (jobz == 'V' && !ifail)){
		free(work); free(iwork); free(ifail);
	        return PyErr_NoMemory();
	    }
            Py_BEGIN_ALLOW_THREADS
	    BLAS_FUNC(dsyevx)(&jobz, &range, &uplo, &n, MAT_BUFD(A)+oA, &ldA, &vl,
                &vu, &il, &iu, &abstol, &m, MAT_BUFD(W)+oW,
		(jobz == 'V') ? MAT_BUFD(Z)+oZ : NULL,  &ldZ, work,
		&lwork, iwork, ifail, &info);
            Py_END_ALLOW_THREADS
	    free(work);   free(iwork);   free(ifail);
            break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("n", (Py_ssize_t) (m));
}


static char doc_heevx[] =
    "Computes selected eigenvalues and eigenvectors of a real symmetric"
    "\nor complex Hermitian matrix (expert driver).\n\n"
    "m = syevx(A, W, jobz='N', range='A', uplo='L', vl=0.0, vu=0.0, \n"
    "          il=1, iu=1, Z=None, n=A.size[0], \n"
    "          ldA = max(1,A.size[0]), ldZ=None, abstol=0.0, \n"
    "          offsetA=0, offsetW=0, offsetZ=0)\n\n"
    "PURPOSE\n"
    "Computes selected eigenvalues/vectors of a real symmetric or\n"
    "complex Hermitian n by n matrix A.\n"
    "If range is 'A', all eigenvalues are computed.\n"
    "If range is 'V', all eigenvalues in the interval (vl,vu] are\n"
    "computed.\n"
    "If range is 'I', all eigenvalues il through iu are computed\n"
    "(sorted in ascending order with 1 <= il <= iu <= n).\n"
    "If jobz is 'N', only the eigenvalues are returned in W.\n"
    "If jobz is 'V', the eigenvectors are also returned in Z.\n"
    "On exit, the content of A is destroyed.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "W         'd' matrix of length at least n.  On exit, contains\n"
    "          the computed eigenvalues in ascending order.\n\n"
    "jobz      'N' or 'V'\n\n"
    "range     'A', 'V' or 'I'\n\n"
    "uplo      'L' or 'U'\n\n"
    "vl,vu     doubles.  Only required when range is 'V'.\n\n"
    "il,iu     integers.  Only required when range is 'I'.\n\n"
    "Z         'd' or 'z' matrix.  Must have the same type as A.\n"
    "          Z is only required when jobz is 'V'.  If range is 'A'\n"
    "          or 'V', Z must have at least n columns.  If range is\n"
    "          'I', Z must have at least iu-il+1 columns.  On exit\n"
    "          the first m columns of Z contain the computed\n"
    "          (normalized) eigenvectors.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "ldZ       nonnegative integer.  ldZ >= 1 if jobz is 'N' and\n"
    "          ldZ >= max(1,n) if jobz is 'V'.  The default value\n"
    "          is 1 if jobz is 'N' and max(1,Z.size[0]) if jobz ='V'.\n"
    "          If zero, the default value is used.\n\n"
    "abstol    double.  Absolute error tolerance for eigenvalues.\n"
    "          If nonpositive, the LAPACK default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetW   nonnegative integer\n\n"
    "offsetZ   nonnegative integer\n\n"
    "m         the number of eigenvalues computed";

static PyObject* heevx(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *W, *Z=NULL;
    CBLAS_INT n=-1, ldA=0, ldZ=0, il=1, iu=1, oA=0, oW=0, oZ=0, info, lwork,
        *iwork, m, *ifail=NULL;
    double vl=0.0, vu=0.0, abstol=0.0, *rwork;
    number wl;
    void *work;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', jobz_ = 'N', range_ = 'A';
#endif
    char uplo = 'L', jobz = 'N', range = 'A';
    char *kwlist[] = {"A", "W", "jobz", "range", "uplo", "vl", "vu",
	"il", "iu", "Z", "n", "ldA", "ldZ", "abstol", "offsetA",
        "offsetW", "offsetZ", NULL};
    Py_ssize_t _il = il, _iu = iu, _n = n, _ldA = ldA, _ldZ = ldZ, _oA = oA, _oW = oW, _oZ = oZ;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCddnnOnnndnnn",
        kwlist, &A, &W, &jobz_, &range_, &uplo_, &vl, &vu, &_il, &_iu, &Z,
	&_n, &_ldA, &_ldZ, &abstol, &_oA, &_oW, &_oZ))
        return NULL;
    jobz = (char) jobz_;
    range = (char) range_;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cccddnnOnnndnnn",
        kwlist, &A, &W, &jobz, &range, &uplo, &vl, &vu, &_il, &_iu, &Z,
	&_n, &_ldA, &_ldZ, &abstol, &_oA, &_oW, &_oZ))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_il, &il) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_iu, &iu) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldZ, &ldZ) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oW, &oW) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oZ, &oZ) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
    if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
    if (range != 'A' && range != 'V' && range != 'I')
	err_char("range", "'A', 'V', 'I'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("n", (Py_ssize_t) (0));
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (range == 'V' && vl >= vu){
        PyErr_SetString(PyExc_ValueError, "vl must be less than vu");
        return NULL;
    }
    if (range == 'I' && (il < 1 || il > iu || iu > n)){
        PyErr_SetString(PyExc_ValueError, "il and iu must satisfy "
            "1 <= il <= iu <= n");
        return NULL;
    }
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oW < 0) err_nn_int("offsetW");
    if (oW + n > len(W)) err_buf_len("W");
    if (jobz == 'V'){
        if (!Z || !Matrix_Check(Z)) err_mtrx("Z");
	if (MAT_ID(Z) != MAT_ID(A)) err_conflicting_ids;
        if (ldZ == 0) ldZ = MAX(1,Z->nrows);
        if (ldZ < MAX(1,n)) err_ld("ldZ");
        if (oZ < 0) err_nn_int("offsetZ");
        if (oZ + ((range == 'I') ? iu-il : n-1)*ldZ + n > len(Z))
	    err_buf_len("Z");
    } else {
        if (ldZ == 0) ldZ = 1;
        if (ldZ < 1) err_ld("ldZ");
    }

    switch (MAT_ID(A)){
        case DOUBLE:
	    lwork = -1;
            Py_BEGIN_ALLOW_THREADS
	    BLAS_FUNC(dsyevx)(&jobz, &range, &uplo, &n, NULL, &ldA, &vl, &vu, &il,
                &iu, &abstol, &m, NULL, NULL, &ldZ, &wl.d, &lwork, NULL,
	       	NULL, &info);
            Py_END_ALLOW_THREADS
	    lwork = (CBLAS_INT) wl.d;
	    work = (void *) calloc(lwork, sizeof(double));
	    iwork = (CBLAS_INT *) calloc(5*n, sizeof(CBLAS_INT));
	    if (jobz == 'V') ifail = (CBLAS_INT *) calloc(n, sizeof(CBLAS_INT));
	    if (!work || !iwork || (jobz == 'V' && !ifail)){
		free(work); free(iwork); free(ifail);
	        return PyErr_NoMemory();
	    }
            Py_BEGIN_ALLOW_THREADS
	    BLAS_FUNC(dsyevx)(&jobz, &range, &uplo, &n, MAT_BUFD(A)+oA, &ldA, &vl,
                &vu, &il, &iu, &abstol, &m, MAT_BUFD(W)+oW,
		(jobz == 'V') ? MAT_BUFD(Z)+oZ : NULL,  &ldZ,
		(double *) work, &lwork, iwork, ifail, &info);
            Py_END_ALLOW_THREADS
	    free(work);  free(iwork);  free(ifail);
            break;

	case COMPLEX:
	    lwork = -1;
            Py_BEGIN_ALLOW_THREADS
	    BLAS_FUNC(zheevx)(&jobz, &range, &uplo, &n, NULL, &ldA, &vl, &vu, &il,
                &iu, &abstol, &m, NULL, NULL, &ldZ, &wl.z, &lwork, NULL,
	       	NULL, NULL, &info);
            Py_END_ALLOW_THREADS
	    lwork = (CBLAS_INT) creal(wl.z);
	    work = (void *) calloc(lwork, sizeof(complex_t));
	    rwork = (double *) calloc(7*n, sizeof(double));
	    iwork = (CBLAS_INT *) calloc(5*n, sizeof(CBLAS_INT));
	    if (jobz == 'V') ifail = (CBLAS_INT *) calloc(n, sizeof(CBLAS_INT));
	    if (!work || !rwork || !iwork || (jobz == 'V' && !ifail)){
		free(work); free(rwork); free(iwork); free(ifail);
	        return PyErr_NoMemory();
	    }
            Py_BEGIN_ALLOW_THREADS
	    BLAS_FUNC(zheevx)(&jobz, &range, &uplo, &n, MAT_BUFZ(A)+oA, &ldA, &vl,
                &vu, &il, &iu, &abstol, &m, MAT_BUFD(W)+oW,
		(jobz == 'V') ? MAT_BUFZ(Z)+oZ : NULL,  &ldZ,
		(complex_t *) work, &lwork, rwork, iwork, ifail, 
                &info);
            Py_END_ALLOW_THREADS
	    free(work);  free(rwork);  free(iwork);  free(ifail);
            break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("n", (Py_ssize_t) (m));
}


static char doc_syevd[] =
    "Eigenvalue decomposition of a real symmetric matrix\n"
    "(divide-and-conquer driver).\n\n"
    "syevd(A, W, jobz='N', uplo='L', n=A.size[0], "
    "ldA = max(1,A.size[0]),\n"
    "      offsetA=0, offsetW=0)\n\n"
    "PURPOSE\n"
    "Returns  eigenvalues/vectors of a real symmetric nxn matrix A.\n"
    "On exit, W contains the eigenvalues in ascending order.\n"
    "If jobz is 'V', the (normalized) eigenvectors are also computed\n"
    "and returned in A.  If jobz is 'N', only the eigenvalues are\n"
    "computed, and the content of A is destroyed.\n"
    "\n\nARGUMENTS\n"
    "A         'd' matrix\n\n"
    "W         'd' matrix of length at least n.  On exit, contains\n"
    "          the computed eigenvalues in ascending order.\n\n"
    "jobz      'N' or 'V'\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* syevd(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *W;
    CBLAS_INT n=-1, ldA=0, oA=0, oW=0, info, lwork, liwork, *iwork, iwl;
    double *work=NULL, wl;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', jobz_ = 'N';
#endif
    char uplo = 'L', jobz = 'N';
    char *kwlist[] = {"A", "W", "jobz", "uplo", "n", "ldA", "offsetA",
        "offsetW", NULL};
    Py_ssize_t _n = n, _ldA = ldA, _oA = oA, _oW = oW;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCnnnn", kwlist,
        &A, &W, &jobz_, &uplo_, &_n, &_ldA, &_oA, &_oW))
        return NULL;
    uplo = (char) uplo_;
    jobz = (char) jobz_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ccnnnn", kwlist,
        &A, &W, &jobz, &uplo, &_n, &_ldA, &_oA, &_oW))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oW, &oW) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(W) || W->id != DOUBLE) err_dbl_mtrx("W");
    if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oW < 0) err_nn_int("offsetW");
    if (oW + n > len(W)) err_buf_len("W");

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            liwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsyevd)(&jobz, &uplo, &n, NULL, &ldA, NULL, &wl, &lwork,
                &iwl, &liwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl;
            liwork = iwl;
            work = (double *) calloc(lwork, sizeof(double));
            iwork = (CBLAS_INT *) calloc(liwork, sizeof(CBLAS_INT));
            if (!work || !iwork){
                free(work);  free(iwork);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
	    BLAS_FUNC(dsyevd)(&jobz, &uplo, &n, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(W)+oW, work, &lwork, iwork, &liwork, &info);
            Py_END_ALLOW_THREADS
            free(work);  free(iwork);
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_heevd[] =
    "Eigenvalue decomposition of a real symmetric or complex Hermitian"
    "\nmatrix (divide-and-conquer driver).\n\n"
    "heevd(A, W, jobz='N', uplo='L', n=A.size[0], "
    "ldA = max(1,A.size[0]),\n"
    "      offsetA=0, offsetW=0)\n\n"
    "PURPOSE\n"
    "Returns  eigenvalues/vectors of a real symmetric or complex\n"
    "Hermitian n by n matrix A.  On exit, W contains the eigenvalues\n"
    "in ascending order.  If jobz is 'V', the (normalized) eigenvectors"
    "\nare also computed and returned in A.  If jobz is 'N', only the\n"
    "eigenvalues are computed, and the content of A is destroyed.\n"
    "\n\nARGUMENTS\n"
    "A         'd' matrix\n\n"
    "W         'd' matrix of length at least n.  On exit, contains\n"
    "          the computed eigenvalues in ascending order.\n\n"
    "jobz      'N' or 'V'\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* heevd(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *W;
    CBLAS_INT n=-1, ldA=0, oA=0, oW=0, info, lwork, liwork, *iwork, iwl,
	lrwork;
    double *rwork, rwl;
    number wl;
    void *work;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', jobz_ = 'N';
#endif
    char uplo = 'L', jobz = 'N';
    char *kwlist[] = {"A", "W", "jobz", "uplo", "n", "ldA", "offsetA",
        "offsetW", NULL};
    Py_ssize_t _n = n, _ldA = ldA, _oA = oA, _oW = oW;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCnnnn", kwlist,
        &A, &W, &jobz_, &uplo_, &_n, &_ldA, &_oA, &_oW))
        return NULL;
    jobz = (char) jobz_;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ccnnnn", kwlist,
        &A, &W, &jobz, &uplo, &_n, &_ldA, &_oA, &_oW))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oW, &oW) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(W) || W->id != DOUBLE) err_dbl_mtrx("W");
    if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oW < 0) err_nn_int("offsetW");
    if (oW + n > len(W)) err_buf_len("W");

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            liwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsyevd)(&jobz, &uplo, &n, NULL, &ldA, NULL, &wl.d, &lwork,
                &iwl, &liwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            liwork = iwl;
            work = (void *) calloc(lwork, sizeof(double));
            iwork = (CBLAS_INT *) calloc(liwork, sizeof(CBLAS_INT));
            if (!work || !iwork){
                free(work);  free(iwork);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsyevd)(&jobz, &uplo, &n, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(W)+oW, (double *) work, &lwork, iwork, &liwork,
                &info);
            Py_END_ALLOW_THREADS
            free(work);   free(iwork);
            break;

        case COMPLEX:
            lwork = -1;
            liwork = -1;
            lrwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zheevd)(&jobz, &uplo, &n, NULL, &ldA, NULL, &wl.z, &lwork,
                &rwl, &lrwork, &iwl, &liwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            lrwork = (CBLAS_INT) rwl;
            liwork = iwl;
            work = (void *) calloc(lwork, sizeof(complex_t));
            rwork = (double *) calloc(lrwork, sizeof(double));
            iwork = (CBLAS_INT *) calloc(liwork, sizeof(CBLAS_INT));
            if (!work || !rwork || !iwork){
                free(work);  free(rwork);  free(iwork);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zheevd)(&jobz, &uplo, &n, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFD(W)+oW, (complex_t *) work, &lwork, rwork,
                &lrwork, iwork, &liwork, &info);
            Py_END_ALLOW_THREADS
            free(work);  free(rwork);  free(iwork);
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_syevr[] =
    "Computes selected eigenvalues and eigenvectors of a real symmetric"
    "\n"
    "matrix (RRR driver).\n\n"
    "m = syevr(A, W, jobz='N', range='A', uplo='L', vl=0.0, vu=0.0, \n"
    "          il=1, iu=1, Z=None, n=A.size[0], ldA=max(1,A.size[0]),\n"
    "          ldZ=None, abstol=0.0, offsetA=0, offsetW=0, offsetZ=0)\n"
    "\n"
    "PURPOSE\n"
    "Computes selected eigenvalues/vectors of a real symmetric n by n\n"
    "matrix A.\n"
    "If range is 'A', all eigenvalues are computed.\n"
    "If range is 'V', all eigenvalues in the interval (vl,vu] are\n"
    "computed.\n"
    "If range is 'I', all eigenvalues il through iu are computed\n"
    "(sorted in ascending order with 1 <= il <= iu <= n).\n"
    "If jobz is 'N', only the eigenvalues are returned in W.\n"
    "If jobz is 'V', the eigenvectors are also returned in Z.\n"
    "On exit, the content of A is destroyed.\n"
    "syevr is usually the fastest of the four eigenvalue routines.\n\n"
    "ARGUMENTS\n"
    "A         'd' matrix\n\n"
    "W         'd' matrix of length at least n.  On exit, contains\n"
    "          the computed eigenvalues in ascending order.\n\n"
    "jobz      'N' or 'V'\n\n"
    "range     'A', 'V' or 'I'\n\n"
    "uplo      'L' or 'U'\n\n"
    "vl,vu     doubles.  Only required when range is 'V'.\n\n"
    "il,iu     integers.  Only required when range is 'I'.\n\n"
    "Z         'd' matrix.  Only required when jobz = 'V'.\n"
    "          If range is 'A' or 'V', Z must have at least n columns."
    "\n"
    "          If range is 'I', Z must have at least iu-il+1 columns.\n"
    "          On exit the first m columns of Z contain the computed\n"
    "          (normalized) eigenvectors.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).\n"
    "          If zero, the default value is used.\n\n"
    "ldZ       nonnegative integer.  ldZ >= 1 if jobz is 'N' and\n"
    "          ldZ >= max(1,n) if jobz is 'V'.  The default value\n"
    "          is 1 if jobz is 'N' and max(1,Z.size[0]) if jobz ='V'.\n"
    "          If zero, the default value is used.\n\n"
    "abstol    double.  Absolute error tolerance for eigenvalues.\n"
    "          If nonpositive, the LAPACK default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetW   nonnegative integer\n\n"
    "offsetZ   nonnegative integer\n\n"
    "m         the number of eigenvalues computed";

static PyObject* syevr(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *W, *Z=NULL;
    CBLAS_INT n=-1, ldA=0, ldZ=0, il=1, iu=1, oA=0, oW=0, oZ=0, info, lwork,
        *iwork=NULL, liwork, m, *isuppz=NULL, iwl;
    double *work=NULL, vl=0.0, vu=0.0, abstol=0.0, wl;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', jobz_ = 'N', range_ = 'A';
#endif
    char uplo = 'L', jobz = 'N', range = 'A';
    char *kwlist[] = {"A", "W", "jobz", "range", "uplo", "vl", "vu",
	"il", "iu", "Z", "n", "ldA", "ldZ", "abstol", "offsetA",
        "offsetW", "offsetZ", NULL};
    Py_ssize_t _il = il, _iu = iu, _n = n, _ldA = ldA, _ldZ = ldZ, _oA = oA, _oW = oW, _oZ = oZ;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCddnnOnnndnnn",
        kwlist, &A, &W, &jobz_, &range_, &uplo_, &vl, &vu, &_il, &_iu, &Z,
	&_n, &_ldA, &_ldZ, &abstol, &_oA, &_oW, &_oZ))
        return NULL;
    jobz = (char) jobz_;
    range = (char) range_;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cccddnnOnnndnnn",
        kwlist, &A, &W, &jobz, &range, &uplo, &vl, &vu, &_il, &_iu, &Z,
	&_n, &_ldA, &_ldZ, &abstol, &_oA, &_oW, &_oZ))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_il, &il) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_iu, &iu) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldZ, &ldZ) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oW, &oW) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oZ, &oZ) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
    if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
    if (range != 'A' && range != 'V' && range != 'I')
	err_char("range", "'A', 'V', 'I'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("n", (Py_ssize_t) (0));
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (range == 'V' && vl >= vu){
        PyErr_SetString(PyExc_ValueError, "vl must be less than vu");
        return NULL;
    }
    if (range == 'I' && (il < 1 || il > iu || iu > n)){
        PyErr_SetString(PyExc_ValueError, "il and iu must satisfy "
            "1 <= il <= iu <= n");
        return NULL;
    }
    if (jobz == 'V'){
        if (!Z || !Matrix_Check(Z) || MAT_ID(Z) != DOUBLE)
            err_dbl_mtrx("Z");
        if (ldZ == 0) ldZ = MAX(1,Z->nrows);
        if (ldZ < MAX(1,n)) err_ld("ldZ");
    } else {
        if (ldZ == 0) ldZ = 1;
        if (ldZ < 1) err_ld("ldZ");
    }
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oW < 0) err_nn_int("offsetW");
    if (oW + n > len(W)) err_buf_len("W");
    if (jobz == 'V'){
        if (oZ < 0) err_nn_int("offsetZ");
        if (oZ + ((range == 'I') ? iu-il : n-1)*ldZ + n > len(Z))
	    err_buf_len("Z");
    }

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            liwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsyevr)(&jobz, &range, &uplo, &n, NULL, &ldA, &vl, &vu, &il,
                &iu, &abstol, &m, NULL, NULL, &ldZ, NULL, &wl, &lwork,
                &iwl, &liwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl;
            liwork = iwl;
            work = (void *) calloc(lwork, sizeof(double));
            iwork = (CBLAS_INT *) calloc(liwork, sizeof(CBLAS_INT));
            if (jobz == 'V')
                isuppz = (CBLAS_INT *) calloc(2*MAX(1, (range == 'I') ?
                   iu-il+1 : n), sizeof(CBLAS_INT));
            if (!work  || !iwork || (jobz == 'V' && !isuppz)){
                free(work);  free(iwork);  free(isuppz);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsyevr)(&jobz, &range, &uplo, &n, MAT_BUFD(A)+oA, &ldA, &vl,
                &vu, &il, &iu, &abstol, &m, MAT_BUFD(W)+oW,
                (jobz == 'V') ? MAT_BUFD(Z)+oZ : NULL,  &ldZ,
                (jobz == 'V') ? isuppz : NULL, work, &lwork, iwork,
                &liwork, &info);
            Py_END_ALLOW_THREADS
            free(work);   free(iwork);   free(isuppz);
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("n", (Py_ssize_t) (m));
}


static char doc_heevr[] =
    "Computes selected eigenvalues and eigenvectors of a real symmetric"
    "\nor complex Hermitian matrix (RRR driver).\n\n"
    "m = syevr(A, W, jobz='N', range='A', uplo='L', vl=0.0, vu=0.0, \n"
    "          il=1, iu=1, Z=None, n=A.size[0], ldA=max(1,A.size[0]),\n"
    "          ldZ=None, abstol=0.0, offsetA=0, offsetW=0, offsetZ=0)\n"
    "\n"
    "PURPOSE\n"
    "Computes selected eigenvalues/vectors of a real symmetric or\n"
    "complex Hermitian n by n matrix A.\n"
    "If range is 'A', all eigenvalues are computed.\n"
    "If range is 'V', all eigenvalues in the interval (vl,vu] are\n"
    "computed.\n"
    "If range is 'I', all eigenvalues il through iu are computed\n"
    "(sorted in ascending order with 1 <= il <= iu <= n).\n"
    "If jobz is 'N', only the eigenvalues are returned in W.\n"
    "If jobz is 'V', the eigenvectors are also returned in Z.\n"
    "On exit, the content of A is destroyed.\n"
    "syevr is usually the fastest of the four eigenvalue routines.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "W         'd' matrix of length at least n.  On exit, contains\n"
    "          the computed eigenvalues in ascending order.\n\n"
    "jobz      'N' or 'V'\n\n"
    "range     'A', 'V' or 'I'\n\n"
    "uplo      'L' or 'U'\n\n"
    "vl,vu     doubles.  Only required when range is 'V'.\n\n"
    "il,iu     integers.  Only required when range is 'I'.\n\n"
    "Z         'd' or 'z' matrix.  Must have the same type as A.\n"
    "          Only required when jobz = 'V'.  If range is 'A' or\n"
    "          'V', Z must have at least n columns.  If range is 'I',\n"
    "          Z must have at least iu-il+1 columns.  On exit the\n"
    "          first m columns of Z contain the computed (normalized)\n"
    "          eigenvectors.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).\n"
    "          If zero, the default value is used.\n\n"
    "ldZ       nonnegative integer.  ldZ >= 1 if jobz is 'N' and\n"
    "          ldZ >= max(1,n) if jobz is 'V'.  The default value\n"
    "          is 1 if jobz is 'N' and max(1,Z.size[0]) if jobz ='V'.\n"
    "          If zero, the default value is used.\n\n"
    "abstol    double.  Absolute error tolerance for eigenvalues.\n"
    "          If nonpositive, the LAPACK default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetW   nonnegative integer\n\n"
    "offsetZ   nonnegative integer\n\n"
    "m         the number of eigenvalues computed";

static PyObject* heevr(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *W, *Z=NULL;
    CBLAS_INT n=-1, ldA=0, ldZ=0, il=1, iu=1, oA=0, oW=0, oZ=0, info,
        lwork, *iwork, liwork, lrwork, m, *isuppz=NULL, iwl;
    double vl=0.0, vu=0.0, abstol=0.0, *rwork, rwl;
    void *work;
    number wl;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', jobz_ = 'N', range_ = 'A';
#endif
    char uplo = 'L', jobz = 'N', range = 'A';
    char *kwlist[] = {"A", "W", "jobz", "range", "uplo", "vl", "vu",
	"il", "iu", "Z", "n", "ldA", "ldZ", "abstol", "offsetA",
        "offsetW", "offsetZ", NULL};
    Py_ssize_t _il = il, _iu = iu, _n = n, _ldA = ldA, _ldZ = ldZ, _oA = oA, _oW = oW, _oZ = oZ;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCddnnOnnndnnn",
        kwlist, &A, &W, &jobz_, &range_, &uplo_, &vl, &vu, &_il, &_iu, &Z,
	&_n, &_ldA, &_ldZ, &abstol, &_oA, &_oW, &_oZ))
        return NULL;
    jobz = (char) jobz_;
    range = (char) range_;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cccddnnOnnndnnn",
        kwlist, &A, &W, &jobz, &range, &uplo, &vl, &vu, &_il, &_iu, &Z,
	&_n, &_ldA, &_ldZ, &abstol, &_oA, &_oW, &_oZ))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_il, &il) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_iu, &iu) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldZ, &ldZ) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oW, &oW) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oZ, &oZ) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
    if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
    if (range != 'A' && range != 'V' && range != 'I')
	err_char("range", "'A', 'V', 'I'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("n", (Py_ssize_t) (0));
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (range == 'V' && vl >= vu){
        PyErr_SetString(PyExc_ValueError, "vl must be less than vu");
        return NULL;
    }
    if (range == 'I' && (il < 1 || il > iu || iu > n)){
        PyErr_SetString(PyExc_ValueError, "il and iu must satisfy "
            "1 <= il <= iu <= n");
        return NULL;
    }
    if (jobz == 'V'){
        if (!Z || !Matrix_Check(Z)) err_mtrx("Z");
	if (MAT_ID(Z) != MAT_ID(A)) err_conflicting_ids;
        if (ldZ == 0) ldZ = MAX(1,Z->nrows);
        if (ldZ < MAX(1,n)) err_ld("ldZ");
    } else {
        if (ldZ == 0) ldZ = 1;
        if (ldZ < 1) err_ld("ldZ");
    }
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oW < 0) err_nn_int("offsetW");
    if (oW + n > len(W)) err_buf_len("W");
    if (jobz == 'V'){
        if (oZ < 0) err_nn_int("offsetZ");
        if (oZ + ((range == 'I') ? iu-il : n-1)*ldZ + n > len(Z))
	    err_buf_len("Z");
    }

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            liwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsyevr)(&jobz, &range, &uplo, &n, NULL, &ldA, &vl, &vu, &il,
                &iu, &abstol, &m, NULL, NULL, &ldZ, NULL, &wl.d, &lwork,
                &iwl, &liwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            liwork = iwl;
            work = (void *) calloc(lwork, sizeof(double));
            iwork = (CBLAS_INT *) calloc(liwork, sizeof(CBLAS_INT));
            if (jobz == 'V')
                isuppz = (CBLAS_INT *) calloc(2*MAX(1, (range == 'I') ?
                   iu-il+1 : n), sizeof(CBLAS_INT));
            if (!work  || !iwork || (jobz == 'V' && !isuppz)){
                free(work);  free(iwork);  free(isuppz);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsyevr)(&jobz, &range, &uplo, &n, MAT_BUFD(A)+oA, &ldA, &vl,
                &vu, &il, &iu, &abstol, &m, MAT_BUFD(W)+oW,
                (jobz == 'V') ? MAT_BUFD(Z)+oZ : NULL,  &ldZ,
                (jobz == 'V') ? isuppz : NULL, (double *) work, &lwork,
	       	iwork, &liwork, &info);
            Py_END_ALLOW_THREADS
            free(work);   free(iwork);   free(isuppz);
            break;

	case COMPLEX:
            lwork = -1;
            liwork = -1;
	    lrwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zheevr)(&jobz, &range, &uplo, &n, NULL, &ldA, &vl, &vu, &il,
                &iu, &abstol, &m, NULL, NULL, &ldZ, NULL, &wl.z, &lwork,
                &rwl, &lrwork, &iwl, &liwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
	    lrwork = (CBLAS_INT) rwl;
            liwork = iwl;
            work = (void *) calloc(lwork, sizeof(complex_t));
            rwork = (double *) calloc(lrwork, sizeof(double));
            iwork = (CBLAS_INT *) calloc(liwork, sizeof(CBLAS_INT));
            if (jobz == 'V')
                isuppz = (CBLAS_INT *) calloc(2*MAX(1, (range == 'I') ?
                   iu-il+1 : n), sizeof(CBLAS_INT));
            if (!work  || !rwork || !iwork || (jobz == 'V' && !isuppz)){
                free(work);  free(rwork);  free(iwork);  free(isuppz);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zheevr)(&jobz, &range, &uplo, &n, MAT_BUFZ(A)+oA, &ldA, &vl,
                &vu, &il, &iu, &abstol, &m, MAT_BUFD(W)+oW,
                (jobz == 'V') ? MAT_BUFZ(Z)+oZ : NULL,  &ldZ,
                (jobz == 'V') ? isuppz : NULL, (complex_t *) work, 
                &lwork, rwork, &lrwork, iwork, &liwork, &info);
            Py_END_ALLOW_THREADS
            free(work);   free(rwork); free(iwork);   free(isuppz);
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("n", (Py_ssize_t) (m));
}


static char doc_sygv[] =
    "Generalized symmetric-definite eigenvalue decomposition with real"
    "\n"
    "matrices.\n\n"
    "sygv(A, B, W, itype=1, jobz='N', uplo='L', n=A.size[0], \n"
    "     ldA = max(1,A.size[0]), ldB = max(1,B.size[0]), offsetA=0, \n"
    "     offsetB=0, offsetW=0)\n\n"
    "PURPOSE\n"
    "Returns eigenvalues/vectors of a real generalized \n"
    "symmetric-definite eigenproblem of order n, with B positive \n"
    "definite. \n"
    "1. If itype is 1: A*x = lambda*B*x.\n"
    "2. If itype is 2: A*Bx = lambda*x.\n"
    "3. If itype is 3: B*Ax = lambda*x.\n\n"
    "On exit, W contains the eigenvalues in ascending order.  If jobz\n"
    "is 'V', the matrix of eigenvectors Z is also computed and\n"
    "returned in A, normalized as follows: \n"
    "1. If itype is 1: Z^T*A*Z = diag(W), Z^T*B*Z = I\n"
    "2. If itype is 2: Z^T*A^{-1}*Z = diag(W)^{-1}, Z^T*B*Z = I\n"
    "3. If itype is 3: Z^T*A*Z = diag(W), Z^T*B^{-1}*Z = I.\n\n"
    "If jobz is 'N', only the eigenvalues are computed, and the\n"
    "contents of A is destroyed.   On exit, the matrix B is replaced\n"
    "by its Cholesky factor.\n\n"
    "ARGUMENTS\n"
    "A         'd' matrix\n\n"
    "B         'd' matrix\n\n"
    "W         'd' matrix of length at least n\n\n"
    "itype     integer 1, 2, or 3\n\n"
    "jobz      'N' or 'V'\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "ldB       nonnegative integer.  ldB >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* sygv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B, *W;
    CBLAS_INT n=-1, itype=1, ldA=0, ldB=0, oA=0, oB=0, oW=0, info, lwork;
    double *work;
    number wl;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', jobz_ = 'N';
#endif
    char uplo = 'L', jobz = 'N';
    char *kwlist[] = {"A", "B", "W", "itype", "jobz", "uplo", "n",
        "ldA", "ldB", "offsetA", "offsetB", "offsetW", NULL};
    Py_ssize_t _itype = itype, _n = n, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB, _oW = oW;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|nCCnnnnnn",
        kwlist, &A, &B, &W, &_itype, &jobz_, &uplo_, &_n, &_ldA, &_ldB, &_oA,
        &_oB, &_oW))
        return NULL;
    jobz = (char) jobz_;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|nccnnnnnn",
        kwlist, &A, &B, &W, &_itype, &jobz, &uplo, &_n, &_ldA, &_ldB, &_oA,
        &_oB, &_oW))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_itype, &itype) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oW, &oW) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B) || MAT_ID(B) != MAT_ID(A)) err_conflicting_ids;
    if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
    if (itype != 1 && itype != 2 && itype != 3)
        err_char("itype", "1, 2, 3");
    if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
        if (A->nrows != n || A->ncols != n){
            PyErr_SetString(PyExc_TypeError, "B must have the same "
                "dimension as A");
            return NULL;
	}
    }
    if (n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (n-1)*ldB + n > len(B)) err_buf_len("B");
    if (oW < 0) err_nn_int("offsetW");
    if (oW + n > len(W)) err_buf_len("W");

    switch (MAT_ID(A)){
	case DOUBLE:
            lwork=-1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsygv)(&itype, &jobz, &uplo, &n, NULL, &ldA, NULL, &ldB,
                NULL, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = calloc(lwork, sizeof(double))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsygv)(&itype, &jobz, &uplo, &n, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(B)+oB, &ldB, MAT_BUFD(W)+oW, work, &lwork,
                &info);
            Py_END_ALLOW_THREADS
            free(work);
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_hegv[] =
    "Generalized symmetric-definite eigenvalue decomposition with\n"
    "real or complex matrices.\n\n"
    "hegv(A, B, W, itype=1, jobz='N', uplo='L', n=A.size[0], \n"
    "     ldA = max(1,A.size[0]), ldB = max(1,B.size[0]), offsetA=0, \n"
    "     offsetB=0, offsetW=0)\n\n"
    "PURPOSE\n"
    "Returns eigenvalues/vectors of a real or complex generalized\n"
    "symmetric-definite eigenproblem of order n, with B positive\n"
    "definite. \n"
    "1. If itype is 1: A*x = lambda*B*x.\n"
    "2. If itype is 2: A*Bx = lambda*x.\n"
    "3. If itype is 3: B*Ax = lambda*x.n\n\n"
    "On exit, W contains the eigenvalues in ascending order.  If jobz\n"
    "is 'V', the matrix of eigenvectors Z is also computed and \n"
    "returned in A, normalized as follows: \n"
    "1. If itype is 1: Z^H*A*Z = diag(W), Z^H*B*Z = I\n"
    "2. If itype is 2: Z^H*A^{-1}*Z = diag(W), Z^H*B*Z = I\n"
    "3. If itype is 3: Z^H*A*Z = diag(W), Z^H*B^{-1}*Z = I.\n\n"
    "If jobz is 'N', only the eigenvalues are computed, and the \n"
    "contents of A is destroyed.   On exit, the matrix B is replaced\n"
    "by its Cholesky factor.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "W         'd' matrix of length at least n\n\n"
    "itype     integer 1, 2, or 3\n\n"
    "jobz      'N' or 'V'\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "ldB       nonnegative integer.  ldB >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* hegv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B, *W;
    CBLAS_INT n=-1, itype=1, ldA=0, ldB=0, oA=0, oB=0, oW=0, info, lwork;
    double *rwork=NULL;
    void *work=NULL;
    number wl;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', jobz_ = 'N';
#endif
    char uplo = 'L', jobz = 'N';
    char *kwlist[] = {"A", "B", "W", "itype", "jobz", "uplo", "n",
        "ldA", "ldB", "offsetA", "offsetB", "offsetW", NULL};
    Py_ssize_t _itype = itype, _n = n, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB, _oW = oW;
#if 0
    CBLAS_INT ispec=1, n2=-1, n3=-1, n4=-1;
    char *name = "zhetrd", *uplol = "L", *uplou = "U";
#endif

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|nCCnnnnnn",
        kwlist, &A, &B, &W, &_itype, &jobz_, &uplo_, &_n, &_ldA, &_ldB, &_oA,
        &_oB, &_oW))
        return NULL;
    uplo = (char) uplo_;
    jobz = (char) jobz_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|nccnnnnnn",
        kwlist, &A, &B, &W, &_itype, &jobz, &uplo, &_n, &_ldA, &_ldB, &_oA,
        &_oB, &_oW))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_itype, &itype) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oW, &oW) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B) || MAT_ID(B) != MAT_ID(A)) err_conflicting_ids;
    if (!Matrix_Check(W) || MAT_ID(W) != DOUBLE) err_dbl_mtrx("W");
    if (itype != 1 && itype != 2 && itype != 3)
        err_char("itype", "1, 2, 3");
    if (jobz != 'N' && jobz != 'V') err_char("jobz", "'N', 'V'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
        if (A->nrows != n || A->ncols != n){
            PyErr_SetString(PyExc_TypeError, "B must have the same "
                "dimension as A");
            return NULL;
	}
    }
    if (n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (n-1)*ldB + n > len(B)) err_buf_len("B");
    if (oW < 0) err_nn_int("offsetW");
    if (oW + n > len(W)) err_buf_len("W");

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork=-1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsygv)(&itype, &jobz, &uplo, &n, NULL, &ldA, NULL, &ldB,
                NULL, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = calloc(lwork, sizeof(double))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dsygv)(&itype, &jobz, &uplo, &n, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(B)+oB, &ldB, MAT_BUFD(W)+oW, work, &lwork,
                &info);
            Py_END_ALLOW_THREADS
            free(work);
            break;

        case COMPLEX:
#if 1
            lwork=-1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zhegv)(&itype, &jobz, &uplo, &n, NULL, &n, NULL, &n, NULL,
                &wl.z, &lwork, NULL, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
#endif
#if 0
            /* zhegv used to handle lwork=-1 incorrectly.
               The following replaces the call to zhegv with lwork=-1 */
            lwork = n * (1 + BLAS_FUNC(ilaenv)(&ispec, &name, (uplo == 'L') ?
                &uplol : &uplou, &n, &n2, &n3, &n4));
#endif

            work = (void *) calloc(lwork, sizeof(complex_t));
            rwork = (double *) calloc(3*n-2, sizeof(double));
            if (!work || !rwork){
                free(work);  free(rwork);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zhegv)(&itype, &jobz, &uplo, &n, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(B)+oB, &ldB, MAT_BUFD(W)+oW, 
                (complex_t *) work, &lwork, rwork, &info);
            Py_END_ALLOW_THREADS
            free(work);  free(rwork);
            break;

        default:
	    err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_gesvd[] =
    "Singular value decomposition of a real or complex matrix.\n\n"
    "gesvd(A, S, jobu='N', jobvt='N', U=None, Vt=None, m=A.size[0],\n"
    "      n=A.size[1], ldA=max(1,A.size[0]), ldU=None, ldVt=None,\n"
    "      offsetA=0, offsetS=0, offsetU=0, offsetVt=0)\n\n"
    "PURPOSE\n"
    "Computes singular values and, optionally, singular vectors of a \n"
    "real or complex m by n matrix A.\n\n"
    "The argument jobu controls how many left singular vectors are\n"
    "computed: \n\n"
    "'N': no left singular vectors are computed.\n"
    "'A': all left singular vectors are computed and returned as\n"
    "     columns of U.\n"
    "'S': the first min(m,n) left singular vectors are computed and\n"
    "     returned as columns of U.\n"
    "'O': the first min(m,n) left singular vectors are computed and\n"
    "     returned as columns of A.\n\n"
    "The argument jobvt controls how many right singular vectors are\n"
    "computed:\n\n"
    "'N': no right singular vectors are computed.\n"
    "'A': all right singular vectors are computed and returned as\n"
    "     rows of Vt.\n"
    "'S': the first min(m,n) right singular vectors are computed and\n"
    "     returned as rows of Vt.\n"
    "'O': the first min(m,n) right singular vectors are computed and\n"
    "     returned as rows of A.\n"
    "Note that the (conjugate) transposes of the right singular \n"
    "vectors are returned in Vt or A.\n\n"
    "On exit (in all cases), the contents of A are destroyed.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "S         'd' matrix of length at least min(m,n).  On exit, \n"
    "          contains the computed singular values in descending\n"
    "          order.\n\n"
    "jobu      'N', 'A', 'S' or 'O'\n\n"
    "jobvt     'N', 'A', 'S' or 'O'\n\n"
    "U         'd' or 'z' matrix.  Must have the same type as A.\n"
    "          Not referenced if jobu is 'N' or 'O'.  If jobu is 'A',\n"
    "          a matrix with at least m columns.   If jobu is 'S', a\n"
    "          matrix with at least min(m,n) columns.\n"
    "          On exit (with jobu 'A' or 'S'), the columns of U\n"
    "          contain the computed left singular vectors.\n\n"
    "Vt        'd' or 'z' matrix.  Must have the same type as A.\n"
    "          Not referenced if jobvt is 'N' or 'O'.  If jobvt is \n"
    "          'A' or 'S', a matrix with at least n columns.\n"
    "          On exit (with jobvt 'A' or 'S'), the rows of Vt\n"
    "          contain the computed right singular vectors, or, in\n"
    "          the complex case, their complex conjugates.\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,m).\n"
    "          If zero, the default value is used.\n\n"
    "ldU       nonnegative integer.\n"
    "          ldU >= 1 if jobu is 'N' or 'O'.\n"
    "          ldU >= max(1,m) if jobu is 'A' or 'S'.\n"
    "          The default value is max(1,U.size[0]) if jobu is 'A' \n"
    "          or 'S', and 1 otherwise.\n"
    "          If zero, the default value is used.\n\n"
    "ldVt      nonnegative integer.\n"
    "          ldVt >= 1 if jobvt is 'N' or 'O'.\n"
    "          ldVt >= max(1,n) if jobvt is 'A'.  \n"
    "          ldVt >= max(1,min(m,n)) if ldVt is 'S'.\n"
    "          The default value is max(1,Vt.size[0]) if jobvt is 'A'\n"
    "          or 'S', and 1 otherwise.\n"
    "          If zero, the default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetS   nonnegative integer\n\n"
    "offsetU   nonnegative integer\n\n"
    "offsetVt  nonnegative integer";

static PyObject* gesvd(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *S, *U=NULL, *Vt=NULL;
    CBLAS_INT m=-1, n=-1, ldA=0, ldU=0, ldVt=0, oA=0, oS=0, oU=0, oVt=0, info,
       	lwork;
    double *rwork=NULL;
    void *work=NULL;
    number wl;
#if PY_MAJOR_VERSION >= 3
    int jobu_ = 'N', jobvt_ = 'N';
#endif
    char jobu = 'N', jobvt = 'N';
    char *kwlist[] = {"A", "S", "jobu", "jobvt", "U", "Vt", "m", "n",
	"ldA", "ldU", "ldVt", "offsetA", "offsetS", "offsetU",
        "offsetVt", NULL};
    Py_ssize_t _m = m, _n = n, _ldA = ldA, _ldU = ldU, _ldVt = ldVt, _oA = oA, _oS = oS, _oU = oU, _oVt = oVt;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCOOnnnnnnnnn",
        kwlist, &A, &S, &jobu_, &jobvt_, &U, &Vt, &_m, &_n, &_ldA, &_ldU,
        &_ldVt, &_oA, &_oS, &_oU, &_oVt))
        return NULL;
    jobu = (char) jobu_;
    jobvt = (char) jobvt_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ccOOnnnnnnnnn",
        kwlist, &A, &S, &jobu, &jobvt, &U, &Vt, &_m, &_n, &_ldA, &_ldU,
        &_ldVt, &_oA, &_oS, &_oU, &_oVt))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldU, &ldU) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldVt, &ldVt) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oS, &oS) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oU, &oU) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oVt, &oVt) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(S) || MAT_ID(S) != DOUBLE) err_dbl_mtrx("S");
    if (jobu != 'N' && jobu != 'A' && jobu != 'O' && jobu != 'S')
        err_char("jobu", "'N', 'A', 'S', 'O'");
    if (jobvt != 'N' && jobvt != 'A' && jobvt != 'O' && jobvt != 'S')
        err_char("jobvt", "'N', 'A', 'S', 'O'");
    if (jobu == 'O' && jobvt == 'O') {
        PyErr_SetString(PyExc_ValueError, "'jobu' and 'jobvt' cannot "
            "both be 'O'");
        return NULL;
    }
    if (m < 0) m = A->nrows;
    if (n < 0) n = A->ncols;
    if (m == 0 || n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,m)) err_ld("ldA");
    if (jobu == 'A' || jobu == 'S'){
        if (!U || !Matrix_Check(U)) err_mtrx("U");
        if (MAT_ID(U) != MAT_ID(A)) err_conflicting_ids;
        if (ldU == 0) ldU = MAX(1,U->nrows);
        if (ldU < MAX(1,m)) err_ld("ldU");
    } else {
        if (ldU == 0) ldU = 1;
        if (ldU < 1) err_ld("ldU");
    }
    if (jobvt == 'A' || jobvt == 'S'){
        if (!Vt || !Matrix_Check(Vt)) err_mtrx("Vt");
	if (MAT_ID(Vt) != MAT_ID(A)) err_conflicting_ids;
        if (ldVt == 0) ldVt = MAX(1,Vt->nrows);
        if (ldVt < ((jobvt == 'A') ?  MAX(1,n) : MAX(1,MIN(m,n))))
            err_ld("ldVt");
    } else {
        if (ldVt == 0) ldVt = 1;
        if (ldVt < 1) err_ld("ldVt");
    }
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
    if (oS < 0) err_nn_int("offsetS");
    if (oS + MIN(m,n) > len(S)) err_buf_len("S");
    if (jobu == 'A' || jobu == 'S'){
        if (oU < 0) err_nn_int("offsetU");
        if (oU + ((jobu == 'A') ? m-1 : MIN(m,n)-1)*ldU + m > len(U))
            err_buf_len("U");
    }
    if (jobvt == 'A' || jobvt == 'S'){
        if (oVt < 0) err_nn_int("offsetVt");
        if (oVt + (n-1)*ldVt + ((jobvt == 'A') ? n : MIN(m,n)) >
            len(Vt)) err_buf_len("Vt");
    }

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
             Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgesvd)(&jobu, &jobvt, &m, &n, NULL, &ldA, NULL, NULL,
                &ldU, NULL, &ldVt, &wl.d, &lwork, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            if (!(work = (void *) calloc(lwork, sizeof(double)))){
                free(work);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgesvd)(&jobu, &jobvt, &m, &n, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(S)+oS,  (jobu == 'A' || jobu == 'S') ?
		MAT_BUFD(U)+oU : NULL, &ldU, (jobvt == 'A' ||
                jobvt == 'S') ?  MAT_BUFD(Vt)+oVt : NULL, &ldVt,
		(double *) work, &lwork, &info);
            Py_END_ALLOW_THREADS
            free(work);
            break;

	case COMPLEX:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgesvd)(&jobu, &jobvt, &m, &n, NULL, &ldA, NULL, NULL,
                &ldU, NULL, &ldVt, &wl.z, &lwork, NULL, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
            work = (void *) calloc(lwork, sizeof(complex_t));
            rwork = (double *) calloc(5*MIN(m,n), sizeof(double));
	    if (!work || !rwork){
                free(work);  free(rwork);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgesvd)(&jobu, &jobvt, &m, &n, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFD(S)+oS, (jobu == 'A' || jobu == 'S') ?
                MAT_BUFZ(U)+oU : NULL,  &ldU, (jobvt == 'A' ||
                jobvt == 'S') ? MAT_BUFZ(Vt)+oVt : NULL, &ldVt,
                (complex_t *) work, &lwork, rwork, &info);
            Py_END_ALLOW_THREADS
            free(work);   free(rwork);
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_gesdd[] =
    "Singular value decomposition of a real or complex matrix\n"
    "(divide-and-conquer driver).\n\n"
    "gesdd(A, S, jobz='N', U=None, V=None, m=A.size[0], n=A.size[1], \n"
    "      ldA=max(1,A.size[0]), ldU=None, ldVt=None, offsetA=0, \n"
    "      offsetS=0, offsetU=0, offsetVt=0)\n\n"
    "PURPOSE\n"
    "Computes singular values and, optionally, singular vectors of a \n"
    "real or complex m by n matrix A.  The argument jobz controls how\n"
    "many singular vectors are computed:\n\n"
    "'N': no singular vectors are computed.\n"
    "'A': all m left singular vectors are computed and returned as\n"
    "     columns of U;  all n right singular vectors are computed \n"
    "     and returned as rows of Vt.\n"
    "'S': the first min(m,n) left and right singular vectors are\n"
    "     computed and returned as columns of U and rows of Vt.\n"
    "'O': if m>=n, the first n left singular vectors are returned as\n"
    "     columns of A and the n right singular vectors are returned\n"
    "     as rows of Vt.  If m<n, the m left singular vectors are\n"
    "     returned as columns of U and the first m right singular\n"
    "     vectors are returned as rows of A.\n\n"
    "Note that the (conjugate) transposes of the right singular \n"
    "vectors are returned in Vt or A.\n\n"
    "On exit (in all cases), the contents of A are destroyed.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "S         'd' matrix of length at least min(m,n).  On exit, \n"
    "          contains the computed singular values in descending\n"
    "          order.\n\n"
    "jobz      'N', 'A', 'S' or 'O'\n\n"
    "U         'd' or 'z' matrix.  Must have the same type as A.\n"
    "          Not referenced if jobz is 'N' or jobz is 'O' and m>=n.\n"
    "          If jobz is 'A' or jobz is 'O' and m<n, a matrix with\n"
    "          at least m columns.   If jobz is 'S', a matrix with at\n"
    "          least min(m,n) columns.  On exit (except when jobz is\n"
    "          'N' or jobz is 'O' and m>=n), contains the computed\n"
    "          left singular vectors stored columnwise.\n\n"
    "Vt        'd' or 'z' matrix.  Must have the same type as A.\n"
    "          Not referenced if jobz is 'N' or jobz is 'O' and m<n.\n"
    "          If jobz is 'A' or 'S' or jobz is 'O' and m>=n, a\n"
    "          matrix with at least n columns.   On exit (except when\n"
    "          jobz is 'N' or jobz is 'O' and m<n), the rows of Vt\n"
    "          contain the computed right singular vectors, or, in\n"
    "          the complex case, their complex conjugates.\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,m).\n"
    "          If zero, the default value is used.\n\n"
    "ldU       nonnegative integer.\n"
    "          ldU >= 1 if jobz is 'N' or 'O'.\n"
    "          ldU >= max(1,m) if jobz is 'S' or 'A' or jobz is 'O'\n"
    "          and m<n.  The default value is max(1,U.size[0]) if\n"
    "          jobz is 'S' or 'A' or jobz is'O' and m<n, and 1\n"
    "          otherwise.\n\n"
    "ldVt      nonnegative integer.\n"
    "          ldVt >= 1 if jobz is 'N'.\n"
    "          ldVt >= max(1,n) if jobz is 'A' or jobz is 'O' and \n"
    "          m>=n.  \n"
    "          ldVt >= max(1,min(m,n)) if ldVt is 'S'.\n"
    "          The default value is max(1,Vt.size[0]) if jobvt is 'A'\n"
    "          or 'S' or jobvt is 'O' and m>=n, and 1 otherwise.\n"
    "          If zero, the default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetS   nonnegative integer\n\n"
    "offsetU   nonnegative integer\n\n"
    "offsetVt  nonnegative integer";

static PyObject* gesdd(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *S, *U=NULL, *Vt=NULL;
    CBLAS_INT m=-1, n=-1, ldA=0, ldU=0, ldVt=0, oA=0, oS=0, oU=0, oVt=0, info,
       	*iwork=NULL, lwork;
    double *rwork=NULL;
    void *work=NULL;
    number wl;
#if PY_MAJOR_VERSION >= 3
    int jobz_ = 'N';
#endif
    char jobz = 'N';
    char *kwlist[] = {"A", "S", "jobz", "U", "Vt", "m", "n", "ldA",
	"ldU", "ldVt", "offsetA", "offsetS", "offsetU", "offsetVt",
        NULL};
    Py_ssize_t _m = m, _n = n, _ldA = ldA, _ldU = ldU, _ldVt = ldVt, _oA = oA, _oS = oS, _oU = oU, _oVt = oVt;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|COOnnnnnnnnn",
        kwlist, &A, &S, &jobz_, &U, &Vt, &_m, &_n, &_ldA, &_ldU, &_ldVt, &_oA,
        &_oS, &_oU, &_oVt))
        return NULL;
    jobz = (char) jobz_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cOOnnnnnnnnn",
        kwlist, &A, &S, &jobz, &U, &Vt, &_m, &_n, &_ldA, &_ldU, &_ldVt, &_oA,
        &_oS, &_oU, &_oVt))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldU, &ldU) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldVt, &ldVt) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oS, &oS) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oU, &oU) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oVt, &oVt) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(S) || MAT_ID(S) != DOUBLE) err_dbl_mtrx("S");
    if (jobz != 'A' && jobz != 'S' && jobz != 'O' && jobz != 'N')
        err_char("jobz", "'A', 'S', 'O', 'N'");
    if (m < 0) m = A->nrows;
    if (n < 0) n = A->ncols;
    if (m == 0 || n == 0) return Py_BuildValue("");
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,m)) err_ld("ldA");
    if (jobz == 'A' || jobz == 'S' || (jobz == 'O' && m<n)){
        if (!U || !Matrix_Check(U)) err_mtrx("U");
        if (MAT_ID(U) != MAT_ID(A)) err_conflicting_ids;
        if (ldU == 0) ldU = MAX(1,U->nrows);
        if (ldU < MAX(1,m)) err_ld("ldU");
    } else {
        if (ldU == 0) ldU = 1;
        if (ldU < 1) err_ld("ldU");
    }
    if (jobz == 'A' || jobz == 'S' || (jobz == 'O' && m>=n)){
        if (!Vt || !Matrix_Check(Vt)) err_mtrx("Vt");
	if (MAT_ID(Vt) != MAT_ID(A)) err_conflicting_ids;
        if (ldVt == 0) ldVt = MAX(1,Vt->nrows);
        if (ldVt < ((jobz == 'A' || jobz == 'O') ?  MAX(1,n) :
            MAX(1,MIN(m,n)))) err_ld("ldVt");
    } else {
        if (ldVt == 0) ldVt = 1;
        if (ldVt < 1) err_ld("ldVt");
    }
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
    if (oS < 0) err_nn_int("offsetS");
    if (oS + MIN(m,n) > len(S)) err_buf_len("S");
    if (jobz == 'A' || jobz == 'S' || (jobz == 'O' && m<n)){
        if (oU < 0) err_nn_int("offsetU");
        if (oU + ((jobz == 'A' || jobz == 'O') ? m-1 : MIN(m,n)-1)*ldU
            + m > len(U))
	    err_buf_len("U");
    }
    if (jobz == 'A' || jobz == 'S' || (jobz == 'O' && m>=n)){
        if (oVt < 0) err_nn_int("offsetVt");
        if (oVt + (n-1)*ldVt + ((jobz == 'A' || jobz == 'O') ? n :
            MIN(m,n)) > len(Vt)) err_buf_len("Vt");
    }

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgesdd)(&jobz, &m, &n, NULL, &ldA, NULL, NULL, &ldU, NULL,
                &ldVt, &wl.d, &lwork, NULL, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) wl.d;
            work = (void *) calloc(lwork, sizeof(double));
            iwork = (CBLAS_INT *) calloc(8*MIN(m,n), sizeof(CBLAS_INT));
	    if (!work || !iwork){
                free(work);  free(iwork);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dgesdd)(&jobz, &m, &n, MAT_BUFD(A)+oA, &ldA, MAT_BUFD(S)+oS,
                (jobz == 'A' || jobz == 'S' || (jobz == 'O' && m<n)) ?
                MAT_BUFD(U)+oU : NULL, &ldU, (jobz == 'A' ||
                jobz == 'S' || (jobz == 'O'  && m>=n)) ?
                MAT_BUFD(Vt)+oVt : NULL, &ldVt, (double *) work, &lwork,
                iwork, &info);
            Py_END_ALLOW_THREADS
            free(work);   free(iwork);
            break;

	case COMPLEX:
            lwork = -1;
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgesdd)(&jobz, &m, &n, NULL, &ldA, NULL, NULL, &ldU, NULL,
                &ldVt, &wl.z, &lwork, NULL, NULL, &info);
            Py_END_ALLOW_THREADS
            lwork = (CBLAS_INT) creal(wl.z);
            work = (void *) calloc(lwork, sizeof(complex_t));
            iwork = (CBLAS_INT *) calloc(8*MIN(m,n), sizeof(CBLAS_INT));
            rwork = (double *) calloc( (jobz == 'N') ? 7*MIN(m,n) :
                5*MIN(m,n)*(MIN(m,n)+1), sizeof(double));
	    if (!work || !iwork || !rwork){
                free(work);  free(iwork); free(rwork);
                return PyErr_NoMemory();
            }
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zgesdd)(&jobz, &m, &n, MAT_BUFZ(A)+oA, &ldA, MAT_BUFD(S)+oS,
                (jobz == 'A' || jobz == 'S' || (jobz == 'O' && m<n)) ?
		MAT_BUFZ(U)+oU : NULL,  &ldU, (jobz == 'A' ||
                jobz == 'S' || (jobz == 'O' && m>=n)) ?
                MAT_BUFZ(Vt)+oVt : NULL, &ldVt, (complex_t *) work,
                &lwork, rwork, iwork, &info);
            Py_END_ALLOW_THREADS
            free(work);  free(iwork); free(rwork);
            break;

        default:
            err_invalid_id;
    }

    if (info) err_lapack
    else return Py_BuildValue("");
}


static char doc_gees[] =
    "Schur factorization of a real of complex matrix.\n\n"
    "sdim = gees(A, w=None, V=None, select=None, n=A.size[0],\n"
    "            ldA=max(1,A.size[0]), ldV=max(1,Vs.size[0]), offsetA=0,\n"
    "            offsetw=0, offsetV=0)\n\n"
    "PURPOSE\n"
    "Computes the real Schur form A = V * S * V^T or the complex Schur\n"
    "form A = V * S * V^H, the eigenvalues, and, optionally, the matrix\n"
    "of Schur vectors of an n by n matrix A.  The real Schur form is \n"
    "computed if A is real, and the conmplex Schur form is computed if \n"
    "A is complex.  On exit, A is replaced with S.  If the argument w is\n"
    "provided, the eigenvalues are returned in w.  If V is provided, the\n"
    "Schur vectors are computed and returned in V.  The argument select\n"
    "can be used to obtain an ordered Schur factorization.  It must be a\n"
    "Python function that can be called as f(s) with s complex, and \n"
    "returns 0 or 1.  The eigenvalues s for which f(s) is 1 will be \n"
    "placed first in the Schur factorization.   For the real case, \n"
    "eigenvalues s for which f(s) or f(conj(s)) is 1, are placed first.\n"
    "If select is provided, gees() returns the number of eigenvalues \n"
    "that satisfy the selection criterion.   Otherwise, it returns 0.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "w         'z' matrix of length at least n\n\n"
    "V         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "select    Python function that takes a complex number as argument\n"
    "          and returns True or False.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).\n"
    "          If zero, the default value is used.\n\n"
    "ldV       nonnegative integer.  ldV >= 1 and ldV >= n if V is\n"
    "          present.  If zero, the default value is used (with \n"
    "          V.size[0] replaced by 0 if V is None).\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetW   nonnegative integer\n\n"
    "offsetV   nonnegative integer\n\n"
    "sdim      number of eigenvalues that satisfy the selection\n"
    "          criterion specified by select.";

static PyObject *py_select_r;
static PyObject *py_select_c;

extern CBLAS_INT fselect_c(complex_t *w)
{
    PyObject *wpy, *result;
    CBLAS_INT a = 0;

    wpy = PyComplex_FromDoubles(creal(*w), cimag(*w));
    if (!(result = PyObject_CallFunctionObjArgs(py_select_c, wpy, NULL))) {
        Py_XDECREF(wpy);
        return -1;
    }
#if PY_MAJOR_VERSION >= 3
    if (PyLong_Check(result)) a = (CBLAS_INT) PyLong_AsLong(result);
#else
    if PyInt_Check(result) a = (CBLAS_INT) PyInt_AsLong(result);
#endif
    else
        PyErr_SetString(PyExc_TypeError, "select() must return an integer "
            "argument");
    Py_XDECREF(wpy);  Py_XDECREF(result);
    return a;
}

extern CBLAS_INT fselect_r(double *wr, double *wi)
{
    PyObject *wpy, *result;
    CBLAS_INT a = 0;

    wpy = PyComplex_FromDoubles(*wr, *wi);
    if (!(result = PyObject_CallFunctionObjArgs(py_select_r, wpy, NULL))) {
        Py_XDECREF(wpy);
        return -1;
    }
#if PY_MAJOR_VERSION >= 3
    if (PyLong_Check(result)) a = (CBLAS_INT) PyLong_AsLong(result);
#else
    if PyInt_Check(result) a = (CBLAS_INT) PyInt_AsLong(result);
#endif
    else
        PyErr_SetString(PyExc_TypeError, "select() must return an integer "
           "argument");
    Py_XDECREF(wpy);  Py_XDECREF(result);
    return a;
}

static PyObject* gees(PyObject *self, PyObject *args, PyObject *kwrds)
{
    PyObject *F=NULL;
    matrix *A, *W=NULL, *Vs=NULL;
    CBLAS_INT n=-1, ldA=0, ldVs=0, oA=0, oVs=0, oW=0, info, lwork, sdim, k,
        *bwork=NULL;
    double *wr=NULL, *wi=NULL, *rwork=NULL;
    complex_t *w=NULL;
    void *work=NULL;
    number wl;
    char *kwlist[] = {"A", "w", "V", "select", "n", "ldA", "ldV",
        "offsetA", "offsetw", "offsetV", NULL};
    Py_ssize_t _n = n, _ldA = ldA, _ldVs = ldVs, _oA = oA, _oW = oW, _oVs = oVs;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|OOOnnnnnn",
        kwlist, &A, &W, &Vs, &F, &_n, &_ldA, &_ldVs, &_oA, &_oW, &_oVs))
        return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldVs, &ldVs) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oW, &oW) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oVs, &oVs) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");

    if (W){
        if (!Matrix_Check(W) || MAT_ID(W) != COMPLEX)
            PY_ERR_TYPE("W must be a matrix with typecode 'z'")
        if (oW < 0) err_nn_int("offsetW");
        if (oW + n > len(W)) err_buf_len("W");
    }

    if (Vs){
        if (!Matrix_Check(Vs)) err_mtrx("Vs");
        if (MAT_ID(Vs) != MAT_ID(A)) err_conflicting_ids;
        if (ldVs == 0) ldVs = MAX(1, Vs->nrows);
        if (ldVs < MAX(1,n)) err_ld("ldVs");
        if (oVs < 0) err_nn_int("offsetVs");
        if (oVs + (n-1)*ldVs + n > len(Vs)) err_buf_len("Vs");
    } else {
        if (ldVs == 0) ldVs = 1;
        if (ldVs < 1) err_ld("ldVs");
    }

    if (F && !PyFunction_Check(F))
        PY_ERR_TYPE("select must be a Python function")


    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            BLAS_FUNC(dgees)(Vs ? "V" : "N", F ? "S" : "N", NULL, &n, NULL, &ldA,
                &sdim, NULL, NULL, NULL, &ldVs, &wl.d, &lwork, NULL,
                &info);
            lwork = (CBLAS_INT) wl.d;
            work = (void *) calloc(lwork, sizeof(double));
            wr = (double *) calloc(n, sizeof(double));
            wi = (double *) calloc(n, sizeof(double));
            if (F) bwork = (CBLAS_INT *) calloc(n, sizeof(CBLAS_INT));
            if (!work || !wr || !wi || (F && !bwork)){
                free(work);  free(wr);  free(wi);  free(bwork);
                return PyErr_NoMemory();
            }
            py_select_r = F;
            BLAS_FUNC(dgees)(Vs ? "V": "N", F ? "S" : "N", F ? &fselect_r : NULL,
                &n, MAT_BUFD(A) + oA, &ldA, &sdim, wr, wi,
                Vs ? MAT_BUFD(Vs) + oVs : NULL, &ldVs, (double *) work,
                &lwork, bwork, &info);
            if (W) for (k=0; k<n; k++)
#ifndef _MSC_VER
                MAT_BUFZ(W)[oW + k] = wr[k] + I * wi[k];
#else
  	        MAT_BUFZ(W)[oW + k] = _Cbuild(wr[k],wi[k]);
#endif
            free(work);  free(wr);  free(wi);  free(bwork);
            break;

	case COMPLEX:
            lwork = -1;
            BLAS_FUNC(zgees)(Vs ? "V" : "N", F ? "S": "N", NULL, &n, NULL, &ldA,
                &sdim, NULL, NULL, &ldVs, &wl.z, &lwork, NULL, NULL,
                &info);
            lwork = (CBLAS_INT) creal(wl.z);
            work = (void *) calloc(lwork, sizeof(complex_t));
            rwork = (double *) calloc(n, sizeof(complex_t));
            if (F) bwork = (CBLAS_INT *) calloc(n, sizeof(CBLAS_INT));
            if (!W) 
                w = (complex_t *) calloc(n, sizeof(complex_t));
	    if (!work || !rwork || (F && !bwork) || (!W && !w) ){
                free(work);  free(rwork); free(bwork);  free(w);
                return PyErr_NoMemory();
            }
            py_select_c = F;
            BLAS_FUNC(zgees)(Vs ? "V": "N", F ? "S" : "N", F ? &fselect_c : NULL,
                &n, MAT_BUFZ(A) + oA, &ldA, &sdim,
                W ? MAT_BUFZ(W) + oW : w, Vs ? MAT_BUFZ(Vs) + oVs : NULL,
                &ldVs, (complex_t *) work, &lwork, 
                (complex_t *) rwork,  bwork, &info);
            free(work);  free(rwork); free(bwork);  free(w);
            break;

        default:
            err_invalid_id;
    }

    if (PyErr_Occurred()) return NULL;

    if (info) err_lapack
    else return Py_BuildValue("n", (Py_ssize_t) (F ? sdim : 0));
}


static char doc_gges[] =
    "Generalized Schur factorization of real or complex matrices.\n\n"
    "sdim = gges(A, B, a=None, b=None, Vl=None, Vr=None, select=None,\n"
    "            n=A.size[0], ldA=max(1,A.size[0]),\n"
    "            ldB=max(1,B.size[0]), ldVl=max(1,Vl.size[0]),\n"
    "            ldVr=max(1,Vr.size[0]), offsetA=0, offestB=0, \n"
    "            offseta=0, offsetb=0, offsetVl=0, offsetVr=0)\n\n"
    "PURPOSE\n"
    "Computes the real generalized Schur form A = Vl * S * Vr^T, \n"
    "B = Vl * T * Vr^T, or the complex generalized Schur form \n"
    "A = Vl * S * Vr^H, B = Vl * T * Vr^H of the square matrices A, B,\n"
    "the generalized eigenvalues, and, optionally, the matrices of left \n"
    "and right Schur vectors.  The real form is computed if A and B are \n"
    "real, and the complex Schur form is computed if A and B are\n"
    "complex.  On exit, A is replaced with S and B is replaced with T.\n"
    "If the arguments a and b are provided, then on return a[i] / b[i] \n"
    "is the ith generalized eigenvalue.  If Vl is provided, then the \n"
    "left Schur vectors are computed and returned in Vl.  If Vr is \n"
    "provided then the right Schur vectors are computed and returned \n"
    "in Vr.  The argument select can be used to obtain an ordered Schur\n"
    "factorization.  It must be a Python function that can be called as\n"
    "f(u,v) with u complex, and v real, and returns 0 or 1.  The \n"
    "eigenvalues u/v for which f(u, v) is 1 will be placed first on the\n"
    "diagonal of S and T.  For the real case, eigenvalues u/v for which\n"
    "f(u, v) or f(conj(u), v) is 1, are placed first.  If select is \n"
    "provided, gges() returns the number of eigenvalues that satisfy the\n"
    "selection criterion.  Otherwise, it returns 0.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "a         'z' matrix of length at least n\n\n"
    "b         'd' matrix of length at least n\n\n"
    "Vl        'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "Vr        'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "select    Python function that takes a complex and a real number \n"
    "          as argument and returns True or False.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).\n"
    "          If zero, the default value is used.\n\n"
    "ldB       nonnegative integer.  ldB >= max(1,n).\n"
    "          If zero, the default value is used.\n\n"
    "ldVl      nonnegative integer.  ldVl >= 1 and ldVl >= n if Vl \n"
    "          is present.  If zero, the default value is used (with \n"
    "          Vl.size[0] replaced by 0 if Vl is None).\n\n"
    "ldVr      nonnegative integer.  ldVr >= 1 and ldVr >= n if Vr \n"
    "          is present.  If zero, the default value is used (with \n"
    "          Vr.size[0] replaced by 0 if Vr is None).\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer\n\n"
    "offseta   nonnegative integer\n\n"
    "offsetb   nonnegative integer\n\n"
    "offsetVl  nonnegative integer\n\n"
    "offsetVr  nonnegative integer\n\n"
    "sdim      number of eigenvalues that satisfy the selection\n"
    "          criterion specified by select.";

static PyObject *py_select_gr;
static PyObject *py_select_gc;

extern CBLAS_INT fselect_gc(complex_t *w, double *v)
{
   PyObject *wpy, *vpy, *result;
   CBLAS_INT a = 0;

   wpy = PyComplex_FromDoubles(creal(*w), cimag(*w));
   vpy = PyFloat_FromDouble(*v);
   if (!(result = PyObject_CallFunctionObjArgs(py_select_gc, wpy, vpy,
       NULL))) {
       Py_XDECREF(wpy); Py_XDECREF(vpy);
       return -1;
   }
#if PY_MAJOR_VERSION >= 3
   if (PyLong_Check(result)) a = (CBLAS_INT) PyLong_AsLong(result);
#else
   if PyInt_Check(result) a = (CBLAS_INT) PyInt_AsLong(result);
#endif
   else
       PyErr_SetString(PyExc_TypeError, "select() must return an integer "
           "argument");
   Py_XDECREF(wpy);  Py_XDECREF(vpy);  Py_XDECREF(result);
   return a;
}

extern CBLAS_INT fselect_gr(double *wr, double *wi, double *v)
{
   PyObject *wpy, *vpy, *result;
   CBLAS_INT a = 0;

   wpy = PyComplex_FromDoubles(*wr, *wi);
   vpy = PyFloat_FromDouble(*v);
   if (!(result = PyObject_CallFunctionObjArgs(py_select_gr, wpy, vpy,
       NULL))) {
       Py_XDECREF(wpy); Py_XDECREF(vpy);
       return -1;
   }
#if PY_MAJOR_VERSION >= 3
   if (PyLong_Check(result)) a = (CBLAS_INT) PyLong_AsLong(result);
#else
   if PyInt_Check(result) a = (CBLAS_INT) PyInt_AsLong(result);
#endif
   else
       PyErr_SetString(PyExc_TypeError, "select() must return an integer "
           "argument");
   Py_XDECREF(wpy);  Py_XDECREF(vpy);  Py_XDECREF(result);
   return a;
}

static PyObject* gges(PyObject *self, PyObject *args, PyObject *kwrds)
{
    PyObject *F=NULL;
    matrix *A, *B, *a=NULL, *b=NULL, *Vsl=NULL, *Vsr=NULL;
    CBLAS_INT n=-1, ldA=0, ldB=0, ldVsl=0, ldVsr=0, oA=0, oB=0, oa=0, ob=0,
        oVsl=0, oVsr=0, info, lwork, sdim, k, *bwork=NULL;
    double *ar=NULL, *ai=NULL, *rwork=NULL;
    complex_t *ac=NULL;
    void *work=NULL, *bc=NULL;
    number wl;

    char *kwlist[] = {"A", "B", "a", "b", "Vl", "Vr", "select", "n",
        "ldA", "ldB", "ldVl", "ldVr", "offsetA", "offsetB", "offseta",
        "offsetb", "offsetVl", "offsetVr", NULL};
    Py_ssize_t _n = n, _ldA = ldA, _ldB = ldB, _ldVsl = ldVsl, _ldVsr = ldVsr, _oA = oA, _oB = oB, _oa = oa, _ob = ob, _oVsl = oVsl, _oVsr = oVsr;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|OOOOOnnnnnnnnnnn",
        kwlist, &A, &B, &a, &b, &Vsl, &Vsr, &F, &_n, &_ldA, &_ldB, &_ldVsl,
        &_ldVsr, &_oA, &_oB, &_oa, &_ob, &_oVsl, &_oVsr)) return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldVsl, &ldVsl) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldVsr, &ldVsr) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oa, &oa) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ob, &ob) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oVsl, &oVsl) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oVsr, &oVsr) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(B) != MAT_ID(A)) err_conflicting_ids;
    if (n < 0){
        n = A->nrows;
        if (n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
        if (n != B->nrows || n != B->ncols){
            PyErr_SetString(PyExc_TypeError, "B must be square and of the "
                "same order as A");
            return NULL;
        }
    }
    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (ldB == 0) ldB = MAX(1, B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (n-1)*ldB + n > len(B)) err_buf_len("B");

    if (a){
        if (!Matrix_Check(a) || MAT_ID(a) != COMPLEX)
            PY_ERR_TYPE("a must be a matrix with typecode 'z'")
        if (oa < 0) err_nn_int("offseta");
        if (oa + n > len(a)) err_buf_len("a");
        if (!b){
            PyErr_SetString(PyExc_ValueError, "'b' must be provided if "
                "'a' is provided");
            return NULL;
        }
    }
    if (b){
        if (!Matrix_Check(b) || MAT_ID(b) != DOUBLE)
            PY_ERR_TYPE("b must be a matrix with typecode 'd'")
        if (ob < 0) err_nn_int("offsetb");
        if (ob + n > len(b)) err_buf_len("b");
        if (!a){
            PyErr_SetString(PyExc_ValueError, "'a' must be provided if "
                "'b' is provided");
            return NULL;
        }
    }

    if (Vsl){
        if (!Matrix_Check(Vsl)) err_mtrx("Vsl");
        if (MAT_ID(Vsl) != MAT_ID(A)) err_conflicting_ids;
        if (ldVsl == 0) ldVsl = MAX(1, Vsl->nrows);
        if (ldVsl < MAX(1,n)) err_ld("ldVsl");
        if (oVsl < 0) err_nn_int("offsetVsl");
        if (oVsl + (n-1)*ldVsl + n > len(Vsl)) err_buf_len("Vsl");
    } else {
        if (ldVsl == 0) ldVsl = 1;
        if (ldVsl < 1) err_ld("ldVsl");
    }

    if (Vsr){
        if (!Matrix_Check(Vsr)) err_mtrx("Vsr");
        if (MAT_ID(Vsr) != MAT_ID(A)) err_conflicting_ids;
        if (ldVsr == 0) ldVsr = MAX(1, Vsr->nrows);
        if (ldVsr < MAX(1,n)) err_ld("ldVsr");
        if (oVsr < 0) err_nn_int("offsetVsr");
        if (oVsr + (n-1)*ldVsr + n > len(Vsr)) err_buf_len("Vsr");
    } else {
        if (ldVsr == 0) ldVsr = 1;
        if (ldVsr < 1) err_ld("ldVsr");
    }

    if (F && !PyFunction_Check(F))
        PY_ERR_TYPE("select must be a Python function")

    switch (MAT_ID(A)){
        case DOUBLE:
            lwork = -1;
            BLAS_FUNC(dgges)(Vsl ? "V" : "N", Vsr ? "V" : "N", F ? "S" : "N", NULL,
                &n, NULL, &ldA, NULL, &ldB, &sdim, NULL, NULL, NULL, NULL,
                &ldVsl, NULL, &ldVsr, &wl.d, &lwork, NULL, &info);
            lwork = (CBLAS_INT) wl.d;
            work = (void *) calloc(lwork, sizeof(double));
            ar = (double *) calloc(n, sizeof(double));
            ai = (double *) calloc(n, sizeof(double));
            if (!b) bc = (double *) calloc(n, sizeof(double));
            if (F) bwork = (CBLAS_INT *) calloc(n, sizeof(CBLAS_INT));
            if (!work || !ar || !ai || (!b && !bc) || (F && !bwork)){
                free(work);  free(ar);  free(ai);  free(b);  free(bwork);
                return PyErr_NoMemory();
            }
            py_select_gr = F;
            BLAS_FUNC(dgges)(Vsl ? "V" : "N", Vsr ? "V" : "N", F ? "S" : "N",
                F ? &fselect_gr : NULL, &n, MAT_BUFD(A) + oA, &ldA,
                MAT_BUFD(B) + oB, &ldB, &sdim, ar, ai,
                b ? MAT_BUFD(b) + ob : (double *) bc,
                Vsl ? MAT_BUFD(Vsl) + oVsl : NULL, &ldVsl,
                Vsr ? MAT_BUFD(Vsr) + oVsr : NULL, &ldVsr,
                (double *) work, &lwork, bwork, &info);
            if (a) for (k=0; k<n; k++)
#ifndef _MSC_VER
                MAT_BUFZ(a)[oa + k] = ar[k] + I * ai[k];
#else
	        MAT_BUFZ(a)[oa + k] = _Cbuild(ar[k],ai[k]);
#endif
            free(work);  free(ar);  free(ai);  free(bc); free(bwork);
            break;

	case COMPLEX:
            lwork = -1;
            BLAS_FUNC(zgges)(Vsl ? "V" : "N", Vsr ? "V" : "N", F ? "S" : "N", NULL,
                &n, NULL, &ldA, NULL, &ldB, &sdim, NULL, NULL, NULL,
                &ldVsl, NULL, &ldVsr, &wl.z, &lwork, NULL, NULL, &info);
            lwork = (CBLAS_INT) creal(wl.z);
            work = (void *) calloc(lwork, sizeof(complex_t));
            rwork = (double *) calloc(8*n, sizeof(double));
            if (F) bwork = (CBLAS_INT *) calloc(n, sizeof(CBLAS_INT));
            if (!a) 
                ac = (complex_t *) calloc(n, sizeof(complex_t));
            bc = (complex_t *) calloc(n, sizeof(complex_t));
	    if (!work || !rwork || (F && !bwork) || (!a && !ac) || !bc){
                free(work);  free(rwork); free(bwork); free(ac); free(bc);
                return PyErr_NoMemory();
            }
            py_select_gc = F;
            BLAS_FUNC(zgges)(Vsl ? "V": "N", Vsr ? "V" : "N", F ? "S" : "N",
                F ? &fselect_gc : NULL, &n, MAT_BUFZ(A) + oA, &ldA,
                MAT_BUFZ(B) + oB, &ldB, &sdim, a ? MAT_BUFZ(a) + oa : ac,
                (complex_t *) bc, 
                Vsl ? MAT_BUFZ(Vsl) + oVsl : NULL, &ldVsl,
                Vsr ? MAT_BUFZ(Vsr) + oVsr : NULL, &ldVsr,
                (complex_t *) work, &lwork, rwork,  bwork, &info);
            if (b) for (k=0; k<n; k++)
                MAT_BUFD(b)[ob + k] = 
                    (double) creal(((complex_t *) bc)[k]);

            free(work);  free(rwork); free(bwork); free(ac);  free(bc);
            break;

        default:
            err_invalid_id;
    }

    if (PyErr_Occurred()) return NULL;

    if (info) err_lapack
    else return Py_BuildValue("n", (Py_ssize_t) (F ? sdim : 0));
}


static char doc_lacpy[] =
    "Copy all or part of a matrix.\n\n"
    "lacpy(A, B, uplo='N', m=A.size[0], n=A.size[1], \n"
    "      ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), offsetA=0, \n"
    "      offsetB=0)\n\n"
    "PURPOSE\n"
    "Copy the m x n matrix A to B.  If uplo is 'U', the upper\n"
    "trapezoidal part of A is copied.  If uplo is 'L', the lower \n"
    "trapezoidal part is copied.  if uplo is 'N', the entire matrix is\n"
    "copied.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'N', 'L' or 'U'\n\n"
    "m         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ldA       positive integer.  ldA >= max(1,m).  If zero, the default\n"
    "          value is used.\n\n"
    "ldB       positive integer.  ldB >= max(1,m).  If zero, the default\n"
    "          value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* lacpy(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B;
    CBLAS_INT m = -1, n = -1, ldA = 0, ldB = 0, oA = 0, oB = 0;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'N';
#endif
    char uplo = 'N';
    char *kwlist[] = {"A", "B", "uplo", "m", "n", "ldA", "ldB", "offsetA",
        "offsetB", NULL};
    Py_ssize_t _m = m, _n = n, _ldA = ldA, _ldB = ldB, _oA = oA, _oB = oB;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Cnnnnnn", kwlist,
        &A, &B, &uplo_, &_m, &_n, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cnnnnnn", kwlist,
        &A, &B, &uplo, &_m, &_n, &_ldA, &_ldB, &_oA, &_oB))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldA, &ldA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldB, &ldB) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oA, &oA) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oB, &oB) < 0)
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;
    if (uplo != 'N' && uplo != 'L' && uplo != 'U')
        err_char("trans", "'N', 'L', 'U'");
    if (m < 0) m = A->nrows;
    if (n < 0) n = A->ncols;
    if (ldA == 0) ldA = MAX(1, A->nrows);
    if (ldA < MAX(1, m)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1, B->nrows);
    if (ldB < MAX(1, m)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (n-1)*ldB + m > len(B)) err_buf_len("B");

    switch (MAT_ID(A)){
        case DOUBLE:
            BLAS_FUNC(dlacpy)(&uplo, &m, &n, MAT_BUFD(A)+oA, &ldA, MAT_BUFD(B)+oB,
                &ldB);
            break;

        case COMPLEX:
            BLAS_FUNC(zlacpy)(&uplo, &m, &n, MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(B)+oB,
                &ldB);
            break;

	default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_larfg[] =
    "Generate an elementary Householder reflector.\n\n"
    "tau = larfg(alpha, x, n=None, offseta=0, offsetx=0)\n\n"
    "PURPOSE\n"
    "Generates a Householder reflector\n\n"
    "    H = I - tau * [1; v] * [1; v]^H\n\n"
    "such that\n\n"
    "    H^H * [alpha; x] = [beta; 0].\n\n"
    "In other words,\n\n"
    "    (I - tau.conjugate() * [1; v] * [1; v]^H) * [alpha; x] = "
    "[beta; 0].\n\n"
    "The matrix H is unitary, so\n\n"
    "    2 * tau.real = abs(tau)**2 * ( 1.0 + ||v||**2).\n\n"
    "On exit x contains the vector v and alpha is overwritten with beta.\n"
    "The parameter tau is returned as the output value of the function."
    "\n\n"
    "ARGUMENTS\n"
    "alpha     'd' or 'z' matrix.  On exit, contains beta.\n\n"
    "x         'd' or 'z' matrix.  Must have the same type as alpha.\n"
    "          On exit, contains v. \n\n"
    "n         positive integer.  The dimension of the vector [alpha; x].\n"
    "          If n <= 0, the default value is used, which is equal to\n"
    "          1 + ( (len(x) - offsetx >= 1) ? len(x) - ox : 0 ).\n\n"
    "offseta   nonnegative integer \n\n"
    "offsetx   nonnegative integer \n\n"
    "tau       scalar of the same type as alpha and x";

static PyObject* larfg(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *a, *x;
    number tau;
    CBLAS_INT n = 0, oa = 0, ox = 0, ix = 1;
    char *kwlist[] = {"alpha", "x", "n", "offseta", "offsetx", NULL};
    Py_ssize_t _n = n, _oa = oa, _ox = ox;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|nnn", kwlist,
        &a, &x, &_n, &_oa, &_ox)) return NULL;

    if (cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oa, &oa) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ox, &ox) < 0)
        return NULL;

    if (!Matrix_Check(a)) err_mtrx("alpha");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (MAT_ID(a) != MAT_ID(x)) err_conflicting_ids;
    if (oa < 0) err_nn_int("offseta");
    if (ox < 0) err_nn_int("offsetx");
    if (n <= 0) n = 1 + ((len(x) >= ox + 1) ? len(x) - ox : 0);
    if (len(x) < ox + n - 1) err_buf_len("x");
    if (len(a) < oa + 1) err_buf_len("alpha");

    switch (MAT_ID(a)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dlarfg)(&n, MAT_BUFD(a)+oa, MAT_BUFD(x)+ox, &ix, &tau.d);
            Py_END_ALLOW_THREADS
            return Py_BuildValue("d", tau.d);
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zlarfg)(&n, MAT_BUFZ(a)+oa, MAT_BUFZ(x)+ox, &ix, &tau.z);
            Py_END_ALLOW_THREADS
            return PyComplex_FromDoubles(creal(tau.z), cimag(tau.z));
            break;

	default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_larfx[] =
    "Apply an elementary Householder reflector to a matrix.\n\n"
    "larfx(v, tau, C, side='L', m=C.size[0], n=C.size[1],\n" 
    "      ldC=max(1,C.size[0]), offsetv=0, offsetC=0)\n\n"
    "PURPOSE\n"
    "Computes H*C (side is 'L') or C*H (side is 'R') where\n\n"
    "    H = I - tau * v * v^H.\n\n"
    "On exit C is overwritten with the result.\n\n"
    "ARGUMENTS\n"
    "v         'd' or 'z' matrix\n\n"
    "tau       number.  Can only be complex if v is complex.\n\n"
    "C         'd' or 'z' matrix of the same type as v\n\n"
    "side      'L' or 'R'\n\n"
    "m         nonnegative integer.  If negative, the default value is \n"
    "          used.\n\n" 
    "n         nonnegative integer.  If negative, the default value is \n"
    "          used.\n\n" 
    "ldC       nonnegative integer.  ldC >= max(1,m).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetv   nonnegative integer \n\n"
    "offsetC   nonnegative integer";

static PyObject* larfx(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *v, *C;
    PyObject *tauo=NULL;
    number tau;
    CBLAS_INT m = -1, n = -1, ov = 0, oC = 0, ldC = 0;
    void *work = NULL;
#if PY_MAJOR_VERSION >= 3
    int side_ = 'L';
#endif
    char side = 'L';
    char *kwlist[] = {"v", "tau", "C", "side", "m", "n", "ldC", "offsetv",
        "offsetC", NULL};
    Py_ssize_t _m = m, _n = n, _ov = ov, _oC = oC, _ldC = ldC;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|Cnnnnn", kwlist,
        &v, &tauo, &C, &side_, &_m, &_n, &_ldC, &_ov, &_oC))
        return NULL;
    side = (char) side_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cnnnnn", kwlist,
        &v, &tauo, &C, &side, &_m, &_n, &_ldC, &_ov, &_oC))
        return NULL;
#endif

    if (cvxopt_cblas_int_from_py_ssize_t(_m, &m) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_n, &n) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ldC, &ldC) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_ov, &ov) < 0 ||
        cvxopt_cblas_int_from_py_ssize_t(_oC, &oC) < 0)
        return NULL;

    if (!Matrix_Check(v)) err_mtrx("v");
    if (!Matrix_Check(C)) err_mtrx("C");
    if (MAT_ID(v) != MAT_ID(C)) err_conflicting_ids;
    if (tauo && number_from_pyobject(tauo, &tau, MAT_ID(v)))
        err_type("tau")

    if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");

    if (m < 0) m = C->nrows;
    if (n < 0) n = C->ncols;

    if (ov < 0) err_nn_int("offsetv");
    if ((side == 'L' && len(v) - ov < m) ||
        (side == 'R' && len(v) - ov < n)) err_buf_len("v")

    if (ldC == 0) ldC = MAX(1, C->nrows);
    if (ldC < MAX(1,m)) err_ld("ldC");
    if (oC < 0) err_nn_int("offsetC");
    if (oC + (n-1)*ldC + m > len(C)) err_buf_len("C");

    switch (MAT_ID(v)){
        case DOUBLE:
            if (!(work = (void *) calloc((side == 'L') ? n : m,
                sizeof(double))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(dlarfx)(&side, &m, &n, MAT_BUFD(v)+ov,
                &tau.d,
                MAT_BUFD(C) + oC, &ldC, (double *) work);
            Py_END_ALLOW_THREADS
            free(work);
            break;

        case COMPLEX:
            if (!(work = (void *) calloc((side == 'L') ? n : m,
                sizeof(complex_t))))
                return PyErr_NoMemory();
            Py_BEGIN_ALLOW_THREADS
            BLAS_FUNC(zlarfx)(&side, &m, &n, MAT_BUFZ(v)+ov,
                &tau.z,
                MAT_BUFZ(C) + oC, &ldC, (complex_t *) work);
            Py_END_ALLOW_THREADS
            free(work);
            break;

	default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}



static PyMethodDef lapack_functions[] = {
{"getrf", (PyCFunction) getrf, METH_VARARGS|METH_KEYWORDS, doc_getrf},
{"getrs", (PyCFunction) getrs, METH_VARARGS|METH_KEYWORDS, doc_getrs},
{"getri", (PyCFunction) getri, METH_VARARGS|METH_KEYWORDS, doc_getri},
{"gesv",  (PyCFunction) gesv,  METH_VARARGS|METH_KEYWORDS, doc_gesv},
{"gbtrf", (PyCFunction) gbtrf, METH_VARARGS|METH_KEYWORDS, doc_gbtrf},
{"gbtrs", (PyCFunction) gbtrs, METH_VARARGS|METH_KEYWORDS, doc_gbtrs},
{"gbsv",  (PyCFunction) gbsv,  METH_VARARGS|METH_KEYWORDS, doc_gbsv},
{"gttrf", (PyCFunction) gttrf, METH_VARARGS|METH_KEYWORDS, doc_gttrf},
{"gttrs", (PyCFunction) gttrs, METH_VARARGS|METH_KEYWORDS, doc_gttrs},
{"gtsv",  (PyCFunction) gtsv,  METH_VARARGS|METH_KEYWORDS, doc_gtsv},
{"potrf", (PyCFunction) potrf, METH_VARARGS|METH_KEYWORDS, doc_potrf},
{"potrs", (PyCFunction) potrs, METH_VARARGS|METH_KEYWORDS, doc_potrs},
{"potri", (PyCFunction) potri, METH_VARARGS|METH_KEYWORDS, doc_potri},
{"posv",  (PyCFunction) posv,  METH_VARARGS|METH_KEYWORDS, doc_posv},
{"pbtrf", (PyCFunction) pbtrf, METH_VARARGS|METH_KEYWORDS, doc_pbtrf},
{"pbtrs", (PyCFunction) pbtrs, METH_VARARGS|METH_KEYWORDS, doc_pbtrs},
{"pbsv",  (PyCFunction) pbsv,  METH_VARARGS|METH_KEYWORDS, doc_pbsv},
{"pttrf", (PyCFunction) pttrf, METH_VARARGS|METH_KEYWORDS, doc_pttrf},
{"pttrs", (PyCFunction) pttrs, METH_VARARGS|METH_KEYWORDS, doc_pttrs},
{"ptsv",  (PyCFunction) ptsv,  METH_VARARGS|METH_KEYWORDS, doc_ptsv},
{"sytrs", (PyCFunction) sytrs, METH_VARARGS|METH_KEYWORDS, doc_sytrs},
{"hetrs", (PyCFunction) hetrs, METH_VARARGS|METH_KEYWORDS, doc_hetrs},
{"sytrf", (PyCFunction) sytrf, METH_VARARGS|METH_KEYWORDS, doc_sytrf},
{"hetrf", (PyCFunction) hetrf, METH_VARARGS|METH_KEYWORDS, doc_hetrf},
{"sytri", (PyCFunction) sytri, METH_VARARGS|METH_KEYWORDS, doc_sytri},
{"sysv",  (PyCFunction) sysv,  METH_VARARGS|METH_KEYWORDS, doc_sysv},
{"hetri", (PyCFunction) hetri, METH_VARARGS|METH_KEYWORDS, doc_hetri},
{"hesv",  (PyCFunction) hesv,  METH_VARARGS|METH_KEYWORDS, doc_hesv},
{"trtrs", (PyCFunction) trtrs, METH_VARARGS|METH_KEYWORDS, doc_trtrs},
{"trtri", (PyCFunction) trtri, METH_VARARGS|METH_KEYWORDS, doc_trtri},
{"tbtrs", (PyCFunction) tbtrs, METH_VARARGS|METH_KEYWORDS, doc_tbtrs},
{"gels",  (PyCFunction) gels,  METH_VARARGS|METH_KEYWORDS, doc_gels},
{"geqrf", (PyCFunction) geqrf, METH_VARARGS|METH_KEYWORDS, doc_geqrf},
{"ormqr", (PyCFunction) ormqr, METH_VARARGS|METH_KEYWORDS, doc_ormqr},
{"unmqr", (PyCFunction) unmqr, METH_VARARGS|METH_KEYWORDS, doc_unmqr},
{"orgqr", (PyCFunction) orgqr, METH_VARARGS|METH_KEYWORDS, doc_orgqr},
{"ungqr", (PyCFunction) ungqr, METH_VARARGS|METH_KEYWORDS, doc_ungqr},
{"gelqf", (PyCFunction) gelqf, METH_VARARGS|METH_KEYWORDS, doc_gelqf},
{"ormlq", (PyCFunction) ormlq, METH_VARARGS|METH_KEYWORDS, doc_ormlq},
{"unmlq", (PyCFunction) unmlq, METH_VARARGS|METH_KEYWORDS, doc_unmlq},
{"orglq", (PyCFunction) orglq, METH_VARARGS|METH_KEYWORDS, doc_orglq},
{"unglq", (PyCFunction) unglq, METH_VARARGS|METH_KEYWORDS, doc_unglq},
{"syev",  (PyCFunction) syev,  METH_VARARGS|METH_KEYWORDS, doc_syev},
{"heev",  (PyCFunction) heev,  METH_VARARGS|METH_KEYWORDS, doc_heev},
{"syevx", (PyCFunction) syevx, METH_VARARGS|METH_KEYWORDS, doc_syevx},
{"heevx", (PyCFunction) heevx, METH_VARARGS|METH_KEYWORDS, doc_heevx},
{"sygv",  (PyCFunction) sygv,  METH_VARARGS|METH_KEYWORDS, doc_sygv},
{"hegv",  (PyCFunction) hegv,  METH_VARARGS|METH_KEYWORDS, doc_hegv},
{"syevd", (PyCFunction) syevd, METH_VARARGS|METH_KEYWORDS, doc_syevd},
{"heevd", (PyCFunction) heevd, METH_VARARGS|METH_KEYWORDS, doc_heevd},
{"syevr", (PyCFunction) syevr, METH_VARARGS|METH_KEYWORDS, doc_syevr},
{"heevr", (PyCFunction) heevr, METH_VARARGS|METH_KEYWORDS, doc_heevr},
{"gesvd", (PyCFunction) gesvd, METH_VARARGS|METH_KEYWORDS, doc_gesvd},
{"gesdd", (PyCFunction) gesdd, METH_VARARGS|METH_KEYWORDS, doc_gesdd},
{"gees", (PyCFunction) gees, METH_VARARGS|METH_KEYWORDS, doc_gees},
{"gges", (PyCFunction) gges, METH_VARARGS|METH_KEYWORDS, doc_gges},
{"lacpy", (PyCFunction) lacpy, METH_VARARGS|METH_KEYWORDS, doc_lacpy},
{"geqp3", (PyCFunction) geqp3, METH_VARARGS|METH_KEYWORDS, doc_geqp3},
{"larfg", (PyCFunction) larfg, METH_VARARGS|METH_KEYWORDS, doc_larfg},
{"larfx", (PyCFunction) larfx, METH_VARARGS|METH_KEYWORDS, doc_larfx},
{NULL}  /* Sentinel */
};


#if PY_MAJOR_VERSION >= 3

static PyModuleDef lapack_module = {
    PyModuleDef_HEAD_INIT,
    "lapack",
    lapack__doc__,
    -1,
    lapack_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_lapack(void)
{
    PyObject *m;
    if (!(m = PyModule_Create(&lapack_module))) return NULL;
    if (import_cvxopt() < 0) return NULL;
    return m;
}

#else

PyMODINIT_FUNC initlapack(void)
{
    PyObject *m;
    m = Py_InitModule3("cvxopt.lapack", lapack_functions, lapack__doc__);
    if (import_cvxopt() < 0) return;
}

#endif
