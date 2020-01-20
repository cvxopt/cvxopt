/*
 * Copyright 2012-2020 M. Andersen and L. Vandenberghe.
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
#include "misc.h"

#ifndef _MSC_VER
typedef complex double complex_t;
#else
typedef _Dcomplex complex_t;
#endif

#define USE_CBLAS_ZDOT 0

PyDoc_STRVAR(blas__doc__,"Interface to the double-precision real and "
    "complex BLAS.\n\n"
    "Double and complex matrices and vectors are stored in CVXOPT \n"
    "matrices using the conventional BLAS storage schemes, with the\n"
    "CVXOPT matrix buffers interpreted as one-dimensional arrays.\n"
    "For each matrix argument X, an additional integer argument\n"
    "offsetX specifies the start of the array, i.e., the pointer\n"
    "X->buffer + offsetX is passed to the BLAS function.  The other \n"
    "arguments (dimensions and options) have the same meaning as in\n"
    "the BLAS definition.  Default values of the dimension arguments\n"
    "are derived from the CVXOPT matrix sizes.");


/* BLAS 1 prototypes */
extern void dswap_(int *n, double *x, int *incx, double *y, int *incy);
extern void zswap_(int *n, complex_t *x, int *incx, complex_t *y,
    int *incy);
extern void dscal_(int *n, double *alpha, double *x, int *incx);
extern void zscal_(int *n, complex_t *alpha, complex_t *x, int *incx);
extern void zdscal_(int *n, double *alpha, complex_t *x, int *incx);
extern void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
extern void zcopy_(int *n, complex_t *x, int *incx, complex_t *y,
    int *incy);
extern void daxpy_(int *n, double *alpha, double *x, int *incx,
    double *y, int *incy);
extern void zaxpy_(int *n, complex_t *alpha, complex_t *x, int *incx,
    complex_t *y, int *incy);
extern double ddot_(int *n, double *x, int *incx, double *y, int *incy);
#if USE_CBLAS_ZDOT
extern void cblas_zdotc_sub(int n, void *x, int incx, void *y,
    int incy, void *result);
extern void cblas_zdotu_sub(int n, void *x, int incx, void *y, int incy,
    void *result);
#endif
extern double dnrm2_(int *n, double *x, int *incx);
extern double dznrm2_(int *n, complex_t *x, int *incx);
extern double dasum_(int *n, double *x, int *incx);
extern double dzasum_(int *n, complex_t *x, int *incx);
extern int idamax_(int *n, double *x, int *incx);
extern int izamax_(int *n, complex_t *x, int *incx);


/* BLAS 2 prototypes */
extern void dgemv_(char* trans, int *m, int *n, double *alpha,
    double *A, int *lda, double *x, int *incx, double *beta, double *y,
    int *incy);
extern void zgemv_(char* trans, int *m, int *n, complex_t *alpha,
    complex_t *A, int *lda, complex_t *x, int *incx, complex_t *beta,
    complex_t *y, int *incy);
extern void dgbmv_(char* trans, int *m, int *n, int *kl, int *ku,
    double *alpha, double *A, int *lda, double *x, int *incx,
    double *beta, double *y,  int *incy);
extern void zgbmv_(char* trans, int *m, int *n, int *kl, int *ku,
    complex_t *alpha, complex_t *A, int *lda, complex_t *x, int *incx,
    complex_t *beta, complex_t *y,  int *incy);
extern void dsymv_(char *uplo, int *n, double *alpha, double *A,
    int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void zhemv_(char *uplo, int *n, complex_t *alpha, complex_t *A,
    int *lda, complex_t *x, int *incx, complex_t *beta, complex_t *y,
    int *incy);
extern void dsbmv_(char *uplo, int *n, int *k, double *alpha, double *A,
    int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void zhbmv_(char *uplo, int *n, int *k, complex_t *alpha,
    complex_t *A, int *lda, complex_t *x, int *incx, complex_t *beta,
    complex_t *y, int *incy);
extern void dtrmv_(char *uplo, char *trans, char *diag, int *n,
    double *A, int *lda, double *x, int *incx);
extern void ztrmv_(char *uplo, char *trans, char *diag, int *n,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dtbmv_(char *uplo, char *trans, char *diag, int *n, int *k,
    double *A, int *lda, double *x, int *incx);
extern void ztbmv_(char *uplo, char *trans, char *diag, int *n, int *k,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dtrsv_(char *uplo, char *trans, char *diag, int *n,
    double *A, int *lda, double *x, int *incx);
extern void ztrsv_(char *uplo, char *trans, char *diag, int *n,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dtbsv_(char *uplo, char *trans, char *diag, int *n, int *k,
    double *A, int *lda, double *x, int *incx);
extern void ztbsv_(char *uplo, char *trans, char *diag, int *n, int *k,
    complex_t *A, int *lda, complex_t *x, int *incx);
extern void dger_(int *m, int *n, double *alpha, double *x, int *incx,
    double *y, int *incy, double *A, int *lda);
extern void zgerc_(int *m, int *n, complex_t *alpha, complex_t *x,
    int *incx, complex_t *y, int *incy, complex_t *A, int *lda);
extern void zgeru_(int *m, int *n, complex_t *alpha, complex_t *x,
    int *incx, complex_t *y, int *incy, complex_t *A, int *lda);
extern void dsyr_(char *uplo, int *n, double *alpha, double *x,
    int *incx, double *A, int *lda);
extern void zher_(char *uplo, int *n, double *alpha, complex_t *x,
    int *incx, complex_t *A, int *lda);
extern void dsyr2_(char *uplo, int *n, double *alpha, double *x,
    int *incx, double *y, int *incy, double *A, int *lda);
extern void zher2_(char *uplo, int *n, complex_t *alpha, complex_t *x,
    int *incx, complex_t *y, int *incy, complex_t *A, int *lda);


/* BLAS 3 prototypes */
extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);
extern void zgemm_(char *transa, char *transb, int *m, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void dsymm_(char *side, char *uplo, int *m, int *n,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);
extern void zsymm_(char *side, char *uplo, int *m, int *n,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void zhemm_(char *side, char *uplo, int *m, int *n,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void dsyrk_(char *uplo, char *trans, int *n, int *k,
    double *alpha, double *A, int *lda, double *beta, double *B,
    int *ldb);
extern void zsyrk_(char *uplo, char *trans, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *beta, complex_t *B,
    int *ldb);
extern void zherk_(char *uplo, char *trans, int *n, int *k,
    double *alpha, complex_t *A, int *lda, double *beta, complex_t *B,
    int *ldb);
extern void dsyr2k_(char *uplo, char *trans, int *n, int *k,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);
extern void zsyr2k_(char *uplo, char *trans, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    complex_t *beta, complex_t *C, int *ldc);
extern void zher2k_(char *uplo, char *trans, int *n, int *k,
    complex_t *alpha, complex_t *A, int *lda, complex_t *B, int *ldb,
    double *beta, complex_t *C, int *ldc);
extern void dtrmm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, double *alpha, double *A, int *lda, double *B,
    int *ldb);
extern void ztrmm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, complex_t *alpha, complex_t *A, int *lda, complex_t *B,
    int *ldb);
extern void dtrsm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, double *alpha, double *A, int *lda, double *B,
    int *ldb);
extern void ztrsm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, complex_t *alpha, complex_t *A, int *lda, complex_t *B,
    int *ldb);


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


static char doc_swap[] =
    "Interchanges two vectors (x <-> y).\n\n"
    "swap(x, y, n=None, incx=1, incy=1, offsetx=0, offsety=0)\n\n"
    "ARGUMENTS\n"
    "x         'd' or 'z' matrix\n\n"
    "y         'd' or 'z' matrix.  Must have the same type as x.\n\n"
    "n         integer.  If n<0, the default value of n is used.\n"
    "          The default value is equal to\n"
    "          len(x)>=offsetx+1 ? 1+(len(x)-offsetx-1)/|incx| : 0.\n"
    "          If the default value is used, it must be equal to\n"
    "          len(y)>=offsety+1 ? 1+(len(y)-offsetx-1)/|incy| : 0.\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer";

static PyObject* swap(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x, *y;
    int n=-1, ix=1, iy=1, ox=0, oy=0;
    char *kwlist[] = {"x", "y", "n", "incx", "incy", "offsetx",
        "offsety", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|iiiii", kwlist,
        &x, &y, &n, &ix, &iy, &ox, &oy)) return NULL;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (ox < 0) err_nn_int("offsetx");
    if (oy < 0) err_nn_int("offsety");

    if (n<0){
        n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/abs(ix) : 0;
        if (n != ((len(y) >= oy+1) ? 1+(len(y)-oy-1)/abs(iy) : 0)){
            PyErr_SetString(PyExc_ValueError, "arrays have unequal "
                "default lengths");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("");

    if (len(x) < ox+1+(n-1)*abs(ix)) err_buf_len("x");
    if (len(y) < oy+1+(n-1)*abs(iy)) err_buf_len("y");

    switch (MAT_ID(x)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            dswap_(&n, MAT_BUFD(x)+ox, &ix, MAT_BUFD(y)+oy, &iy);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            zswap_(&n, MAT_BUFZ(x)+ox, &ix, MAT_BUFZ(y)+oy, &iy);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_scal[] =
    "Scales a vector by a constant (x := alpha*x).\n\n"
    "scal(alpha, x, n=None, inc=1, offset=0)\n\n"
    "ARGUMENTS\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if x is complex.\n\n"
    "x         'd' or 'z' matrix\n\n"
    "n         integer.  If n<0, the default value of n is used.\n"
    "          The default value is equal to\n"
    "          (len(x)>=offset+1) ? 1+(len-offset-1)/inc : 0.\n\n"
    "inc       positive integer\n\n"
    "offset    nonnegative integer";

static PyObject* scal(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x;
    PyObject *ao;
    number a;
    int n=-1, ix=1, ox=0;
    char *kwlist[] = {"alpha", "x", "n", "inc", "offset", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|iii", kwlist,
        &ao, &x, &n, &ix, &ox)) return NULL;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (ix <= 0) err_p_int("inc");
    if (ox < 0) err_nn_int("offset");
    if (n < 0) n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/ix : 0;
    if (n == 0) return Py_BuildValue("");
    if (len(x) < ox+1+(n-1)*ix) err_buf_len("x");

    switch (MAT_ID(x)){
        case DOUBLE:
            if (number_from_pyobject(ao, &a, MAT_ID(x)))
                err_type("alpha");
            Py_BEGIN_ALLOW_THREADS
            dscal_(&n, &a.d, MAT_BUFD(x)+ox, &ix);
            Py_END_ALLOW_THREADS
	    break;

        case COMPLEX:
            if (!number_from_pyobject(ao, &a, DOUBLE))
                Py_BEGIN_ALLOW_THREADS
                zdscal_(&n, &a.d, MAT_BUFZ(x)+ox, &ix);
                Py_END_ALLOW_THREADS
            else if (!number_from_pyobject(ao, &a, COMPLEX))
                Py_BEGIN_ALLOW_THREADS
                zscal_(&n, &a.z, MAT_BUFZ(x)+ox, &ix);
                Py_END_ALLOW_THREADS
            else
                err_type("alpha");
	    break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_copy[] =
    "Copies a vector x to a vector y (y := x).\n\n"
    "copy(x, y, n=None, incx=1, incy=1, offsetx=0, offsety=0)\n\n"
    "ARGUMENTS\n"
    "x         'd' or 'z' matrix\n\n"
    "y         'd' or 'z' matrix.  Must have the same type as x.\n\n"
    "n         integer.  If n<0, the default value of n is used.\n"
    "          The default value is given by\n"
    "          (len(x)>=offsetx+1) ? 1+(len(x)-offsetx-1)/incx : 0\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer";

static PyObject* copy(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x, *y;
    int n=-1, ix=1, iy=1, ox=0, oy=0;
    char *kwlist[] = {"x", "y", "n", "incx", "incy", "offsetx",
        "offsety", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|iiiii", kwlist,
        &x, &y, &n, &ix, &iy, &ox, &oy))
        return NULL;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (ox < 0 ) err_nn_int("offsetx");
    if (oy < 0 ) err_nn_int("offsety");

    if (n < 0) n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/abs(ix) : 0;
    if (n == 0) return Py_BuildValue("");

    if (len(x) < ox+1+(n-1)*abs(ix)) err_buf_len("x");
    if (len(y) < oy+1+(n-1)*abs(iy)) err_buf_len("y");

    switch (MAT_ID(x)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            dcopy_(&n, MAT_BUFD(x)+ox, &ix, MAT_BUFD(y)+oy, &iy);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            zcopy_(&n, MAT_BUFZ(x)+ox, &ix, MAT_BUFZ(y)+oy, &iy);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_axpy[] =
    "Constant times a vector plus a vector (y := alpha*x+y).\n\n"
    "axpy(x, y, alpha=1.0, n=None, incx=1, incy=1, offsetx=0, "
    "offsety=0)\n\n"
    "ARGUMENTS\n"
    "x         'd' or 'z' matrix\n\n"
    "y         'd' or 'z' matrix.  Must have the same type as x.\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if x is complex.\n\n"
    "n         integer.  If n<0, the default value of n is used.\n"
    "          The default value is equal to\n"
    "          (len(x)>=offsetx+1) ? 1+(len(x)-offsetx-1)/incx : 0.\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer";

static PyObject* axpy(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x, *y;
    PyObject *ao=NULL;
    number a;
    int n=-1, ix=1, iy=1, ox=0, oy=0;
    char *kwlist[] = {"x", "y", "alpha", "n", "incx", "incy", "offsetx",
        "offsety", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Oiiiii", kwlist,
        &x, &y, &ao, &n, &ix, &iy, &ox, &oy)) return NULL;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (ox < 0) err_nn_int("offsetx");
    if (oy < 0) err_nn_int("offsety");

    if (n < 0) n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/abs(ix) : 0;
    if (n == 0) return Py_BuildValue("");

    if (len(x) < ox + 1+(n-1)*abs(ix)) err_buf_len("x");
    if (len(y) < oy + 1+(n-1)*abs(iy)) err_buf_len("y");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(x)))
        err_type("alpha");

    switch (MAT_ID(x)){
        case DOUBLE:
            if (!ao) a.d=1.0;
            Py_BEGIN_ALLOW_THREADS
            daxpy_(&n, &a.d, MAT_BUFD(x)+ox, &ix, MAT_BUFD(y)+oy, &iy);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!ao) a.z=1.0;
#else
            if (!ao) a.z=_Cbuild(1.0,0.0);
#endif
            Py_BEGIN_ALLOW_THREADS
            zaxpy_(&n, &a.z, MAT_BUFZ(x)+ox, &ix, MAT_BUFZ(y)+oy, &iy);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_dot[] =
    "Returns x^H*y for real or complex x, y.\n\n"
    "dot(x, y, n=None, incx=1, incy=1, offsetx=0, offsety=0)\n\n"
    "ARGUMENTS\n"
    "x         'd' or 'z' matrix\n\n"
    "y         'd' or 'z' matrix.  Must have the same type as x.\n\n"
    "n         integer.  If n<0, the default value of n is used.\n"
    "          The default value is equal to\n"
    "          (len(x)>=offsetx+1) ? 1+(len(x)-offsetx-1)/incx : 0.\n"
    "          If the default value is used, it must be equal to\n"
    "          len(y)>=offsety+1 ? 1+(len(y)-offsetx-1)/|incy| : 0.\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer\n\n"
    "Returns 0 if n=0.";

static PyObject* dot(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x, *y;
    number val;
    int n=-1, ix=1, iy=1, ox=0, oy=0;
    char *kwlist[] = {"x", "y", "n", "incx", "incy", "offsetx",
        "offsety", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|iiiii", kwlist,
        &x, &y, &n, &ix, &iy, &ox, &oy))
        return NULL;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (ox < 0) err_nn_int("offsetx");
    if (oy < 0) err_nn_int("offsety");

    if (n<0){
        n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/abs(ix) : 0;
        if (n != ((len(y) >= oy+1) ? 1+(len(y)-oy-1)/abs(iy) : 0)){
            PyErr_SetString(PyExc_ValueError, "arrays have unequal "
                "default lengths");
            return NULL;
        }
    }

    if (n && len(x) < ox + 1 + (n-1)*abs(ix)) err_buf_len("x");
    if (n && len(y) < oy + 1 + (n-1)*abs(iy)) err_buf_len("y");

    switch (MAT_ID(x)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            val.d = (n==0) ? 0.0 : ddot_(&n, MAT_BUFD(x)+ox, &ix,
                MAT_BUFD(y)+oy, &iy);
            Py_END_ALLOW_THREADS
            return Py_BuildValue("d", val.d);

        case COMPLEX:
#ifndef _MSC_VER
	        if (n==0) val.z = 0.0;
#else
	        if (n==0) val.z = _Cbuild(0.0,0.0);
#endif
	        else
#if USE_CBLAS_ZDOT
                cblas_zdotc_sub(n, MAT_BUFZ(x)+ox, ix, MAT_BUFZ(y)+oy,
                    iy, &val.z);
#else
                ix *= 2;
                iy *= 2;
                Py_BEGIN_ALLOW_THREADS
#ifndef _MSC_VER
                val.z = (ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
                    MAT_BUFD(y)+2*oy, &iy) +
                    ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
                    MAT_BUFD(y)+2*oy + 1, &iy)) +
                    I*(ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
                    MAT_BUFD(y)+2*oy + 1, &iy) -
                    ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
                    MAT_BUFD(y)+2*oy, &iy));
#else
                val.z = _Cbuild(ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
				      MAT_BUFD(y)+2*oy, &iy) +
				ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
				      MAT_BUFD(y)+2*oy + 1, &iy),
				ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
				      MAT_BUFD(y)+2*oy + 1, &iy) -
				ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
				      MAT_BUFD(y)+2*oy, &iy));
#endif
                Py_END_ALLOW_THREADS
#endif
	    return PyComplex_FromDoubles(creal(val.z),cimag(val.z));

        default:
            err_invalid_id;
    }
}


static char doc_dotu[] =
    "Returns x^T*y for real or complex x, y.\n\n"
    "dotu(x, y, n=None, incx=1, incy=1, offsetx=0, offsety=0)\n\n"
    "ARGUMENTS\n"
    "x         'd' or 'z' matrix\n\n"
    "y         'd' or 'z' matrix.  Must have the same type as x.\n\n"
    "n         integer.  If n<0, the default value of n is used.\n"
    "          The default value is equal to\n"
    "          (len(x)>=offsetx+1) ? 1+(len(x)-offsetx-1)/incx : 0.\n"
    "          If the default value is used, it must be equal to\n"
    "          len(y)>=offsety+1 ? 1+(len(y)-offsetx-1)/|incy| : 0.\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer\n\n"
    "Returns 0 if n=0.";

static PyObject* dotu(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x, *y;
    number val;
    int n=-1, ix=1, iy=1, ox=0, oy=0;
    char *kwlist[] = {"x", "y", "n", "incx", "incy", "offsetx",
        "offsety", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|iiiii", kwlist,
        &x, &y, &n, &ix, &iy, &ox, &oy))
        return NULL;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (ox < 0) err_nn_int("offsetx");
    if (oy < 0) err_nn_int("offsety");

    if (n<0){
        n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/abs(ix) : 0;
        if (n != ((len(y) >= oy+1) ? 1+(len(y)-oy-1)/abs(iy) : 0)){
            PyErr_SetString(PyExc_ValueError, "arrays have unequal "
                "default lengths");
            return NULL;
        }
    }

    if (n && len(x) < ox + 1 + (n-1)*abs(ix)) err_buf_len("x");
    if (n && len(y) < oy + 1 + (n-1)*abs(iy)) err_buf_len("y");

    switch (MAT_ID(x)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            val.d = (n==0) ? 0.0 : ddot_(&n, MAT_BUFD(x)+ox, &ix,
                MAT_BUFD(y)+oy, &iy);
            Py_END_ALLOW_THREADS
            return Py_BuildValue("d", val.d);

        case COMPLEX:
#ifndef _MSC_VER
	        if (n==0) val.z = 0.0;
#else
	        if (n==0) val.z = _Cbuild(0.0,0.0);
#endif
	        else
#if USE_CBLAS_ZDOT
                Py_BEGIN_ALLOW_THREADS
                cblas_zdotu_sub(n, MAT_BUFZ(x)+ox, ix, MAT_BUFZ(y)+oy,
                    iy, &val.z);
                Py_END_ALLOW_THREADS
#else
                ix *= 2;
                iy *= 2;
                Py_BEGIN_ALLOW_THREADS
#ifndef _MSC_VER
                val.z = (ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
                    MAT_BUFD(y)+2*oy, &iy) -
                    ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
                    MAT_BUFD(y)+2*oy + 1, &iy)) +
                    I*(ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
                    MAT_BUFD(y)+2*oy + 1, &iy) +
                    ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
                    MAT_BUFD(y)+2*oy, &iy));
#else
                val.z = _Cbuild(ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
				      MAT_BUFD(y)+2*oy, &iy) -
				ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
				      MAT_BUFD(y)+2*oy + 1, &iy),
				ddot_(&n, MAT_BUFD(x)+2*ox, &ix,
				      MAT_BUFD(y)+2*oy + 1, &iy) +
				ddot_(&n, MAT_BUFD(x)+2*ox + 1, &ix,
				      MAT_BUFD(y)+2*oy, &iy));
#endif
                Py_END_ALLOW_THREADS
#endif
	    return PyComplex_FromDoubles(creal(val.z),cimag(val.z));

        default:
            err_invalid_id;
    }
}


static char doc_nrm2[] =
    "Returns the Euclidean norm of a vector (returns ||x||_2).\n\n"
    "nrm2(x, n=None, inc=1, offset=0)\n\n"
    "ARGUMENTS\n"
    "x         'd' or 'z' matrix\n\n"
    "n         integer.  If n<0, the default value of n is used.\n"
    "          The default value is equal to\n"
    "          (len(x)>=offsetx+1) ? 1+(len(x)-offsetx-1)/incx : 0.\n\n"
    "inc       positive integer\n\n"
    "offset    nonnegative integer\n\n"
    "Returns 0 if n=0.";

static PyObject* nrm2(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x;
    int n=-1, ix=1, ox=0;
    char *kwlist[] = {"x", "n", "inc", "offset", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|iii", kwlist, &x,
        &n, &ix, &ox)) return NULL;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (ix <= 0) err_p_int("incx");
    if (ox < 0) err_nn_int("offsetx");
    if (n < 0) n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/ix : 0;
    if (n == 0) return Py_BuildValue("d", 0.0);
    if (len(x) < ox + 1+(n-1)*ix) err_buf_len("x");

    switch (MAT_ID(x)){
        case DOUBLE:
            return Py_BuildValue("d", dnrm2_(&n, MAT_BUFD(x)+ox, &ix));

        case COMPLEX:
            return Py_BuildValue("d", dznrm2_(&n, MAT_BUFZ(x)+ox, &ix));

        default:
            err_invalid_id;
    }
}


static char doc_asum[] =
    "Returns ||Re x||_1 + ||Im x||_1.\n\n"
    "asum(x, n=None, inc=1, offset=0)\n\n"
    "ARGUMENTS\n"
    "x         'd' or 'z' matrix\n\n"
    "n         integer.  If n<0, the default value of n is used.\n"
    "          The default value is equal to\n"
    "          n = (len(x)>=offset+1) ? 1+(len(x)-offset-1)/inc : 0.\n"
    "\n"
    "inc       positive integer\n\n"
    "offset    nonnegative integer\n\n"
    "Returns 0 if n=0.";

static PyObject* asum(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x;
    int n=-1, ix=1, ox=0;
    char *kwlist[] = {"x", "n", "inc", "offset", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|iii", kwlist,
        &x, &n, &ix, &ox))
        return NULL;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (ix <= 0) err_p_int("inc");
    if (ox < 0) err_nn_int("offset");
    if (n < 0) n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/ix : 0;
    if (n == 0) return Py_BuildValue("d", 0.0);
    if (len(x) < ox + 1+(n-1)*ix) err_buf_len("x");

    double val;
    switch (MAT_ID(x)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
	    val = dasum_(&n, MAT_BUFD(x)+ox, &ix);
            Py_END_ALLOW_THREADS
            return Py_BuildValue("d", val);

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
	    val = dzasum_(&n, MAT_BUFZ(x)+ox, &ix);
            Py_END_ALLOW_THREADS
            return Py_BuildValue("d", val);

        default:
            err_invalid_id;
    }
}


static char doc_iamax[] =
    "Returns the index (in {0,...,n-1}) of the coefficient with \n"
    "maximum value of |Re x_k| + |Im x_k|.\n\n"
    "iamax(x, n=None, inc=1, offset=0)\n\n"
    "ARGUMENTS\n"
    "x         'd' or 'z' matrix\n\n"
    "n         integer.  If n<0, the default value of n is used.\n"
    "          The default value is equal to\n"
    "          (len(x)>=offset+1) ? 1+(len(x)-offset-1)/inc : 0.\n\n"
    "inc       positive integer\n\n"
    "offset    nonnegative integer\n\n"
    "In the case of ties, the index of the first maximizer is \n"
    "returned.  If n=0, iamax returns 0.";

static PyObject* iamax(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x;
    int n=-1, ix=1, ox=0;
    char *kwlist[] = {"x", "n", "inc", "offset", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|iii", kwlist,
        &x, &n, &ix, &ox))
        return NULL;

    if (!Matrix_Check(x)) err_mtrx("x");
    if (ix <= 0) err_p_int("inc");
    if (ox < 0) err_nn_int("offset");
    if (n < 0) n = (len(x) >= ox+1) ? 1+(len(x)-ox-1)/ix : 0;
    if (n == 0) return Py_BuildValue("i", 0);
    if (len(x) < ox + 1+(n-1)*ix) err_buf_len("x");

#if PY_MAJOR_VERSION >= 3
    double val;
#endif
    switch (MAT_ID(x)){
        case DOUBLE:
#if PY_MAJOR_VERSION >= 3
            Py_BEGIN_ALLOW_THREADS
            val = idamax_(&n, MAT_BUFD(x)+ox, &ix)-1;
            Py_END_ALLOW_THREADS
            return Py_BuildValue("i", val);
#else
            return Py_BuildValue("i", idamax_(&n, MAT_BUFD(x)+ox, &ix)-1);
#endif

        case COMPLEX:
#if PY_MAJOR_VERSION >= 3
            Py_BEGIN_ALLOW_THREADS
            val = izamax_(&n, MAT_BUFZ(x)+ox, &ix)-1;
            Py_END_ALLOW_THREADS
            return Py_BuildValue("i", val);
#else
            return Py_BuildValue("i", izamax_(&n, MAT_BUFZ(x)+ox, &ix)-1);
#endif

        default:
            err_invalid_id;
    }
}


static char doc_gemv[] =
    "General matrix-vector product. \n\n"
    "gemv(A, x, y, trans='N', alpha=1.0, beta=0.0, m=A.size[0],\n"
    "     n=A.size[1], ldA=max(1,A.size[0]), incx=1, incy=1, \n"
    "     offsetA=0, offsetx=0, offsety=0)\n\n"
    "PURPOSE\n"
    "If trans is 'N', computes y := alpha*A*x + beta*y.\n"
    "If trans is 'T', computes y := alpha*A^T*x + beta*y.\n"
    "If trans is 'C', computes y := alpha*A^H*x + beta*y.\n"
    "The matrix A is m by n.\n"
    "Returns immediately if n=0 and trans is 'T' or 'C', or if m=0 \n"
    "and trans is 'N'.\n"
    "Computes y := beta*y if n=0, m>0 and trans is 'N', or if m=0, \n"
    "n>0 and trans is 'T' or 'C'.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "x         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "y         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "trans     'N', 'T' or 'C'\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "beta      number (int, float or complex).  Complex beta is only\n"
    "          allowed if A is complex.\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,m).\n"
    "          If zero, the default value is used.\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer";

static PyObject* gemv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x, *y;
    PyObject *ao=NULL, *bo=NULL;
    number a, b;
    int m=-1, n=-1, ldA=0, ix=1, iy=1, oA=0, ox=0, oy=0;
#if PY_MAJOR_VERSION >= 3
    int trans_ = 'N';
#endif
    char trans='N';
    char *kwlist[] = {"A", "x", "y", "trans", "alpha", "beta", "m", "n",
        "ldA", "incx", "incy", "offsetA", "offsetx", "offsety", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|COOiiiiiiii",
        kwlist, &A, &x, &y, &trans_, &ao, &bo, &m, &n, &ldA, &ix, &iy,
        &oA, &ox, &oy))
        return NULL;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cOOiiiiiiii",
        kwlist, &A, &x, &y, &trans, &ao, &bo, &m, &n, &ldA, &ix, &iy,
        &oA, &ox, &oy))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(A) != MAT_ID(x) || MAT_ID(A) != MAT_ID(y) ||
        MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N','T','C'");

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (m < 0) m = A->nrows;
    if (n < 0) n = A->ncols;
    if ((!m && trans == 'N') || (!n && (trans == 'T' || trans == 'C')))
        return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,m)) err_ld("ldA");

    if (oA < 0) err_nn_int("offsetA");
    if (n > 0 && m > 0 && oA + (n-1)*ldA + m > len(A)) err_buf_len("A");

    if (ox < 0) err_nn_int("offsetx");
    if ((trans == 'N' && n > 0 && ox + (n-1)*abs(ix) + 1 > len(x)) ||
	((trans == 'T' || trans == 'C') && m > 0 &&
        ox + (m-1)*abs(ix) + 1 > len(x))) err_buf_len("x");

    if (oy < 0) err_nn_int("offsety");
    if ((trans == 'N' && oy + (m-1)*abs(iy) + 1 > len(y)) ||
        ((trans == 'T' || trans == 'C') &&
        oy + (n-1)*abs(iy) + 1 > len(y))) err_buf_len("y");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(x)))
        err_type("alpha");
    if (bo && number_from_pyobject(bo, &b, MAT_ID(x)))
        err_type("beta");

    switch (MAT_ID(x)){
        case DOUBLE:
            if (!ao) a.d=1.0;
            if (!bo) b.d=0.0;
            if (trans == 'N' && n == 0)
                Py_BEGIN_ALLOW_THREADS
                dscal_(&m, &b.d, MAT_BUFD(y)+oy, &iy);
                Py_END_ALLOW_THREADS
            else if ((trans == 'T' || trans == 'C') && m == 0)
                Py_BEGIN_ALLOW_THREADS
                dscal_(&n, &b.d, MAT_BUFD(y)+oy, &iy);
                Py_END_ALLOW_THREADS
            else
                Py_BEGIN_ALLOW_THREADS
                dgemv_(&trans, &m, &n, &a.d, MAT_BUFD(A)+oA, &ldA,
                    MAT_BUFD(x)+ox, &ix, &b.d, MAT_BUFD(y)+oy, &iy);
                Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!ao) a.z=1.0;
            if (!bo) b.z=0.0;
#else
            if (!ao) a.z=_Cbuild(1.0,0.0);
            if (!bo) b.z=_Cbuild(0.0,0.0);
#endif
            if (trans == 'N' && n == 0)
                Py_BEGIN_ALLOW_THREADS
                zscal_(&m, &b.z, MAT_BUFZ(y)+oy, &iy);
                Py_END_ALLOW_THREADS
            else if ((trans == 'T' || trans == 'C') && m == 0)
                Py_BEGIN_ALLOW_THREADS
                zscal_(&n, &b.z, MAT_BUFZ(y)+oy, &iy);
                Py_END_ALLOW_THREADS
            else
                Py_BEGIN_ALLOW_THREADS
                zgemv_(&trans, &m, &n, &a.z, MAT_BUFZ(A)+oA, &ldA,
                    MAT_BUFZ(x)+ox, &ix, &b.z, MAT_BUFZ(y)+oy, &iy);
                Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_gbmv[] =
    "Matrix-vector product with a general banded matrix.\n\n"
    "gbmv(A, m, kl, x, y, trans='N', alpha=1.0, beta=0.0, n=A.size[1],\n"
    "     ku=A.size[0]-kl-1, ldA=max(1,A.size[0]), incx=1, incy=1, \n"
    "     offsetA=0, offsetx=0, offsety=0)\n\n"
    "PURPOSE\n"
    "If trans is 'N', computes y := alpha*A*x + beta*y.\n"
    "If trans is 'T', computes y := alpha*A^T*x + beta*y.\n"
    "If trans is 'C', computes y := alpha*A^H*x + beta*y.\n"
    "The matrix A is m by n with upper bandwidth ku and lower\n"
    "bandwidth kl.\n"
    "Returns immediately if n=0 and trans is 'T' or 'C', or if m=0 \n"
    "and trans is 'N'.\n"
    "Computes y := beta*y if n=0, m>0, and trans is 'N', or if m=0, n>0,\n"
    "and trans is 'T' or 'C'.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "m         nonnegative integer\n\n"
    "kl        nonnegative integer\n\n"
    "x         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "y         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "trans     'N', 'T' or 'C'\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "beta      number (int, float or complex).  Complex beta is only\n"
    "          allowed if A is complex.\n\n"
    "n         nonnegative integer.  If negative, the default value is\n"
    "          used.\n\n"
    "ku        nonnegative integer.  If negative, the default value is\n"
    "          used.\n"
    "ldA       positive integer.  ldA >= kl+ku+1. If zero, the default\n"
    "          value is used.\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer";

static PyObject* gbmv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x, *y;
    PyObject *ao=NULL, *bo=NULL;
    number a, b;
    int m, kl, ku=-1, n=-1, ldA=0, ix=1, iy=1, oA=0, ox=0, oy=0;
#if PY_MAJOR_VERSION >= 3
    int trans_ = 'N';
#endif
    char trans = 'N';
    char *kwlist[] = {"A", "m", "kl", "x", "y", "trans", "alpha", "beta",
        "n", "ku", "ldA", "incx", "incy", "offsetA", "offsetx", "offsety",
        NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OiiOO|COOiiiiiiii",
        kwlist, &A, &m, &kl, &x, &y, &trans_, &ao, &bo, &n, &ku, &ldA, &ix,
        &iy, &oA, &ox, &oy))
        return NULL;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OiiOO|cOOiiiiiiii",
        kwlist, &A, &m, &kl, &x, &y, &trans, &ao, &bo, &n, &ku, &ldA, &ix,
        &iy, &oA, &ox, &oy))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(A) != MAT_ID(x) || MAT_ID(A) != MAT_ID(y) ||
        MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");
    if (n < 0) n = A->ncols;
    if ((!m && trans == 'N') || (!n && (trans == 'T' || trans == 'C')))
       return Py_BuildValue("");

    if (kl < 0) err_nn_int("kl");
    if (ku < 0) ku = A->nrows - 1 - kl;
    if (ku < 0) err_nn_int("ku");

    if (ldA == 0) ldA = A->nrows;
    if (ldA < kl+ku+1) err_ld("ldA");

    if (oA < 0) err_nn_int("offsetA");
    if (m>0 && n>0 && oA + (n-1)*ldA + kl + ku + 1 > len(A))
        err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if ((trans == 'N' && n > 0 && ox + (n-1)*abs(ix) + 1 > len(x)) ||
        ((trans == 'T' || trans == 'C') && m > 0 &&
        ox + (m-1)*abs(ix) + 1 > len(x))) err_buf_len("x");
    if (oy < 0) err_nn_int("offsety");
    if ((trans == 'N' && oy + (m-1)*abs(iy) + 1 > len(y)) ||
	((trans == 'T' || trans == 'C') &&
        oy + (n-1)*abs(iy) + 1 > len(y))) err_buf_len("y");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(x)))
        err_type("alpha");
    if (bo && number_from_pyobject(bo, &b, MAT_ID(x)))
        err_type("beta");

    switch (MAT_ID(x)){
        case DOUBLE:
            if (!ao) a.d=1.0;
            if (!bo) b.d=0.0;
            if (trans == 'N' && n == 0)
                Py_BEGIN_ALLOW_THREADS
                dscal_(&m, &b.d, MAT_BUFD(y)+oy, &iy);
                Py_END_ALLOW_THREADS
            else if ((trans == 'T' || trans == 'C') && m == 0)
                Py_BEGIN_ALLOW_THREADS
                dscal_(&n, &b.d, MAT_BUFD(y)+oy, &iy);
                Py_END_ALLOW_THREADS
            else
                Py_BEGIN_ALLOW_THREADS
                dgbmv_(&trans, &m, &n, &kl, &ku, &a.d, MAT_BUFD(A)+oA,
                    &ldA, MAT_BUFD(x)+ox, &ix, &b.d, MAT_BUFD(y)+oy, &iy);
	        Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!ao) a.z=1.0;
            if (!bo) b.z=0.0;
#else
            if (!ao) a.z=_Cbuild(1.0,0.0);
            if (!bo) b.z=_Cbuild(0.0,0.0);
#endif
            if (trans == 'N' && n == 0)
                Py_BEGIN_ALLOW_THREADS
                zscal_(&m, &b.z, MAT_BUFZ(y)+oy, &iy);
                Py_END_ALLOW_THREADS
            else if ((trans == 'T' || trans == 'C') && m == 0)
                Py_BEGIN_ALLOW_THREADS
                zscal_(&n, &b.z, MAT_BUFZ(y)+oy, &iy);
                Py_END_ALLOW_THREADS
            else
                Py_BEGIN_ALLOW_THREADS
                zgbmv_(&trans, &m, &n, &kl, &ku, &a.z, MAT_BUFZ(A)+oA,
                    &ldA, MAT_BUFZ(x)+ox, &ix, &b.z, MAT_BUFZ(y)+oy, &iy);
                Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_symv[] =
    "Matrix-vector product with a real symmetric matrix.\n\n"
    "symv(A, x, y, uplo='L', alpha=1.0, beta=0.0, n=A.size[0], \n"
    "     ldA=max(1,A.size[0]), incx=1, incy=1, offsetA=0, offsetx=0,\n"
    "     offsety=0)\n\n"
    "PURPOSE\n"
    "Computes y := alpha*A*x + beta*y with A real symmetric of order n."
    "\n\n"
    "ARGUMENTS\n"
    "A         'd' matrix\n\n"
    "x         'd' matrix\n\n"
    "y         'd' matrix\n\n"
    "uplo      'L' or 'U'\n\n"
    "alpha     real number (int or float)\n\n"
    "beta      real number (int or float)\n\n"
    "n         integer.  If negative, the default value is used.\n"
    "          If the default value is used, we require that\n"
    "          A.size[0]=A.size[1].\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).\n"
    "          If zero, the default value is used.\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer";

static PyObject* symv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x, *y;
    PyObject *ao=NULL, *bo=NULL;
    number a, b;
    int n=-1, ldA=0, ix=1, iy=1, oA=0, ox=0, oy=0;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "x", "y", "uplo", "alpha", "beta", "n",
        "ldA", "incx", "incy", "offsetA", "offsetx", "offsety", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|COOiiiiiii",
        kwlist, &A, &x, &y, &uplo_, &ao, &bo, &n, &ldA, &ix, &iy, &oA,
        &ox, &oy))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cOOiiiiiii",
        kwlist, &A, &x, &y, &uplo, &ao, &bo, &n, &ldA, &ix, &iy, &oA,
        &ox, &oy))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(A) != MAT_ID(x) || MAT_ID(A) != MAT_ID(y) ||
        MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (n < 0){
        if (A->nrows != A->ncols){
            PyErr_SetString(PyExc_ValueError, "A is not square");
            return NULL;
        }
        n = A->nrows;
    }
    if (n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (n-1)*abs(ix) + 1 > len(x)) err_buf_len("x");
    if (oy < 0) err_nn_int("offsety");
    if (oy + (n-1)*abs(iy) + 1 > len(y)) err_buf_len("y");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(x)))
        err_type("alpha");
    if (bo && number_from_pyobject(bo, &b, MAT_ID(x)))
        err_type("beta");

    switch (MAT_ID(x)){
        case DOUBLE:
            if (!ao) a.d=1.0;
            if (!bo) b.d=0.0;
            Py_BEGIN_ALLOW_THREADS
            dsymv_(&uplo, &n, &a.d, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(x)+ox, &ix, &b.d, MAT_BUFD(y)+oy, &iy);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}



static char doc_hemv[] =
    "Matrix-vector product with a real symmetric or complex Hermitian\n"
    "matrix.\n\n"
    "hemv(A, x, y, uplo='L', alpha=1.0, beta=0.0, n=A.size[0],\n"
    "     ldA=max(1,A.size[0]), incx=1, incy=1, offsetA=0, offsetx=0,\n"
    "     offsety=0)\n\n"
    "PURPOSE\n"
    "Computes y := alpha*A*x + beta*y, with A real symmetric or\n"
    "complex Hermitian of order n.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "x         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "y         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "n         integer.  If negative, the default value is used.\n"
    "          If the default value is used, we require that\n"
    "          A.size[0]=A.size[1].\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).\n"
    "          If zero, the default value is used.\n\n"
    "incx      nonzero integer\n\n"
    "beta      number (int, float or complex).  Complex beta is only\n"
    "          allowed if A is complex.\n\n"
    "incy      nonzero integer\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer";


static PyObject* hemv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x, *y;
    PyObject *ao=NULL, *bo=NULL;
    number a, b;
    int n=-1, ldA=0, ix=1, iy=1, oA=0, ox=0, oy=0;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "x", "y", "uplo", "alpha", "beta", "n",
        "ldA", "incx", "incy", "offsetA", "offsetx", "offsety", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|COOiiiiiii",
        kwlist, &A, &x, &y, &uplo_, &ao, &bo, &n, &ldA, &ix, &iy, &oA,
        &ox, &oy))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cOOiiiiiii",
        kwlist, &A, &x, &y, &uplo, &ao, &bo, &n, &ldA, &ix, &iy, &oA,
        &ox, &oy))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(A) != MAT_ID(x) || MAT_ID(A) != MAT_ID(y) ||
        MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (n < 0){
        if (A->nrows != A->ncols){
            PyErr_SetString(PyExc_ValueError, "A is not square");
            return NULL;
        }
        n = A->nrows;
    }
    if (n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (n-1)*abs(ix) + 1 > len(x)) err_buf_len("x");
    if (oy < 0) err_nn_int("offsety");
    if (oy + (n-1)*abs(iy) + 1 > len(y)) err_buf_len("y");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(x)))
        err_type("alpha");
    if (bo && number_from_pyobject(bo, &b, MAT_ID(x)))
        err_type("beta");

    switch (MAT_ID(x)){
        case DOUBLE:
            if (!ao) a.d=1.0;
            if (!bo) b.d=0.0;
            Py_BEGIN_ALLOW_THREADS
            dsymv_(&uplo, &n, &a.d, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(x)+ox, &ix, &b.d, MAT_BUFD(y)+oy, &iy);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!ao) a.z=1.0;
            if (!bo) b.z=0.0;
#else
	    if (!ao) a.z=_Cbuild(1.0,0.0);
            if (!bo) b.z=_Cbuild(0.0,0.0);
#endif
            Py_BEGIN_ALLOW_THREADS
            zhemv_(&uplo, &n, &a.z, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(x)+ox, &ix, &b.z, MAT_BUFZ(y)+oy, &iy);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_sbmv[] =
    "Matrix-vector product with a real symmetric band matrix.\n\n"
    "sbmv(A, x, y, uplo='L', alpha=1.0, beta=0.0, n=A.size[1], \n"
    "     k=None, ldA=A.size[0], incx=1, incy=1, offsetA=0,\n"
    "     offsetx=0, offsety=0)\n\n"
    "PURPOSE\n"
    "Computes y := alpha*A*x + beta*y with A real symmetric and \n"
    "banded of order n and with bandwidth k.\n\n"
    "ARGUMENTS\n"
    "A         'd' matrix\n\n"
    "x         'd' matrix\n\n"
    "y         'd' matrix\n\n"
    "uplo      'L' or 'U'\n\n"
    "alpha     real number (int or float)\n\n"
    "beta      real number (int or float)\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "k         integer.  If negative, the default value is used.\n"
    "          The default value is k = max(0,A.size[0]-1).\n\n"
    "ldA       nonnegative integer.  ldA >= k+1.\n"
    "          If zero, the default vaule is used.\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer\n\n";

static PyObject* sbmv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x, *y;
    PyObject *ao=NULL, *bo=NULL;
    number a, b;
    int n=-1, k=-1, ldA=0, ix=1, iy=1, oA=0, ox=0, oy=0;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "x", "y", "uplo", "alpha", "beta", "n", "k",
        "ldA", "incx", "incy", "offsetA", "offsetx", "offsety", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|COOiiiiiiii",
        kwlist, &A, &x, &y, &uplo_, &ao, &bo, &n, &k, &ldA, &ix, &iy,
        &oA, &ox, &oy))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cOOiiiiiiii",
        kwlist, &A, &x, &y, &uplo, &ao, &bo, &n, &k, &ldA, &ix, &iy,
        &oA, &ox, &oy))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(A) != MAT_ID(x) || MAT_ID(A) != MAT_ID(y) ||
        MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (n < 0) n = A->ncols;
    if (n == 0) return Py_BuildValue("");

    if (k < 0) k = MAX(0, A->nrows-1);
    if (ldA == 0) ldA = A->nrows;
    if (ldA < 1+k) err_ld("ldA");

    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + k+1 > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (n-1)*abs(ix) + 1 > len(x)) err_buf_len("x");
    if (oy < 0) err_nn_int("offsety");
    if (oy + (n-1)*abs(iy) + 1 > len(y)) err_buf_len("y");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(x)))
        err_type("alpha");
    if (bo && number_from_pyobject(bo, &b, MAT_ID(x)))
        err_type("beta");

    switch (MAT_ID(x)){
        case DOUBLE:
            if (!ao) a.d=1.0;
            if (!bo) b.d=0.0;
            Py_BEGIN_ALLOW_THREADS
            dsbmv_(&uplo, &n, &k, &a.d, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(x)+ox, &ix, &b.d, MAT_BUFD(y)+oy, &iy);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_hbmv[] =
    "Matrix-vector product with a real symmetric or complex Hermitian\n"
    "band matrix.\n\n"
    "hbmv(A, x, y, uplo='L', alpha=1.0, beta=0.0, n=A.size[1], \n"
    "     k=None, ldA=A.size[0], incx=1, incy=1, offsetA=0, \n"
    "     offsetx=0, offsety=0)\n\n"
    "PURPOSE\n"
    "Computes y := alpha*A*x + beta*y with A real symmetric or \n"
    "complex Hermitian and banded of order n and with bandwidth k.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "x         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "y         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "beta      number (int, float or complex).  Complex beta is only\n"
    "          allowed if A is complex.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "k         integer.  If negative, the default value is used.\n"
    "          The default value is k = max(0,A.size[0]-1).\n\n"
    "ldA       nonnegative integer.  ldA >= k+1.\n"
    "          If zero, the default vaule is used.\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "offsetA   nonnegative integer.\n\n"
    "offsetx   nonnegative integer.\n\n"
    "offsety   nonnegative integer.\n\n";

static PyObject* hbmv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x, *y;
    PyObject *ao=NULL, *bo=NULL;
    number a, b;
    int n=-1, k=-1, ldA=0, ix=1, iy=1, oA=0, ox=0, oy=0;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"A", "x", "y", "uplo", "alpha", "beta", "n", "k",
        "ldA", "incx", "incy", "offsetA", "offsetx", "offsety", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|COOiiiiiiii",
        kwlist, &A, &x, &y, &uplo_, &ao, &bo, &n, &k, &ldA, &ix, &iy,
        &oA, &ox, &oy))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cOOiiiiiiii",
        kwlist, &A, &x, &y, &uplo, &ao, &bo, &n, &k, &ldA, &ix, &iy,
        &oA, &ox, &oy))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(A) != MAT_ID(x) || MAT_ID(A) != MAT_ID(y) ||
        MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (n < 0) n = A->ncols;
    if (n == 0) return Py_BuildValue("");

    if (k < 0) k = MAX(0, A->nrows-1);
    if (ldA == 0) ldA = A->nrows;
    if (ldA < 1+k) err_ld("ldA");

    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + k+1 > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (n-1)*abs(ix) + 1 > len(x)) err_buf_len("x");
    if (oy < 0) err_nn_int("offsety");
    if (oy + (n-1)*abs(iy) + 1 > len(y)) err_buf_len("y");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(x)))
        err_type("alpha");
    if (bo && number_from_pyobject(bo, &b, MAT_ID(x)))
        err_type("beta");

    switch (MAT_ID(x)){
        case DOUBLE:
            if (!ao) a.d=1.0;
            if (!bo) b.d=0.0;
            Py_BEGIN_ALLOW_THREADS
            dsbmv_(&uplo, &n, &k, &a.d, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(x)+ox, &ix, &b.d, MAT_BUFD(y)+oy, &iy);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!ao) a.z=1.0;
            if (!bo) b.z=0.0;
#else
            if (!ao) a.z=_Cbuild(1.0,0.0);
            if (!bo) b.z=_Cbuild(0.0,0.0);
#endif
            Py_BEGIN_ALLOW_THREADS
            zhbmv_(&uplo, &n, &k, &a.z, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(x)+ox, &ix, &b.z, MAT_BUFZ(y)+oy, &iy);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_trmv[] =
    "Matrix-vector product with a triangular matrix.\n\n"
    "trmv(A, x, uplo='L', trans='N', diag='N', n=A.size[0],\n"
    "     ldA=max(1,A.size[0]), incx=1, offsetA=0, offsetx=0)\n\n"
    "PURPOSE\n"
    "If trans is 'N', computes x := A*x.\n"
    "If trans is 'T', computes x := A^T*x.\n"
    "If trans is 'C', computes x := A^H*x.\n"
    "A is triangular of order n.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "x         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "trans     'N' or 'T'\n\n"
    "diag      'N' or 'U'\n\n"
    "n         integer.  If negative, the default value is used.\n"
    "          If the default value is used, we require that\n"
    "          A.size[0] = A.size[1].\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).\n"
    "          If zero the default value is used.\n\n"
    "incx      nonzero integer\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetx   nonnegative integer";

static PyObject* trmv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x;
    int n=-1, ldA=0, ix=1, oA=0, ox=0;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', trans_ = 'N', diag_ = 'N';
#endif
    char uplo = 'L', trans = 'N', diag = 'N';
    char *kwlist[] = {"A", "x", "uplo", "trans", "diag", "n", "ldA",
        "incx", "offsetA", "offsetx", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCiiiii",
        kwlist, &A, &x, &uplo_, &trans_, &diag_, &n, &ldA, &ix, &oA, &ox))
        return NULL;
    uplo = (char) uplo_;
    trans = (char) trans_;
    diag = (char) diag_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ccciiiii",
        kwlist, &A, &x, &uplo, &trans, &diag, &n, &ldA, &ix, &oA, &ox))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (MAT_ID(A) != MAT_ID(x)) err_conflicting_ids;

    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'U', 'N'");

    if (ix == 0) err_nz_int("incx");

    if (n < 0){
        if (A->nrows != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A is not square");
            return NULL;
        }
        n = A->nrows;
    }
    if (n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (n-1)*abs(ix) + 1 > len(x)) err_buf_len("offsetx");

    switch (MAT_ID(x)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            dtrmv_(&uplo, &trans, &diag, &n, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(x)+ox, &ix);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            ztrmv_(&uplo, &trans, &diag, &n, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(x)+ox, &ix);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_tbmv[] =
    "Matrix-vector product with a triangular band matrix.\n\n"
    "tbmv(A, x, uplo='L', trans='N', diag='N', n=A.size[1],\n"
    "     k=max(0,A.size[0]-1), ldA=A.size[0], incx=1, offsetA=0,\n"
    "     offsetx=0)\n\n"
    "PURPOSE\n"
    "If trans is 'N', computes x := A*x.\n"
    "If trans is 'T', computes x := A^T*x.\n"
    "If trans is 'C', computes x := A^H*x.\n"
    "A is banded triangular of order n and with bandwith k.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "x         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "trans     'N', 'T' or 'C'\n\n"
    "diag      'N' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value\n"
    "          is used.\n\n"
    "k         nonnegative integer.  If negative, the default value\n"
    "          is used.\n\n"
    "ldA       nonnegative integer.  lda >= 1+k.\n"
    "          If zero the default value is used.\n\n"
    "incx      nonzero integer\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetx   nonnegative integer";

static PyObject* tbmv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x;
    int n=-1, k=-1, ldA=0, ix=1, oA=0, ox=0;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', trans_ = 'N', diag_ = 'N';
#endif
    char uplo = 'L', trans = 'N', diag = 'N';
    char *kwlist[] = {"A", "x", "uplo", "trans", "diag", "n", "k",
        "ldA", "incx", "offsetA", "offsetx", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCiiiiii",
        kwlist, &A, &x, &uplo_, &trans_, &diag_, &n, &k, &ldA, &ix, &oA,
        &ox))
        return NULL;
    uplo = (char) uplo_;
    trans = (char) trans_;
    diag = (char) diag_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ccciiiiii",
        kwlist, &A, &x, &uplo, &trans, &diag, &n, &k, &ldA, &ix, &oA,
        &ox))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (MAT_ID(A) != MAT_ID(x)) err_conflicting_ids;

    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'U', 'N'");

    if (ix == 0) err_nz_int("incx");

    if (n < 0) n = A->ncols;
    if (n == 0) return Py_BuildValue("");
    if (k < 0) k = MAX(0,A->nrows-1);

    if (ldA == 0) ldA = A->nrows;
    if (ldA < k+1)  err_ld("ldA");

    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + k + 1 > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (n-1)*abs(ix) + 1 > len(x)) err_buf_len("x");

    switch (MAT_ID(x)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            dtbmv_(&uplo, &trans, &diag, &n, &k, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(x)+ox, &ix);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            ztbmv_(&uplo, &trans, &diag, &n, &k, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(x)+ox, &ix);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_trsv[] =
    "Solution of a triangular set of equations with one righthand side."
    "\n\n"
    "trsv(A, x, uplo='L', trans='N', diag='N', n=A.size[0],\n"
    "     ldA=max(1,A.size[0]), incx=1, offsetA=0, offsetx=0)\n\n"
    "PURPOSE\n"
    "If trans is 'N', computes x := A^{-1}*x.\n"
    "If trans is 'T', computes x := A^{-T}*x.\n"
    "If trans is 'C', computes x := A^{-H}*x.\n"
    "A is triangular of order n.  The code does not verify whether A\n"
    "is nonsingular.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "x         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "trans     'N', 'T' or 'C'\n\n"
    "diag      'N' or 'U'\n\n"
    "n         integer.  If negative, the default value is used.\n"
    "          If the default value is used, we require that\n"
    "          A.size[0] = A.size[1].\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).\n"
    "          If zero, the default value is used.\n\n"
    "incx      nonzero integer\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetx   nonnegative integer";

static PyObject* trsv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x;
    int n=-1, ldA=0, ix=1, oA=0, ox=0;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', trans_ = 'N', diag_ = 'N';
#endif
    char uplo = 'L', trans = 'N', diag = 'N';
    char *kwlist[] = {"A", "x", "uplo", "trans", "diag", "n", "ldA",
        "incx", "offsetA", "offsetx", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCiiiii",
        kwlist, &A, &x, &uplo_, &trans_, &diag_, &n, &ldA, &ix, &oA, &ox))
        return NULL;
    uplo = (char) uplo_;
    trans = (char) trans_;
    diag = (char) diag_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ccciiiii",
        kwlist, &A, &x, &uplo, &trans, &diag, &n, &ldA, &ix, &oA, &ox))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (MAT_ID(A) != MAT_ID(x)) err_conflicting_ids;

    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'N', 'U'");

    if (ix == 0) err_nz_int("incx");

    if (n < 0){
        if (A->nrows != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A is not square");
            return NULL;
        }
        n = A->nrows;
    }
    if (n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");

    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (n-1)*abs(ix) + 1 > len(x)) err_buf_len("x");

    switch (MAT_ID(x)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            dtrsv_(&uplo, &trans, &diag, &n, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(x)+ox, &ix);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            ztrsv_(&uplo, &trans, &diag, &n, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(x)+ox, &ix);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_tbsv[] =
    "Solution of a triangular and banded set of equations.\n\n"
    "tbsv(A, x, uplo='L', trans='N', diag='N', n=A.size[1],\n"
    "     k=max(0,A.size[0]-1), ldA=A.size[0], incx=1, offsetA=0,\n"
    "     offsetx=0)\n\n"
    "PURPOSE\n"
    "If trans is 'N', computes x := A^{-1}*x.\n"
    "If trans is 'T', computes x := A^{-T}*x.\n"
    "If trans is 'C', computes x := A^{-H}*x.\n"
    "A is banded triangular of order n and with bandwidth k.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "x         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "trans     'N', 'T' or 'C'\n\n"
    "diag      'N' or 'U'\n\n"
    "n         nonnegative integer.  If negative, the default value\n"
    "          is used.\n\n"
    "k         nonnegative integer.  If negative, the default value\n"
    "          is used.\n\n"
    "ldA       nonnegative integer.  ldA >= 1+k.\n"
    "          If zero the default value is used.\n\n"
    "incx      nonzero integer\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetx   nonnegative integer";

static PyObject* tbsv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x;
    int n=-1, k=-1, ldA=0, ix=1, oA=0, ox=0;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L', trans_ = 'N', diag_ = 'N';
#endif
    char uplo='L', trans='N', diag='N';
    char *kwlist[] = {"A", "x", "uplo", "trans", "diag", "n", "k",
        "ldA", "incx", "offsetA", "offsetx", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCiiiiii",
        kwlist, &A, &x, &uplo_, &trans_, &diag_, &n, &k, &ldA, &ix, &oA,
        &ox))
        return NULL;
    uplo = (char) uplo_;
    trans = (char) trans_;
    diag = (char) diag_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ccciiiiii",
        kwlist, &A, &x, &uplo, &trans, &diag, &n, &k, &ldA, &ix, &oA,
        &ox))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (MAT_ID(A) != MAT_ID(x)) err_conflicting_ids;

    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'N', 'U'");

    if (ix == 0) err_nz_int("incx");

    if (n < 0) n = A->ncols;
    if (n == 0) return Py_BuildValue("");
    if (k < 0) k = MAX(0, A->nrows-1);

    if (ldA == 0) ldA = A->nrows;
    if (ldA < k+1) err_ld("ldA");

    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + k + 1 > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (n-1)*abs(ix) + 1 > len(x)) err_buf_len("x");

    switch (MAT_ID(x)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            dtbsv_(&uplo, &trans, &diag, &n, &k, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(x)+ox, &ix);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            ztbsv_(&uplo, &trans, &diag, &n, &k, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(x)+ox, &ix);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_ger[] =
    "General rank-1 update.\n\n"
    "ger(x, y, A, alpha=1.0, m=A.size[0], n=A.size[1], incx=1,\n"
    "    incy=1, ldA=max(1,A.size[0]), offsetx=0, offsety=0,\n"
    "    offsetA=0)\n\n"
    "PURPOSE\n"
    "Computes A := A + alpha*x*y^H with A m by n, real or complex.\n\n"
    "ARGUMENTS\n"
    "x         'd' or 'z' matrix\n\n"
    "y         'd' or 'z' matrix.  Must have the same type as x.\n\n"
    "A         'd' or 'z' matrix.  Must have the same type as x.\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,m).\n"
    "          If zero, the default value is used.\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer\n\n"
    "offsetA   nonnegative integer";

static PyObject* ger(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x, *y;
    PyObject *ao=NULL;
    number a;
    int m=-1, n=-1, ldA=0, ix=1, iy=1, oA=0, ox=0, oy=0;
    char *kwlist[] = {"x", "y", "A", "alpha", "m", "n", "incx", "incy",
        "ldA", "offsetx", "offsety", "offsetA", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|Oiiiiiiii",
        kwlist, &x, &y, &A, &ao, &m, &n, &ix, &iy, &ldA, &ox, &oy, &oA))
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(A) != MAT_ID(x) || MAT_ID(A) != MAT_ID(y) ||
        MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (m < 0) m = A->nrows;
    if (n < 0) n = A->ncols;
    if (m == 0 || n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,m)) err_ld("ldA");

    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (m-1)*abs(ix) + 1 > len(x)) err_buf_len("x");
    if (oy < 0) err_nn_int("offsety");
    if (oy + (n-1)*abs(iy) + 1 > len(y)) err_buf_len("y");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(x)))
        err_type("alpha");

    switch (MAT_ID(x)){
        case DOUBLE:
            if (!ao) a.d = 1.0;
            Py_BEGIN_ALLOW_THREADS
            dger_(&m, &n, &a.d, MAT_BUFD(x)+ox, &ix, MAT_BUFD(y)+oy,
                &iy, MAT_BUFD(A)+oA, &ldA);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!ao) a.z = 1.0;
#else
            if (!ao) a.z = _Cbuild(1.0,0.0);
#endif
            Py_BEGIN_ALLOW_THREADS
            zgerc_(&m, &n, &a.z, MAT_BUFZ(x)+ox, &ix, MAT_BUFZ(y)+oy, &iy,
                MAT_BUFZ(A)+oA, &ldA);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_geru[] =
    "General rank-1 update.\n\n"
    "geru(x, y, A, m=A.size[0], n=A.size[1], alpha=1.0, incx=1,\n"
    "     incy=1, ldA=max(1,A.size[0]), offsetx=0, offsety=0,\n"
    "     offsetA=0)\n\n"
    "PURPOSE\n"
    "Computes A := A + alpha*x*y^T with A m by n, real or complex.\n\n"
    "ARGUMENTS\n"
    "x         'd' or 'z' matrix\n\n"
    "y         'd' or 'z' matrix.  Must have the same type as x.\n\n"
    "A         'd' or 'z' matrix.  Must have the same type as x.\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,m).\n"
    "          If zero, the default value is used.\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer\n\n"
    "offsetA   nonnegative integer";

static PyObject* geru(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x, *y;
    PyObject *ao=NULL;
    number a;
    int m=-1, n=-1, ldA=0, ix=1, iy=1, oA=0, ox=0, oy=0;
    char *kwlist[] = {"x", "y", "A", "alpha", "m", "n", "incx", "incy",
        "ldA", "offsetx", "offsety", "offsetA", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|Oiiiiiiii",
        kwlist, &x, &y, &A, &ao, &m, &n, &ix, &iy, &ldA, &ox, &oy, &oA))
        return NULL;

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(A) != MAT_ID(x) || MAT_ID(A) != MAT_ID(y) ||
        MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (m < 0) m = A->nrows;
    if (n < 0) n = A->ncols;
    if (m == 0 || n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,m)) err_ld("ldA");

    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + m > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (m-1)*abs(ix) + 1 > len(x)) err_buf_len("x");
    if (oy < 0) err_nn_int("offsety");
    if (oy + (n-1)*abs(iy) + 1 > len(y)) err_buf_len("y");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(x)))
        err_type("alpha");

    switch (MAT_ID(x)){
        case DOUBLE:
            if (!ao) a.d = 1.0;
            Py_BEGIN_ALLOW_THREADS
            dger_(&m, &n, &a.d, MAT_BUFD(x)+ox, &ix, MAT_BUFD(y)+oy, &iy,
                MAT_BUFD(A)+oA, &ldA);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!ao) a.z = 1.0;
#else
            if (!ao) a.z = _Cbuild(1.0,0.0);
#endif
            Py_BEGIN_ALLOW_THREADS
            zgeru_(&m, &n, &a.z, MAT_BUFZ(x)+ox, &ix, MAT_BUFZ(y)+oy,
                &iy, MAT_BUFZ(A)+oA, &ldA);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_syr[] =
    "Symmetric rank-1 update.\n\n"
    "syr(x, A, uplo='L', alpha=1.0, n=A.size[0], incx=1,\n"
    "    ldA=max(1,A.size[0]), offsetx=0, offsetA=0)\n\n"
    "PURPOSE\n"
    "Computes A := A + alpha*x*x^T with A real symmetric of order n."
    "\n\n"
    "ARGUMENTS\n"
    "x         'd' matrix\n\n"
    "A         'd' matrix\n\n"
    "uplo      'L' or 'U'\n\n"
    "alpha     real number (int or float)\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "incx      nonzero integer\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).\n"
    "          If zero, the default value is used.\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsetA   nonnegative integer";


static PyObject* syr(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x;
    PyObject *ao=NULL;
    number a;
    int n=-1, ldA=0, ix=1, oA=0, ox=0;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo='L';
    char *kwlist[] = {"x", "A", "uplo", "alpha", "n", "incx", "ldA",
        "offsetx", "offsetA", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|COiiiii", kwlist,
        &x, &A, &uplo_, &ao, &n, &ix, &ldA, &ox, &oA))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cOiiiii", kwlist,
        &x, &A, &uplo, &ao, &n, &ix, &ldA, &ox, &oA))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (MAT_ID(A) != MAT_ID(x)) err_conflicting_ids;

    if (ix == 0)  err_nz_int("incx");

    if (n < 0){
        if (A->nrows != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A is not square");
            return NULL;
        }
        n = A->nrows;
    }
    if (n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");

    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (n-1)*abs(ix) + 1 > len(x)) err_buf_len("x");

    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");

    if (ao && number_from_pyobject(ao, &a, DOUBLE)) err_type("alpha");
    if (!ao) a.d = 1.0;

    switch (MAT_ID(x)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            dsyr_(&uplo, &n, &a.d, MAT_BUFD(x)+ox, &ix, MAT_BUFD(A)+oA,
                &ldA);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_her[] =
    "Hermitian rank-1 update.\n\n"
    "her(x, A, uplo='L', alpha=1.0, n=A.size[0], incx=1,\n"
    "    ldA=max(1,A.size[0]), offsetx=0, offsetA=0)\n\n"
    "PURPOSE\n"
    "Computes A := A + alpha*x*x^H with A real symmetric or complex\n"
    "Hermitian of order n.\n\n"
    "ARGUMENTS\n"
    "x         'd' or 'z' matrix\n\n"
    "A         'd' or 'z' matrix.  Must have the same type as x.\n\n"
    "uplo      'L' or 'U'\n\n"
    "alpha     real number (int or float)\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "incx      nonzero integer\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).\n"
    "          If zero, the default value is used.\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsetA   nonnegative integer";


static PyObject* her(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x;
    PyObject *ao=NULL;
    number a;
    int n=-1, ldA=0, ix=1, oA=0, ox=0;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"x", "A", "uplo", "alpha", "n", "incx", "ldA",
        "offsetx", "offsetA", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|COiiiii",
        kwlist, &x, &A, &uplo_, &ao, &n, &ix, &ldA, &ox, &oA))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cOiiiii",
        kwlist, &x, &A, &uplo, &ao, &n, &ix, &ldA, &ox, &oA))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (MAT_ID(A) != MAT_ID(x)) err_conflicting_ids;

    if (ix == 0)  err_nz_int("incx");

    if (n < 0){
        if (A->nrows != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A is not square");
            return NULL;
        }
        n = A->nrows;
    }
    if (n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");

    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (n-1)*abs(ix) + 1 > len(x)) err_buf_len("x");

    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");

    if (ao && number_from_pyobject(ao, &a, DOUBLE)) err_type("alpha");
    if (!ao) a.d = 1.0;

    switch (MAT_ID(x)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            dsyr_(&uplo, &n, &a.d, MAT_BUFD(x)+ox, &ix, MAT_BUFD(A)+oA,
                &ldA);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            zher_(&uplo, &n, &a.d, MAT_BUFZ(x)+ox, &ix, MAT_BUFZ(A)+oA,
                &ldA);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}



static char doc_syr2[] =
    "Symmetric rank-2 update.\n\n"
    "syr2(x, y, A, uplo='L', alpha=1.0, n=A.size[0], incx=1, incy=1,\n"
    "    ldA=max(1,A.size[0]), offsetx=0, offsety=0, offsetA=0)\n\n"
    "PURPOSE\n"
    "Computes A := A + alpha*(x*y^T + y*x^T) with A real symmetric.\n\n"
    "ARGUMENTS\n"
    "x         'd' matrix\n\n"
    "y         'd' matrix\n\n"
    "A         'd' matrix\n\n"
    "uplo      'L' or 'U'\n\n"
    "alpha     real number (int or float)\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).\n"
    "          If zero the default value is used.\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer\n\n"
    "offsetA   nonnegative integer";

static PyObject* syr2(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x, *y;
    PyObject *ao=NULL;
    number a;
    int n=-1, ldA=0, ix=1, iy=1, ox=0, oy=0, oA=0;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"x", "y", "A", "uplo", "alpha", "n", "incx",
        "incy", "ldA", "offsetx", "offsety", "offsetA", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|COiiiiiii",
        kwlist, &x, &y, &A, &uplo_, &ao, &n, &ix, &iy, &ldA, &ox, &oy,
        &oA))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cOiiiiiii",
        kwlist, &x, &y, &A, &uplo, &ao, &n, &ix, &iy, &ldA, &ox, &oy,
        &oA))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(A) != MAT_ID(x) || MAT_ID(A) != MAT_ID(y) ||
        MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (n < 0){
        if (A->nrows != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A is not square");
            return NULL;
        }
        n = A->nrows;
    }
    if (n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (n-1)*abs(ix) + 1 > len(x)) err_buf_len("x");
    if (oy < 0) err_nn_int("offsety");
    if (oy + (n-1)*abs(iy) + 1 > len(y)) err_buf_len("y");

    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L','U'");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(x)))
        err_type("alpha");

    switch (MAT_ID(x)){
        case DOUBLE:
            if (!ao) a.d = 1.0;
            Py_BEGIN_ALLOW_THREADS
            dsyr2_(&uplo, &n, &a.d, MAT_BUFD(x)+ox, &ix, MAT_BUFD(y)+oy,
                &iy, MAT_BUFD(A)+oA, &ldA);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_her2[] =
    "Hermitian rank-2 update.\n\n"
    "her2(x, y, A, uplo='L', alpha=1.0, n=A.size[0], incx=1, incy=1,\n"
    "     ldA=max(1,A.size[0]), offsetx=0, offsety=0, offsetA=0)\n\n"
    "PURPOSE\n"
    "Computes A := A + alpha*x*y^H + conj(alpha)*y*x^H with A \n"
    "real symmetric or complex Hermitian of order n.\n\n"
    "ARGUMENTS\n"
    "x         'd' or 'z' matrix\n\n"
    "y         'd' or 'z' matrix.  Must have the same type as x.\n\n"
    "A         'd' or 'z' matrix.  Must have the same type as x.\n\n"
    "uplo      'L' or 'U'\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,n).\n"
    "          If zero the default value is used.\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer\n\n"
    "offsetA   nonnegative integer";

static PyObject* her2(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *x, *y;
    PyObject *ao=NULL;
    number a;
    int n=-1, ldA=0, ix=1, iy=1, ox=0, oy=0, oA=0;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    char *kwlist[] = {"x", "y", "A", "uplo", "alpha", "n", "incx",
        "incy", "ldA", "offsetx", "offsety", "offsetA", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|COiiiiiii",
        kwlist, &x, &y, &A, &uplo_, &ao, &n, &ix, &iy, &ldA, &ox, &oy,
        &oA))
        return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cOiiiiiii",
        kwlist, &x, &y, &A, &uplo, &ao, &n, &ix, &iy, &ldA, &ox, &oy,
        &oA))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(x)) err_mtrx("x");
    if (!Matrix_Check(y)) err_mtrx("y");
    if (MAT_ID(A) != MAT_ID(x) || MAT_ID(A) != MAT_ID(y) ||
        MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

    if (ix == 0) err_nz_int("incx");
    if (iy == 0) err_nz_int("incy");

    if (n < 0){
        if (A->nrows != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A is not square");
            return NULL;
        }
        n = A->nrows;
    }
    if (n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1,n)) err_ld("ldA");
    if (oA < 0) err_nn_int("offsetA");
    if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
    if (ox < 0) err_nn_int("offsetx");
    if (ox + (n-1)*abs(ix) + 1 > len(x)) err_buf_len("x");
    if (oy < 0) err_nn_int("offsety");
    if (oy + (n-1)*abs(iy) + 1 > len(y)) err_buf_len("y");

    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L','U'");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(x)))
        err_type("alpha");

    switch (MAT_ID(x)){
        case DOUBLE:
            if (!ao) a.d = 1.0;
            Py_BEGIN_ALLOW_THREADS
            dsyr2_(&uplo, &n, &a.d, MAT_BUFD(x)+ox, &ix, MAT_BUFD(y)+oy,
                &iy, MAT_BUFD(A)+oA, &ldA);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!ao) a.z = 1.0;
#else
            if (!ao) a.z = _Cbuild(1.0,0.0);
#endif
            Py_BEGIN_ALLOW_THREADS
            zher2_(&uplo, &n, &a.z, MAT_BUFZ(x)+ox, &ix, MAT_BUFZ(y)+oy,
                &iy, MAT_BUFZ(A)+oA, &ldA);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_gemm[] =
    "General matrix-matrix product.\n\n"
    "gemm(A, B, C, transA='N', transB='N', alpha=1.0, beta=0.0, \n"
    "     m=None, n=None, k=None, ldA=max(1,A.size[0]), \n"
    "     ldB=max(1,B.size[0]), ldC=max(1,C.size[0]), offsetA=0, \n"
    "     offsetB=0, offsetC=0) \n\n"
    "PURPOSE\n"
    "Computes \n"
    "C := alpha*A*B + beta*C     if transA = 'N' and transB = 'N'.\n"
    "C := alpha*A^T*B + beta*C   if transA = 'T' and transB = 'N'.\n"
    "C := alpha*A^H*B + beta*C   if transA = 'C' and transB = 'N'.\n"
    "C := alpha*A*B^T + beta*C   if transA = 'N' and transB = 'T'.\n"
    "C := alpha*A^T*B^T + beta*C if transA = 'T' and transB = 'T'.\n"
    "C := alpha*A^H*B^T + beta*C if transA = 'C' and transB = 'T'.\n"
    "C := alpha*A*B^H + beta*C   if transA = 'N' and transB = 'C'.\n"
    "C := alpha*A^T*B^H + beta*C if transA = 'T' and transB = 'C'.\n"
    "C := alpha*A^H*B^H + beta*C if transA = 'C' and transB = 'C'.\n"
    "The number of rows of the matrix product is m.  The number of \n"
    "columns is n.  The inner dimension is k.  If k=0, this reduces \n"
    "to C := beta*C.\n\n"
    "ARGUMENTS\n\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "C         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "transA    'N', 'T' or 'C'\n\n"
    "transB    'N', 'T' or 'C'\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "beta      number (int, float or complex).  Complex beta is only\n"
    "          allowed if A is complex.\n\n"
    "m         integer.  If negative, the default value is used.\n"
    "          The default value is\n"
    "          m = (transA == 'N') ? A.size[0] : A.size[1].\n\n"
    "n         integer.  If negative, the default value is used.\n"
    "          The default value is\n"
    "          n = (transB == 'N') ? B.size[1] : B.size[0].\n\n"
    "k         integer.  If negative, the default value is used.\n"
    "          The default value is\n"
    "          (transA == 'N') ? A.size[1] : A.size[0], transA='N'.\n"
    "          If the default value is used it should also be equal to\n"
    "          (transB == 'N') ? B.size[0] : B.size[1].\n\n"
    "ldA       nonnegative integer.  ldA >= max(1,(transA == 'N') ? m : k).\n"
    "          If zero, the default value is used.\n\n"
    "ldB       nonnegative integer.  ldB >= max(1,(transB == 'N') ? k : n).\n"
    "          If zero, the default value is used.\n\n"
    "ldC       nonnegative integer.  ldC >= max(1,m).\n"
    "          If zero, the default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer\n\n"
    "offsetC   nonnegative integer";

static PyObject* gemm(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B, *C;
    PyObject *ao=NULL, *bo=NULL;
    number a, b;
    int m=-1, n=-1, k=-1, ldA=0, ldB=0, ldC=0, oA=0, oB=0, oC=0;
#if PY_MAJOR_VERSION >= 3
    int transA_ = 'N', transB_ = 'N';
#endif
    char transA = 'N', transB = 'N';
    char *kwlist[] = {"A", "B", "C", "transA", "transB", "alpha",
        "beta", "m", "n", "k", "ldA", "ldB", "ldC", "offsetA",
        "offsetB", "offsetC", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|CCOOiiiiiiiii",
        kwlist, &A, &B, &C, &transA_, &transB_, &ao, &bo, &m, &n, &k,
        &ldA, &ldB, &ldC, &oA, &oB, &oC))
        return NULL;
    transA = (char) transA_;
    transB = (char) transB_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ccOOiiiiiiiii",
        kwlist, &A, &B, &C, &transA, &transB, &ao, &bo, &m, &n, &k,
        &ldA, &ldB, &ldC, &oA, &oB, &oC))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (!Matrix_Check(C)) err_mtrx("C");
    if (MAT_ID(A) != MAT_ID(B) || MAT_ID(A) != MAT_ID(C) ||
        MAT_ID(B) != MAT_ID(C)) err_conflicting_ids;

    if (transA != 'N' && transA != 'T' && transA != 'C')
        err_char("transA", "'N', 'T', 'C'");
    if (transB != 'N' && transB != 'T' && transB != 'C')
        err_char("transB", "'N', 'T', 'C'");

    if (m < 0) m = (transA == 'N') ? A->nrows : A->ncols;
    if (n < 0) n = (transB == 'N') ? B->ncols : B->nrows;
    if (k < 0){
        k = (transA == 'N') ? A->ncols : A->nrows;
        if (k != ((transB == 'N') ? B->nrows : B->ncols)){
             PyErr_SetString(PyExc_TypeError, "dimensions of A and B "
                  "do not match");
             return NULL;
        }
    }
    if (m == 0 || n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (k > 0 && ldA < MAX(1, (transA == 'N') ? m : k)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (k > 0 && ldB < MAX(1, (transB == 'N') ? k : n)) err_ld("ldB");
    if (ldC == 0) ldC = MAX(1,C->nrows);
    if (ldC < MAX(1,m)) err_ld("ldB");

    if (oA < 0) err_nn_int("offsetA");
    if (k > 0 && ((transA == 'N' && oA + (k-1)*ldA + m > len(A)) ||
        ((transA == 'T' || transA == 'C') &&
        oA + (m-1)*ldA + k > len(A)))) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (k > 0 && ((transB == 'N' && oB + (n-1)*ldB + k > len(B)) ||
        ((transB == 'T' || transB == 'C') &&
        oB + (k-1)*ldB + n > len(B)))) err_buf_len("B");
    if (oC < 0) err_nn_int("offsetC");
    if (oC + (n-1)*ldC + m > len(C)) err_buf_len("C");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(A)))
        err_type("alpha");
    if (bo && number_from_pyobject(bo, &b, MAT_ID(A)))
        err_type("beta");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (!ao) a.d = 1.0;
            if (!bo) b.d = 0.0;
            Py_BEGIN_ALLOW_THREADS
            dgemm_(&transA, &transB, &m, &n, &k, &a.d,
                MAT_BUFD(A)+oA, &ldA, MAT_BUFD(B)+oB, &ldB, &b.d,
                MAT_BUFD(C)+oC, &ldC);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!ao) a.z = 1.0;
            if (!bo) b.z = 0.0;
#else
            if (!ao) a.z = _Cbuild(1.0,0.0);
            if (!bo) b.z = _Cbuild(0.0,0.0);
#endif
            Py_BEGIN_ALLOW_THREADS
            zgemm_(&transA, &transB, &m, &n, &k, &a.z,
                MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(B)+oB, &ldB, &b.z,
                MAT_BUFZ(C)+oC, &ldC);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_symm[] =
    "Matrix-matrix product where one matrix is symmetric."
    "\n\n"
    "symm(A, B, C, side='L', uplo='L', alpha=1.0, beta=0.0, \n"
    "     m=B.size[0], n=B.size[1], ldA=max(1,A.size[0]), \n"
    "     ldB=max(1,B.size[0]), ldC=max(1,C.size[0]), offsetA=0, \n"
    "     offsetB=0, offsetC=0)\n\n"
    "PURPOSE\n"
    "If side is 'L', computes C := alpha*A*B + beta*C.\n"
    "If side is 'R', computes C := alpha*B*A + beta*C.\n"
    "C is m by n and A is real or complex symmetric.  (Use hemm for\n"
    "Hermitian A).\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "C         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "side      'L' or 'R'\n\n"
    "uplo      'L' or 'U'\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "beta      number (int, float or complex).  Complex beta is only\n"
    "          allowed if A is complex.\n\n"
    "m         integer.  If negative, the default value is used.\n"
    "          If the default value is used and side = 'L', then m\n"
    "          must be equal to A.size[0] and A.size[1].\n\n"
    "n         integer.  If negative, the default value is used.\n"
    "          If the default value is used and side = 'R', then \n\n"
    "          must be equal to A.size[0] and A.size[1].\n\n"
    "ldA       nonnegative integer.\n"
    "          ldA >= max(1, (side == 'L') ? m : n).  If zero, the\n"
    "          default value is used.\n\n"
    "ldB       nonnegative integer.\n"
    "          ldB >= max(1, (side == 'L') ? n : m).  If zero, the\n"
    "          default value is used.\n\n"
    "ldC       nonnegative integer.  ldC >= max(1,m).\n"
    "          If zero, the default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer\n\n"
    "offsetC   nonnegative integer";

static PyObject* symm(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B, *C;
    PyObject *ao=NULL, *bo=NULL;
    number a, b;
    int m=-1, n=-1, ldA=0, ldB=0, ldC=0, oA=0, oB=0, oC = 0;
#if PY_MAJOR_VERSION >= 3
    int side_  = 'L', uplo_ = 'L';
#endif
    char side = 'L', uplo = 'L';
    char *kwlist[] = {"A", "B", "C", "side", "uplo", "alpha", "beta",
        "m", "n", "ldA", "ldB", "ldC", "offsetA", "offsetB", "offsetC",
        NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|CCOOiiiiiiii",
        kwlist, &A, &B, &C, &side_, &uplo_, &ao, &bo, &m, &n, &ldA, &ldB,
        &ldC, &oA, &oB, &oC))
        return NULL;
    side = (char) side_;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ccOOiiiiiiii",
        kwlist, &A, &B, &C, &side, &uplo, &ao, &bo, &m, &n, &ldA, &ldB,
        &ldC, &oA, &oB, &oC))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (!Matrix_Check(C)) err_mtrx("C");
    if (MAT_ID(A) != MAT_ID(B) || MAT_ID(A) != MAT_ID(C) ||
        MAT_ID(B) != MAT_ID(C)) err_conflicting_ids;

    if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");

    if (m < 0){
        m = B->nrows;
        if (side == 'L' && (m != A->nrows || m != A->ncols)){
            PyErr_SetString(PyExc_TypeError, "dimensions of A and B "
                "do not match");
            return NULL;
        }
    }
    if (n < 0){
        n = B->ncols;
        if (side == 'R' && (n != A->nrows || n != A->ncols)){
            PyErr_SetString(PyExc_TypeError, "dimensions of A and B "
                "do not match");
            return NULL;
        }
    }
    if (m == 0 || n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1, (side == 'L') ? m : n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,m)) err_ld("ldB");
    if (ldC == 0) ldC = MAX(1,C->nrows);
    if (ldC < MAX(1,m)) err_ld("ldC");

    if (oA < 0) err_nn_int("offsetA");
    if ((side == 'L' && oA + (m-1)*ldA + m > len(A)) ||
        (side == 'R' && oA + (n-1)*ldA + n > len(A))) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (n-1)*ldB + m > len(B)) err_buf_len("B");
    if (oC < 0) err_nn_int("offsetC");
    if (oC + (n-1)*ldC + m > len(C)) err_buf_len("C");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(A)))
        err_type("alpha");
    if (bo && number_from_pyobject(bo, &b, MAT_ID(A)))
        err_type("beta");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (!ao) a.d = 1.0;
            if (!bo) b.d = 0.0;
            Py_BEGIN_ALLOW_THREADS
            dsymm_(&side, &uplo, &m, &n, &a.d, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(B)+oB, &ldB, &b.d, MAT_BUFD(C)+oC, &ldC);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!ao) a.z = 1.0;
            if (!bo) b.z = 0.0;
#else
            if (!ao) a.z = _Cbuild(1.0,0.0);
            if (!bo) b.z = _Cbuild(0.0,0.0);
#endif
            Py_BEGIN_ALLOW_THREADS
            zsymm_(&side, &uplo, &m, &n, &a.z, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(B)+oB, &ldB, &b.z, MAT_BUFZ(C)+oC, &ldC);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_hemm[] =
    "Matrix-matrix product where one matrix is real symmetric or\n"
    "complex Hermitian."
    "\n\n"
    "hemm(A, B, C, side='L', uplo='L', alpha=1.0, beta=0.0, \n"
    "     m=B.size[0], n=B.size[1], ldA=max(1,A.size[0]), \n"
    "     ldB=max(1,B.size[0]), ldC=max(1,C.size[0]), offsetA=0, \n"
    "     offsetB=0, offsetC=0)\n\n"
    "PURPOSE\n"
    "If side is 'L', computes C := alpha*A*B + beta*C.\n"
    "If side is 'R', computes C := alpha*B*A + beta*C.\n"
    "C is m by n and A is real symmetric or complex Hermitian.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "C         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "side      'L' or 'R'\n\n"
    "uplo      'L' or 'U'\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "beta      number (int, float or complex).  Complex beta is only\n"
    "          allowed if A is complex.\n\n"
    "m         integer.  If negative, the default value is used.\n"
    "          If the default value is used and side = 'L', then m\n"
    "          must be equal to A.size[0] and A.size[1].\n\n"
    "n         integer.  If negative, the default value is used.\n"
    "          If the default value is used and side = 'R', then \n\n"
    "          must be equal to A.size[0] and A.size[1].\n\n"
    "ldA       nonnegative integer.\n"
    "          ldA >= max(1, (side == 'L') ? m : n).  If zero, the\n"
    "          default value is used.\n\n"
    "ldB       nonnegative integer.\n"
    "          ldB >= max(1, (side == 'L') ? n : m).  If zero, the\n"
    "          default value is used.\n\n"
    "ldC       nonnegative integer.  ldC >= max(1,m).\n"
    "          If zero, the default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer\n\n"
    "offsetC   nonnegative integer";

static PyObject* hemm(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B, *C;
    PyObject *ao=NULL, *bo=NULL;
    number a, b;
    int m=-1, n=-1, ldA=0, ldB=0, ldC=0, oA=0, oB=0, oC = 0;
#if PY_MAJOR_VERSION >= 3
    int side_ = 'L', uplo_ = 'L';
#endif
    char side = 'L', uplo = 'L';
    char *kwlist[] = {"A", "B", "C", "side", "uplo", "alpha", "beta",
        "m", "n", "ldA", "ldB", "ldC", "offsetA", "offsetB", "offsetC",
        NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|CCOOiiiiiiii",
        kwlist, &A, &B, &C, &side_, &uplo_, &ao, &bo, &m, &n, &ldA, &ldB,
        &ldC, &oA, &oB, &oC))
        return NULL;
    side = (char) side_;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ccOOiiiiiiii",
        kwlist, &A, &B, &C, &side, &uplo, &ao, &bo, &m, &n, &ldA, &ldB,
        &ldC, &oA, &oB, &oC))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (!Matrix_Check(C)) err_mtrx("C");
    if (MAT_ID(A) != MAT_ID(B) || MAT_ID(A) != MAT_ID(C) ||
        MAT_ID(B) != MAT_ID(C)) err_conflicting_ids;

    if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");

    if (m < 0){
        m = B->nrows;
        if (side == 'L' && (m != A->nrows || m != A->ncols)){
            PyErr_SetString(PyExc_TypeError, "dimensions of A and B "
                "do not match");
            return NULL;
        }
    }
    if (n < 0){
        n = B->ncols;
        if (side == 'R' && (n != A->nrows || n != A->ncols)){
            PyErr_SetString(PyExc_TypeError, "dimensions of A and B "
                "do not match");
            return NULL;
        }
    }
    if (m == 0 || n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1, (side == 'L') ? m : n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,m)) err_ld("ldB");
    if (ldC == 0) ldC = MAX(1,C->nrows);
    if (ldC < MAX(1,m)) err_ld("ldC");

    if (oA < 0) err_nn_int("offsetA");
    if ((side == 'L' && oA + (m-1)*ldA + m > len(A)) ||
        (side == 'R' && oA + (n-1)*ldA + n > len(A))) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (n-1)*ldB + m > len(B)) err_buf_len("B");
    if (oC < 0) err_nn_int("offsetC");
    if (oC + (n-1)*ldC + m > len(C)) err_buf_len("C");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(A)))
        err_type("alpha");
    if (bo && number_from_pyobject(bo, &b, MAT_ID(A)))
        err_type("beta");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (!ao) a.d = 1.0;
            if (!bo) b.d = 0.0;
            Py_BEGIN_ALLOW_THREADS
            dsymm_(&side, &uplo, &m, &n, &a.d, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(B)+oB, &ldB, &b.d, MAT_BUFD(C)+oC, &ldC);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!ao) a.z = 1.0;
            if (!bo) b.z = 0.0;
#else
            if (!ao) a.z = _Cbuild(1.0,0.0);
            if (!bo) b.z = _Cbuild(0.0,0.0);
#endif
            Py_BEGIN_ALLOW_THREADS
            zhemm_(&side, &uplo, &m, &n, &a.z, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(B)+oB, &ldB, &b.z, MAT_BUFZ(C)+oC, &ldC);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}



static char doc_syrk[] =
    "Rank-k update of symmetric matrix.\n\n"
    "syrk(A, C, uplo='L', trans='N', alpha=1.0, beta=0.0, n=None, \n"
    "     k=None, ldA=max(1,A.size[0]), ldC=max(1,C.size[0]),\n"
    "     offsetA=0, offsetB=0)\n\n"
    "PURPOSE   \n"
    "If trans is 'N', computes C := alpha*A*A^T + beta*C.\n"
    "If trans is 'T', computes C := alpha*A^T*A + beta*C.\n"
    "C is symmetric (real or complex) of order n. \n"
    "The inner dimension of the matrix product is k.  If k=0 this is\n"
    "interpreted as C := beta*C.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "C         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "trans     'N' or 'T'\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "beta      number (int, float or complex).  Complex beta is only\n"
    "          allowed if A is complex.\n\n"
    "n         integer.  If negative, the default value is used.\n"
    "          The default value is\n"
    "          n = (trans == N) ? A.size[0] : A.size[1].\n\n"
    "k         integer.  If negative, the default value is used.\n"
    "          The default value is\n"
    "          k = (trans == 'N') ? A.size[1] : A.size[0].\n\n"
    "ldA       nonnegative integer.\n"
    "          ldA >= max(1, (trans == 'N') ? n : k).  If zero,\n"
    "          the default value is used.\n\n"
    "ldC       nonnegative integer.  ldC >= max(1,n).\n"
    "          If zero, the default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetC   nonnegative integer";

static PyObject* syrk(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *C;
    PyObject *ao=NULL, *bo=NULL;
    number a, b;
    int n=-1, k=-1, ldA=0, ldC=0, oA = 0, oC = 0;
#if PY_MAJOR_VERSION >= 3
    int trans_ = 'N', uplo_ = 'L';
#endif
    char trans = 'N', uplo = 'L';
    char *kwlist[] = {"A", "C", "uplo", "trans", "alpha", "beta", "n",
        "k", "ldA", "ldC", "offsetA", "offsetC", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCOOiiiiii",
        kwlist, &A, &C, &uplo_, &trans_, &ao, &bo, &n, &k, &ldA, &ldC,
	&oA, &oC))
        return NULL;
    uplo = (char) uplo_;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ccOOiiiiii",
        kwlist, &A, &C, &uplo, &trans, &ao, &bo, &n, &k, &ldA, &ldC,
	&oA, &oC))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(C)) err_mtrx("C");
    if (MAT_ID(A) != MAT_ID(C)) err_conflicting_ids;

    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (MAT_ID(A) == DOUBLE && trans != 'N' && trans != 'T' &&
        trans != 'C') err_char("trans", "'N', 'T', 'C'");
    if (MAT_ID(A) == COMPLEX && trans != 'N' && trans != 'T')
	err_char("trans", "'N', 'T'");

    if (n < 0) n = (trans == 'N') ? A->nrows : A->ncols;
    if (k < 0) k = (trans == 'N') ? A->ncols : A->nrows;
    if (n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (k > 0 && ldA < MAX(1, (trans == 'N') ? n : k)) err_ld("ldA");
    if (ldC == 0) ldC = MAX(1,C->nrows);
    if (ldC < MAX(1,n)) err_ld("ldC");
    if (oA < 0) err_nn_int("offsetA");
    if (k > 0 && ((trans == 'N' && oA + (k-1)*ldA + n > len(A)) ||
        ((trans == 'T' || trans == 'C') &&
	oA + (n-1)*ldA + k > len(A))))
        err_buf_len("A");
    if (oC < 0) err_nn_int("offsetC");
    if (oC + (n-1)*ldC + n > len(C)) err_buf_len("C");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(A)))
        err_type("alpha");
    if (bo && number_from_pyobject(bo, &b, MAT_ID(A)))
        err_type("beta");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (!ao) a.d = 1.0;
            if (!bo) b.d = 0.0;
            Py_BEGIN_ALLOW_THREADS
            dsyrk_(&uplo, &trans, &n, &k, &a.d, MAT_BUFD(A)+oA, &ldA,
                &b.d, MAT_BUFD(C)+oC, &ldC);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!ao) a.z = 1.0;
            if (!bo) b.z = 0.0;
#else
            if (!ao) a.z = _Cbuild(1.0,0.0);
            if (!bo) b.z = _Cbuild(0.0,0.0);
#endif
            Py_BEGIN_ALLOW_THREADS
            zsyrk_(&uplo, &trans, &n, &k, &a.z, MAT_BUFZ(A)+oA, &ldA,
                &b.z, MAT_BUFZ(C)+oC, &ldC);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_herk[] =
    "Rank-k update of Hermitian matrix.\n\n"
    "herk(A, C, uplo='L', trans='N', alpha=1.0, beta=0.0, n=None, \n"
    "     k=None, ldA=max(1,A.size[0]), ldC=max(1,C.size[0]),\n"
    "     offsetA=0, offsetB=0)\n\n"
    "PURPOSE   \n"
    "If trans is 'N', computes C := alpha*A*A^H + beta*C.\n"
    "If trans is 'C', computes C := alpha*A^H*A + beta*C.\n"
    "C is real symmetric or Hermitian of order n.  The inner \n"
    "dimension of the matrix product is k.\n"
    "If k=0 this is interpreted as C := beta*C.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "C         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "trans     'N' or 'C'\n\n"
    "alpha     real number (int or float)\n\n"
    "beta      number (int, float or complex)\n\n"
    "n         integer.  If negative, the default value is used.\n"
    "          The default value is\n"
    "          n = (trans == N) ? A.size[0] : A.size[1].\n\n"
    "k         integer.  If negative, the default value is used.\n"
    "          The default value is\n"
    "          k = (trans == 'N') ? A.size[1] : A.size[0].\n\n"
    "ldA       nonnegative integer.\n"
    "          ldA >= max(1, (trans == 'N') ? n : k).  If zero,\n"
    "          the default value is used.\n\n"
    "ldC       nonnegative integer.  ldC >= max(1,n).\n"
    "          If zero, the default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetC   nonnegative integer";

static PyObject* herk(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *C;
    PyObject *ao=NULL, *bo=NULL;
    number a, b;
    int n=-1, k=-1, ldA=0, ldC=0, oA = 0, oC = 0;
#if PY_MAJOR_VERSION >= 3
    int trans_ = 'N', uplo_ = 'L';
#endif
    char trans = 'N', uplo = 'L';
    char *kwlist[] = {"A", "C", "uplo", "trans", "alpha", "beta", "n",
        "k", "ldA", "ldC", "offsetA", "offsetC", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCOOiiiiii",
        kwlist, &A, &C, &uplo_, &trans_, &ao, &bo, &n, &k, &ldA, &ldC,
	&oA, &oC))
        return NULL;
    uplo = (char) uplo_;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ccOOiiiiii",
        kwlist, &A, &C, &uplo, &trans, &ao, &bo, &n, &k, &ldA, &ldC,
	&oA, &oC))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(C)) err_mtrx("C");
    if (MAT_ID(A) != MAT_ID(C)) err_conflicting_ids;

    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (MAT_ID(A) == DOUBLE && trans != 'N' && trans != 'T' &&
        trans != 'C') err_char("trans", "'N', 'T', 'C'");
    if (MAT_ID(A) == COMPLEX && trans != 'N' && trans != 'C')
	err_char("trans", "'N', 'C'");

    if (n < 0) n = (trans == 'N') ? A->nrows : A->ncols;
    if (k < 0) k = (trans == 'N') ? A->ncols : A->nrows;
    if (n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (k > 0 && ldA < MAX(1, (trans == 'N') ? n : k)) err_ld("ldA");
    if (ldC == 0) ldC = MAX(1,C->nrows);
    if (ldC < MAX(1,n)) err_ld("ldC");
    if (oA < 0) err_nn_int("offsetA");
    if (k > 0 && ((trans == 'N' && oA + (k-1)*ldA + n > len(A)) ||
        ((trans == 'T' || trans == 'C') &&
	oA + (n-1)*ldA + k > len(A))))
        err_buf_len("A");
    if (oC < 0) err_nn_int("offsetC");
    if (oC + (n-1)*ldC + n > len(C)) err_buf_len("C");

    if (ao && number_from_pyobject(ao, &a, DOUBLE)) err_type("alpha");
    if (bo && number_from_pyobject(bo, &b, DOUBLE)) err_type("beta");
    if (!ao) a.d = 1.0;
    if (!bo) b.d = 0.0;

    switch (MAT_ID(A)){
        case DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            dsyrk_(&uplo, &trans, &n, &k, &a.d, MAT_BUFD(A)+oA, &ldA,
                &b.d, MAT_BUFD(C)+oC, &ldC);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
            Py_BEGIN_ALLOW_THREADS
            zherk_(&uplo, &trans, &n, &k, &a.d, MAT_BUFZ(A)+oA, &ldA,
                &b.d, MAT_BUFZ(C)+oC, &ldC);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_syr2k[] =
    "Rank-2k update of symmetric matrix.\n\n"
    "syr2k(A, B, C, uplo='L', trans='N', alpha=1.0, beta=0.0, n=None,\n"
    "      k=None, ldA=max(1,A.size[0]), ldB=max(1,B.size[0]), \n"
    "      ldC=max(1,C.size[0])), offsetA=0, offsetB=0, offsetC=0)\n\n"
    "PURPOSE\n"
    "If trans is 'N', computes C := alpha*(A*B^T + B*A^T) + beta*C.\n"
    "If trans is 'T', computes C := alpha*(A^T*B + B^T*A) + beta*C.\n"
    "C is symmetric (real or complex) of order n.\n"
    "The inner dimension of the matrix product is k.  If k=0 this is\n"
    "interpreted as C := beta*C.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "C         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "trans     'N', 'T' or 'C' ('C' is only allowed when in the real\n"
    "          case and means the same as 'T')\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "beta      number (int, float or complex).  Complex beta is only\n"
    "          allowed if A is complex.\n\n"
    "n         integer.  If negative, the default value is used.\n"
    "          The default value is\n"
    "          n = (trans == 'N') ? A.size[0] : A.size[1].\n"
    "          If the default value is used, it should be equal to\n"
    "          (trans == 'N') ? B.size[0] : B.size[1].\n\n"
    "k         integer.  If negative, the default value is used.\n"
    "          The default value is\n"
    "          k = (trans == 'N') ? A.size[1] : A.size[0].\n"
    "          If the default value is used, it should be equal to\n"
    "          (trans == 'N') ? B.size[1] : B.size[0].\n\n"
    "ldA       nonnegative integer.\n"
    "          ldA >= max(1, (trans=='N') ? n : k).\n"
    "          If zero, the default value is used.\n\n"
    "ldB       nonnegative integer.\n"
    "          ldB >= max(1, (trans=='N') ? n : k).\n"
    "          If zero, the default value is used.\n\n"
    "ldC       nonnegative integer.  ldC >= max(1,n).\n"
    "          If zero, the default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer\n\n"
    "offsetC   nonnegative integer";

static PyObject* syr2k(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B, *C;
    PyObject *ao=NULL, *bo=NULL;
    number a, b;
    int n=-1, k=-1, ldA=0, ldB=0, ldC=0, oA = 0, oB = 0, oC = 0;
#if PY_MAJOR_VERSION >= 3
    int trans_ = 'N', uplo_ = 'L';
#endif
    char trans = 'N', uplo = 'L';
    char *kwlist[] = {"A", "B", "C", "uplo", "trans", "alpha", "beta",
        "n", "k", "ldA", "ldB", "ldC", "offsetA", "offsetB", "offsetC",
        NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|CCOOiiiiiiii",
        kwlist, &A, &B, &C, &uplo_, &trans_, &ao, &bo, &n, &k, &ldA, &ldB,
	&ldC, &oA, &oB, &oC))
        return NULL;
    uplo = (char) uplo_;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ccOOiiiiiiii",
        kwlist, &A, &B, &C, &uplo, &trans, &ao, &bo, &n, &k, &ldA, &ldB,
	&ldC, &oA, &oB, &oC))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (!Matrix_Check(C)) err_mtrx("C");
    if (MAT_ID(A) != MAT_ID(B) || MAT_ID(A) != MAT_ID(C) ||
        MAT_ID(B) != MAT_ID(C)) err_conflicting_ids;

    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (MAT_ID(A) == DOUBLE && trans != 'N' && trans != 'T' &&
        trans != 'C') err_char("trans", "'N', 'T', 'C'");
    if (MAT_ID(A) == COMPLEX && trans != 'N' && trans != 'T')
	err_char("trans", "'N', 'T'");

    if (n < 0){
        n = (trans == 'N') ? A->nrows : A->ncols;
        if (n != ((trans == 'N') ? B->nrows : B->ncols)){
            PyErr_SetString(PyExc_TypeError, "dimensions of A and B "
                "do not match");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("");
    if (k < 0){
        k = (trans == 'N') ? A->ncols : A->nrows;
        if (k != ((trans == 'N') ? B->ncols : B->nrows)){
            PyErr_SetString(PyExc_TypeError, "dimensions of A and B "
                "do not match");
            return NULL;
        }
    }

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (k > 0 && ldA < MAX(1, (trans == 'N') ? n : k)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (k > 0 && ldB < MAX(1, (trans == 'N') ? n : k)) err_ld("ldB");
    if (ldC == 0) ldC = MAX(1,C->nrows);
    if (ldC < MAX(1,n)) err_ld("ldC");

    if (oA < 0) err_nn_int("offsetA");
    if (k > 0 && ((trans == 'N' && oA + (k-1)*ldA + n > len(A)) ||
        ((trans == 'T' || trans == 'C') &&
	oA + (n-1)*ldA + k > len(A))))
        err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (k > 0 && ((trans == 'N' && oB + (k-1)*ldB + n > len(B)) ||
        ((trans == 'T' || trans == 'C') &&
	oB + (n-1)*ldB + k > len(B))))
        err_buf_len("B");
    if (oC < 0) err_nn_int("offsetC");
    if (oC + (n-1)*ldC + n > len(C))  err_buf_len("C");


    if (ao && number_from_pyobject(ao, &a, MAT_ID(A)))
        err_type("alpha");
    if (bo && number_from_pyobject(bo, &b, MAT_ID(A)))
        err_type("beta");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (!ao) a.d = 1.0;
            if (!bo) b.d = 0.0;
            Py_BEGIN_ALLOW_THREADS
            dsyr2k_(&uplo, &trans, &n, &k, &a.d, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(B)+oB, &ldB, &b.d, MAT_BUFD(C)+oC, &ldC);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
            if (!ao) a.z = 1.0;
            if (!bo) b.z = 0.0;
#else
            if (!ao) a.z = _Cbuild(1.0,0.0);
            if (!bo) b.z = _Cbuild(0.0,0.0);
#endif
            Py_BEGIN_ALLOW_THREADS
            zsyr2k_(&uplo, &trans, &n, &k, &a.z, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(B)+oB, &ldB, &b.z, MAT_BUFZ(C)+oC, &ldC);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}



static char doc_her2k[] =
    "Rank-2k update of Hermitian matrix.\n\n"
    "her2k(A, B, C, alpha=1.0, beta=0.0, uplo='L', trans='N', n=None,\n"
    "      k=None, ldA=max(1,A.size[0]), ldB=max(1,B.size[0]),\n"
    "      ldC=max(1,C.size[0])), offsetA=0, offsetB=0, offsetC=0)\n\n"
    "PURPOSE\n"
    "Computes\n"
    "C := alpha*A*B^H + conj(alpha)*B*A^H + beta*C  (trans=='N')\n"
    "C := alpha*A^H*B + conj(alpha)*B^H*A + beta*C  (trans=='C')\n"
    "C is real symmetric or complex Hermitian of order n.  The inner\n"
    "dimension of the matrix product is k.  If k=0 this is interpreted\n"
    "as C := beta*C.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "C         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "trans     'N' or 'C'\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "beta      real number (int or float)\n\n"
    "n         integer.  If negative, the default value is used.\n"
    "          The default value is\n"
    "          n = (trans == 'N') ? A.size[0] : A.size[1].\n"
    "          If the default value is used, it should be equal to\n"
    "          (trans == 'N') ? B.size[0] : B.size[1].\n\n"
    "k         integer.  If negative, the default value is used.\n"
    "          The default value is\n"
    "          k = (trans == 'N') ? A.size[1] : A.size[0].\n"
    "          If the default value is used, it should be equal to\n"
    "          (trans == 'N') ? B.size[1] : B.size[0].\n\n"
    "ldA       nonnegative integer.\n"
    "          ldA >= max(1, (trans=='N') ? n : k).\n"
    "          If zero, the default value is used.\n\n"
    "ldB       nonnegative integer.\n"
    "          ldB >= max(1, (trans=='N') ? n : k).\n"
    "          If zero, the default value is used.\n\n"
    "ldC       nonnegative integer.  ldC >= max(1,n).\n"
    "          If zero, the default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer\n\n"
    "offsetC   nonnegative integer";

static PyObject* her2k(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B, *C;
    PyObject *ao=NULL, *bo=NULL;
    number a, b;
    int n=-1, k=-1, ldA=0, ldB=0, ldC=0, oA = 0, oB = 0, oC = 0;
#if PY_MAJOR_VERSION >= 3
    int trans_ = 'N', uplo_ = 'L';
#endif
    char trans = 'N', uplo = 'L';
    char *kwlist[] = {"A", "B", "C", "uplo", "trans", "alpha", "beta",
        "n", "k", "ldA", "ldB", "ldC", "offsetA", "offsetB", "offsetC",
        NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|CCOOiiiiiiii",
        kwlist, &A, &B, &C, &uplo_, &trans_, &ao, &bo, &n, &k, &ldA, &ldB,
	&ldC, &oA, &oB, &oC))
        return NULL;
    uplo = (char) uplo_;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ccOOiiiiiiii",
        kwlist, &A, &B, &C, &uplo, &trans, &ao, &bo, &n, &k, &ldA, &ldB,
	&ldC, &oA, &oB, &oC))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (!Matrix_Check(C)) err_mtrx("C");
    if (MAT_ID(A) != MAT_ID(B) || MAT_ID(A) != MAT_ID(C) ||
        MAT_ID(B) != MAT_ID(C)) err_conflicting_ids;

    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (MAT_ID(A) == DOUBLE && trans != 'N' && trans != 'T' &&
        trans != 'C') err_char("trans", "'N', 'T', 'C'");
    if (MAT_ID(A) == COMPLEX && trans != 'N' && trans != 'C')
	err_char("trans", "'N', 'C'");

    if (n < 0){
        n = (trans == 'N') ? A->nrows : A->ncols;
        if (n != ((trans == 'N') ? B->nrows : B->ncols)){
            PyErr_SetString(PyExc_TypeError, "dimensions of A and B "
                "do not match");
            return NULL;
        }
    }
    if (n == 0) return Py_BuildValue("");
    if (k < 0){
        k = (trans == 'N') ? A->ncols : A->nrows;
        if (k != ((trans == 'N') ? B->ncols : B->nrows)){
            PyErr_SetString(PyExc_TypeError, "dimensions of A and B "
                "do not match");
            return NULL;
        }
    }

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (k > 0 && ldA < MAX(1, (trans == 'N') ? n : k)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (k > 0 && ldB < MAX(1, (trans == 'N') ? n : k)) err_ld("ldB");
    if (ldC == 0) ldC = MAX(1,C->nrows);
    if (ldC < MAX(1,n)) err_ld("ldC");

    if (oA < 0) err_nn_int("offsetA");
    if (k > 0 && ((trans == 'N' && oA + (k-1)*ldA + n > len(A)) ||
        ((trans == 'T' || trans == 'C') &&
	oA + (n-1)*ldA + k > len(A))))
        err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (k > 0 && ((trans == 'N' && oB + (k-1)*ldB + n > len(B)) ||
        ((trans == 'T' || trans == 'C') &&
	oB + (n-1)*ldB + k > len(B))))
        err_buf_len("B");
    if (oC < 0) err_nn_int("offsetC");
    if (oC + (n-1)*ldC + n > len(C))  err_buf_len("C");


    if (ao && number_from_pyobject(ao, &a, MAT_ID(A)))
        err_type("alpha");
    if (bo && number_from_pyobject(bo, &b, MAT_ID(A)))
        err_type("beta");
    if (!bo) b.d = 0.0;

    switch (MAT_ID(A)){
        case DOUBLE:
            if (!ao) a.d = 1.0;
            Py_BEGIN_ALLOW_THREADS
            dsyr2k_(&uplo, &trans, &n, &k, &a.d, MAT_BUFD(A)+oA, &ldA,
                MAT_BUFD(B)+oB, &ldB, &b.d, MAT_BUFD(C)+oC, &ldC);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
	    if (!ao) a.z = 1.0;
#else
	    if (!ao) a.z = _Cbuild(1.0,0.0);
#endif
            Py_BEGIN_ALLOW_THREADS
            zher2k_(&uplo, &trans, &n, &k, &a.z, MAT_BUFZ(A)+oA, &ldA,
                MAT_BUFZ(B)+oB, &ldB, &b.d, MAT_BUFZ(C)+oC, &ldC);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static char doc_trmm[] =
    "Matrix-matrix product where one matrix is triangular.\n\n"
    "trmm(A, B, side='L', uplo='L', transA='N', diag='N', alpha=1.0,\n"
    "     m=None, n=None, ldA=max(1,A.size[0]), ldB=max(1,B.size[0]),\n"
    "     offsetA=0, offsetB=0)\n\n"
    "PURPOSE\n"
    "Computes\n"
    "B := alpha*A*B   if transA is 'N' and side = 'L'.\n"
    "B := alpha*B*A   if transA is 'N' and side = 'R'.\n"
    "B := alpha*A^T*B if transA is 'T' and side = 'L'.\n"
    "B := alpha*B*A^T if transA is 'T' and side = 'R'.\n"
    "B := alpha*A^H*B if transA is 'C' and side = 'L'.\n"
    "B := alpha*B*A^H if transA is 'C' and side = 'R'.\n"
    "B is m by n and A is triangular.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "side      'L' or 'R'\n\n"
    "uplo      'L' or 'U'\n\n"
    "transA    'N' or 'T'\n\n"
    "diag      'N' or 'U'\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "m         integer.  If negative, the default value is used.\n"
    "          The default value is\n"
    "          m = (side == 'L') ? A.size[0] : B.size[0].\n"
    "          If the default value is used and side is 'L', m must\n"
    "          be equal to A.size[1].\n\n"
    "n         integer.  If negative, the default value is used.\n"
    "          The default value is\n"
    "          n = (side == 'L') ? B.size[1] : A.size[0].\n"
    "          If the default value is used and side is 'R', n must\n"
    "          be equal to A.size[1].\n\n"
    "ldA       nonnegative integer.\n"
    "          ldA >= max(1, (side == 'L') ? m : n).\n"
    "          If zero, the default value is used. \n\n"
    "ldB       nonnegative integer.  ldB >= max(1,m).\n"
    "          If zero, the default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* trmm(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B;
    PyObject *ao=NULL;
    number a;
    int m=-1, n=-1, ldA=0, ldB=0, oA=0, oB=0;
#if PY_MAJOR_VERSION >= 3
    int side_ = 'L', uplo_ = 'L', transA_ = 'N', diag_ = 'N';
#endif
    char side = 'L', uplo = 'L', transA = 'N', diag = 'N';
    char *kwlist[] = {"A", "B", "side", "uplo", "transA", "diag",
        "alpha", "m", "n", "ldA", "ldB", "offsetA", "offsetB", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCCOiiiiii",
        kwlist, &A, &B, &side_, &uplo_, &transA_, &diag_, &ao, &m, &n,
        &ldA, &ldB, &oA, &oB))
        return NULL;
    side = (char) side_;
    uplo = (char) uplo_;
    transA = (char) transA_;
    diag = (char) diag_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ccccOiiiiii",
        kwlist, &A, &B, &side, &uplo, &transA, &diag, &ao, &m, &n, &ldA,
        &ldB, &oA, &oB))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;

    if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'N', 'U'");
    if (transA != 'N' && transA != 'T' && transA != 'C')
        err_char("transA", "'N', 'T', 'C'");

    if (n < 0){
        n = (side == 'L') ? B->ncols : A->nrows;
        if (side != 'L' && n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (m < 0){
        m = (side == 'L') ? A->nrows: B->nrows;
        if (side == 'L' && m != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (m == 0 || n == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1, (side == 'L') ? m : n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1, m)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if ((side == 'L' && oA + (m-1)*ldA + m > len(A)) ||
        (side == 'R' && oA + (n-1)*ldA + n > len(A))) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (n-1)*ldB + m > len(B)) err_buf_len("B");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(A)))
        err_type("alpha");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (!ao) a.d = 1.0;
            Py_BEGIN_ALLOW_THREADS
            dtrmm_(&side, &uplo, &transA, &diag, &m, &n, &a.d,
                MAT_BUFD(A)+oA, &ldA, MAT_BUFD(B)+oB, &ldB);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
   	    if (!ao) a.z = 1.0;
#else
   	    if (!ao) a.z = _Cbuild(1.0,0.0);
#endif
            Py_BEGIN_ALLOW_THREADS
            ztrmm_(&side, &uplo, &transA, &diag, &m, &n, &a.z,
                MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(B)+oB, &ldB);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}



static char doc_trsm[] =
    "Solution of a triangular system of equations with multiple \n"
    "righthand sides.\n\n"
    "trsm(A, B, side='L', uplo='L', transA='N', diag='N', alpha=1.0,\n"
    "     m=None, n=None, ldA=max(1,A.size[0]), ldB=max(1,B.size[0]),\n"
    "     offsetA=0, offsetB=0)\n\n"
    "PURPOSE\n"
    "Computes\n"
    "B := alpha*A^{-1}*B if transA is 'N' and side = 'L'.\n"
    "B := alpha*B*A^{-1} if transA is 'N' and side = 'R'.\n"
    "B := alpha*A^{-T}*B if transA is 'T' and side = 'L'.\n"
    "B := alpha*B*A^{-T} if transA is 'T' and side = 'R'.\n"
    "B := alpha*A^{-H}*B if transA is 'C' and side = 'L'.\n"
    "B := alpha*B*A^{-H} if transA is 'C' and side = 'R'.\n"
    "B is m by n and A is triangular.  The code does not verify \n"
    "whether A is nonsingular.\n\n"
    "ARGUMENTS\n"
    "A         'd' or 'z' matrix\n\n"
    "B         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "side      'L' or 'R'\n\n"
    "uplo      'L' or 'U'\n\n"
    "transA    'N' or 'T'\n\n"
    "diag      'N' or 'U'\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "m         integer.  If negative, the default value is used.\n"
    "          The default value is\n"
    "          m = (side == 'L') ? A.size[0] : B.size[0].\n"
    "          If the default value is used and side is 'L', m must\n"
    "          be equal to A.size[1].\n\n"
    "n         integer.  If negative, the default value is used.\n"
    "          The default value is\n"
    "          n = (side == 'L') ? B.size[1] : A.size[0].\n"
    "          If the default value is used and side is 'R', n must\n"
    "          be equal to A.size[1].\n\n"
    "ldA       nonnegative integer.\n"
    "          ldA >= max(1, (side == 'L') ? m : n).\n"
    "          If zero, the default value is used.\n\n"
    "ldB       nonnegative integer.  ldB >= max(1,m).\n"
    "          If zero, the default value is used.\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetB   nonnegative integer";

static PyObject* trsm(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *A, *B;
    PyObject *ao=NULL;
    number a;
    int m=-1, n=-1, ldA=0, ldB=0, oA=0, oB=0;
#if PY_MAJOR_VERSION >= 3
    int side_ = 'L', uplo_ = 'L', transA_ = 'N', diag_ = 'N';
#endif
    char side = 'L', uplo = 'L', transA = 'N', diag = 'N';
    char *kwlist[] = {"A", "B", "side", "uplo", "transA", "diag",
        "alpha", "m", "n", "ldA", "ldB", "offsetA", "offsetB", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCCCOiiiiii",
        kwlist, &A, &B, &side_, &uplo_, &transA_, &diag_, &ao, &m, &n,
        &ldA, &ldB, &oA, &oB))
        return NULL;
    side = (char) side_;
    uplo = (char) uplo_;
    transA = (char) transA_;
    diag = (char) diag_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ccccOiiiiii",
        kwlist, &A, &B, &side, &uplo, &transA, &diag, &ao, &m, &n, &ldA,
        &ldB, &oA, &oB))
        return NULL;
#endif

    if (!Matrix_Check(A)) err_mtrx("A");
    if (!Matrix_Check(B)) err_mtrx("B");
    if (MAT_ID(A) != MAT_ID(B)) err_conflicting_ids;

    if (side != 'L' && side != 'R') err_char("side", "'L', 'R'");
    if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
    if (diag != 'N' && diag != 'U') err_char("diag", "'N', 'U'");
    if (transA != 'N' && transA != 'T' && transA != 'C')
        err_char("transA", "'N', 'T', 'C'");

    if (n < 0){
        n = (side == 'L') ? B->ncols : A->nrows;
        if (side != 'L' && n != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (m < 0){
        m = (side == 'L') ? A->nrows: B->nrows;
        if (side == 'L' && m != A->ncols){
            PyErr_SetString(PyExc_TypeError, "A must be square");
            return NULL;
        }
    }
    if (n == 0 || m == 0) return Py_BuildValue("");

    if (ldA == 0) ldA = MAX(1,A->nrows);
    if (ldA < MAX(1, (side == 'L') ? m : n)) err_ld("ldA");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,m)) err_ld("ldB");
    if (oA < 0) err_nn_int("offsetA");
    if ((side == 'L' && oA + (m-1)*ldA + m > len(A)) ||
        (side == 'R' && oA + (n-1)*ldA + n > len(A))) err_buf_len("A");
    if (oB < 0) err_nn_int("offsetB");
    if (oB < 0 || oB + (n-1)*ldB + m > len(B)) err_buf_len("B");

    if (ao && number_from_pyobject(ao, &a, MAT_ID(A)))
        err_type("alpha");

    switch (MAT_ID(A)){
        case DOUBLE:
            if (!ao) a.d = 1.0;
             Py_BEGIN_ALLOW_THREADS
            dtrsm_(&side, &uplo, &transA, &diag, &m, &n, &a.d,
                MAT_BUFD(A)+oA, &ldA, MAT_BUFD(B)+oB, &ldB);
            Py_END_ALLOW_THREADS
            break;

        case COMPLEX:
#ifndef _MSC_VER
  	    if (!ao) a.z = 1.0;
#else
  	    if (!ao) a.z = _Cbuild(1.0,0.0);
#endif
            Py_BEGIN_ALLOW_THREADS
            ztrsm_(&side, &uplo, &transA, &diag, &m, &n, &a.z,
                MAT_BUFZ(A)+oA, &ldA, MAT_BUFZ(B)+oB, &ldB);
            Py_END_ALLOW_THREADS
            break;

        default:
            err_invalid_id;
    }

    return Py_BuildValue("");
}


static PyMethodDef blas_functions[] = {
  {"swap", (PyCFunction) swap,  METH_VARARGS|METH_KEYWORDS, doc_swap},
  {"scal", (PyCFunction) scal,  METH_VARARGS|METH_KEYWORDS, doc_scal},
  {"copy", (PyCFunction) copy,  METH_VARARGS|METH_KEYWORDS, doc_copy},
  {"axpy", (PyCFunction) axpy,  METH_VARARGS|METH_KEYWORDS, doc_axpy},
  {"dot",  (PyCFunction) dot,   METH_VARARGS|METH_KEYWORDS, doc_dot},
  {"dotu", (PyCFunction) dotu,  METH_VARARGS|METH_KEYWORDS, doc_dotu},
  {"nrm2", (PyCFunction) nrm2,  METH_VARARGS|METH_KEYWORDS, doc_nrm2},
  {"asum", (PyCFunction) asum,  METH_VARARGS|METH_KEYWORDS, doc_asum},
  {"iamax",(PyCFunction) iamax, METH_VARARGS|METH_KEYWORDS, doc_iamax},
  {"gemv", (PyCFunction) gemv,  METH_VARARGS|METH_KEYWORDS, doc_gemv},
  {"gbmv", (PyCFunction) gbmv,  METH_VARARGS|METH_KEYWORDS, doc_gbmv},
  {"symv", (PyCFunction) symv,  METH_VARARGS|METH_KEYWORDS, doc_symv},
  {"hemv", (PyCFunction) hemv,  METH_VARARGS|METH_KEYWORDS, doc_hemv},
  {"sbmv", (PyCFunction) sbmv,  METH_VARARGS|METH_KEYWORDS, doc_sbmv},
  {"hbmv", (PyCFunction) hbmv,  METH_VARARGS|METH_KEYWORDS, doc_hbmv},
  {"trmv", (PyCFunction) trmv,  METH_VARARGS|METH_KEYWORDS, doc_trmv},
  {"tbmv", (PyCFunction) tbmv,  METH_VARARGS|METH_KEYWORDS, doc_tbmv},
  {"trsv", (PyCFunction) trsv,  METH_VARARGS|METH_KEYWORDS, doc_trsv},
  {"tbsv", (PyCFunction) tbsv,  METH_VARARGS|METH_KEYWORDS, doc_tbsv},
  {"ger",  (PyCFunction) ger,   METH_VARARGS|METH_KEYWORDS, doc_ger},
  {"geru", (PyCFunction) geru,  METH_VARARGS|METH_KEYWORDS, doc_geru},
  {"syr",  (PyCFunction) syr,   METH_VARARGS|METH_KEYWORDS, doc_syr},
  {"her",  (PyCFunction) her,   METH_VARARGS|METH_KEYWORDS, doc_her},
  {"syr2", (PyCFunction) syr2,  METH_VARARGS|METH_KEYWORDS, doc_syr2},
  {"her2", (PyCFunction) her2,  METH_VARARGS|METH_KEYWORDS, doc_her2},
  {"gemm", (PyCFunction) gemm,  METH_VARARGS|METH_KEYWORDS, doc_gemm},
  {"symm", (PyCFunction) symm,  METH_VARARGS|METH_KEYWORDS, doc_symm},
  {"hemm", (PyCFunction) hemm,  METH_VARARGS|METH_KEYWORDS, doc_hemm},
  {"syrk", (PyCFunction) syrk,  METH_VARARGS|METH_KEYWORDS, doc_syrk},
  {"herk", (PyCFunction) herk,  METH_VARARGS|METH_KEYWORDS, doc_herk},
  {"syr2k",(PyCFunction) syr2k, METH_VARARGS|METH_KEYWORDS, doc_syr2k},
  {"her2k",(PyCFunction) her2k, METH_VARARGS|METH_KEYWORDS, doc_her2k},
  {"trmm", (PyCFunction) trmm,  METH_VARARGS|METH_KEYWORDS, doc_trmm},
  {"trsm", (PyCFunction) trsm,  METH_VARARGS|METH_KEYWORDS, doc_trsm},
  {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static PyModuleDef blas_module = {
    PyModuleDef_HEAD_INIT,
    "blas",
    blas__doc__,
    -1,
    blas_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_blas(void)
{
  PyObject *m;
  if (!(m = PyModule_Create(&blas_module))) return NULL;
  if (import_cvxopt() < 0) return NULL;
  return m;
}

#else

PyMODINIT_FUNC initblas(void)
{
  PyObject *m;
  m = Py_InitModule3("cvxopt.blas", blas_functions, blas__doc__);
  if (import_cvxopt() < 0) return ;
}

#endif
