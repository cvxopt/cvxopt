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

#define BASE_MODULE

#include "Python.h"
#include "cvxopt.h"
#include "misc.h"

#include <complexobject.h>

PyDoc_STRVAR(base__doc__,"Convex optimization package");

extern PyTypeObject matrix_tp ;
extern PyTypeObject matrixiter_tp ;
matrix * Matrix_New(int, int, int) ;
matrix * Matrix_NewFromMatrix(matrix *, int) ;
matrix * Matrix_NewFromSequence(PyObject *, int) ;
matrix * Matrix_NewFromPyBuffer(PyObject *, int, int *) ;

extern PyTypeObject spmatrix_tp ;
extern PyTypeObject spmatrixiter_tp ;
spmatrix * SpMatrix_New(int_t, int_t, int_t, int ) ;
spmatrix * SpMatrix_NewFromMatrix(matrix *, int) ;
spmatrix * SpMatrix_NewFromSpMatrix(spmatrix *, int) ;
spmatrix * SpMatrix_NewFromIJV(matrix *, matrix *, matrix *, int_t, int_t, int) ;
void free_ccs(ccs *);
int get_id(void *val, int val_type);

extern int (*sp_axpy[])(number, void *, void *, int, int, int, void **) ;

extern int (*sp_gemm[])(char, char, number, void *, void *, number, void *,
    int, int, int, int, void **, int, int, int);

extern int (*sp_gemv[])(char, int, int, number, void *, int, void *, int,
    number, void *, int) ;

extern int (*sp_symv[])(char, int, number, ccs *, int, void *, int,
    number, void *, int) ;

extern int (*sp_syrk[])(char, char, number, void *, number,
    void *, int, int, int, int, void **) ;

#ifndef _MSC_VER
const int  E_SIZE[] = { sizeof(int_t), sizeof(double), sizeof(double complex) };
#else
const int  E_SIZE[] = { sizeof(int_t), sizeof(double), sizeof(_Dcomplex) };
#endif

const char TC_CHAR[][2] = {"i","d","z"} ;

/*
 *  Helper routines and definitions to implement type transparency.
 */

number One[3], MinusOne[3], Zero[3];
int intOne = 1;

static void write_inum(void *dest, int i, void *src, int j) {
  ((int_t *)dest)[i]  = ((int_t *)src)[j];
}

static void write_dnum(void *dest, int i, void *src, int j) {
  ((double *)dest)[i]  = ((double *)src)[j];
}

static void write_znum(void *dest, int i, void *src, int j) {
#ifndef _MSC_VER
  ((double complex *)dest)[i]  = ((double complex *)src)[j];
#else
  ((_Dcomplex *)dest)[i]  = ((_Dcomplex *)src)[j];
#endif
}

void (*write_num[])(void *, int, void *, int) = {
    write_inum, write_dnum, write_znum };

static PyObject * inum2PyObject(void *src, int i) {
  return Py_BuildValue("l", ((int_t *)src)[i]);
}

static PyObject * dnum2PyObject(void *src, int i) {
  return Py_BuildValue("d", ((double *)src)[i]);
}

static PyObject * znum2PyObject(void *src, int i) {
  Py_complex z;
#ifndef _MSC_VER
  z.real = creal (((double complex *)src)[i]);
  z.imag = cimag (((double complex *)src)[i]);
#else
  z.real = creal (((_Dcomplex *)src)[i]);
  z.imag = cimag (((_Dcomplex *)src)[i]);
#endif
  return Py_BuildValue("D", &z);
}

PyObject * (*num2PyObject[])(void *, int) = {
    inum2PyObject, dnum2PyObject, znum2PyObject };

/* val_id: 0 = matrix, 1 = PyNumber */
static int
convert_inum(void *dest, void *val, int val_id, int_t offset)
{
  if (val_id==0) { /* 1x1 matrix */
    switch (MAT_ID(val)) {
    case INT:
      *(int_t *)dest = MAT_BUFI(val)[offset]; return 0;
      //case DOUBLE:
      //*(int_t *)dest = (int_t)round(MAT_BUFD(val)[offset]); return 0;
    default: PY_ERR_INT(PyExc_TypeError,"cannot cast argument as integer");
    }
  } else { /* PyNumber */
#if PY_MAJOR_VERSION >= 3
    if (PyLong_Check((PyObject *)val)) {
      *(int_t *)dest = PyLong_AS_LONG((PyObject *)val); return 0;
    }
#else
    if (PyInt_Check((PyObject *)val)) {
      *(int_t *)dest = PyInt_AS_LONG((PyObject *)val); return 0;
    }
#endif
    else PY_ERR_INT(PyExc_TypeError,"cannot cast argument as integer");
  }
}

static int
convert_dnum(void *dest, void *val, int val_id, int_t offset)
{
  if (val_id==0) { /* matrix */
    switch (MAT_ID(val)) {
    case INT:    *(double *)dest = MAT_BUFI(val)[offset]; return 0;
    case DOUBLE: *(double *)dest = MAT_BUFD(val)[offset]; return 0;
    default: PY_ERR_INT(PyExc_TypeError, "cannot cast argument as double");
    }
  } else { /* PyNumber */
#if PY_MAJOR_VERSION >= 3
    if (PyLong_Check((PyObject *)val) || PyFloat_Check((PyObject *)val)) {
#else
    if (PyInt_Check((PyObject *)val) || PyFloat_Check((PyObject *)val)) {
#endif
      *(double *)dest = PyFloat_AsDouble((PyObject *)val);
      return 0;
    }
    else PY_ERR_INT(PyExc_TypeError,"cannot cast argument as double");
  }
}

static int
convert_znum(void *dest, void *val, int val_id, int_t offset)
{
  if (val_id==0) { /* 1x1 matrix */
    switch (MAT_ID(val)) {
    case INT:
#ifndef _MSC_VER
      *(double complex *)dest = MAT_BUFI(val)[offset]; return 0;
#else
      *(_Dcomplex *)dest = _Cbuild((double)MAT_BUFI(val)[offset],0.0); return 0;
#endif
    case DOUBLE:
#ifndef _MSC_VER
      *(double complex *)dest = MAT_BUFD(val)[offset]; return 0;
#else
      *(_Dcomplex *)dest = _Cbuild(MAT_BUFD(val)[offset],0.0); return 0;
#endif
    case COMPLEX:
#ifndef _MSC_VER
      *(double complex *)dest = MAT_BUFZ(val)[offset]; return 0;
#else
      *(_Dcomplex *)dest = MAT_BUFZ(val)[offset]; return 0;
#endif
    default: return -1;
    }
  } else { /* PyNumber */
    Py_complex c = PyComplex_AsCComplex((PyObject *)val);
#ifndef _MSC_VER
    *(double complex *)dest = c.real + I*c.imag;
#else
    *(_Dcomplex *)dest = _Cbuild(c.real,c.imag);
#endif
    return 0;
  }
}

int (*convert_num[])(void *, void *, int, int_t) = {
    convert_inum, convert_dnum, convert_znum };

extern void daxpy_(int *, void *, void *, int *, void *, int *) ;
extern void zaxpy_(int *, void *, void *, int *, void *, int *) ;

static void i_axpy(int *n, void *a, void *x, int *incx, void *y, int *incy) {
  int i;
  for (i=0; i < *n; i++) {
    ((int_t *)y)[i*(*incy)] += *((int_t *)a)*((int_t *)x)[i*(*incx)];
  }
}

void (*axpy[])(int *, void *, void *, int *, void *, int *) = {
    i_axpy, daxpy_, zaxpy_ };

extern void dscal_(int *, void *, void *, int *) ;
extern void zscal_(int *, void *, void *, int *) ;

/* we dont implement a BLAS iscal */
static void i_scal(int *n, void *a, void *x, int *incx) {
  int i;
  for (i=0; i < *n; i++) {
    ((int_t *)x)[i*(*incx)] *= *((int_t *)a);
  }
}

void (*scal[])(int *, void *, void *, int *) = { i_scal, dscal_, zscal_ };

extern void dgemm_(char *, char *, int *, int *, int *, void *, void *,
    int *, void *, int *, void *, void *, int *) ;
extern void zgemm_(char *, char *, int *, int *, int *, void *, void *,
    int *, void *, int *, void *, void *, int *) ;

/* we dont implement a BLAS igemm */
static void i_gemm(char *transA, char *transB, int *m, int *n, int *k,
    void *alpha, void *A, int *ldA, void *B, int *ldB, void *beta,
    void *C, int *ldC)
{
  int i, j, l;
  for (j=0; j<*n; j++) {
    for (i=0; i<*m; i++) {
      ((int_t *)C)[i+j*(*m)] = 0;
      for (l=0; l<*k; l++)
        ((int_t *)C)[i+j*(*m)]+=((int_t *)A)[i+l*(*m)]*((int_t *)B)[j*(*k)+l];
    }
  }
}

void (*gemm[])(char *, char *, int *, int *, int *, void *, void *, int *,
    void *, int *, void *, void *, int *) = { i_gemm, dgemm_, zgemm_ };

extern void dgemv_(char *, int *, int *, void *, void *, int *, void *,
    int *, void *, void *, int *);
extern void zgemv_(char *, int *, int *, void *, void *, int *, void *,
    int *, void *, void *, int *);
static void (*gemv[])(char *, int *, int *, void *, void *, int *, void *,
    int *, void *, void *, int *) = { NULL, dgemv_, zgemv_ };

extern void dsyrk_(char *, char *, int *, int *, void *, void *,
    int *, void *, void *, int *);
extern void zsyrk_(char *, char *, int *, int *, void *, void *,
    int *, void *, void *, int *);
void (*syrk[])(char *, char *, int *, int *, void *, void *,
    int *, void *, void *, int *) = { NULL, dsyrk_, zsyrk_ };

extern void dsymv_(char *, int *, void *, void *, int *, void *, int *,
    void *, void *, int *);
extern void zsymv_(char *, int *, void *, void *, int *, void *, int *,
    void *, void *, int *);
void (*symv[])(char *, int *, void *, void *, int *, void *, int *,
    void *, void *, int *) = { NULL, dsymv_, zsymv_ };

static void mtx_iabs(void *src, void *dest, int n) {
  int i;
  for (i=0; i<n; i++)
    ((int_t *)dest)[i] = labs(((int_t *)src)[i]);
}

static void mtx_dabs(void *src, void *dest, int n) {
  int i;
  for (i=0; i<n; i++)
    ((double *)dest)[i] = fabs(((double *)src)[i]);
}

static void mtx_zabs(void *src, void *dest, int n) {
  int i;
  for (i=0; i<n; i++)
#ifndef _MSC_VER
    ((double *)dest)[i] = cabs(((double complex *)src)[i]);
#else
    ((double *)dest)[i] = cabs(((_Dcomplex *)src)[i]);
#endif
}

void (*mtx_abs[])(void *, void *, int) = { mtx_iabs, mtx_dabs, mtx_zabs };

static int idiv(void *dest, number a, int n) {
  if (a.i==0) PY_ERR_INT(PyExc_ZeroDivisionError, "division by zero");
  int i;
  for (i=0; i<n; i++)
    ((int_t *)dest)[i] /= a.i;

  return 0;
}

static int ddiv(void *dest, number a, int n) {
  if (a.d==0.0) PY_ERR_INT(PyExc_ZeroDivisionError, "division by zero");
  int _n = n, int1 = 1;
  double _a = 1/a.d;
  dscal_(&_n, (void *)&_a, dest, &int1);
  return 0;
}

static int zdiv(void *dest, number a, int n) {
  if (cabs(a.z) == 0.0)
    PY_ERR_INT(PyExc_ZeroDivisionError, "division by zero");

  int _n = n, int1 = 1;
#ifndef _MSC_VER
  double complex _a = 1.0/a.z;
#else
  _Dcomplex _a = _Cmulcr(conj(a.z),1.0/norm(a.z));
#endif
  zscal_(&_n, (void *)&_a, dest, &int1);
  return 0;
}

int (*div_array[])(void *, number, int) = { idiv, ddiv, zdiv };

static int mtx_irem(void *dest, number a, int n) {
  if (a.i==0) PY_ERR_INT(PyExc_ZeroDivisionError, "division by zero");
  int i;
  for (i=0; i<n; i++)
    ((int_t *)dest)[i] %= a.i;

  return 0;
}

static int mtx_drem(void *dest, number a, int n) {
  if (a.d==0.0) PY_ERR_INT(PyExc_ZeroDivisionError, "division by zero");
  int i;
  for (i=0; i<n; i++)
    ((double *)dest)[i] -= floor(((double *)dest)[i]/a.d)*a.d;

  return 0;
}

int (*mtx_rem[])(void *, number, int) = { mtx_irem, mtx_drem };

/* val_type = 0: (sp)matrix if type = 0, PY_NUMBER if type = 1 */
int get_id(void *val, int val_type) {
  if (!val_type) {
    if Matrix_Check((PyObject *)val)
    return MAT_ID((matrix *)val);
    else
      return SP_ID((spmatrix *)val);
  }
#if PY_MAJOR_VERSION >= 3
  else if (PyLong_Check((PyObject *)val))
#else
  else if (PyInt_Check((PyObject *)val))
#endif
    return INT;
  else if (PyFloat_Check((PyObject *)val))
    return DOUBLE;
  else
    return COMPLEX;
}



static int Matrix_Check_func(void *o) {
  return Matrix_Check((PyObject*)o);
}

static int SpMatrix_Check_func(void *o) {
  return SpMatrix_Check((PyObject *)o);
}


static char doc_axpy[] =
    "Constant times a vector plus a vector (y := alpha*x+y).\n\n"
    "axpy(x, y, n=None, alpha=1.0)\n\n"
    "ARGUMENTS\n"
    "x         'd' or 'z' (sp)matrix\n\n"
    "y         'd' or 'z' (sp)matrix.  Must have the same type as x.\n\n"
    "n         integer.  If n<0, the default value of n is used.\n"
    "          The default value is equal to\n"
    "          (len(x)>=offsetx+1) ? 1+(len(x)-offsetx-1)/incx : 0.\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if x is complex";

PyObject * base_axpy(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *x, *y, *partial = NULL;
  PyObject *ao=NULL;
  number a;
  char *kwlist[] = {"x", "y", "alpha", "partial", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|OO:axpy", kwlist,
      &x, &y, &ao, &partial)) return NULL;

  if (!Matrix_Check(x) && !SpMatrix_Check(x)) err_mtrx("x");
  if (!Matrix_Check(y) && !SpMatrix_Check(y)) err_mtrx("y");
  if (partial && !PyBool_Check(partial)) err_bool("partial");

  if (X_ID(x) != X_ID(y)) err_conflicting_ids;
  int id = X_ID(x);

  if (X_NROWS(x) != X_NROWS(y) || X_NCOLS(x) != X_NCOLS(y))
    PY_ERR_TYPE("dimensions of x and y do not match");

  if (ao && convert_num[id](&a, ao, 1, 0)) err_type("alpha");

  if (Matrix_Check(x) && Matrix_Check(y)) {
    int n = X_NROWS(x)*X_NCOLS(x);
    axpy[id](&n, (ao ? &a : &One[id]), MAT_BUF(x), &intOne,
        MAT_BUF(y), &intOne);
  }
  else {

    void *z = NULL;
    if (sp_axpy[id]((ao ? a : One[id]),
        Matrix_Check(x) ? MAT_BUF(x): ((spmatrix *)x)->obj,
            Matrix_Check(y) ? MAT_BUF(y): ((spmatrix *)y)->obj,
                SpMatrix_Check(x), SpMatrix_Check(y),
#if PY_MAJOR_VERSION >= 3
                partial ? PyLong_AS_LONG(partial) : 0, &z))
#else
                partial ? PyInt_AS_LONG(partial) : 0, &z))
#endif
      return PyErr_NoMemory();

    if (z) {
      free_ccs( ((spmatrix *)y)->obj );
      ((spmatrix *)y)->obj = z;
    }
  }

  return Py_BuildValue("");
}

static char doc_gemm[] =
    "General matrix-matrix product.\n\n"
    "gemm(A, B, C, transA='N', transB='N', alpha=1.0, beta=0.0, \n"
    "     partial=False) \n\n"
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
    "If k=0, this reduces to C := beta*C.\n\n"
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
    "partial   boolean. If C is sparse and partial is True, then only the\n"
    "          nonzero elements of C are updated irrespective of the\n"
    "          sparsity patterns of A and B.";

PyObject* base_gemm(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *A, *B, *C, *partial=NULL;
  PyObject *ao=NULL, *bo=NULL;
  number a, b;
  int m, n, k;
#if PY_MAJOR_VERSION >= 3
  int transA='N', transB='N';
  char transA_, transB_;
#else
  char transA='N', transB='N';
#endif
  char *kwlist[] = {"A", "B", "C", "transA", "transB", "alpha", "beta",
      "partial", NULL};

#if PY_MAJOR_VERSION >= 3
  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|CCOOO:gemm",
      kwlist, &A, &B, &C, &transA, &transB, &ao, &bo, &partial))
#else
  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ccOOO:gemm",
      kwlist, &A, &B, &C, &transA, &transB, &ao, &bo, &partial))
#endif
    return NULL;

  if (!(Matrix_Check(A) || SpMatrix_Check(A)))
    PY_ERR_TYPE("A must a matrix or spmatrix");
  if (!(Matrix_Check(B) || SpMatrix_Check(B)))
    PY_ERR_TYPE("B must a matrix or spmatrix");
  if (!(Matrix_Check(C) || SpMatrix_Check(C)))
    PY_ERR_TYPE("C must a matrix or spmatrix");
  if (partial && !PyBool_Check(partial)) err_bool("partial");

  if (X_ID(A) != X_ID(B) || X_ID(A) != X_ID(C) ||
      X_ID(B) != X_ID(C)) err_conflicting_ids;

  if (transA != 'N' && transA != 'T' && transA != 'C')
    err_char("transA", "'N', 'T', 'C'");
  if (transB != 'N' && transB != 'T' && transB != 'C')
    err_char("transB", "'N', 'T', 'C'");

  m = (transA == 'N') ? X_NROWS(A) : X_NCOLS(A);
  n = (transB == 'N') ? X_NCOLS(B) : X_NROWS(B);
  k = (transA == 'N') ? X_NCOLS(A) : X_NROWS(A);
  if (k != ((transB == 'N') ? X_NROWS(B) : X_NCOLS(B)))
    PY_ERR_TYPE("dimensions of A and B do not match");

  if (m == 0 || n == 0) return Py_BuildValue("");

  if (ao && convert_num[X_ID(A)](&a, ao, 1, 0)) err_type("alpha");
  if (bo && convert_num[X_ID(A)](&b, bo, 1, 0)) err_type("beta");

#if PY_MAJOR_VERSION >= 3
  transA_ = transA;
  transB_ = transB;
#endif

  int id = X_ID(A);
  if (Matrix_Check(A) && Matrix_Check(B) && Matrix_Check(C)) {

    int ldA = MAX(1,MAT_NROWS(A));
    int ldB = MAX(1,MAT_NROWS(B));
    int ldC = MAX(1,MAT_NROWS(C));
    if (id == INT) err_invalid_id;
#if PY_MAJOR_VERSION >= 3
    gemm[id](&transA_, &transB_, &m, &n, &k, (ao ? &a : &One[id]),
        MAT_BUF(A), &ldA, MAT_BUF(B), &ldB, (bo ? &b : &Zero[id]),
        MAT_BUF(C), &ldC);
#else
    gemm[id](&transA, &transB, &m, &n, &k, (ao ? &a : &One[id]),
        MAT_BUF(A), &ldA, MAT_BUF(B), &ldB, (bo ? &b : &Zero[id]),
        MAT_BUF(C), &ldC);
#endif
  } else {

    void *z = NULL;
#if PY_MAJOR_VERSION >= 3
    if (sp_gemm[id](transA_, transB_, (ao ? a : One[id]),
        Matrix_Check(A) ? MAT_BUF(A) : ((spmatrix *)A)->obj,
            Matrix_Check(B) ? MAT_BUF(B) : ((spmatrix *)B)->obj,
                (bo ? b : Zero[id]),
                Matrix_Check(C) ? MAT_BUF(C) : ((spmatrix *)C)->obj,
                    SpMatrix_Check(A), SpMatrix_Check(B), SpMatrix_Check(C),
                    partial ? PyLong_AS_LONG(partial) : 0, &z, m, n, k))
      return PyErr_NoMemory();
#else
    if (sp_gemm[id](transA, transB, (ao ? a : One[id]),
        Matrix_Check(A) ? MAT_BUF(A) : ((spmatrix *)A)->obj,
            Matrix_Check(B) ? MAT_BUF(B) : ((spmatrix *)B)->obj,
                (bo ? b : Zero[id]),
                Matrix_Check(C) ? MAT_BUF(C) : ((spmatrix *)C)->obj,
                    SpMatrix_Check(A), SpMatrix_Check(B), SpMatrix_Check(C),
                    partial ? PyInt_AS_LONG(partial) : 0, &z, m, n, k))
      return PyErr_NoMemory();
#endif

    if (z) {
      free_ccs( ((spmatrix *)C)->obj );
      ((spmatrix *)C)->obj = z;
    }
  }

  return Py_BuildValue("");
}

static char doc_gemv[] =
    "General matrix-vector product for sparse and dense matrices. \n\n"
    "gemv(A, x, y, trans='N', alpha=1.0, beta=0.0, m=A.size[0],\n"
    "     n=A.size[1], incx=1, incy=1, offsetA=0, offsetx=0, offsety=0)\n\n"
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
    "A         'd' or 'z' (sp)matrix\n\n"
    "x         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "y         'd' or 'z' matrix.  Must have the same type as A.\n\n"
    "trans     'N', 'T' or 'C'\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "beta      number (int, float or complex).  Complex beta is only\n"
    "          allowed if A is complex.\n\n"
    "m         integer.  If negative, the default value is used.\n\n"
    "n         integer.  If negative, the default value is used.\n\n"
    "          If zero, the default value is used.\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer\n\n"
    "This sparse version of GEMV requires that\n"
    "  m <= A.size[0] - (offsetA % A.size[0])";

static PyObject* base_gemv(PyObject *self, PyObject *args, PyObject *kwrds)
{
  matrix *x, *y;
  PyObject *A;
  PyObject *ao=NULL, *bo=NULL;
  number a, b;
  int m=-1, n=-1, ix=1, iy=1, oA=0, ox=0, oy=0;
#if PY_MAJOR_VERSION >= 3
  int trans='N';
  char trans_;
#else
  char trans='N';
#endif
  char *kwlist[] = {"A", "x", "y", "trans", "alpha", "beta", "m", "n",
      "incx", "incy", "offsetA", "offsetx",
      "offsety", NULL};

#if PY_MAJOR_VERSION >= 3
  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|COOiiiiiii:gemv",
      kwlist, &A, &x, &y, &trans, &ao, &bo, &m, &n, &ix, &iy,
      &oA, &ox, &oy))
    return NULL;
#else
  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cOOiiiiiii:gemv",
      kwlist, &A, &x, &y, &trans, &ao, &bo, &m, &n, &ix, &iy,
      &oA, &ox, &oy))
    return NULL;
#endif

  if (!Matrix_Check(A) && !SpMatrix_Check(A))
    PY_ERR(PyExc_TypeError, "A must be a dense or sparse matrix");
  if (!Matrix_Check(x)) err_mtrx("x");
  if (!Matrix_Check(y)) err_mtrx("y");

  if (MAT_ID(x) == INT || MAT_ID(y) == INT || X_ID(A) == INT)
    PY_ERR_TYPE("invalid matrix types");

  if (X_ID(A) != MAT_ID(x) || X_ID(A) != MAT_ID(y))
    err_conflicting_ids;

  if (trans != 'N' && trans != 'T' && trans != 'C')
    err_char("trans", "'N','T','C'");

  if (ix == 0) err_nz_int("incx");
  if (iy == 0) err_nz_int("incy");

  int id = MAT_ID(x);
  if (m < 0) m = X_NROWS(A);
  if (n < 0) n = X_NCOLS(A);
  if ((!m && trans == 'N') || (!n && (trans == 'T' || trans == 'C')))
    return Py_BuildValue("");

  if (oA < 0) err_nn_int("offsetA");
  if (n > 0 && m > 0 && oA + (n-1)*MAX(1,X_NROWS(A)) + m >
  X_NROWS(A)*X_NCOLS(A))
    err_buf_len("A");

  if (ox < 0) err_nn_int("offsetx");
  if ((trans == 'N' && n > 0 && ox + (n-1)*abs(ix) + 1 > MAT_LGT(x)) ||
      ((trans == 'T' || trans == 'C') && m > 0 &&
          ox + (m-1)*abs(ix) + 1 > MAT_LGT(x))) err_buf_len("x");

  if (oy < 0) err_nn_int("offsety");
  if ((trans == 'N' && oy + (m-1)*abs(iy) + 1 > MAT_LGT(y)) ||
      ((trans == 'T' || trans == 'C') &&
          oy + (n-1)*abs(iy) + 1 > MAT_LGT(y))) err_buf_len("y");

  if (ao && convert_num[MAT_ID(x)](&a, ao, 1, 0)) err_type("alpha");
  if (bo && convert_num[MAT_ID(x)](&b, bo, 1, 0)) err_type("beta");

#if PY_MAJOR_VERSION >= 3
  trans_ = trans;
#endif

  if (Matrix_Check(A)) {
    int ldA = MAX(1,X_NROWS(A));
    if (trans == 'N' && n == 0)
      scal[id](&m, (bo ? &b : &Zero[id]), (unsigned char*)MAT_BUF(y)+oy*E_SIZE[id], &iy);
    else if ((trans == 'T' || trans == 'C') && m == 0)
      scal[id](&n, (bo ? &b : &Zero[id]), (unsigned char*)MAT_BUF(y)+oy*E_SIZE[id], &iy);
    else
#if PY_MAJOR_VERSION >= 3
      gemv[id](&trans_, &m, &n, (ao ? &a : &One[id]),
	       (unsigned char*)MAT_BUF(A) + oA*E_SIZE[id], &ldA,
	       (unsigned char*)MAT_BUF(x) + ox*E_SIZE[id], &ix, (bo ? &b : &Zero[id]),
	       (unsigned char*)MAT_BUF(y) + oy*E_SIZE[id], &iy);
#else
      gemv[id](&trans, &m, &n, (ao ? &a : &One[id]),
	       (unsigned char*)MAT_BUF(A) + oA*E_SIZE[id], &ldA,
	       (unsigned char*)MAT_BUF(x) + ox*E_SIZE[id], &ix, (bo ? &b : &Zero[id]),
	       (unsigned char*)MAT_BUF(y) + oy*E_SIZE[id], &iy);
#endif
  } else {
#if PY_MAJOR_VERSION >= 3
    if (sp_gemv[id](trans_, m, n, (ao ? a : One[id]), ((spmatrix *)A)->obj,
		    oA, (unsigned char*)MAT_BUF(x) + ox*E_SIZE[id], ix, (bo ? b : Zero[id]),
		    (unsigned char*)MAT_BUF(y) + oy*E_SIZE[id], iy))
      return PyErr_NoMemory();
#else
    if (sp_gemv[id](trans, m, n, (ao ? a : One[id]), ((spmatrix *)A)->obj,
		    oA, (unsigned char*)MAT_BUF(x) + ox*E_SIZE[id], ix, (bo ? b : Zero[id]),
		    (unsigned char*)MAT_BUF(y) + oy*E_SIZE[id], iy))
      return PyErr_NoMemory();
#endif
  }

  return Py_BuildValue("");
}

static char doc_syrk[] =
    "Rank-k update of symmetric sparse or dense matrix.\n\n"
    "syrk(A, C, uplo='L', trans='N', alpha=1.0, beta=0.0, partial=False)\n\n"
    "PURPOSE   \n"
    "If trans is 'N', computes C := alpha*A*A^T + beta*C.\n"
    "If trans is 'T', computes C := alpha*A^T*A + beta*C.\n"
    "C is symmetric (real or complex) of order n. \n"
    "ARGUMENTS\n"
    "A         'd' or 'z' (sp)matrix\n\n"
    "C         'd' or 'z' (sp)matrix.  Must have the same type as A.\n\n"
    "uplo      'L' or 'U'\n\n"
    "trans     'N' or 'T'\n\n"
    "alpha     number (int, float or complex).  Complex alpha is only\n"
    "          allowed if A is complex.\n\n"
    "beta      number (int, float or complex).  Complex beta is only\n"
    "          allowed if A is complex.\n\n"
    "partial   boolean. If C is sparse and partial is True, then only the\n"
    "          nonzero elements of C are updated irrespective of the\n"
    "          sparsity patterns of A.";

static PyObject* base_syrk(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *A, *C, *partial=NULL, *ao=NULL, *bo=NULL;
  number a, b;
#if PY_MAJOR_VERSION >= 3
  int trans='N', uplo='L';
  char trans_, uplo_;
#else
  char trans='N', uplo='L';
#endif
  char *kwlist[] = {"A", "C", "uplo", "trans", "alpha", "beta", "partial",
      NULL};

#if PY_MAJOR_VERSION >= 3
  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CCOOO:syrk", kwlist,
      &A, &C, &uplo, &trans, &ao, &bo, &partial))
#else
  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ccOOO:syrk", kwlist,
      &A, &C, &uplo, &trans, &ao, &bo, &partial))
#endif
    return NULL;

  if (!(Matrix_Check(A) || SpMatrix_Check(A)))
    PY_ERR_TYPE("A must be a dense or sparse matrix");
  if (!(Matrix_Check(C) || SpMatrix_Check(C)))
    PY_ERR_TYPE("C must be a dense or sparse matrix");

  int id = X_ID(A);
  if (id == INT) PY_ERR_TYPE("invalid matrix types");
  if (id != X_ID(C)) err_conflicting_ids;

  if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");
  if (id == DOUBLE && trans != 'N' && trans != 'T' &&
      trans != 'C') err_char("trans", "'N', 'T', 'C'");
  if (id == COMPLEX && trans != 'N' && trans != 'T')
    err_char("trans", "'N', 'T'");

  if (partial && !PyBool_Check(partial)) err_bool("partial");

  int n = (trans == 'N') ? X_NROWS(A) : X_NCOLS(A);
  int k = (trans == 'N') ? X_NCOLS(A) : X_NROWS(A);
  if (n == 0) return Py_BuildValue("");

  if (ao && convert_num[id](&a, ao, 1, 0)) err_type("alpha");
  if (bo && convert_num[id](&b, bo, 1, 0)) err_type("beta");

#if PY_MAJOR_VERSION >= 3
  trans_ = trans;
  uplo_ = uplo;
#endif

  if (Matrix_Check(A) && Matrix_Check(C)) {

    int ldA = MAX(1,MAT_NROWS(A));
    int ldC = MAX(1,MAT_NROWS(C));

#if PY_MAJOR_VERSION >= 3
    syrk[id](&uplo_, &trans_, &n, &k, (ao ? &a : &One[id]),
        MAT_BUF(A), &ldA, (bo ? &b : &Zero[id]), MAT_BUF(C), &ldC);
#else
    syrk[id](&uplo, &trans, &n, &k, (ao ? &a : &One[id]),
        MAT_BUF(A), &ldA, (bo ? &b : &Zero[id]), MAT_BUF(C), &ldC);
#endif
  } else {

    void *z = NULL;
#if PY_MAJOR_VERSION >= 3
    if (sp_syrk[id](uplo_, trans_,
        (ao ? a : One[id]),
        Matrix_Check(A) ? MAT_BUF(A) : ((spmatrix *)A)->obj,
            (bo ? b : Zero[id]),
            Matrix_Check(C) ? MAT_BUF(C) : ((spmatrix *)C)->obj,
                SpMatrix_Check(A), SpMatrix_Check(C),
                partial ? PyLong_AS_LONG(partial) : 0,
                    (trans == 'N' ? X_NCOLS(A) : X_NROWS(A)), &z))
#else
    if (sp_syrk[id](uplo, trans,
        (ao ? a : One[id]),
        Matrix_Check(A) ? MAT_BUF(A) : ((spmatrix *)A)->obj,
            (bo ? b : Zero[id]),
            Matrix_Check(C) ? MAT_BUF(C) : ((spmatrix *)C)->obj,
                SpMatrix_Check(A), SpMatrix_Check(C),
                partial ? PyInt_AS_LONG(partial) : 0,
                    (trans == 'N' ? X_NCOLS(A) : X_NROWS(A)), &z))
#endif
      return PyErr_NoMemory();

    if (z) {
      free_ccs( ((spmatrix *)C)->obj );
      ((spmatrix *)C)->obj = z;
    }
  }

  return Py_BuildValue("");
}

static char doc_symv[] =
    "Matrix-vector product with a real symmetric dense or sparse matrix.\n\n"
    "symv(A, x, y, uplo='L', alpha=1.0, beta=0.0, n=A.size[0], \n"
    "     ldA=max(1,A.size[0]), incx=1, incy=1, offsetA=0, offsetx=0,\n"
    "     offsety=0)\n\n"
    "PURPOSE\n"
    "Computes y := alpha*A*x + beta*y with A real symmetric of order n."
    "n\n"
    "ARGUMENTS\n"
    "A         'd' (sp)matrix\n\n"
    "x         'd' matrix\n\n"
    "y         'd' matrix\n\n"
    "uplo      'L' or 'U'\n\n"
    "alpha     real number (int or float)\n\n"
    "beta      real number (int or float)\n\n"
    "n         integer.  If negative, the default value is used.\n"
    "          If the default value is used, we require that\n"
    "          A.size[0]=A.size[1].\n\n"
    "incx      nonzero integer\n\n"
    "incy      nonzero integer\n\n"
    "offsetA   nonnegative integer\n\n"
    "offsetx   nonnegative integer\n\n"
    "offsety   nonnegative integer\n\n"
    "This sparse version of SYMV requires that\n"
    "  m <= A.size[0] - (offsetA % A.size[0])";

static PyObject* base_symv(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *A, *ao=NULL, *bo=NULL;
  matrix *x, *y;
  number a, b;
  int n=-1, ix=1, iy=1, oA=0, ox=0, oy=0, ldA;
#if PY_MAJOR_VERSION >= 3
  int uplo='L';
  char uplo_;
#else
  char uplo='L';
#endif
  char *kwlist[] = {"A", "x", "y", "uplo", "alpha", "beta", "n",
      "incx", "incy", "offsetA", "offsetx", "offsety", NULL};

#if PY_MAJOR_VERSION >= 3
  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|COOiiiiii:symv",
      kwlist, &A, &x, &y, &uplo, &ao, &bo, &n, &ix, &iy, &oA, &ox, &oy))
#else
  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|cOOiiiiii:symv",
      kwlist, &A, &x, &y, &uplo, &ao, &bo, &n, &ix, &iy, &oA, &ox, &oy))
#endif
    return NULL;

  if (!Matrix_Check(A) && !SpMatrix_Check(A))
    PY_ERR_TYPE("A must be a dense or sparse matrix");

  ldA = MAX(1,X_NROWS(A));

  if (!Matrix_Check(x)) err_mtrx("x");
  if (!Matrix_Check(y)) err_mtrx("y");
  if (X_ID(A) != MAT_ID(x) || X_ID(A) != MAT_ID(y) ||
      MAT_ID(x) != MAT_ID(y)) err_conflicting_ids;

  int id = X_ID(A);
  if (id == INT) PY_ERR_TYPE("invalid matrix types");

  if (uplo != 'L' && uplo != 'U') err_char("uplo", "'L', 'U'");

  if (ix == 0) err_nz_int("incx");
  if (iy == 0) err_nz_int("incy");

  if (n < 0) {
    if (X_NROWS(A) != X_NCOLS(A)) {
      PyErr_SetString(PyExc_ValueError, "A is not square");
      return NULL;
    }
    n = X_NROWS(A);
  }
  if (n == 0) return Py_BuildValue("");

  if (oA < 0) err_nn_int("offsetA");
  if (oA + (n-1)*ldA + n > len(A)) err_buf_len("A");
  if (ox < 0) err_nn_int("offsetx");
  if (ox + (n-1)*abs(ix) + 1 > len(x)) err_buf_len("x");
  if (oy < 0) err_nn_int("offsety");
  if (oy + (n-1)*abs(iy) + 1 > len(y)) err_buf_len("y");

  if (ao && convert_num[id](&a, ao, 1, 0)) err_type("alpha");
  if (bo && convert_num[id](&b, bo, 1, 0)) err_type("beta");

#if PY_MAJOR_VERSION >= 3
  uplo_ = uplo;
#endif

  if (Matrix_Check(A)) {

#if PY_MAJOR_VERSION >= 3
    symv[id](&uplo_, &n, (ao ? &a : &One[id]),
	     (unsigned char*)MAT_BUF(A) + oA*E_SIZE[id], &ldA, (unsigned char*)MAT_BUF(x) + ox*E_SIZE[id],
	     &ix, (bo ? &b : &Zero[id]), (unsigned char*)MAT_BUF(y) + oy*E_SIZE[id], &iy);
#else
    symv[id](&uplo, &n, (ao ? &a : &One[id]),
	     (unsigned char*)MAT_BUF(A) + oA*E_SIZE[id], &ldA, ((char*)MAT_BUF(x)) + ox*E_SIZE[id],
	     &ix, (bo ? &b : &Zero[id]), (unsigned char*)MAT_BUF(y) + oy*E_SIZE[id], &iy);
#endif
  }
  else {

#if PY_MAJOR_VERSION >= 3
    if (sp_symv[id](uplo_, n, (ao ? a : One[id]), ((spmatrix *)A)->obj,
		    oA, (unsigned char*)MAT_BUF(x) + ox*E_SIZE[id], ix,
		    (bo ? b : Zero[id]), (unsigned char*)MAT_BUF(y) + oy*E_SIZE[id], iy))
#else
    if (sp_symv[id](uplo, n, (ao ? a : One[id]), ((spmatrix *)A)->obj,
		    oA, (unsigned char*)MAT_BUF(x) + ox*E_SIZE[id], ix,
		    (bo ? b : Zero[id]), (unsigned char*)MAT_BUF(y) + oy*E_SIZE[id], iy))
#endif
      return PyErr_NoMemory();
  }

  return Py_BuildValue("");
}

spmatrix * sparse_concat(PyObject *L, int id_arg) ;

static char doc_sparse[] =
    "Constructs a sparse block matrix.\n\n"
    "sparse(x, tc = None)\n\n"
    "PURPOSE\n"
    "Constructs a sparse block matrix from a list of block matrices.  If a\n"
    "single matrix is given as argument,  then the matrix is converted to\n"
    "sparse format, optionally with a different typcode.  If a single list\n"
    "of subblocks is specified, then a block column matrix is created;\n"
    "otherwise when a list of lists is specified, then the inner lists\n"
    "specify the different block-columns.  Each block element must be either\n"
    "a dense or sparse matrix, or a scalar,  and dense matrices are converted\n"
    "to sparse format by removing 0 elements.\n\n"
    "ARGUMENTS\n"
    "x       a single matrix, or a list of matrices and scalars, or a list of\n"
    "        lists of matrices and scalars\n\n"
    "tc      typecode character 'd' or 'z'.";


static PyObject *
sparse(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  PyObject *Objx = NULL;
  static char *kwlist[] = { "x", "tc", NULL};

#if PY_MAJOR_VERSION >= 3
  int tc = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|C:sparse", kwlist,
      &Objx, &tc))
#else
  char tc = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|c:sparse", kwlist,
      &Objx, &tc))
#endif
    return NULL;

  if (tc && !(VALID_TC_SP(tc))) PY_ERR_TYPE("tc must be 'd' or 'z'");
  int id = (tc ? TC2ID(tc) : -1);

  spmatrix *ret = NULL;
  /* a matrix */
  if (Matrix_Check(Objx)) {

    int m = MAT_NROWS(Objx), n = MAT_NCOLS(Objx);
    ret = SpMatrix_NewFromMatrix((matrix *)Objx,
        (id == -1 ? MAX(DOUBLE,MAT_ID(Objx)) : id));

    MAT_NROWS(Objx) = m; MAT_NCOLS(Objx) = n;
  }

  /* sparse matrix */
  else if (SpMatrix_Check(Objx)) {

    int_t nnz = 0, ik, jk;

    for (jk=0; jk<SP_NCOLS(Objx); jk++) {
      for (ik=SP_COL(Objx)[jk]; ik<SP_COL(Objx)[jk+1]; ik++) {
#ifndef _MSC_VER
        if (((SP_ID(Objx) == DOUBLE) && (SP_VALD(Objx)[ik] != 0.0)) ||
            ((SP_ID(Objx) == COMPLEX) && (SP_VALZ(Objx)[ik] != 0.0)))
#else
        if (((SP_ID(Objx) == DOUBLE) && (SP_VALD(Objx)[ik] != 0.0)) ||
            ((SP_ID(Objx) == COMPLEX) && (creal(SP_VALZ(Objx)[ik]) != 0.0 || cimag(SP_VALZ(Objx)[ik]) != 0.0)))
#endif
          nnz++;
      }
    }

    ret = SpMatrix_New(SP_NROWS(Objx), SP_NCOLS(Objx), nnz, SP_ID(Objx));
    if (!ret) return NULL;

    nnz = 0;
    for (jk=0; jk<SP_NCOLS(Objx); jk++) {
      for (ik=SP_COL(Objx)[jk]; ik<SP_COL(Objx)[jk+1]; ik++) {
        if ((SP_ID(Objx) == DOUBLE) && (SP_VALD(Objx)[ik] != 0.0)) {
          SP_VALD(ret)[nnz] = SP_VALD(Objx)[ik];
          SP_ROW(ret)[nnz++] = SP_ROW(Objx)[ik];
          SP_COL(ret)[jk+1]++;
        }
#ifndef _MSC_VER
        else if ((SP_ID(Objx) == COMPLEX) && (SP_VALZ(Objx)[ik] != 0.0)) {
#else
        else if ((SP_ID(Objx) == COMPLEX) && (creal(SP_VALZ(Objx)[ik]) != 0.0 || cimag(SP_VALZ(Objx)[ik]) != 0.0)) {
#endif
          SP_VALZ(ret)[nnz] = SP_VALZ(Objx)[ik];
          SP_ROW(ret)[nnz++] = SP_ROW(Objx)[ik];
          SP_COL(ret)[jk+1]++;
        }
      }
    }

    for (jk=0; jk<SP_NCOLS(Objx); jk++)
      SP_COL(ret)[jk+1] += SP_COL(ret)[jk];
  }

  /* x is a list of lists */
  else if (PyList_Check(Objx))
    ret = sparse_concat(Objx, id);

  else PY_ERR_TYPE("invalid matrix initialization");

  return (PyObject *)ret;
}

static char doc_spdiag[] =
    "Constructs a square block diagonal sparse matrix.\n\n"
    "spdiag(diag)\n\n"
    "ARGUMENTS\n"
    "diag      a matrix with a single row or column,  or a list of matrices\n"
    "          and scalars.";

static PyObject *
spdiag(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  PyObject *diag = NULL, *Dk;
  static char *kwlist[] = { "diag", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:spdiag", kwlist, &diag))
    return NULL;

  if ((!PyList_Check(diag) && !Matrix_Check(diag) && !SpMatrix_Check(diag)) ||
      (Matrix_Check(diag) &&
          (MAT_LGT(diag) != MAX(MAT_NROWS(diag),MAT_NCOLS(diag)) )) ||
          (SpMatrix_Check(diag) &&
              (SP_LGT(diag) != MAX(SP_NROWS(diag),SP_NCOLS(diag)) )) )
    PY_ERR_TYPE("invalid diag argument");

  if (Matrix_Check(diag)) {
    int j, id = MAX(DOUBLE, MAT_ID(diag)), n = MAT_LGT(diag);

    spmatrix *ret = SpMatrix_New((int_t)n, (int_t)n, (int_t)n, id);
    if (!ret) return NULL;
    SP_COL(ret)[0] = 0;

    for (j=0; j<n; j++) {
      SP_COL(ret)[j+1] = j+1;
      SP_ROW(ret)[j] = j;

      if (MAT_ID(diag) == INT)
        SP_VALD(ret)[j] = MAT_BUFI(diag)[j];
      else if (MAT_ID(diag) == DOUBLE)
        SP_VALD(ret)[j] = MAT_BUFD(diag)[j];
      else
        SP_VALZ(ret)[j] = MAT_BUFZ(diag)[j];
    }
    return (PyObject *)ret;
  }
  else if (SpMatrix_Check(diag)) {

    int k, id = MAX(DOUBLE, SP_ID(diag));
    int_t n = SP_LGT(diag);

    spmatrix *ret = SpMatrix_New(n, n, SP_NNZ(diag), id);
    if (!ret) return NULL;
    SP_COL(ret)[0] = 0;

    for (k=0; k<SP_NNZ(diag); k++) {

      SP_COL(ret)[SP_ROW(diag)[k]+1] = 1;
      SP_ROW(ret)[k] = SP_ROW(diag)[k];
      if (SP_ID(diag) == DOUBLE)
        SP_VALD(ret)[k] = SP_VALD(diag)[k];
      else
        SP_VALZ(ret)[k] = SP_VALZ(diag)[k];
    }

    for (k=0; k<n; k++) SP_COL(ret)[k+1] += SP_COL(ret)[k];

    return (PyObject *)ret;

  }

  int j, k, l, idx, id=DOUBLE;
  int_t n=0, nnz=0;

  for (k=0; k<PyList_GET_SIZE(diag); k++) {
    Dk = PyList_GET_ITEM(diag, k);
    if (!Matrix_Check(Dk) && !SpMatrix_Check(Dk) && !PyNumber_Check(Dk))
      PY_ERR_TYPE("invalid element in diag");

    if (PyNumber_Check(Dk)) {
      int scalarid = (PyComplex_Check(Dk) ? COMPLEX :
      (PyFloat_Check(Dk) ? DOUBLE : INT));
      id = MAX(id, scalarid);
      nnz += 1;
      n += 1;
    } else {
      if (X_NROWS(Dk) != X_NCOLS(Dk))
        PY_ERR_TYPE("the elements in diag must be square");

      n   += X_NCOLS(Dk);
      nnz += (Matrix_Check(Dk) ? X_NROWS(Dk)*X_NROWS(Dk) : SP_NNZ(Dk));
      id   = MAX(id, X_ID(Dk));
    }
  }

  spmatrix *ret = SpMatrix_New(n, n, nnz, id);
  if (!ret) return NULL;
  SP_COL(ret)[0] = 0;

  n = 0, idx = 0;
  for (k=0; k<PyList_GET_SIZE(diag); k++) {
    Dk = PyList_GET_ITEM(diag, k);

    if (PyNumber_Check(Dk)) {
      SP_COL(ret)[n+1] = SP_COL(ret)[n] + 1;
      SP_ROW(ret)[idx] = n;

      number val;
      convert_num[id](&val, Dk, 1, 0);
      write_num[id](SP_VAL(ret), idx, &val, 0);
      idx += 1;
      n   += 1;
    }
    else {
      for (j=0; j<X_NCOLS(Dk); j++) {

        if (Matrix_Check(Dk)) {

          SP_COL(ret)[j+n+1] = SP_COL(ret)[j+n] + X_NROWS(Dk);
          for (l=0; l<X_NROWS(Dk); l++) {
            SP_ROW(ret)[idx] = n + l;
            if (id == DOUBLE)
              SP_VALD(ret)[idx] = (MAT_ID(Dk) == DOUBLE ?
                  MAT_BUFD(Dk)[l + j*MAT_NROWS(Dk)] :
              MAT_BUFI(Dk)[l + j*MAT_NROWS(Dk)]);
            else
#ifndef _MSC_VER
              SP_VALZ(ret)[idx] = (MAT_ID(Dk) == COMPLEX ?
                  MAT_BUFZ(Dk)[l + j*MAT_NROWS(Dk)] :
              (MAT_ID(Dk) == DOUBLE ? MAT_BUFD(Dk)[l + j*MAT_NROWS(Dk)] :
              MAT_BUFI(Dk)[l + j*MAT_NROWS(Dk)]));
#else
              SP_VALZ(ret)[idx] = (MAT_ID(Dk) == COMPLEX ?
                  MAT_BUFZ(Dk)[l + j*MAT_NROWS(Dk)] :
              (MAT_ID(Dk) == DOUBLE ? _Cbuild(MAT_BUFD(Dk)[l + j*MAT_NROWS(Dk)],0.0) :
	       _Cbuild((double)MAT_BUFI(Dk)[l + j*MAT_NROWS(Dk)],0.0)));
#endif
            idx++;
          }
        } else {

          SP_COL(ret)[j+n+1] = SP_COL(ret)[j+n] + SP_COL(Dk)[j+1]-SP_COL(Dk)[j];
          for (l=SP_COL(Dk)[j]; l<SP_COL(Dk)[j+1]; l++) {
            SP_ROW(ret)[idx] = n + SP_ROW(Dk)[l];
            if (id == DOUBLE)
              SP_VALD(ret)[idx] = SP_VALD(Dk)[l];
            else
#ifndef _MSC_VER
              SP_VALZ(ret)[idx] = (SP_ID(Dk) == COMPLEX ?
                  SP_VALZ(Dk)[l] : SP_VALD(Dk)[l]);
#else
              SP_VALZ(ret)[idx] = (SP_ID(Dk) == COMPLEX ?
		  SP_VALZ(Dk)[l] : _Cbuild(SP_VALD(Dk)[l],0.0));
#endif
            idx++;
          }
        }
      }
      n += X_NCOLS(Dk);
    }
  }
  return (PyObject *)ret;
}

matrix * dense(spmatrix *self);

PyObject * matrix_elem_max(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *A, *B;
  if (!PyArg_ParseTuple(args, "OO:emax", &A, &B)) return NULL;

  if (!(X_Matrix_Check(A) || PyNumber_Check(A)) ||
      !(X_Matrix_Check(B) || PyNumber_Check(B)))
    PY_ERR_TYPE("arguments must be either matrices or python numbers");

  if (PyComplex_Check(A) || (X_Matrix_Check(A) && X_ID(A)==COMPLEX))
    PY_ERR_TYPE("ordering not defined for complex numbers");

  if (PyComplex_Check(B) || (X_Matrix_Check(B) && X_ID(B)==COMPLEX))
    PY_ERR_TYPE("ordering not defined for complex numbers");

  int a_is_number = PyNumber_Check(A) ||
      (Matrix_Check(A) && MAT_LGT(A) == 1) ||
      (SpMatrix_Check(A) && SP_LGT(A) == 1);
  int b_is_number = PyNumber_Check(B) ||
      (Matrix_Check(B) && MAT_LGT(B) == 1) ||
      (SpMatrix_Check(B) && SP_LGT(B) == 1);

  int ida = PyNumber_Check(A) ? PyFloat_Check(A) : X_ID(A);
  int idb = PyNumber_Check(B) ? PyFloat_Check(B) : X_ID(B);
  int id  = MAX( ida, idb );

  number a, b;
  if (a_is_number) {
    if (PyNumber_Check(A) || Matrix_Check(A))
      convert_num[id](&a, A, PyNumber_Check(A), 0);
    else
      a.d = ((SP_LGT(A) > 0) ? SP_VALD(A)[0] : 0.0);
  }
  if (b_is_number) {
    if (PyNumber_Check(B) || Matrix_Check(B))
      convert_num[id](&b, B, PyNumber_Check(B), 0);
    else
      b.d = ((SP_LGT(B) > 0) ? SP_VALD(B)[0] : 0.0);
  }

  if ((a_is_number && b_is_number) &&
      (!X_Matrix_Check(A) && !X_Matrix_Check(B))) {
    if (id == INT)
      return Py_BuildValue("i", MAX(a.i, b.i) );
    else
      return Py_BuildValue("d", MAX(a.d, b.d) );
  }

  if (!(a_is_number || b_is_number)) {
    if (X_NROWS(A) != X_NROWS(B) || X_NCOLS(A) != X_NCOLS(B))
      PY_ERR_TYPE("incompatible dimensions");
  }

  int_t m = ( !a_is_number ? X_NROWS(A) : (!b_is_number ? X_NROWS(B) : 1));
  int_t n = ( !a_is_number ? X_NCOLS(A) : (!b_is_number ? X_NCOLS(B) : 1));

  if ((Matrix_Check(A) || a_is_number) || (Matrix_Check(B) || b_is_number)) {

    int freeA = SpMatrix_Check(A) && (SP_LGT(A) > 1);
    int freeB = SpMatrix_Check(B) && (SP_LGT(B) > 1);
    if (freeA) {
      if (!(A = (PyObject *)dense((spmatrix *)A)) ) return NULL;
    }
    if (freeB) {
      if (!(B = (PyObject *)dense((spmatrix *)B)) ) return NULL;
    }

    PyObject *ret = (PyObject *)Matrix_New((int)m, (int)n, id);
    if (!ret) {
      if (freeA) { Py_DECREF(A); }
      if (freeB) { Py_DECREF(B); }
      return NULL;
    }
    int_t i;
    for (i=0; i<m*n; i++) {
      if (!a_is_number) convert_num[id](&a, A, 0, i);
      if (!b_is_number) convert_num[id](&b, B, 0, i);

      if (id == INT)
        MAT_BUFI(ret)[i] = MAX(a.i, b.i);
      else
        MAT_BUFD(ret)[i] = MAX(a.d, b.d);
    }

    if (freeA) { Py_DECREF(A); }
    if (freeB) { Py_DECREF(B); }
    return ret;

  } else {

    spmatrix *ret = SpMatrix_New(m, n, 0, DOUBLE);
    if (!ret) return NULL;

    int_t j, ka = 0, kb = 0, kret = 0;
    for (j=0; j<n; j++) {

      while (ka < SP_COL(A)[j+1] && kb < SP_COL(B)[j+1]) {

        if (SP_ROW(A)[ka] < SP_ROW(B)[kb]) {
          if (SP_VALD(A)[ka++] > 0.0) SP_COL(ret)[j+1]++;
        }
        else if (SP_ROW(A)[ka] > SP_ROW(B)[kb]) {
          if (SP_VALD(B)[kb++] > 0.0) SP_COL(ret)[j+1]++;
        }
        else {
          SP_COL(ret)[j+1]++; ka++; kb++;
        }
      }

      while (ka < SP_COL(A)[j+1]) {
        if (SP_VALD(A)[ka++] > 0.0) SP_COL(ret)[j+1]++;
      }

      while (kb < SP_COL(B)[j+1]) {
        if (SP_VALD(B)[kb++] > 0.0) SP_COL(ret)[j+1]++;
      }

    }

    for (j=0; j<n; j++) SP_COL(ret)[j+1] += SP_COL(ret)[j];

    int_t *newrow = malloc( sizeof(int_t)*SP_COL(ret)[n] );
    double *newval = malloc( sizeof(double)*SP_COL(ret)[n] );
    if (!newrow || !newval) {
      free(newrow); free(newval); Py_DECREF(ret);
      return PyErr_NoMemory();
    }
    free( ret->obj->rowind );
    free( ret->obj->values );
    ret->obj->rowind = newrow;
    ret->obj->values = newval;

    ka = 0; kb = 0;
    for (j=0; j<n; j++) {

      while (ka < SP_COL(A)[j+1] && kb < SP_COL(B)[j+1]) {

        if (SP_ROW(A)[ka] < SP_ROW(B)[kb]) {
          if (SP_VALD(A)[ka] > 0.0) {
            SP_ROW(ret)[kret] = SP_ROW(A)[ka];
            SP_VALD(ret)[kret++] = SP_VALD(A)[ka];
          }
          ka++;
        }
        else if (SP_ROW(A)[ka] > SP_ROW(B)[kb]) {
          if (SP_VALD(B)[kb] > 0.0) {
            SP_ROW(ret)[kret] = SP_ROW(B)[kb];
            SP_VALD(ret)[kret++] = SP_VALD(B)[kb];
          }
          kb++;
        }
        else {
          SP_ROW(ret)[kret] = SP_ROW(A)[ka];
          SP_VALD(ret)[kret] = MAX(SP_VALD(A)[ka], SP_VALD(B)[kb]);
          kret++; ka++; kb++;
        }
      }

      while (ka < SP_COL(A)[j+1]) {
        if (SP_VALD(A)[ka] > 0.0) {
          SP_ROW(ret)[kret] = SP_ROW(A)[ka];
          SP_VALD(ret)[kret++] = SP_VALD(A)[ka];
        }
        ka++;
      }

      while (kb < SP_COL(B)[j+1]) {
        if (SP_VALD(B)[kb] > 0.0) {
          SP_ROW(ret)[kret] = SP_ROW(B)[kb];
          SP_VALD(ret)[kret++] = SP_VALD(B)[kb];
        }
        kb++;
      }
    }

    return (PyObject *)ret;
  }
}

PyObject * matrix_elem_min(PyObject *self, PyObject *args, PyObject *kwrds)
{
  PyObject *A, *B;
  if (!PyArg_ParseTuple(args, "OO:emin", &A, &B)) return NULL;

  if (!(X_Matrix_Check(A) || PyNumber_Check(A)) ||
      !(X_Matrix_Check(B) || PyNumber_Check(B)))
    PY_ERR_TYPE("arguments must be either matrices or python numbers");

  if (PyComplex_Check(A) || (X_Matrix_Check(A) && X_ID(A)==COMPLEX))
    PY_ERR_TYPE("ordering not defined for complex numbers");

  if (PyComplex_Check(B) || (X_Matrix_Check(B) && X_ID(B)==COMPLEX))
    PY_ERR_TYPE("ordering not defined for complex numbers");

  int a_is_number = PyNumber_Check(A) ||
      (Matrix_Check(A) && MAT_LGT(A) == 1) ||
      (SpMatrix_Check(A) && SP_LGT(A) == 1);
  int b_is_number = PyNumber_Check(B) ||
      (Matrix_Check(B) && MAT_LGT(B) == 1) ||
      (SpMatrix_Check(B) && SP_LGT(B) == 1);

  int ida = PyNumber_Check(A) ? PyFloat_Check(A) : X_ID(A);
  int idb = PyNumber_Check(B) ? PyFloat_Check(B) : X_ID(B);
  int id  = MAX( ida, idb );

  number a, b;
  if (a_is_number) {
    if (PyNumber_Check(A) || Matrix_Check(A))
      convert_num[id](&a, A, PyNumber_Check(A), 0);
    else
      a.d = ((SP_LGT(A) > 0) ? SP_VALD(A)[0] : 0.0);
  }
  if (b_is_number) {
    if (PyNumber_Check(B) || Matrix_Check(B))
      convert_num[id](&b, B, PyNumber_Check(B), 0);
    else
      b.d = ((SP_LGT(B) > 0) ? SP_VALD(B)[0] : 0.0);
  }

  if ((a_is_number && b_is_number) &&
      (!X_Matrix_Check(A) && !X_Matrix_Check(B))) {
    if (id == INT)
      return Py_BuildValue("i", MIN(a.i, b.i) );
    else
      return Py_BuildValue("d", MIN(a.d, b.d) );
  }

  if (!(a_is_number || b_is_number)) {
    if (X_NROWS(A) != X_NROWS(B) || X_NCOLS(A) != X_NCOLS(B))
      PY_ERR_TYPE("incompatible dimensions");
  }

  int_t m = ( !a_is_number ? X_NROWS(A) : (!b_is_number ? X_NROWS(B) : 1));
  int_t n = ( !a_is_number ? X_NCOLS(A) : (!b_is_number ? X_NCOLS(B) : 1));

  if ((Matrix_Check(A) || a_is_number) || (Matrix_Check(B) || b_is_number)) {

    int freeA = SpMatrix_Check(A) && (SP_LGT(A) > 1);
    int freeB = SpMatrix_Check(B) && (SP_LGT(B) > 1);
    if (freeA) {
      if (!(A = (PyObject *)dense((spmatrix *)A)) ) return NULL;
    }
    if (freeB) {
      if (!(B = (PyObject *)dense((spmatrix *)B)) ) return NULL;
    }

    PyObject *ret = (PyObject *)Matrix_New(m, n, id);
    if (!ret) {
      if (freeA) { Py_DECREF(A); }
      if (freeB) { Py_DECREF(B); }
      return NULL;
    }
    int_t i;
    for (i=0; i<m*n; i++) {
      if (!a_is_number) convert_num[id](&a, A, 0, i);
      if (!b_is_number) convert_num[id](&b, B, 0, i);

      if (id == INT)
        MAT_BUFI(ret)[i] = MIN(a.i, b.i);
      else
        MAT_BUFD(ret)[i] = MIN(a.d, b.d);
    }

    if (freeA) { Py_DECREF(A); }
    if (freeB) { Py_DECREF(B); }
    return ret;

  } else {

    spmatrix *ret = SpMatrix_New(m, n, 0, DOUBLE);
    if (!ret) return NULL;

    int_t j, ka = 0, kb = 0, kret = 0;
    for (j=0; j<n; j++) {

      while (ka < SP_COL(A)[j+1] && kb < SP_COL(B)[j+1]) {

        if (SP_ROW(A)[ka] < SP_ROW(B)[kb]) {
          if (SP_VALD(A)[ka++] < 0.0) SP_COL(ret)[j+1]++;
        }
        else if (SP_ROW(A)[ka] > SP_ROW(B)[kb]) {
          if (SP_VALD(B)[kb++] < 0.0) SP_COL(ret)[j+1]++;
        }
        else {
          SP_COL(ret)[j+1]++; ka++; kb++;
        }
      }

      while (ka < SP_COL(A)[j+1]) {
        if (SP_VALD(A)[ka++] < 0.0) SP_COL(ret)[j+1]++;
      }

      while (kb < SP_COL(B)[j+1]) {
        if (SP_VALD(B)[kb++] < 0.0) SP_COL(ret)[j+1]++;
      }

    }

    for (j=0; j<n; j++) SP_COL(ret)[j+1] += SP_COL(ret)[j];

    int_t *newrow = malloc( sizeof(int_t)*SP_COL(ret)[n] );
    double *newval = malloc( sizeof(double)*SP_COL(ret)[n] );
    if (!newrow || !newval) {
      free(newrow); free(newval); Py_DECREF(ret);
      return PyErr_NoMemory();
    }
    free( ret->obj->rowind );
    free( ret->obj->values );
    ret->obj->rowind = newrow;
    ret->obj->values = newval;

    ka = 0; kb = 0;
    for (j=0; j<n; j++) {

      while (ka < SP_COL(A)[j+1] && kb < SP_COL(B)[j+1]) {

        if (SP_ROW(A)[ka] < SP_ROW(B)[kb]) {
          if (SP_VALD(A)[ka] < 0.0) {
            SP_ROW(ret)[kret] = SP_ROW(A)[ka];
            SP_VALD(ret)[kret++] = SP_VALD(A)[ka];
          }
          ka++;
        }
        else if (SP_ROW(A)[ka] > SP_ROW(B)[kb]) {
          if (SP_VALD(B)[kb] < 0.0) {
            SP_ROW(ret)[kret] = SP_ROW(B)[kb];
            SP_VALD(ret)[kret++] = SP_VALD(B)[kb];
          }
          kb++;
        }
        else {
          SP_ROW(ret)[kret] = SP_ROW(A)[ka];
          SP_VALD(ret)[kret] = MIN(SP_VALD(A)[ka], SP_VALD(B)[kb]);
          kret++; ka++; kb++;
        }
      }

      while (ka < SP_COL(A)[j+1]) {
        if (SP_VALD(A)[ka] < 0.0) {
          SP_ROW(ret)[kret] = SP_ROW(A)[ka];
          SP_VALD(ret)[kret++] = SP_VALD(A)[ka];
        }
        ka++;
      }

      while (kb < SP_COL(B)[j+1]) {
        if (SP_VALD(B)[kb] < 0.0) {
          SP_ROW(ret)[kret] = SP_ROW(B)[kb];
          SP_VALD(ret)[kret++] = SP_VALD(B)[kb];
        }
        kb++;
      }
    }

    return (PyObject *)ret;
  }
}

PyObject * matrix_elem_mul(matrix *self, PyObject *args, PyObject *kwrds)
{
  PyObject *A, *B;
  if (!PyArg_ParseTuple(args, "OO:emul", &A, &B)) return NULL;

  if (!(X_Matrix_Check(A) || PyNumber_Check(A)) ||
      !(X_Matrix_Check(B) || PyNumber_Check(B)))
    PY_ERR_TYPE("arguments must be either matrices or python numbers");

  int a_is_number = PyNumber_Check(A) || (Matrix_Check(A) && MAT_LGT(A) == 1);
  int b_is_number = PyNumber_Check(B) || (Matrix_Check(B) && MAT_LGT(B) == 1);

  int ida, idb;
#if PY_MAJOR_VERSION >= 3
  if (PyLong_Check(A)) { ida = INT; }
#else
  if (PyInt_Check(A)) { ida = INT; }
#endif
  else if (PyFloat_Check(A)) { ida = DOUBLE; }
  else if (PyComplex_Check(A)) { ida = COMPLEX; }
  else { ida = X_ID(A); }

#if PY_MAJOR_VERSION >= 3
  if (PyLong_Check(B)) { idb = INT; }
#else
  if (PyInt_Check(B)) { idb = INT; }
#endif
  else if (PyFloat_Check(B)) { idb = DOUBLE; }
  else if (PyComplex_Check(B)) { idb = COMPLEX; }
  else { idb = X_ID(B); }

  int id  = MAX( ida, idb );

  number a, b;
  if (a_is_number) convert_num[id](&a, A, PyNumber_Check(A), 0);
  if (b_is_number) convert_num[id](&b, B, PyNumber_Check(B), 0);

  if (a_is_number && b_is_number &&
      (!X_Matrix_Check(A) && !X_Matrix_Check(B))) {
    if (!X_Matrix_Check(A) && !X_Matrix_Check(B)) {
      if (id == INT)
        return Py_BuildValue("i", a.i*b.i );
      else if (id == DOUBLE)
        return Py_BuildValue("d", a.d*b.d );
      else {
        number c;
#ifndef _MSC_VER
        c.z = a.z*b.z;
#else
        c.z = _Cmulcc(a.z,b.z);
#endif
        return znum2PyObject(&c, 0);
      }
    }
  }

  if (!(a_is_number || b_is_number)) {
    if (X_NROWS(A) != X_NROWS(B) || X_NCOLS(A) != X_NCOLS(B))
      PY_ERR_TYPE("incompatible dimensions");
  }

  int_t m = ( !a_is_number ? X_NROWS(A) : (!b_is_number ? X_NROWS(B) : 1));
  int_t n = ( !a_is_number ? X_NCOLS(A) : (!b_is_number ? X_NCOLS(B) : 1));

  if ((Matrix_Check(A) || a_is_number) && (Matrix_Check(B) || b_is_number)) {

    PyObject *ret = (PyObject *)Matrix_New(m, n, id);
    if (!ret) return NULL;

    int_t i;
    for (i=0; i<m*n; i++) {
      if (!a_is_number) convert_num[id](&a, A, 0, i);
      if (!b_is_number) convert_num[id](&b, B, 0, i);

      if (id == INT)
        MAT_BUFI(ret)[i] = a.i*b.i;
      else if (id == DOUBLE)
        MAT_BUFD(ret)[i] = a.d*b.d;
      else
#ifndef _MSC_VER
        MAT_BUFZ(ret)[i] = a.z*b.z;
#else
        MAT_BUFZ(ret)[i] = _Cmulcc(a.z,b.z);
#endif
    }

    return ret;

  }
  else if (SpMatrix_Check(A) && !SpMatrix_Check(B)) {

    PyObject *ret = (PyObject *)SpMatrix_NewFromSpMatrix((spmatrix *)A, id);
    if (!ret) return NULL;

    int_t j, k;
    for (j=0; j<SP_NCOLS(A); j++) {
      for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1]; k++) {
        if (!b_is_number) convert_num[id](&b, B, 0, j*m + SP_ROW(A)[k]);

        if (id == DOUBLE)
          SP_VALD(ret)[k] *= b.d;
        else
#ifndef _MSC_VER
          SP_VALZ(ret)[k] *= b.z;
#else
          SP_VALZ(ret)[k] = _Cmulcc(SP_VALZ(ret)[k],b.z);
#endif
      }
    }

    return ret;
  }
  else if (SpMatrix_Check(B) && !SpMatrix_Check(A)) {

    PyObject *ret = (PyObject *)SpMatrix_NewFromSpMatrix((spmatrix *)B, id);
    if (!ret) return NULL;

    int_t j, k;
    for (j=0; j<SP_NCOLS(B); j++) {
      for (k=SP_COL(B)[j]; k<SP_COL(B)[j+1]; k++) {
        if (!a_is_number) convert_num[id](&a, A, 0, j*m + SP_ROW(B)[k]);

        if (id == DOUBLE)
          SP_VALD(ret)[k] *= a.d;
        else
#ifndef _MSC_VER
          SP_VALZ(ret)[k] *= a.z;
#else
          SP_VALZ(ret)[k] = _Cmulcc(SP_VALZ(ret)[k],a.z);
#endif
      }
    }

    return ret;
  }

  else {

    spmatrix *ret = SpMatrix_New(m, n, 0, id);
    if (!ret) return NULL;

    int_t j, ka = 0, kb = 0, kret = 0;
    for (j=0; j<n; j++) {

      while (ka < SP_COL(A)[j+1] && kb < SP_COL(B)[j+1]) {

        if (SP_ROW(A)[ka] < SP_ROW(B)[kb]) {
          ka++;
        }
        else if (SP_ROW(A)[ka] > SP_ROW(B)[kb]) {
          kb++;
        }
        else {
          SP_COL(ret)[j+1]++; ka++; kb++;
        }
      }

      ka = SP_COL(A)[j+1];
      kb = SP_COL(B)[j+1];
    }

    for (j=0; j<n; j++) SP_COL(ret)[j+1] += SP_COL(ret)[j];

    int_t *newrow = malloc( sizeof(int_t)*SP_COL(ret)[n] );
    double *newval = malloc( E_SIZE[id]*SP_COL(ret)[n] );
    if (!newrow || !newval) {
      free(newrow); free(newval); Py_DECREF(ret);
      return PyErr_NoMemory();
    }
    free( ret->obj->rowind );
    free( ret->obj->values );
    ret->obj->rowind = newrow;
    ret->obj->values = newval;

    ka = 0; kb = 0;
    for (j=0; j<n; j++) {

      while (ka < SP_COL(A)[j+1] && kb < SP_COL(B)[j+1]) {

        if (SP_ROW(A)[ka] < SP_ROW(B)[kb]) {
          ka++;
        }
        else if (SP_ROW(A)[ka] > SP_ROW(B)[kb]) {
          kb++;
        }
        else {
          SP_ROW(ret)[kret] = SP_ROW(A)[ka];
          if (id == DOUBLE)
            SP_VALD(ret)[kret] = SP_VALD(A)[ka]*SP_VALD(B)[kb];
          else
#ifndef _MSC_VER
            SP_VALZ(ret)[kret] =
                (X_ID(A) == DOUBLE ? SP_VALD(A)[ka] : SP_VALZ(A)[ka])*
                (X_ID(B) == DOUBLE ? SP_VALD(B)[kb] : SP_VALZ(B)[kb]);
#else
	  SP_VALZ(ret)[kret] = _Cmulcc(
		(X_ID(A) == DOUBLE ? _Cbuild(SP_VALD(A)[ka],0.0) : SP_VALZ(A)[ka]),
		(X_ID(B) == DOUBLE ? _Cbuild(SP_VALD(B)[kb],0.0) : SP_VALZ(B)[kb]));
#endif

          kret++; ka++; kb++;
        }
      }

      ka = SP_COL(A)[j+1];
      kb = SP_COL(B)[j+1];
    }

    return (PyObject *)ret;
  }
}

PyObject * matrix_elem_div(matrix *self, PyObject *args, PyObject *kwrds)
{
  PyObject *A, *B, *ret;
  if (!PyArg_ParseTuple(args, "OO:ediv", &A, &B)) return NULL;

  if (!(X_Matrix_Check(A) || PyNumber_Check(A)) ||
      !(X_Matrix_Check(B) || PyNumber_Check(B)))
    PY_ERR_TYPE("arguments must be either matrices or python numbers");

  if (SpMatrix_Check(B))
    PY_ERR_TYPE("elementwise division with sparse matrix\n");

  int a_is_number = PyNumber_Check(A) || (Matrix_Check(A) && MAT_LGT(A) == 1);
  int b_is_number = PyNumber_Check(B) || (Matrix_Check(B) && MAT_LGT(B) == 1);

  int ida, idb;
#if PY_MAJOR_VERSION >= 3
  if (PyLong_Check(A)) { ida = INT; }
#else
  if (PyInt_Check(A)) { ida = INT; }
#endif
  else if (PyFloat_Check(A)) { ida = DOUBLE; }
  else if (PyComplex_Check(A)) { ida = COMPLEX; }
  else { ida = X_ID(A); }

#if PY_MAJOR_VERSION >= 3
  if (PyLong_Check(B)) { idb = INT; }
#else
  if (PyInt_Check(B)) { idb = INT; }
#endif
  else if (PyFloat_Check(B)) { idb = DOUBLE; }
  else if (PyComplex_Check(B)) { idb = COMPLEX; }
  else { idb = X_ID(B); }

#if PY_MAJOR_VERSION >= 3
  int id  = MAX( DOUBLE, MAX( ida, idb ) ) ;
#else
  int id  = MAX( ida, idb );
#endif

  number a, b;
  if (a_is_number) convert_num[id](&a, A, PyNumber_Check(A), 0);
  if (b_is_number) convert_num[id](&b, B, PyNumber_Check(B), 0);

  if ((a_is_number && b_is_number) &&
      (!X_Matrix_Check(A) && !Matrix_Check(B))) {
    if (id == INT) {
      if (b.i == 0) PY_ERR(PyExc_ArithmeticError, "division by zero");
      return Py_BuildValue("i", a.i/b.i );
    }
    else if (id == DOUBLE) {
      if (b.d == 0.0) PY_ERR(PyExc_ArithmeticError, "division by zero");
      return Py_BuildValue("d", a.d/b.d );
    }
    else {
#ifndef _MSC_VER
      if (b.z == 0.0) PY_ERR(PyExc_ArithmeticError, "division by zero");
#else
      if (creal(b.z) == 0.0 && cimag(b.z) == 0.0) PY_ERR(PyExc_ArithmeticError, "division by zero");
#endif
      number c;
#ifndef _MSC_VER
      c.z = a.z/b.z;
#else
      c.z = _Cmulcc(a.z, _Cmulcr(conj(b.z),1.0/norm(b.z)));
#endif
      return znum2PyObject(&c, 0);
    }
  }

  if (!(a_is_number || b_is_number)) {
    if (X_NROWS(A) != MAT_NROWS(B) || X_NCOLS(A) != MAT_NCOLS(B))
      PY_ERR_TYPE("incompatible dimensions");
  }

  int m = ( !a_is_number ? X_NROWS(A) : (!b_is_number ? X_NROWS(B) : 1));
  int n = ( !a_is_number ? X_NCOLS(A) : (!b_is_number ? X_NCOLS(B) : 1));

  if ((Matrix_Check(A) || a_is_number) && (Matrix_Check(B) || b_is_number)) {

    if (!(ret = (PyObject *)Matrix_New(m, n, id)))
      return NULL;

    int i;
    for (i=0; i<m*n; i++) {
      if (!a_is_number) convert_num[id](&a, A, 0, i);
      if (!b_is_number) convert_num[id](&b, B, 0, i);

      if (id == INT) {
        if (b.i == 0) goto divzero;
        MAT_BUFI(ret)[i] = a.i/b.i;
      }
      else if (id == DOUBLE) {
        if (b.d == 0) goto divzero;
        MAT_BUFD(ret)[i] = a.d/b.d;
      }
      else {
#ifndef _MSC_VER
        if (b.z == 0) goto divzero;
        MAT_BUFZ(ret)[i] = a.z/b.z;
#else
        if (creal(b.z) == 0 && cimag(b.z)== 0) goto divzero;
        MAT_BUFZ(ret)[i] = _Cmulcc(a.z,_Cmulcr(conj(b.z),1.0/norm(b.z)));
#endif
      }
    }

    return ret;
  }
  else { // (SpMatrix_Check(A) && !SpMatrix_Check(B)) {

    if (!(ret = (PyObject *)SpMatrix_NewFromSpMatrix((spmatrix *)A, id)))
      return NULL;

    int j, k;
    for (j=0; j<SP_NCOLS(A); j++) {
      for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1]; k++) {
        if (!b_is_number) convert_num[id](&b, B, 0, j*m + SP_ROW(A)[k]);

        if (id == DOUBLE) {
          if (b.d == 0.0) goto divzero;
          SP_VALD(ret)[k] /= b.d;
        }
        else {
#ifndef _MSC_VER
         if (b.z == 0) goto divzero;
         SP_VALZ(ret)[k] /= b.z;
#else
         if (creal(b.z) == 0 && cimag(b.z)== 0) goto divzero;
         SP_VALZ(ret)[k] = _Cmulcc(SP_VALZ(ret)[k],_Cmulcr(conj(b.z),1.0/norm(b.z)));
#endif
        }
      }
    }

    return ret;
  }

  divzero:
  Py_DECREF(ret);
  PY_ERR(PyExc_ArithmeticError, "division by zero");
}

extern PyObject * matrix_exp(matrix *, PyObject *, PyObject *) ;
extern PyObject * matrix_log(matrix *, PyObject *, PyObject *) ;
extern PyObject * matrix_sqrt(matrix *, PyObject *, PyObject *) ;
extern PyObject * matrix_cos(matrix *, PyObject *, PyObject *) ;
extern PyObject * matrix_sin(matrix *, PyObject *, PyObject *) ;

static PyMethodDef base_functions[] = {
    {"exp", (PyCFunction)matrix_exp, METH_VARARGS|METH_KEYWORDS,
        "Computes the element-wise expontial of a matrix"},
    {"log", (PyCFunction)matrix_log, METH_VARARGS|METH_KEYWORDS,
        "Computes the element-wise logarithm of a matrix"},
    {"sqrt", (PyCFunction)matrix_sqrt, METH_VARARGS|METH_KEYWORDS,
        "Computes the element-wise square-root of a matrix"},
    {"cos", (PyCFunction)matrix_cos, METH_VARARGS|METH_KEYWORDS,
        "Computes the element-wise cosine of a matrix"},
    {"sin", (PyCFunction)matrix_sin, METH_VARARGS|METH_KEYWORDS,
        "Computes the element-wise sine of a matrix"},
    {"axpy", (PyCFunction)base_axpy, METH_VARARGS|METH_KEYWORDS, doc_axpy},
    {"gemm", (PyCFunction)base_gemm, METH_VARARGS|METH_KEYWORDS, doc_gemm},
    {"gemv", (PyCFunction)base_gemv, METH_VARARGS|METH_KEYWORDS, doc_gemv},
    {"syrk", (PyCFunction)base_syrk, METH_VARARGS|METH_KEYWORDS, doc_syrk},
    {"symv", (PyCFunction)base_symv, METH_VARARGS|METH_KEYWORDS, doc_symv},
    {"emul", (PyCFunction)matrix_elem_mul, METH_VARARGS|METH_KEYWORDS,
        "elementwise product of two matrices"},
    {"ediv", (PyCFunction)matrix_elem_div, METH_VARARGS|METH_KEYWORDS,
        "elementwise division between two matrices"},
    {"emin", (PyCFunction)matrix_elem_min, METH_VARARGS|METH_KEYWORDS,
        "elementwise minimum between two matrices"},
    {"emax", (PyCFunction)matrix_elem_max, METH_VARARGS|METH_KEYWORDS,
        "elementwise maximum between two matrices"},
    {"sparse", (PyCFunction)sparse, METH_VARARGS|METH_KEYWORDS, doc_sparse},
    {"spdiag", (PyCFunction)spdiag, METH_VARARGS|METH_KEYWORDS, doc_spdiag},
    {NULL}		/* sentinel */
};

#if PY_MAJOR_VERSION >= 3

static PyModuleDef base_module = {
    PyModuleDef_HEAD_INIT,
    "base",
    base__doc__,
    -1,
    base_functions,
    NULL, NULL, NULL, NULL
};
#define INITERROR return NULL
PyMODINIT_FUNC PyInit_base(void)

#else

#define INITERROR return
PyMODINIT_FUNC initbase(void)

#endif
{
  static void *base_API[8];
  PyObject *base_mod, *c_api_object;

#if PY_MAJOR_VERSION >= 3
  base_mod = PyModule_Create(&base_module);
  if (base_mod == NULL)
#else
  if (!(base_mod = Py_InitModule3("base", base_functions, base__doc__)))
#endif
    INITERROR;

  /* for MS VC++ compatibility */
  matrix_tp.tp_alloc = PyType_GenericAlloc;
  matrix_tp.tp_free = PyObject_Del;
  if (PyType_Ready(&matrix_tp) < 0)
    INITERROR;

  if (PyType_Ready(&matrixiter_tp) < 0)
    INITERROR;

  Py_INCREF(&matrix_tp);
  if (PyModule_AddObject(base_mod, "matrix", (PyObject *) &matrix_tp) < 0)
    INITERROR;

  spmatrix_tp.tp_alloc = PyType_GenericAlloc;
  spmatrix_tp.tp_free = PyObject_Del;
  if (PyType_Ready(&spmatrix_tp) < 0)
    INITERROR;

  if (PyType_Ready(&spmatrixiter_tp) < 0)
    INITERROR;

  Py_INCREF(&spmatrix_tp);
  if (PyModule_AddObject(base_mod, "spmatrix", (PyObject *) &spmatrix_tp) < 0)
    INITERROR;

#ifndef _MSC_VER
  One[INT].i = 1; One[DOUBLE].d = 1.0; One[COMPLEX].z = 1.0;
#else
  One[INT].i = 1; One[DOUBLE].d = 1.0; One[COMPLEX].z = _Cbuild(1.0,0.0);
#endif

#ifndef _MSC_VER
  MinusOne[INT].i = -1; MinusOne[DOUBLE].d = -1.0; MinusOne[COMPLEX].z = -1.0;
#else
  MinusOne[INT].i = -1; MinusOne[DOUBLE].d = -1.0; MinusOne[COMPLEX].z = _Cbuild(-1.0,0.0);
#endif

#ifndef _MSC_VER
  Zero[INT].i = 0; Zero[DOUBLE].d = 0.0; Zero[COMPLEX].z = 0.0;
#else
  Zero[INT].i = 0; Zero[DOUBLE].d = 0.0; Zero[COMPLEX].z = _Cbuild(0.0,0.0);
#endif

  /* initialize the C API object */
  base_API[0] = (void *)Matrix_New;
  base_API[1] = (void *)Matrix_NewFromMatrix;
  base_API[2] = (void *)Matrix_NewFromSequence;
  base_API[3] = (void *)Matrix_Check_func;
  base_API[4] = (void *)SpMatrix_New;
  base_API[5] = (void *)SpMatrix_NewFromSpMatrix;
  base_API[6] = (void *)SpMatrix_NewFromIJV;
  base_API[7] = (void *)SpMatrix_Check_func;

#if PY_MAJOR_VERSION >= 3
  /* Create a Capsule containing the API pointer array's address */
  c_api_object = PyCapsule_New((void *)base_API, "base_API", NULL);
#else
  /* Create a CObject containing the API pointer array's address */
  c_api_object = PyCObject_FromVoidPtr((void *)base_API, NULL);
#endif

  if (c_api_object != NULL)
    PyModule_AddObject(base_mod, "_C_API", c_api_object);

#if PY_MAJOR_VERSION >= 3
  return base_mod;
#endif
}
