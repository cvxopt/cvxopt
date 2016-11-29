/*
 * Copyright 2012-2016 M. Andersen and L. Vandenberghe.
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

#include "cvxopt.h"
#include "misc.h"
#include <fftw3.h>

PyDoc_STRVAR(fftw__doc__, "Interface to the FFTW3 library.\n");

extern void zscal_(int *n, double complex *alpha, double complex *x, int *incx);
extern void dscal_(int *n, double *alpha, double *x, int *incx);

static char doc_dft[] =
    "DFT of a matrix,  X := dft(X)\n\n"
    "PURPOSE\n"
    "Computes the DFT of a dense matrix X column by column.\n\n"
    "ARGUMENTS\n"
    "X         A dense matrix of typecode 'z'.";

static PyObject *dft(PyObject *self, PyObject *args, PyObject *kwrds)
{
  matrix *X;
  char *kwlist[] = {"X", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O", kwlist, &X))
    return NULL;

  if (!(Matrix_Check(X) && MAT_ID(X) == COMPLEX))
    PY_ERR(PyExc_ValueError, "X must be a dense matrix with type 'z'");

  int m = X->nrows, n = X->ncols;
  if (m == 0) return Py_BuildValue("");

  fftw_plan p = fftw_plan_many_dft(1, &m, n,
      X->buffer, &m, 1, m,
      X->buffer, &m, 1, m,
      FFTW_FORWARD, FFTW_ESTIMATE);

  Py_BEGIN_ALLOW_THREADS
  fftw_execute(p);
  Py_END_ALLOW_THREADS
  fftw_destroy_plan(p);
  return Py_BuildValue("");
}

static char doc_dftn[] =
    "N-dimensional DFT of a matrix.\n"
    "X := dftn(X, dims)\n\n"
    "PURPOSE\n"
    "Computes the DFT of an N-dimensional array represented by a dense\n"
    "matrix X. The shape of the matrix X does not matter, but the data\n"
    "must be arranged in row-major-order.  The total dimension (defined\n"
    "as mul(dims)) must equal the length of the array X.\n\n"
    "ARGUMENTS\n"
    "X         A dense matrix of typecode 'd'.\n\n"
    "dims      a tuple with the dimensions of the array.";

static PyObject *dftn(PyObject *self, PyObject *args, PyObject *kwrds)
{
  matrix *X;
  PyObject *dims = NULL;
  char *kwlist[] = {"X", "dims", NULL};

  int *dimarr;
  int free_dims = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|O:dftn", kwlist, &X, &dims))
    return NULL;

  if (!(Matrix_Check(X) && MAT_ID(X) == COMPLEX))
    PY_ERR_TYPE("X must be a dense matrix with type 'z'");

  if (!dims) {
    dims = PyTuple_New(2);
    if (!dims) return PyErr_NoMemory();

#if PY_MAJOR_VERSION >= 3
    PyTuple_SET_ITEM(dims, 0, PyLong_FromLong(MAT_NCOLS(X)));
    PyTuple_SET_ITEM(dims, 1, PyLong_FromLong(MAT_NROWS(X)));
#else
    PyTuple_SET_ITEM(dims, 0, PyInt_FromLong(MAT_NCOLS(X)));
    PyTuple_SET_ITEM(dims, 1, PyInt_FromLong(MAT_NROWS(X)));
#endif
    free_dims = 1;
  }

  if (!PyTuple_Check(dims)) {
    if (free_dims) { Py_DECREF(dims); }
    PY_ERR_TYPE("invalid dimension tuple");
  }

  int len = PySequence_Size(dims);
  PyObject *seq = PySequence_Fast(dims, "list is not iterable");

  if (!(dimarr = malloc(len*sizeof(int)))) {
    if (free_dims) { Py_DECREF(dims); }
    Py_DECREF(seq);
    return PyErr_NoMemory();
  }

  int i, proddim = 1;
  for (i=0; i<len; i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(seq, i);

#if PY_MAJOR_VERSION >= 3
    if (!PyLong_Check(item)) {
#else
    if (!PyInt_Check(item)) {
#endif
      if (free_dims) { Py_DECREF(dims); }
      Py_DECREF(seq);
      free(dimarr);
      PY_ERR_TYPE("non-integer in dimension tuple");
    }

#if PY_MAJOR_VERSION >= 3
    dimarr[len-i-1] = PyLong_AS_LONG(item);
#else
    dimarr[len-i-1] = PyInt_AS_LONG(item);
#endif
    if (dimarr[len-i-1] < 0) {
      if (free_dims) { Py_DECREF(dims); }
      Py_DECREF(seq);
      free(dimarr);
      PY_ERR(PyExc_ValueError, "negative dimension");
    }
    proddim *= dimarr[len-i-1];
  }

  if (free_dims) { Py_DECREF(dims); }

  Py_DECREF(seq);

  if (proddim != MAT_LGT(X)) {
    free(dimarr);
    PY_ERR_TYPE("length of X does not match dimensions");
  }

  if (proddim == 0) {
    free(dimarr);
    return Py_BuildValue("");
  }

  fftw_plan p = fftw_plan_dft(len, dimarr,
      X->buffer, X->buffer, FFTW_FORWARD, FFTW_ESTIMATE);

  Py_BEGIN_ALLOW_THREADS
  fftw_execute(p);
  Py_END_ALLOW_THREADS
  fftw_destroy_plan(p);
  free(dimarr);
  return Py_BuildValue("");
}

static char doc_idft[] =
    "IDFT of a matrix,  X := idft(X)\n\n"
    "PURPOSE\n"
    "Computes the inverse DFT of a dense matrix X column by column.\n\n"
    "ARGUMENTS\n"
    "X         A dense matrix of typecode 'z'.";

static PyObject *idft(PyObject *self, PyObject *args, PyObject *kwrds)
{
  matrix *X;
  char *kwlist[] = {"X", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O", kwlist, &X))
    return NULL;

  if (!(Matrix_Check(X) && MAT_ID(X) == COMPLEX))
    PY_ERR(PyExc_ValueError, "X must be a dense matrix with type 'z'");

  int m = X->nrows, n = X->ncols;
  if (m == 0) return Py_BuildValue("");

  fftw_plan p = fftw_plan_many_dft(1, &m, n,
      X->buffer, &m, 1, m,
      X->buffer, &m, 1, m,
      FFTW_BACKWARD, FFTW_ESTIMATE);

  Py_BEGIN_ALLOW_THREADS
  fftw_execute(p);
  Py_END_ALLOW_THREADS

  number a;
  a.z = 1.0/m;
  int mn = m*n, ix = 1;
  zscal_(&mn, &a.z, MAT_BUFZ(X), &ix);

  fftw_destroy_plan(p);
  return Py_BuildValue("");
}

static char doc_idftn[] =
    "Inverse N-dimensional DFT of a matrix.\n"
    "X := idftn(X, dims)\n\n"
    "PURPOSE\n"
    "Computes the IDFT of an N-dimensional array represented by a dense\n"
    "matrix X. The shape of the matrix X does not matter, but the data\n"
    "must be arranged in row-major-order.  The total dimension (defined\n"
    "as mul(dims)) must equal the length of the array X.\n\n"
    "ARGUMENTS\n"
    "X         A dense matrix of typecode 'd'.\n\n"
    "dims      a tuple with the dimensions of the array.";

static PyObject *idftn(PyObject *self, PyObject *args, PyObject *kwrds)
{
  matrix *X;
  PyObject *dims = NULL;
  char *kwlist[] = {"X", "dims", NULL};

  int *dimarr;
  int free_dims = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|O:idftn", 
      kwlist, &X, &dims))
    return NULL;

  if (!(Matrix_Check(X) && MAT_ID(X) == COMPLEX))
    PY_ERR_TYPE("X must be a dense matrix with type 'z'");

  if (!dims) {
    dims = PyTuple_New(2);
    if (!dims) return PyErr_NoMemory();

#if PY_MAJOR_VERSION >= 3
    PyTuple_SET_ITEM(dims, 0, PyLong_FromLong(MAT_NCOLS(X)));
    PyTuple_SET_ITEM(dims, 1, PyLong_FromLong(MAT_NROWS(X)));
#else
    PyTuple_SET_ITEM(dims, 0, PyInt_FromLong(MAT_NCOLS(X)));
    PyTuple_SET_ITEM(dims, 1, PyInt_FromLong(MAT_NROWS(X)));
#endif
    free_dims = 1;
  }

  if (!PyTuple_Check(dims))
    PY_ERR_TYPE("invalid dimension tuple");

  int len = PySequence_Size(dims);
  PyObject *seq = PySequence_Fast(dims, "list is not iterable");

  if (!(dimarr = malloc(len*sizeof(int)))) {
    if (free_dims) { Py_DECREF(dims); }
    Py_DECREF(seq);
    return PyErr_NoMemory();
  }

  int i, proddim = 1;
  for (i=0; i<len; i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(seq, i);

#if PY_MAJOR_VERSION >= 3
    if (!PyLong_Check(item)) {
#else
    if (!PyInt_Check(item)) {
#endif
      if (free_dims) { Py_DECREF(dims); }
      Py_DECREF(seq);
      free(dimarr);
      PY_ERR_TYPE("non-integer in dimension tuple");
    }

#if PY_MAJOR_VERSION >= 3
    dimarr[len-i-1] = PyLong_AS_LONG(item);
#else
    dimarr[len-i-1] = PyInt_AS_LONG(item);
#endif
    if (dimarr[len-i-1] < 0) {
      if (free_dims) { Py_DECREF(dims); }
      Py_DECREF(seq);
      free(dimarr);
      PY_ERR(PyExc_ValueError, "negative dimension");
    }
    proddim *= dimarr[len-i-1];
  }

  Py_DECREF(seq);

  if (free_dims) { Py_DECREF(dims); }

  if (proddim != MAT_LGT(X)) {
    free(dimarr);
    PY_ERR_TYPE("length of X does not match dimensions");
  }

  if (proddim == 0) {
    free(dimarr);
    return Py_BuildValue("");
  }

  number a;
  a.z = 1.0/proddim;

  int ix = 1;
  zscal_(&proddim, &a.z, MAT_BUFZ(X), &ix);

  fftw_plan p = fftw_plan_dft(len, dimarr,
      X->buffer, X->buffer, FFTW_BACKWARD, FFTW_ESTIMATE);

  Py_BEGIN_ALLOW_THREADS
  fftw_execute(p);
  Py_END_ALLOW_THREADS
  fftw_destroy_plan(p);
  free(dimarr);
  return Py_BuildValue("");
}

static char doc_dct[] =
    "DCT of a matrix.\n"
    "X := dct(X, type=2)\n\n"
    "PURPOSE\n"
    "Computes the DCT of a dense matrix X column by column.\n\n"
    "ARGUMENTS\n"
    "X         A dense matrix of typecode 'd'.\n\n"
    "type      integer from 1 to 4; chooses either DCT-I, DCT-II, \n"
    "          DCT-III or DCT-IV.";

static PyObject *dct(PyObject *self, PyObject *args, PyObject *kwrds)
{
  matrix *X;
  int type = 2;
  char *kwlist[] = {"X", "type", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|i", kwlist, &X, &type))
    return NULL;

  if (!(Matrix_Check(X) && MAT_ID(X) == DOUBLE))
    PY_ERR(PyExc_ValueError, "X must be a dense matrix with type 'd'");

  int m = X->nrows, n = X->ncols;
  if (m == 0) return Py_BuildValue("");

  fftw_r2r_kind kind;
  switch(type) {
  case 1:
    kind = FFTW_REDFT00;
    if (m <= 1) PY_ERR(PyExc_ValueError, "m must be greater than 1 for DCT-I");
    break;
  case 2: kind = FFTW_REDFT10; break;
  case 3: kind = FFTW_REDFT01; break;
  case 4: kind = FFTW_REDFT11; break;
  default: PY_ERR(PyExc_ValueError, "type must be between 1 and 4");
  }
  fftw_plan p = fftw_plan_many_r2r(1, &m, n,
      X->buffer, &m, 1, m,
      X->buffer, &m, 1, m,
      &kind, FFTW_ESTIMATE);

  Py_BEGIN_ALLOW_THREADS
  fftw_execute(p);
  Py_END_ALLOW_THREADS
  fftw_destroy_plan(p);
  return Py_BuildValue("");
}

static char doc_dctn[] =
    "N-dimensional DCT of a matrix.\n"
    "X := dctn(X, dims, type=2)\n\n"
    "PURPOSE\n"
    "Computes the DCT of an N-dimensional array represented by a dense\n"
    "matrix X. The shape of the matrix X does not matter, but the data\n"
    "must be arranged in row-major-order.  The total dimension (defined\n"
    "as mul(dims)) must equal the length of the array X.\n\n"
    "ARGUMENTS\n"
    "X         A dense matrix of typecode 'd'.\n\n"
    "dims      a tuple with the dimensions of the array.\n\n"
    "type      integer from 1 to 4; chooses either DCT-I, DCT-II, \n"
    "          DCT-III or DCT-IV.";

static PyObject *dctn(PyObject *self, PyObject *args, PyObject *kwrds)
{
  matrix *X;
  PyObject *dims = NULL, *type = NULL;
  char *kwlist[] = {"X", "dims", "type", NULL};

  int *dimarr;
  fftw_r2r_kind *kindarr;
  int free_dims = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|OO:dctn", kwlist,
      &X, &dims, &type))
    return NULL;

  if (!(Matrix_Check(X) && MAT_ID(X) == DOUBLE))
    PY_ERR_TYPE("X must be a dense matrix with type 'd'");

  if (!dims) {
    dims = PyTuple_New(2);
    if (!dims) return PyErr_NoMemory();

#if PY_MAJOR_VERSION >= 3
    PyTuple_SET_ITEM(dims, 0, PyLong_FromLong(MAT_NCOLS(X)));
    PyTuple_SET_ITEM(dims, 1, PyLong_FromLong(MAT_NROWS(X)));
#else
    PyTuple_SET_ITEM(dims, 0, PyInt_FromLong(MAT_NCOLS(X)));
    PyTuple_SET_ITEM(dims, 1, PyInt_FromLong(MAT_NROWS(X)));
#endif
    free_dims = 1;
  }

  if (!PyTuple_Check(dims))
    PY_ERR_TYPE("invalid dimension tuple");

  if (type && !PyTuple_Check(type)) {
    if (free_dims) { Py_DECREF(dims); }
    PY_ERR_TYPE("invalid type tuple");
  }

  int len = PySequence_Size(dims);
  if (type && PySequence_Size(type) != len) {
    if (free_dims) { Py_DECREF(dims); }
    PY_ERR_TYPE("dimensions and type tuples must have same length");
  }

  PyObject *seq = PySequence_Fast(dims, "list is not iterable");

  if (!(dimarr = malloc(len*sizeof(int)))) {
    if (free_dims) { Py_DECREF(dims); }
    Py_DECREF(seq);
    return PyErr_NoMemory();
  }

  if (!(kindarr = malloc(len*sizeof(fftw_r2r_kind)))) {
    if (free_dims) { Py_DECREF(dims); }
    Py_DECREF(seq);
    free(dimarr);
    return PyErr_NoMemory();
  }

  int i, proddim = 1;
  for (i=0; i<len; i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(seq, i);

#if PY_MAJOR_VERSION >= 3
    if (!PyLong_Check(item)) {
#else
    if (!PyInt_Check(item)) {
#endif
      if (free_dims) { Py_DECREF(dims); }
      Py_DECREF(seq);
      free(dimarr); free(kindarr);
      PY_ERR_TYPE("non-integer in dimension tuple");
    }

#if PY_MAJOR_VERSION >= 3
    dimarr[len-i-1] = PyLong_AS_LONG(item);
#else
    dimarr[len-i-1] = PyInt_AS_LONG(item);
#endif
    if (dimarr[len-i-1] < 0) {
      if (free_dims) { Py_DECREF(dims); }
      Py_DECREF(seq);
      free(dimarr); free(kindarr);
      PY_ERR(PyExc_ValueError, "negative dimension");
    }
    proddim *= dimarr[len-i-1];
  }

  if (free_dims) { Py_DECREF(dims); }

  if (proddim != MAT_LGT(X)) {
    Py_DECREF(seq);
    free(dimarr); free(kindarr);
    PY_ERR_TYPE("length of X does not match dimensions");
  }

  if (proddim == 0) {
    Py_DECREF(seq);
    free(dimarr); free(kindarr);
    return Py_BuildValue("");
  }

  Py_DECREF(seq);

  if (type == NULL) {
    for (i=0; i<len; i++)
      kindarr[i] = FFTW_REDFT10;
  } else {

    seq = PySequence_Fast(type, "list is not iterable");

    for (i=0; i<len; i++) {
      PyObject *item = PySequence_Fast_GET_ITEM(seq, i);

#if PY_MAJOR_VERSION >= 3
      if (!PyLong_Check(item)) {
#else
      if (!PyInt_Check(item)) {
#endif
        Py_DECREF(seq);
        free(dimarr); free(kindarr);
        PY_ERR_TYPE("non-integer in type tuple");
      }

#if PY_MAJOR_VERSION >= 3
      switch(PyLong_AS_LONG(item)) {
#else
      switch(PyInt_AS_LONG(item)) {
#endif
      case 1:
          kindarr[len-i-1] = FFTW_REDFT00;
          if (dimarr[len-i-1] <= 1) {
              Py_DECREF(seq);
              free(dimarr); free(kindarr);
              PY_ERR(PyExc_ValueError, 
                  "dimension must be greater than 1 for DCT-I");
          }
          break;
      case 2: kindarr[len-i-1] = FFTW_REDFT10; break;
      case 3: kindarr[len-i-1] = FFTW_REDFT01; break;
      case 4: kindarr[len-i-1] = FFTW_REDFT11; break;
      default:
          Py_DECREF(seq);
          free(dimarr); free(kindarr);
          PY_ERR(PyExc_ValueError, "type must be between 1 and 4");
      }
    }

    Py_DECREF(seq);
  }

  fftw_plan p = fftw_plan_r2r(len, dimarr,
      X->buffer, X->buffer, kindarr, FFTW_ESTIMATE);

  Py_BEGIN_ALLOW_THREADS
  fftw_execute(p);
  Py_END_ALLOW_THREADS
  fftw_destroy_plan(p);
  free(dimarr); free(kindarr);
  return Py_BuildValue("");
}

static char doc_idct[] =
    "Inverse DCT of a matrix.\n"
    "X := idct(X, type=2)\n\n"
    "PURPOSE\n"
    "Computes the IDCT of a dense matrix X column by column.\n\n"
    "ARGUMENTS\n"
    "X         A dense matrix of typecode 'd'.\n\n"
    "type      integer from 1 to 4; chooses either DCT-I, DCT-II, \n"
    "          DCT-III or DCT-IV.";

static PyObject *idct(PyObject *self, PyObject *args, PyObject *kwrds)
{
  matrix *X;
  int type = 2;
  char *kwlist[] = {"X", "type", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|i", kwlist, &X, &type))
    return NULL;

  if (!(Matrix_Check(X) && MAT_ID(X) == DOUBLE))
    PY_ERR(PyExc_ValueError, "X must be a dense matrix with type 'd'");

  int m = X->nrows, n = X->ncols;
  if (m == 0) return Py_BuildValue("");

  fftw_r2r_kind kind;
  switch(type) {
      case 1: kind = FFTW_REDFT00;
          if (m <= 1) 
            PY_ERR(PyExc_ValueError, "m must be greater than 1 for DCT-I");
          break;
      case 2: kind = FFTW_REDFT01; break;
      case 3: kind = FFTW_REDFT10; break;
      case 4: kind = FFTW_REDFT11; break;
      default: PY_ERR(PyExc_ValueError, "type must be between 1 and 4");
  }
  fftw_plan p = fftw_plan_many_r2r(1, &m, n, X->buffer, &m, 1, m,
      X->buffer, &m, 1, m, &kind, FFTW_ESTIMATE);

  Py_BEGIN_ALLOW_THREADS
  fftw_execute(p);
  Py_END_ALLOW_THREADS

  double a = 1.0/(type == 1 ? MAX(1,2*(m-1)) : 2*m);
  int mn = m*n, ix = 1;
  dscal_(&mn, &a, MAT_BUFD(X), &ix);

  fftw_destroy_plan(p);
  return Py_BuildValue("");
}

static char doc_idctn[] =
    "Inverse N-dimensional DCT of a matrix.\n"
    "X := idctn(X, dims, type=2)\n\n"
    "PURPOSE\n"
    "Computes the IDCT of an N-dimensional array represented by a dense\n"
    "matrix X. The shape of the matrix X does not matter, but the data\n"
    "must be arranged in row-major-order.  The total dimension (defined\n"
    "as mul(dims)) must equal the length of the array X.\n\n"
    "ARGUMENTS\n"
    "X         A dense matrix of typecode 'd'.\n\n"
    "dims      a tuple with the dimensions of the array.\n\n"
    "type      integer from 1 to 4; chooses either DCT-I, DCT-II, \n"
    "          DCT-III or DCT-IV.";

static PyObject *idctn(PyObject *self, PyObject *args, PyObject *kwrds)
{
  matrix *X;
  PyObject *dims = NULL, *type = NULL;
  char *kwlist[] = {"X", "dims", "type", NULL};

  int *dimarr;
  fftw_r2r_kind *kindarr;
  int free_dims = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|OO:idctn", kwlist,
      &X, &dims, &type))
    return NULL;

  if (!(Matrix_Check(X) && MAT_ID(X) == DOUBLE))
    PY_ERR_TYPE("X must be a dense matrix with type 'd'");

  if (!dims) {
    dims = PyTuple_New(2);
    if (!dims) return PyErr_NoMemory();

#if PY_MAJOR_VERSION >= 3
    PyTuple_SET_ITEM(dims, 0, PyLong_FromLong(MAT_NCOLS(X)));
    PyTuple_SET_ITEM(dims, 1, PyLong_FromLong(MAT_NROWS(X)));
#else
    PyTuple_SET_ITEM(dims, 0, PyInt_FromLong(MAT_NCOLS(X)));
    PyTuple_SET_ITEM(dims, 1, PyInt_FromLong(MAT_NROWS(X)));
#endif
    free_dims = 1;
  }

  if (!PyTuple_Check(dims))
    PY_ERR_TYPE("invalid dimension tuple");

  if (type && !PyTuple_Check(type)) {
    if (free_dims) { Py_DECREF(dims); }
    PY_ERR_TYPE("invalid type tuple");
  }

  int len = PySequence_Size(dims);
  if (type && PySequence_Size(type) != len) {
    if (free_dims) { Py_DECREF(dims); }
    PY_ERR_TYPE("dimensions and type tuples must have same length");
  }

  PyObject *seq = PySequence_Fast(dims, "list is not iterable");

  if (!(dimarr = malloc(len*sizeof(int)))) {
    if (free_dims) { Py_DECREF(dims); }
    Py_DECREF(seq);
    return PyErr_NoMemory();
  }

  if (!(kindarr = malloc(len*sizeof(fftw_r2r_kind)))) {
    if (free_dims) { Py_DECREF(dims); }
    Py_DECREF(seq);
    free(dimarr);
    return PyErr_NoMemory();
  }

  int i, proddim = 1;
  for (i=0; i<len; i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(seq, i);

#if PY_MAJOR_VERSION >= 3
    if (!PyLong_Check(item)) {
#else
    if (!PyInt_Check(item)) {
#endif
      if (free_dims) { Py_DECREF(dims); }
      Py_DECREF(seq);
      free(dimarr); free(kindarr);
      PY_ERR_TYPE("non-integer in dimension tuple");
    }

#if PY_MAJOR_VERSION >= 3
    dimarr[len-i-1] = PyLong_AS_LONG(item);
#else
    dimarr[len-i-1] = PyInt_AS_LONG(item);
#endif
    if (dimarr[len-i-1] < 0) {
      if (free_dims) { Py_DECREF(dims); }
      Py_DECREF(seq);
      free(dimarr); free(kindarr);
      PY_ERR(PyExc_ValueError, "negative dimension");
    }
    proddim *= dimarr[len-i-1];
  }

  if (free_dims) { Py_DECREF(dims); }

  Py_DECREF(seq);

  if (proddim != MAT_LGT(X)) {
    free(dimarr); free(kindarr);
    PY_ERR_TYPE("length of X does not match dimensions");
  }

  if (proddim == 0) {
    free(dimarr); free(kindarr);
    return Py_BuildValue("");
  }

  if (type == NULL) {
    for (i=0; i<len; i++)
      kindarr[i] = FFTW_REDFT01;
  } else {

    seq = PySequence_Fast(type, "list is not iterable");

    for (i=0; i<len; i++) {
      PyObject *item = PySequence_Fast_GET_ITEM(seq, i);

#if PY_MAJOR_VERSION >= 3
      if (!PyLong_Check(item)) {
#else
      if (!PyInt_Check(item)) {
#endif
        Py_DECREF(seq);
        free(dimarr); free(kindarr);
        PY_ERR_TYPE("non-integer in type tuple");
      }

#if PY_MAJOR_VERSION >= 3
      switch(PyLong_AS_LONG(item)) {
#else
      switch(PyInt_AS_LONG(item)) {
#endif
      case 1:
          kindarr[len-i-1] = FFTW_REDFT00;
          if (dimarr[len-i-1] <= 1) {
              Py_DECREF(seq);
              free(dimarr); free(kindarr);
              PY_ERR(PyExc_ValueError,
                  "dimension must be greater than 1 for DCT-I");
          }
          break;
      case 2: kindarr[len-i-1] = FFTW_REDFT01; break;
      case 3: kindarr[len-i-1] = FFTW_REDFT10; break;
      case 4: kindarr[len-i-1] = FFTW_REDFT11; break;
      default:
          Py_DECREF(seq);
          free(dimarr); free(kindarr);
          PY_ERR(PyExc_ValueError, "type must be between 1 and 4");
      }
    }

    Py_DECREF(seq);
  }

  double a = 1.0;
  for (i=0; i<len; i++)
    a /= (kindarr[i] == FFTW_REDFT00 ? MAX(1,2*(dimarr[i]-1)) : 2*dimarr[i]);

  int ix = 1;
  dscal_(&proddim, &a, MAT_BUFD(X), &ix);

  fftw_plan p = fftw_plan_r2r(len, dimarr,
      X->buffer, X->buffer, kindarr, FFTW_ESTIMATE);

  Py_BEGIN_ALLOW_THREADS
  fftw_execute(p);
  Py_END_ALLOW_THREADS
  fftw_destroy_plan(p);
  free(dimarr); free(kindarr);
  return Py_BuildValue("");
}

static char doc_dst[] =
    "DST of a matrix.\n"
    "X := dst(X, type=1)\n\n"
    "PURPOSE\n"
    "Computes the DST of a dense matrix X column by column.\n\n"
    "ARGUMENTS\n"
    "X         A dense matrix of typecode 'd'.\n\n"
    "type      integer from 1 to 4; chooses either DST-I, DST-II, \n"
    "          DST-III or DST-IV.";

static PyObject *dst(PyObject *self, PyObject *args, PyObject *kwrds)
{
  matrix *X;
  int type = 1;
  char *kwlist[] = {"X", "type", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|i", kwlist, &X, &type))
    return NULL;

  if (!(Matrix_Check(X) && MAT_ID(X) == DOUBLE))
    PY_ERR(PyExc_ValueError, "X must be a dense matrix with type 'd'");

  int m = X->nrows, n = X->ncols;
  if (m == 0) return Py_BuildValue("");

  fftw_r2r_kind kind;
  switch(type) {
  case 1: kind = FFTW_RODFT00; break;
  case 2: kind = FFTW_RODFT10; break;
  case 3: kind = FFTW_RODFT01; break;
  case 4: kind = FFTW_RODFT11; break;
  default: PY_ERR(PyExc_ValueError, "type must be between 1 and 4");
  }
  fftw_plan p = fftw_plan_many_r2r(1, &m, n,
      X->buffer, &m, 1, m,
      X->buffer, &m, 1, m,
      &kind, FFTW_ESTIMATE);

  Py_BEGIN_ALLOW_THREADS
  fftw_execute(p);
  Py_END_ALLOW_THREADS
  fftw_destroy_plan(p);
  return Py_BuildValue("");
}

static char doc_dstn[] =
    "N-dimensional DST of a matrix.\n"
    "X := dstn(X, dims, type=1)\n\n"
    "PURPOSE\n"
    "Computes the DST of an N-dimensional array represented by a dense\n"
    "matrix X. The shape of the matrix X does not matter, but the data\n"
    "must be arranged in row-major-order.  The total dimension (defined\n"
    "as mul(dims)) must equal the length of the array X.\n\n"
    "ARGUMENTS\n"
    "X         A dense matrix of typecode 'd'.\n\n"
    "dims      a tuple with the dimensions of the array.\n\n"
    "type      integer from 1 to 4; chooses either DST-I, DST-II, \n"
    "          DST-III or DST-IV.";

static PyObject *dstn(PyObject *self, PyObject *args, PyObject *kwrds)
{
  matrix *X;
  PyObject *dims = NULL, *type = NULL;
  char *kwlist[] = {"X", "dims", "type", NULL};

  int *dimarr;
  fftw_r2r_kind *kindarr;
  int free_dims = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|OO:dctn", kwlist,
	  &X, &dims, &type))
    return NULL;

  if (!(Matrix_Check(X) && MAT_ID(X) == DOUBLE))
    PY_ERR_TYPE("X must be a dense matrix with type 'd'");

  if (!dims) {
    dims = PyTuple_New(2);
    if (!dims) return PyErr_NoMemory();

#if PY_MAJOR_VERSION >= 3
    PyTuple_SET_ITEM(dims, 0, PyLong_FromLong(MAT_NCOLS(X)));
    PyTuple_SET_ITEM(dims, 1, PyLong_FromLong(MAT_NROWS(X)));
#else
    PyTuple_SET_ITEM(dims, 0, PyInt_FromLong(MAT_NCOLS(X)));
    PyTuple_SET_ITEM(dims, 1, PyInt_FromLong(MAT_NROWS(X)));
#endif
    free_dims = 1;
  }

  if (!PyTuple_Check(dims))
    PY_ERR_TYPE("invalid dimension tuple");

  if (type && !PyTuple_Check(type))
    PY_ERR_TYPE("invalid type tuple");

  int len = PySequence_Size(dims);
  if (type && PySequence_Size(type) != len) {
    if (free_dims) { Py_DECREF(dims); }
    PY_ERR_TYPE("dimensions and type tuples must have same length");
  }

  PyObject *seq = PySequence_Fast(dims, "list is not iterable");

  if (!(dimarr = malloc(len*sizeof(int)))) {
    if (free_dims) { Py_DECREF(dims); }
    return PyErr_NoMemory();
  }

  if (!(kindarr = malloc(len*sizeof(fftw_r2r_kind)))) {
    if (free_dims) { Py_DECREF(dims); }
    free(dimarr);
    return PyErr_NoMemory();
  }

  int i, proddim = 1;
  for (i=0; i<len; i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(seq, i);

#if PY_MAJOR_VERSION >= 3
    if (!PyLong_Check(item)) {
#else
    if (!PyInt_Check(item)) {
#endif
      if (free_dims) { Py_DECREF(dims); }
      free(dimarr); free(kindarr);
      PY_ERR_TYPE("non-integer in dimension tuple");
    }

#if PY_MAJOR_VERSION >= 3
    dimarr[len-i-1] = PyLong_AS_LONG(item);
#else
    dimarr[len-i-1] = PyInt_AS_LONG(item);
#endif
    if (dimarr[len-i-1] < 0) {
      if (free_dims) { Py_DECREF(dims); }
      free(dimarr); free(kindarr);
      PY_ERR(PyExc_ValueError, "negative dimension");
    }
    proddim *= dimarr[len-i-1];
  }

  if (free_dims) { Py_DECREF(dims); }

  if (proddim != MAT_LGT(X)) {
    free(dimarr); free(kindarr);
    PY_ERR_TYPE("length of X does not match dimensions");
  }

  if (proddim == 0) {
    free(dimarr); free(kindarr);
    return Py_BuildValue("");
  }

  if (type == NULL) {
    for (i=0; i<len; i++)
      kindarr[i] = FFTW_RODFT00;
  } else {

    seq = PySequence_Fast(type, "list is not iterable");

    for (i=0; i<len; i++) {
      PyObject *item = PySequence_Fast_GET_ITEM(seq, i);

#if PY_MAJOR_VERSION >= 3
      if (!PyLong_Check(item)) {
#else
      if (!PyInt_Check(item)) {
#endif
	free(dimarr); free(kindarr);
	PY_ERR_TYPE("non-integer in type tuple");
      }

#if PY_MAJOR_VERSION >= 3
      switch(PyLong_AS_LONG(item)) {
#else
      switch(PyInt_AS_LONG(item)) {
#endif
      case 1: kindarr[len-i-1] = FFTW_RODFT00; break;
      case 2: kindarr[len-i-1] = FFTW_RODFT10; break;
      case 3: kindarr[len-i-1] = FFTW_RODFT01; break;
      case 4: kindarr[len-i-1] = FFTW_RODFT11; break;
      default:
          free(dimarr); free(kindarr);
          PY_ERR(PyExc_ValueError, "type must be between 1 and 4");
      }
    }
  }

  fftw_plan p = fftw_plan_r2r(len, dimarr,
      X->buffer, X->buffer, kindarr, FFTW_ESTIMATE);

  Py_BEGIN_ALLOW_THREADS
  fftw_execute(p);
  Py_END_ALLOW_THREADS
  fftw_destroy_plan(p);
  free(dimarr); free(kindarr);
  return Py_BuildValue("");
}

static char doc_idst[] =
    "IDST of a matrix.\n"
    "X := idst(X, type=1)\n\n"
    "PURPOSE\n"
    "Computes the IDST of a dense matrix X column by column.\n\n"
    "ARGUMENTS\n"
    "X         A dense matrix of typecode 'd'.\n\n"
    "type      integer from 1 to 4; chooses the inverse transform for\n"
    "          either DST-I, DST-II, DST-III or DST-IV.";

static PyObject *idst(PyObject *self, PyObject *args, PyObject *kwrds)
{
  matrix *X;
  int type = 1;
  char *kwlist[] = {"X", "type", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|i", kwlist, &X, &type))
    return NULL;

  if (!(Matrix_Check(X) && MAT_ID(X) == DOUBLE))
    PY_ERR(PyExc_ValueError, "X must be a dense matrix with type 'd'");

  int m = X->nrows, n = X->ncols;
  if (m == 0) return Py_BuildValue("");

  fftw_r2r_kind kind;
  switch(type) {
  case 1: kind = FFTW_RODFT00; break;
  case 2: kind = FFTW_RODFT01; break;
  case 3: kind = FFTW_RODFT10; break;
  case 4: kind = FFTW_RODFT11; break;
  default: PY_ERR(PyExc_ValueError, "type must be between 1 and 4");
  }
  fftw_plan p = fftw_plan_many_r2r(1, &m, n,
      X->buffer, &m, 1, m,
      X->buffer, &m, 1, m,
      &kind, FFTW_ESTIMATE);

  Py_BEGIN_ALLOW_THREADS
  fftw_execute(p);
  Py_END_ALLOW_THREADS

  double a = 1.0/(type == 1 ? MAX(1,2*(m+1)) : 2*m);
  int mn = m*n, ix = 1;
  dscal_(&mn, &a, MAT_BUFD(X), &ix);

  fftw_destroy_plan(p);
  return Py_BuildValue("");
}

static char doc_idstn[] =
    "Inverse N-dimensional DST of a matrix.\n"
    "X := idstn(X, dims, type=1)\n\n"
    "PURPOSE\n"
    "Computes the IDST of an N-dimensional array represented by a dense\n"
    "matrix X. The shape of the matrix X does not matter, but the data\n"
    "must be arranged in row-major-order.  The total dimension (defined\n"
    "as mul(dims)) must equal the length of the array X.\n\n"
    "ARGUMENTS\n"
    "X         A dense matrix of typecode 'd'.\n\n"
    "dims      a tuple with the dimensions of the array.\n\n"
    "type      integer from 1 to 4; chooses either DST-I, DST-II, \n"
    "          DST-III or DST-IV.";

static PyObject *idstn(PyObject *self, PyObject *args, PyObject *kwrds)
{
  matrix *X;
  PyObject *dims = NULL, *type = NULL;
  char *kwlist[] = {"X", "dims", "type", NULL};

  int *dimarr;
  fftw_r2r_kind *kindarr;
  int free_dims = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|OO:dctn", kwlist,
	  &X, &dims, &type))
    return NULL;

  if (!(Matrix_Check(X) && MAT_ID(X) == DOUBLE))
    PY_ERR_TYPE("X must be a dense matrix with type 'd'");

  if (!dims) {
    dims = PyTuple_New(2);
    if (!dims) return PyErr_NoMemory();

#if PY_MAJOR_VERSION >= 3
    PyTuple_SET_ITEM(dims, 0, PyLong_FromLong(MAT_NCOLS(X)));
    PyTuple_SET_ITEM(dims, 1, PyLong_FromLong(MAT_NROWS(X)));
#else
    PyTuple_SET_ITEM(dims, 0, PyInt_FromLong(MAT_NCOLS(X)));
    PyTuple_SET_ITEM(dims, 1, PyInt_FromLong(MAT_NROWS(X)));
#endif
    free_dims = 1;
  }

  if (!PyTuple_Check(dims))
    PY_ERR_TYPE("invalid dimension tuple");

  if (type && !PyTuple_Check(type)) {
    if (free_dims) { Py_DECREF(dims); }
    PY_ERR_TYPE("invalid type tuple");
  }

  int len = PySequence_Size(dims);
  if (type && PySequence_Size(type) != len) {
    if (free_dims) { Py_DECREF(dims); }
    PY_ERR_TYPE("dimensions and type tuples must have same length");
  }

  PyObject *seq = PySequence_Fast(dims, "list is not iterable");

  if (!(dimarr = malloc(len*sizeof(int)))) {
    if (free_dims) { Py_DECREF(dims); }
    return PyErr_NoMemory();
  }

  if (!(kindarr = malloc(len*sizeof(fftw_r2r_kind)))) {
    if (free_dims) { Py_DECREF(dims); }
    free(dimarr);
    return PyErr_NoMemory();
  }

  int i, proddim = 1;
  for (i=0; i<len; i++) {
    PyObject *item = PySequence_Fast_GET_ITEM(seq, i);

#if PY_MAJOR_VERSION >= 3
    if (!PyLong_Check(item)) {
#else
    if (!PyInt_Check(item)) {
#endif
      if (free_dims) { Py_DECREF(dims); }
      free(dimarr); free(kindarr);
      PY_ERR_TYPE("non-integer in dimension tuple");
    }

#if PY_MAJOR_VERSION >= 3
    dimarr[len-i-1] = PyLong_AS_LONG(item);
#else
    dimarr[len-i-1] = PyInt_AS_LONG(item);
#endif
    if (dimarr[len-i-1] < 0) {
      if (free_dims) { Py_DECREF(dims); }
      free(dimarr); free(kindarr);
      PY_ERR(PyExc_ValueError, "negative dimension");
    }
    proddim *= dimarr[len-i-1];
  }

  if (free_dims) { Py_DECREF(dims); }

  if (proddim != MAT_LGT(X)) {
    free(dimarr); free(kindarr);
    PY_ERR_TYPE("length of X does not match dimensions");
  }

  if (proddim == 0) {
    free(dimarr); free(kindarr);
    return Py_BuildValue("");
  }

  if (type == NULL) {
    for (i=0; i<len; i++)
      kindarr[i] = FFTW_RODFT00;
  } else {

    seq = PySequence_Fast(type, "list is not iterable");

    for (i=0; i<len; i++) {
      PyObject *item = PySequence_Fast_GET_ITEM(seq, i);

#if PY_MAJOR_VERSION >= 3
      if (!PyLong_Check(item)) {
#else
      if (!PyInt_Check(item)) {
#endif
	free(dimarr); free(kindarr);
	PY_ERR_TYPE("non-integer in type tuple");
      }

#if PY_MAJOR_VERSION >= 3
      switch(PyLong_AS_LONG(item)) {
#else
      switch(PyInt_AS_LONG(item)) {
#endif
      case 1: kindarr[len-i-1] = FFTW_RODFT00; break;
      case 2: kindarr[len-i-1] = FFTW_RODFT10; break;
      case 3: kindarr[len-i-1] = FFTW_RODFT01; break;
      case 4: kindarr[len-i-1] = FFTW_RODFT11; break;
      default:
	free(dimarr); free(kindarr);
	PY_ERR(PyExc_ValueError, "type must be between 1 and 4");
      }
    }
  }

  double a = 1.0;
  for (i=0; i<len; i++)
    a /= (kindarr[i] == FFTW_RODFT00 ? MAX(1,2*(dimarr[i]+1)) : 2*dimarr[i]);

  int ix = 1;
  dscal_(&proddim, &a, MAT_BUFD(X), &ix);

  fftw_plan p = fftw_plan_r2r(len, dimarr,
      X->buffer, X->buffer, kindarr, FFTW_ESTIMATE);

  Py_BEGIN_ALLOW_THREADS
  fftw_execute(p);
  Py_END_ALLOW_THREADS
  fftw_destroy_plan(p);
  free(dimarr); free(kindarr);
  return Py_BuildValue("");
}


static PyMethodDef fftw_functions[] = {
    {"dft", (PyCFunction) dft, METH_VARARGS|METH_KEYWORDS, doc_dft},
    {"dftn", (PyCFunction) dftn, METH_VARARGS|METH_KEYWORDS, doc_dftn},
    {"idft", (PyCFunction) idft, METH_VARARGS|METH_KEYWORDS, doc_idft},
    {"idftn", (PyCFunction) idftn, METH_VARARGS|METH_KEYWORDS, doc_idftn},
    {"dct", (PyCFunction) dct, METH_VARARGS|METH_KEYWORDS, doc_dct},
    {"dctn", (PyCFunction) dctn, METH_VARARGS|METH_KEYWORDS, doc_dctn},
    {"idct", (PyCFunction) idct, METH_VARARGS|METH_KEYWORDS, doc_idct},
    {"idctn", (PyCFunction) idctn, METH_VARARGS|METH_KEYWORDS, doc_idctn},
    {"dst", (PyCFunction) dst, METH_VARARGS|METH_KEYWORDS, doc_dst},
    {"dstn", (PyCFunction) dstn, METH_VARARGS|METH_KEYWORDS, doc_dstn},
    {"idst", (PyCFunction) idst, METH_VARARGS|METH_KEYWORDS, doc_idst},
    {"idstn", (PyCFunction) idstn, METH_VARARGS|METH_KEYWORDS, doc_idstn},
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static PyModuleDef fftw_module = {
    PyModuleDef_HEAD_INIT,
    "fftw",
    fftw__doc__,
    -1,
    fftw_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_fftw(void)
{
  PyObject *m;
  if (!(m = PyModule_Create(&fftw_module))) return NULL;
  if (import_cvxopt() < 0) return NULL;
  return m;
}

#else

PyMODINIT_FUNC initfftw(void)
{
  PyObject *m;
  m = Py_InitModule3("cvxopt.fftw", fftw_functions, fftw__doc__);
  if (import_cvxopt() < 0) return;
}

#endif
