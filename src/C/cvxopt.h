/*
 * Copyright 2012-2018 M. Andersen and L. Vandenberghe.
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
#include "structmember.h"
#include "blas_redefines.h"

#include "assert.h"

/* ANSI99 complex is disabled during build of CHOLMOD */

#if !defined(NO_ANSI99_COMPLEX) 
#include "complex.h"
#if !defined(_MSC_VER)
#define MAT_BUFZ(O)  ((double complex *)((matrix *)O)->buffer)
#else
#define MAT_BUFZ(O)  ((_Dcomplex *)((matrix *)O)->buffer)
#endif
#endif

#ifndef __CVXOPT__
#define __CVXOPT__

#define INT       0
#define DOUBLE    1
#define COMPLEX   2

#define int_t     Py_ssize_t

typedef struct {
  PyObject_HEAD
  void *buffer;          /* in column-major-mode array of type 'id' */
  int nrows, ncols;      /* number of rows and columns */
  int   id;              /* DOUBLE, INT, COMPLEX */
  int_t shape[2];
  int_t strides[2];
  int_t ob_exports;
} matrix;

typedef struct {
  void  *values;      /* value list */
  int_t *colptr;      /* column pointer list */
  int_t *rowind;      /* row index list */
  int_t nrows, ncols; /* number of rows and columns */
  int   id;           /* DOUBLE, COMPLEX */
} ccs;

typedef struct {
  PyObject_HEAD
  ccs *obj;
} spmatrix;

#ifdef BASE_MODULE

#define Matrix_Check(self) PyObject_TypeCheck(self, &matrix_tp)
#define SpMatrix_Check(self) PyObject_TypeCheck(self, &spmatrix_tp)

#else

static void **cvxopt_API;

#define Matrix_New (*(matrix * (*)(int, int, int)) cvxopt_API[0])
#define Matrix_NewFromMatrix (*(matrix * (*)(matrix *, int)) cvxopt_API[1])
#define Matrix_NewFromList (*(matrix * (*)(PyObject *, int)) cvxopt_API[2])
#define Matrix_Check (*(int * (*)(void *)) cvxopt_API[3])

#define SpMatrix_New (*(spmatrix * (*)(int_t, int_t, int_t, int)) cvxopt_API[4])
#define SpMatrix_NewFromSpMatrix \
  (*(spmatrix * (*)(spmatrix *, int)) cvxopt_API[5])
#define SpMatrix_NewFromIJV \
  (*(spmatrix * (*)(matrix *, matrix *, matrix *, int_t, int_t, int)) \
      cvxopt_API[6])
#define SpMatrix_Check (*(int * (*)(void *)) cvxopt_API[7])

/* Return -1 and set exception on error, 0 on success. */
static int
import_cvxopt(void)
{
  PyObject *module = PyImport_ImportModule("cvxopt.base");

  if (module != NULL) {
    PyObject *c_api_object = PyObject_GetAttrString(module, "_C_API");
#if PY_MAJOR_VERSION >= 3
    if (!c_api_object || !PyCapsule_IsValid(c_api_object, "base_API"))
        return -1;
    cvxopt_API = (void **) PyCapsule_GetPointer(c_api_object, "base_API");
#else
    if (!c_api_object || !(PyCObject_Check(c_api_object)))
        return -1;
    cvxopt_API = (void **) PyCObject_AsVoidPtr(c_api_object);
#endif
    Py_DECREF(c_api_object);
  }
  return 0;
}

#endif

/*
 * Below this line are non-essential convenience macros
 */

#define MAT_BUF(O)   ((matrix *)O)->buffer
#define MAT_BUFI(O)  ((int_t *)((matrix *)O)->buffer)
#define MAT_BUFD(O)  ((double *)((matrix *)O)->buffer)
#ifndef _MSC_VER
#define MAT_BUFZ(O)  ((double complex *)((matrix *)O)->buffer)
#else
#define MAT_BUFZ(O)  ((_Dcomplex *)((matrix *)O)->buffer)
#endif
#define MAT_NROWS(O) ((matrix *)O)->nrows
#define MAT_NCOLS(O) ((matrix *)O)->ncols
#define MAT_LGT(O)   (MAT_NROWS(O)*MAT_NCOLS(O))
#define MAT_ID(O)    ((matrix *)O)->id

#define SP_NCOLS(O)  ((spmatrix *)O)->obj->ncols
#define SP_NROWS(O)  ((spmatrix *)O)->obj->nrows
#define SP_LGT(O)    (SP_NROWS(O)*SP_NCOLS(O))
#define SP_NNZ(O)    ((spmatrix *)O)->obj->colptr[SP_NCOLS(O)]
#define SP_ID(O)     ((spmatrix *)O)->obj->id
#define SP_COL(O)    ((spmatrix *)O)->obj->colptr
#define SP_ROW(O)    ((spmatrix *)O)->obj->rowind
#define SP_VAL(O)    ((spmatrix *)O)->obj->values
#define SP_VALD(O)   ((double *)((spmatrix *)O)->obj->values)
#ifndef _MSC_VER
#define SP_VALZ(O)   ((double complex *)((spmatrix *)O)->obj->values)
#else
#define SP_VALZ(O)   ((_Dcomplex *)((spmatrix *)O)->obj->values)
#endif

#define CCS_NROWS(O) ((ccs *)O)->nrows
#define CCS_NCOLS(O) ((ccs *)O)->ncols
#define CCS_NNZ(O)   ((ccs *)O)->colptr[CCS_NCOLS(O)]

#endif
