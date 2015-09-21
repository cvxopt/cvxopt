/*
 * Copyright 2004-2009 J. Dahl and L. Vandenberghe.
 *
 * This file is part of CVXOPT version 1.1.6
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

/*** not upgraded to Python 3.3. ***/

#include "cvxopt.h"
#include "metis.h"
#include "misc.h"

/* Backward compatibility with Metis 4 */
#ifndef METIS_VER_MAJOR  
typedef idxtype idx_t; 
#endif

PyDoc_STRVAR(metis__doc__,"Interface to the METIS library.\n\n"
    "Multilevel nested dissection ordering of sparse matrices.\n\n"
    "Currently only default optional arguments are implemented.");

#if PY_MAJOR_VERSION < 3
static  PyObject *metis_module;
#endif



static ccs * alloc_ccs(int_t nrows, int_t ncols, int_t nnz)
{
  ccs *obj = malloc(sizeof(ccs));
  if (!obj) return NULL;

  obj->nrows = nrows;
  obj->ncols = ncols;
  obj->id = DOUBLE;

  obj->values = malloc(sizeof(double)*nnz);
  obj->colptr = calloc(ncols+1,sizeof(int_t));
  obj->rowind = malloc(sizeof(int_t)*nnz);

  if (!obj->values || !obj->colptr || !obj->rowind) {
    free(obj->values); free(obj->colptr); free(obj->rowind); free(obj);
    return NULL;
  }

  return obj;
}

static void free_ccs(ccs *obj) {
  free(obj->values);
  free(obj->rowind);
  free(obj->colptr);
  free(obj);
}

static ccs * transpose(ccs *A, int_t *p) {

  ccs *B = alloc_ccs(A->ncols, A->nrows, A->colptr[A->ncols]);
  if (!B) return NULL;

  int_t i, j, *buf = calloc(A->nrows,sizeof(int_t));
  if (!buf) { free_ccs(B); return NULL; }

  /* Run through matrix and count number of elms in each row */
  if (p) {
    for (j=0; j<A->ncols; j++) {
      for (i=A->colptr[p[j]]; i<A->colptr[p[j]+1]; i++)
        buf[ A->rowind[i] ]++;
    }
  } else {
    for (j=0; j<A->ncols; j++) {
      for (i=A->colptr[j]; i<A->colptr[j+1]; i++)
        buf[ A->rowind[i] ]++;
    }
  }

  /* generate new colptr */
  for (i=0; i<B->ncols; i++)
    B->colptr[i+1] = B->colptr[i] + buf[i];

  /* fill in rowind and values */
  for (i=0; i<A->nrows; i++) buf[i] = 0;
  for (j=0; j<A->ncols; j++) {
    for (i=A->colptr[p ? p[j] : j]; i<A->colptr[p ? p[j]+1 : j+1]; i++) {
      B->rowind[ B->colptr[A->rowind[i]] + buf[A->rowind[i]] ] = j;
      ((double *)B->values)[B->colptr[A->rowind[i]] + buf[A->rowind[i]]++] =
        ((double *)A->values)[i];
    }
  }

  free(buf);
  return B;
}

static ccs * symmetrize(ccs *A) {

  ccs *At = transpose(A, NULL);
  if (!At) return NULL;

  ccs *B = alloc_ccs(A->nrows, A->nrows, 0);
  if (!B) {
    free_ccs(At); free_ccs(B); return NULL;
  }

  int j, k, cnt = 0;
  for (j=0; j<B->ncols; j++)
    B->colptr[j+1] = B->colptr[j] +
    A->colptr[j+1]-A->colptr[j] + At->colptr[j+1]-At->colptr[j] -
    ((A->colptr[j+1]>A->colptr[j]) && (A->rowind[A->colptr[j]] == j));

  double *newval = malloc(B->colptr[B->ncols]*sizeof(double));
  int_t *newrow = malloc(B->colptr[B->ncols]*sizeof(int_t));
  if (!newval || !newrow) {
    free_ccs(At); free_ccs(B); free(newval); free(newrow); return NULL;
  }

  for (j=0; j<B->ncols; j++) {

    for (k=At->colptr[j]; k<At->colptr[j+1]; k++) {
      newrow[cnt] = At->rowind[k];
      newval[cnt] = ((double *)At->values)[k];
      cnt++;
    }

    for (k=A->colptr[j]; k<A->colptr[j+1]; k++) {
      if (A->rowind[k] > j) {
        newrow[cnt] = A->rowind[k];
        newval[cnt] = ((double *)A->values)[k];
        cnt++;
      }
    }
  }

  free(B->rowind);
  free(B->values);
  B->rowind = newrow;
  B->values = newval;

  free_ccs(At);
  return B;
}

static char doc_order[] =
  "Computes the multilevel nested dissection ordering of a square "
  "matrix.\n\n"
  "p = order(A)\n\n"
  "PURPOSE\n"
  "Computes a permutation p that reduces fill-in in the Cholesky\n"
  "factorization of A[p,p].\n\n"
  "ARGUMENTS\n"
  "A         square sparse lower-triangular matrix. The matrix A\n"
  "          must have non-zero diagonal.\n\n"
  "p         'i' matrix of length equal to the order of A";

static PyObject* order_c(PyObject *self, PyObject *args, PyObject *kwrds)
{
  spmatrix *A;
  char *kwlist[] = {"A", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O", kwlist, &A)) return NULL;

  if ((SP_ID(A) != DOUBLE) || (SP_NROWS(A) != SP_NCOLS(A)))
    PY_ERR_TYPE("A must be sparse lower triangular matrix of doubles");

  int i, j, k, n = SP_NCOLS(A);
  for (j=0; j<n; j++) {
    for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1]; k++) {
      if (SP_ROW(A)[k] < j)
        PY_ERR_TYPE("A must be sparse lower triangular matrix of doubles");

      if (SP_ROW(A)[k] != j)
        PY_ERR_TYPE("A must include diagonal elements");

      break;
    }
  }

  ccs *Asym = symmetrize(A->obj);
  if (!Asym) return PyErr_NoMemory();

  idx_t *xadj   = malloc((n+1)*sizeof(int));
  idx_t *perm   = malloc(n*sizeof(int));
  idx_t *iperm  = malloc(n*sizeof(int));
  if (!xadj || !perm || !iperm) {
    free_ccs(Asym);
    free(xadj); free(perm); free(iperm);
    return PyErr_NoMemory();
  }

  int numflag = 0;
  int options[8] = {0,0,0,0,0,0,0,0};

  xadj[0] = 0;
  for (i=1; i<n+1; i++)
    xadj[i] = xadj[i-1] + (Asym->colptr[i]-Asym->colptr[i-1]-1);

  idx_t *adjncy = malloc(xadj[n]*sizeof(int));
  if (!adjncy) {
    free_ccs(Asym);
    free(xadj); free(perm); free(iperm);
    return PyErr_NoMemory();
  }

  for (j=0, k=0; j<n; j++) {
    for (i=Asym->colptr[j]; i<Asym->colptr[j+1]; i++) {
      if (Asym->rowind[i] != j)
        adjncy[k++] = Asym->rowind[i];
    }
  }

  METIS_NodeND(&n, xadj, adjncy, &numflag, options, perm, iperm);

  matrix *p = Matrix_New(n, 1, INT);
  for (i=0; i<n; i++)
    MAT_BUFI(p)[i] = perm[i];

  free(xadj);
  free(adjncy);
  free(perm);
  free(iperm);
  free_ccs(Asym);
  return (PyObject *)p;
}

static PyMethodDef metis_functions[] = {
    {"order", (PyCFunction) order_c, METH_VARARGS|METH_KEYWORDS, doc_order},
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static PyModuleDef metis_module = {
    PyModuleDef_HEAD_INIT,
    "metis",
    metis__doc__,
    -1,
    metis_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_metis(void)
{
  PyObject *m;
  if (!(m = PyModule_Create(&metis_module))) return NULL;
  if (import_cvxopt() < 0) return NULL;
  return m;
}

#else

PyMODINIT_FUNC initmetis(void)
{
  metis_module = Py_InitModule3("cvxopt.metis", metis_functions,
      metis__doc__);
  PyModule_AddObject(metis_module, "options", PyDict_New());
  if (import_cvxopt() < 0) return;
}

#endif
