/*
 * Copyright 2004-2009 J. Dahl and L. Vandenberghe.
 *
 * This file is part of CVXOPT version 1.1.2
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

/*** Not upgraded to python 3.3. ***/

#include "cvxopt.h"
#include "misc.h"

#if PY_MAJOR_VERSION < 3
static  PyObject *ma57_module;
#endif

extern void ma57id_(double *cntl, int *icnt);

extern void ma57ad_(int *n, int *ne, int *irn, int *jcn, int *lkeep, int *keep,
                    int *iwork, int *icntl, int *info, double *rinfo);

extern void ma57bd_(int *n, int *ne, double *A, double *fact, int *lfact,
                    int *ifact, int *lifact, int *lkeep, int *keep, int *iwork,
                    int *icntl, double *cntl, int *info, double *rinfo);

extern void ma57cd_(int *job, int *n, double *fact, int *lfact, int *ifact,
                    int *lifact, int *nrhs, double *rhs, int *lrhs,
                    double *work, int *lwork, int *iwork, int *icntl,
                    int *info);

typedef struct {
  int n;

  double cntl[5], rinfo[20];
  int icntl[20], info[40];

  int lkeep, lfact, lifact;
  int *keep, *ifact, *iwork;
  double *fact;

} ma57_factor;

PyDoc_STRVAR(ma57__doc__, "Interface to the MA57 library.\n\n");

static void free_factor(void *L, void *descr)
{
  free(((ma57_factor *)L)->keep);
  free(((ma57_factor *)L)->ifact);
  free(((ma57_factor *)L)->iwork);
  free(((ma57_factor *)L)->fact);
  free(L);
}

static char doc_factorize[] =
    "Factorization of a real symmetric sparse matrix.\n\n"
    "F = factorize(A)\n\n"
    "PURPOSE\n"
    "Computes a P * L * D * L^T * P^T factorization of a real\n"
    "symmetric matrix.\n\n"
    "ARGUMENTS\n"
    "A         square sparse matrix. Only the lower triangular\n"
    "          part is used.\n\n"
    "RETURNS\n"
    "F         An opaque object reprenting the numeric\n"
    "          factorization.";

static PyObject* factorize(PyObject *self, PyObject *args)
{
    spmatrix *A;

    if (!PyArg_ParseTuple(args, "O", &A)) return NULL;

    if (!SpMatrix_Check(A) || SP_ID(A) != DOUBLE ||
          SP_NROWS(A) != SP_NCOLS(A))
        PY_ERR_TYPE("A is not a sparse square 'd' matrix");

    int j, k, nnz = 0;
    for (j=0; j<SP_NCOLS(A); j++) {
      for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1]; k++) {
        if (SP_ROW(A)[k] >= j) nnz++;
      }
    }

    int *irn = malloc(nnz*sizeof(int));
    int *jcn = malloc(nnz*sizeof(int));
    double *a = malloc(nnz*sizeof(double));
    if (!irn || !jcn || !a) {
      free(irn); free(jcn); free(a);
      return PyErr_NoMemory();
    }

    nnz = 0;
    for (j=0; j<SP_NCOLS(A); j++) {
      for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1]; k++) {
        if (SP_ROW(A)[k] >= j) {
          irn[nnz] = SP_ROW(A)[k] + 1;
          jcn[nnz] = j + 1;
          a[nnz] = SP_VALD(A)[k];
          nnz++;
        }
      }
    }

    ma57_factor *S = malloc(sizeof(ma57_factor));
    if (!S) return PyErr_NoMemory();

    S->n = SP_NROWS(A);
    S->lkeep = 5*S->n + nnz + MAX(S->n, nnz) + 42;
    if (!(S->keep = malloc(S->lkeep*sizeof(int))) ) {

      free(irn); free(jcn); free(a); free(S);
      return PyErr_NoMemory();
    }

    /* set default values for control parameters */
    ma57id_(S->cntl, S->icntl);

    S->iwork = malloc(5*S->n*sizeof(int));
    if (!S->iwork) {
      free(S->keep); free(S);
      return PyErr_NoMemory();
    }

    ma57ad_(&S->n, &nnz, irn, jcn, &S->lkeep, S->keep, S->iwork,
        S->icntl, S->info, S->rinfo);

    S->lfact = S->info[8];
    S->lifact = S->info[9];
    S->fact = malloc(S->lfact*sizeof(double));
    S->ifact = malloc(S->lifact*sizeof(int));
    if (!S->fact || !S->ifact) {

      free(irn); free(jcn); free(a);
      free(S->keep); free(S->iwork); free(S->fact); free(S->ifact); free(S);
      return PyErr_NoMemory();
    }

    ma57bd_(&S->n, &nnz, a, S->fact, &S->lfact, S->ifact, &S->lifact,
        &S->lkeep, S->keep, S->iwork, S->icntl, S->cntl, S->info, S->rinfo);

    free(irn); free(jcn); free(a);
    if (S->info[0] == -3 || S->info[0] == -4) {
      free(S->keep); free(S->iwork); free(S->fact); free(S->ifact); free(S);
      return PyErr_NoMemory();
    }

    if (S->info[0] == -5) {
      free(S->keep); free(S->iwork); free(S->fact); free(S->ifact); free(S);
      PY_ERR(PyExc_ArithmeticError, "Singular matrix error");
    }

    if (S->info[0] < 0) {
      free(S->keep); free(S->iwork); free(S->fact); free(S->ifact); free(S);
      PY_ERR_TYPE("MA57 returned with an error");
    }

    return (PyObject *) PyCObject_FromVoidPtrAndDesc((void *) S,
        "MA57 FACTOR", free_factor);
}

static char doc_solve[] =
    "Solves a sparse set of linear equations with a factored\n"
    "coefficient matrix and an dense matrix as right-hand side.\n\n"
    "solve(F, B, nrhs=B.size[1], ldB=max(1,B.size[0], "
    "offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves the A*X = B using the factorization computed by\n"
    "ma57.factorize.\n"
    "ARGUMENTS\n"
    "F         the factorization object of A computed by\n"
    "          ma57.factorize\n\n"
    "B         dense 'd' matrix.\n\n"
    "nrhs      integer.  If negative, the default value is used.\n\n"
    "ldB       nonnegative integer.  ldB >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetB   nonnegative integer";

static PyObject* solve(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *B;
    PyObject *F;
    int n, oB=0, ldB=0, nrhs=-1;
    char *descr;
    char *kwlist[] = {"F", "B", "nrhs", "ldB", "offsetB", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|iii", kwlist,
        &F, &B, &nrhs, &ldB, &oB)) return NULL;

    if (!PyCObject_Check(F)) err_CO("F");
    descr = PyCObject_GetDesc(F);
    if (!descr || strncmp(descr, "MA57 FACTOR", 14))
        PY_ERR_TYPE("F is not a MA57 factor");

    ma57_factor *S = (ma57_factor *) PyCObject_AsVoidPtr(F);

    n = S->n;

    if (!Matrix_Check(B) || MAT_ID(B) != DOUBLE)
      PY_ERR_TYPE("B must a dense matrix with typecode 'd'");

    if (nrhs < 0) nrhs = MAT_NCOLS(B);
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldB == 0) ldB = MAX(1,MAT_NROWS(B));
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > MAT_LGT(B)) err_buf_len("B");

    int lwork = S->n*nrhs;
    double *work = malloc(lwork*sizeof(double));
    if (!work) return PyErr_NoMemory();

    /* Solve the equations */
    int job = 1;
    ma57cd_(&job, &n, S->fact, &S->lfact, S->ifact, &S->lifact,
        &nrhs, MAT_BUFD(B) + oB, &ldB, work, &lwork, S->iwork, S->icntl, S->info);

    free(work);

    if (S->info[0] >= 0) {
      return Py_BuildValue("");
    } else {
      PY_ERR_TYPE("MA57 returned with an error");
    }
}

static PyMethodDef ma57_functions[] = {
  {"factorize", (PyCFunction) factorize, METH_VARARGS|METH_KEYWORDS,
   doc_factorize},
  {"solve", (PyCFunction) solve, METH_VARARGS|METH_KEYWORDS,
   doc_solve},
  {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static PyModuleDef ma57_module = {
    PyModuleDef_HEAD_INIT,
    "ma57",
    ma57__doc__,
    -1,
    ma57_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_ma57(void)
{
  PyObject *m;
  if (!(m = PyModule_Create(&ma57_module))) return NULL;
  if (import_cvxopt() < 0) return NULL;
  return m;
}

#else

PyMODINIT_FUNC initma57(void)
{
  ma57_module = Py_InitModule3("cvxopt.ma57", ma57_functions, ma57__doc__);
  if (import_cvxopt() < 0) return;
}
#endif
