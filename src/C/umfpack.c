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

#include "cvxopt.h"
#include "umfpack.h"
#include "misc.h"

#if (SIZEOF_INT < SIZEOF_LONG)
#define UMFD(name) umfpack_dl_ ## name
#define UMFZ(name) umfpack_zl_ ## name
#else
#define UMFD(name) umfpack_di_ ## name
#define UMFZ(name) umfpack_zi_ ## name
#endif

const int E_SIZE[] = {sizeof(int_t), sizeof(double), sizeof(complex)};

static char umfpack_error[20];

PyDoc_STRVAR(umfpack__doc__,"Interface to the UMFPACK library.\n\n"
    "Routines for symbolic and numeric LU factorization of sparse\n"
    "matrices and for solving sparse sets of linear equations.\n"
    "The default control settings of UMPFACK are used.\n\n"
    "See also http://www.cise.ufl.edu/research/sparse/umfpack.");

static void free_umfpack_d_symbolic(void *F, void *descr)
{
    UMFD(free_symbolic)(&F);
}

static void free_umfpack_z_symbolic(void *F, void *descr)
{
    UMFZ(free_symbolic)(&F);
}

static void free_umfpack_d_numeric(void *F, void *descr)
{
    UMFD(free_numeric)(&F);
}

static void free_umfpack_z_numeric(void *F, void *descr)
{
    UMFZ(free_numeric)(&F);
}

static char doc_linsolve[] =
    "Solves a sparse set of linear equations.\n\n"
    "linsolve(A, B, trans='N', nrhs=B.size[1], ldB=max(1,B.size[0]),\n"
    "         offsetB=0)\n\n"
    "PURPOSE\n"
    "If trans is 'N', solves A*X = B.\n"
    "If trans is 'T', solves A^T*X = B.\n"
    "If trans is 'C', solves A^H*X = B.\n"
    "A is a sparse n by n matrix, and B is n by nrhs.\n"
    "On exit B is replaced by the solution.  A is not modified.\n\n"
    "ARGUMENTS\n"
    "A         square sparse matrix\n\n"
    "B         dense matrix of the same type as A, stored following \n"
    "          the BLAS conventions\n\n"
    "trans     'N', 'T' or 'C'\n\n"
    "nrhs      integer.  If negative, the default value is used.\n\n"
    "ldB       nonnegative integer.  ldB >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetB   nonnegative integer";

static PyObject* linsolve(PyObject *self, PyObject *args,
    PyObject *kwrds)
{
    spmatrix *A;
    matrix *B;
    char trans='N';
    double info[UMFPACK_INFO];
    int oB=0, n, nrhs=-1, ldB=0, k;
    void *symbolic, *numeric, *x;
    char *kwlist[] = {"A", "B", "trans", "nrhs", "ldB", "offsetB",
        NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ciii", kwlist,
        &A, &B, &trans, &nrhs, &ldB, &oB)) return NULL;

    if (!SpMatrix_Check(A) || SP_NROWS(A) != SP_NCOLS(A))
        PY_ERR_TYPE("A must be a square sparse matrix");
    n = SP_NROWS(A);
    if (!Matrix_Check(B) || MAT_ID(B) != SP_ID(A))
        PY_ERR_TYPE("B must a dense matrix of the same numeric type "
            "as A");

    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("i", 0);
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > MAT_LGT(B)) err_buf_len("B");

    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");

    if (SP_ID(A) == DOUBLE)
        UMFD(symbolic)(n, n, SP_COL(A), SP_ROW(A), SP_VAL(A), &symbolic,
            NULL, info);
    else
        UMFZ(symbolic)(n, n, SP_COL(A), SP_ROW(A), SP_VAL(A), NULL,
            &symbolic, NULL, info);

    if (info[UMFPACK_STATUS] != UMFPACK_OK){
        if (SP_ID(A) == DOUBLE)
            UMFD(free_symbolic)(&symbolic);
        else
            UMFZ(free_symbolic)(&symbolic);
        if (info[UMFPACK_STATUS] == UMFPACK_ERROR_out_of_memory)
            return PyErr_NoMemory();
        else {
            snprintf(umfpack_error,20,"%s %i","UMFPACK ERROR",
                (int) info[UMFPACK_STATUS]);
            PyErr_SetString(PyExc_ValueError, umfpack_error);
            return NULL;
        }
    }

    if (SP_ID(A) == DOUBLE) {
        UMFD(numeric)(SP_COL(A), SP_ROW(A), SP_VAL(A), symbolic,
            &numeric, NULL, info);
        UMFD(free_symbolic)(&symbolic);
    } else {
        UMFZ(numeric)(SP_COL(A), SP_ROW(A), SP_VAL(A), NULL, symbolic,
            &numeric, NULL, info);
        UMFZ(free_symbolic)(&symbolic);
    }
    if (info[UMFPACK_STATUS] != UMFPACK_OK){
        if (SP_ID(A) == DOUBLE)
            UMFD(free_numeric)(&numeric);
        else
            UMFZ(free_numeric)(&numeric);
        if (info[UMFPACK_STATUS] == UMFPACK_ERROR_out_of_memory)
            return PyErr_NoMemory();
        else {
            if (info[UMFPACK_STATUS] == UMFPACK_WARNING_singular_matrix)
                PyErr_SetString(PyExc_ArithmeticError, "singular "
                    "matrix");
            else {
                snprintf(umfpack_error,20,"%s %i","UMFPACK ERROR",
                    (int) info[UMFPACK_STATUS]);
                PyErr_SetString(PyExc_ValueError, umfpack_error);
            }
            return NULL;
        }
    }

    if (!(x = malloc(n*E_SIZE[SP_ID(A)]))) {
        if (SP_ID(A) == DOUBLE)
            UMFD(free_numeric)(&numeric);
        else
            UMFZ(free_numeric)(&numeric);
        return PyErr_NoMemory();
    }
    for (k=0; k<nrhs; k++){
        if (SP_ID(A) == DOUBLE)
            UMFD(solve)(trans == 'N' ? UMFPACK_A: UMFPACK_Aat,
                SP_COL(A), SP_ROW(A), SP_VAL(A), x,
                MAT_BUFD(B) + k*ldB + oB, numeric, NULL, info);
        else
            UMFZ(solve)(trans == 'N' ? UMFPACK_A: trans == 'C' ?
                UMFPACK_At : UMFPACK_Aat, SP_COL(A), SP_ROW(A),
                SP_VAL(A), NULL, x, NULL,
                (double *)(MAT_BUFZ(B) + k*ldB + oB), NULL, numeric,
                NULL, info);
        if (info[UMFPACK_STATUS] == UMFPACK_OK)
            memcpy(B->buffer + (k*ldB + oB)*E_SIZE[SP_ID(A)], x,
                n*E_SIZE[SP_ID(A)]);
        else
	    break;
    }
    free(x);
    if (SP_ID(A) == DOUBLE)
        UMFD(free_numeric)(&numeric);
    else
        UMFZ(free_numeric)(&numeric);

    if (info[UMFPACK_STATUS] != UMFPACK_OK){
        if (info[UMFPACK_STATUS] == UMFPACK_ERROR_out_of_memory)
            return PyErr_NoMemory();
        else {
            if (info[UMFPACK_STATUS] == UMFPACK_WARNING_singular_matrix)
                PyErr_SetString(PyExc_ArithmeticError, "singular "
                    "matrix");
            else {
                snprintf(umfpack_error,20,"%s %i","UMFPACK ERROR",
                    (int) info[UMFPACK_STATUS]);
                PyErr_SetString(PyExc_ValueError, umfpack_error);
            }
        return NULL;
        }
    }
    return Py_BuildValue("");
}


static char doc_symbolic[] =
    "Symbolic LU factorization of a sparse matrix.\n\n"
    "F = symbolic(A)\n\n"
    "ARGUMENTS\n"
    "A         sparse matrix with at least one row and at least one\n"
    "          column.  A may be rectangular.\n\n"
    "F         the symbolic factorization as an opaque C object";

static PyObject* symbolic(PyObject *self, PyObject *args)
{
    spmatrix *A;
    double info[UMFPACK_INFO];
    void *symbolic;

    if (!PyArg_ParseTuple(args, "O", &A)) return NULL;

    if (!SpMatrix_Check(A)) PY_ERR_TYPE("A must be a sparse matrix");
    if (SP_NCOLS(A) == 0 || SP_NROWS(A) == 0) {
        PyErr_SetString(PyExc_ValueError, "A must have at least one "
            "row and column");
        return NULL;
    }

    switch (SP_ID(A)){
        case DOUBLE:
            UMFD(symbolic)(SP_NROWS(A), SP_NCOLS(A), SP_COL(A),
                SP_ROW(A), SP_VAL(A), &symbolic, NULL, info);
            if (info[UMFPACK_STATUS] == UMFPACK_OK)
                return (PyObject *) PyCObject_FromVoidPtrAndDesc(
                    (void *) symbolic, "UMFPACK SYM D FACTOR",
                    free_umfpack_d_symbolic);
            else
                UMFD(free_symbolic)(&symbolic);
            break;

        case COMPLEX:
            UMFZ(symbolic)(SP_NROWS(A), SP_NCOLS(A), SP_COL(A),
                SP_ROW(A), SP_VAL(A), NULL, &symbolic, NULL, info);
            if (info[UMFPACK_STATUS] == UMFPACK_OK)
                return (PyObject *) PyCObject_FromVoidPtrAndDesc(
                    (void *) symbolic, "UMFPACK SYM Z FACTOR",
                free_umfpack_z_symbolic);
            else
                UMFZ(free_symbolic)(&symbolic);
            break;
    }

    if (info[UMFPACK_STATUS] == UMFPACK_ERROR_out_of_memory)
        return PyErr_NoMemory();
    else {
        snprintf(umfpack_error,20,"%s %i","UMFPACK ERROR",
            (int) info[UMFPACK_STATUS]);
        PyErr_SetString(PyExc_ValueError, umfpack_error);
        return NULL;
    }
}


static char doc_numeric[] =
    "Numeric LU factorization of a sparse matrix, given a symbolic\n"
    "factorization computed by umfpack.symbolic.  Raises an\n"
    "ArithmeticError if A is singular.\n\n"
    "Fn = numeric(A, Fs)\n\n"
    "ARGUMENTS\n"
    "A         sparse matrix; may be rectangular\n\n"
    "Fs        symbolic factorization of A, or a matrix with the same\n"
    "          sparsity pattern, dimensions, and typecode as A, \n"
    "          created by umfpack.symbolic\n\n"
    "Fn        the numeric factorization, as an opaque C object";

static PyObject* numeric(PyObject *self, PyObject *args)
{
    spmatrix *A;
    PyObject *Fs;
    double info[UMFPACK_INFO];
    void *numeric;

    if (!PyArg_ParseTuple(args, "OO", &A, &Fs)) return NULL;

    if (!SpMatrix_Check(A)) PY_ERR_TYPE("A must be a sparse matrix");
    if (!PyCObject_Check(Fs)) err_CO("Fs");

    switch (SP_ID(A)) {
	case DOUBLE:
            TypeCheck_CObject(Fs, "UMFPACK SYM D FACTOR", "Fs is not "
                "the UMFPACK symbolic factor of a 'd' matrix");
            UMFD(numeric)(SP_COL(A), SP_ROW(A), SP_VAL(A),
	        (void *) PyCObject_AsVoidPtr(Fs), &numeric, NULL, info);
            if (info[UMFPACK_STATUS] == UMFPACK_OK)
                return (PyObject *) PyCObject_FromVoidPtrAndDesc(
                    (void *) numeric, "UMFPACK NUM D FACTOR",
                    free_umfpack_d_numeric);
            else
                UMFD(free_numeric)(&numeric);
	    break;

        case COMPLEX:
            TypeCheck_CObject(Fs, "UMFPACK SYM Z FACTOR", "Fs is not "
                "the UMFPACK symbolic factor of a 'z' matrix");
            UMFZ(numeric)(SP_COL(A), SP_ROW(A), SP_VAL(A), NULL,
	         (void *) PyCObject_AsVoidPtr(Fs), &numeric, NULL,
		 info);
            if (info[UMFPACK_STATUS] == UMFPACK_OK)
                return (PyObject *) PyCObject_FromVoidPtrAndDesc(
                    (void *) numeric, "UMFPACK NUM Z FACTOR",
                    free_umfpack_z_numeric);
	    else
                 UMFZ(free_numeric)(&numeric);
	    break;
    }

    if (info[UMFPACK_STATUS] == UMFPACK_ERROR_out_of_memory)
        return PyErr_NoMemory();
    else {
        if (info[UMFPACK_STATUS] == UMFPACK_WARNING_singular_matrix)
	    PyErr_SetString(PyExc_ArithmeticError, "singular matrix");
        else {
	    snprintf(umfpack_error,20,"%s %i","UMFPACK ERROR",
	        (int) info[UMFPACK_STATUS]);
	    PyErr_SetString(PyExc_ValueError, umfpack_error);
        }
        return NULL;
    }
}


static char doc_solve[] =
    "Solves a factored set of linear equations.\n\n"
    "solve(A, F, B, trans='N', nrhs=B.size[1], ldB=max(1,B.size[0]),\n"
    "      offsetB=0)\n\n"
    "PURPOSE\n"
    "If trans is 'N', solves A*X = B.\n"
    "If trans is 'T', solves A^T*X = B.\n"
    "If trans is 'C', solves A^H*X = B.\n"
    "A is a sparse n by n matrix, and B is n by nrhs.  F is the\n"
    "numeric factorization of A, computed by umfpack.numeric.\n"
    "On exit B is replaced by the solution.  A is not modified.\n\n"
    "ARGUMENTS\n"
    "A         square sparse matrix\n\n"
    "F         numeric factorization, as returned by umfpack.numeric\n"
    "\n"
    "B         dense matrix of the same type as A, stored following \n"
    "          the BLAS conventions\n\n"
    "trans     'N', 'T' or 'C'\n\n"
    "nrhs      integer.  If negative, the default value is used.\n\n"
    "ldB       nonnegative integer.  ldB >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetB   nonnegative integer";

static PyObject* solve(PyObject *self, PyObject *args, PyObject *kwrds)
{
    spmatrix *A;
    PyObject *F;
    matrix *B;
    char trans='N';
    double *x, info[UMFPACK_INFO];
    int oB=0, n, ldB=0, nrhs=-1, k;
    char *kwlist[] = {"A", "F", "B", "trans", "nrhs", "ldB", "offsetB",
        NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ciii", kwlist,
        &A, &F, &B, &trans, &nrhs, &ldB, &oB)) return NULL;

    if (!SpMatrix_Check(A) || SP_NROWS(A) != SP_NCOLS(A))
        PY_ERR_TYPE("A must a square sparse matrix");
    n = SP_NROWS(A);

    if (!PyCObject_Check(F)) err_CO("F");
    if (SP_ID(A) == DOUBLE) {
        TypeCheck_CObject(F, "UMFPACK NUM D FACTOR", "F is not the "
            "UMFPACK numeric factor of a 'd' matrix");
    }
    else {
        TypeCheck_CObject(F, "UMFPACK NUM Z FACTOR", "F is not the "
            "UMFPACK numeric factor of a 'z' matrix");
    }

    if (!Matrix_Check(B) || MAT_ID(B) != SP_ID(A))
        PY_ERR_TYPE("B must a dense matrix of the same numeric type "
            "as A");
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > MAT_LGT(B)) err_buf_len("B");

    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");

    if (!(x = malloc(n*E_SIZE[SP_ID(A)]))) return PyErr_NoMemory();

    for (k=0; k<nrhs; k++) {
        if (SP_ID(A) == DOUBLE)
            UMFD(solve)(trans == 'N' ? UMFPACK_A : UMFPACK_Aat,
                SP_COL(A), SP_ROW(A), SP_VAL(A), x,
                MAT_BUFD(B) + k*ldB + oB,
                (void *) PyCObject_AsVoidPtr(F), NULL, info);
        else
            UMFZ(solve)(trans == 'N' ? UMFPACK_A : trans == 'C' ?
                UMFPACK_At : UMFPACK_Aat, SP_COL(A), SP_ROW(A),
                SP_VAL(A), NULL, x, NULL,
                (double *)(MAT_BUFZ(B) + k*ldB + oB), NULL,
                (void *) PyCObject_AsVoidPtr(F), NULL, info);
        if (info[UMFPACK_STATUS] == UMFPACK_OK)
            memcpy(B->buffer + (k*ldB + oB)*E_SIZE[SP_ID(A)], x,
                n*E_SIZE[SP_ID(A)]);
        else
	    break;
    }
    free(x);

    if (info[UMFPACK_STATUS] != UMFPACK_OK){
        if (info[UMFPACK_STATUS] == UMFPACK_ERROR_out_of_memory)
            return PyErr_NoMemory();
        else {
            if (info[UMFPACK_STATUS] == UMFPACK_WARNING_singular_matrix)
                PyErr_SetString(PyExc_ArithmeticError,
                    "singular matrix");
            else {
                snprintf(umfpack_error,20,"%s %i","UMFPACK ERROR",
                    (int) info[UMFPACK_STATUS]);
                PyErr_SetString(PyExc_ValueError, umfpack_error);
            }
            return NULL;
        }
    }

    return Py_BuildValue("");
}

static PyMethodDef umfpack_functions[] = {
{"linsolve", (PyCFunction) linsolve, METH_VARARGS|METH_KEYWORDS,
    doc_linsolve},
{"symbolic", (PyCFunction) symbolic, METH_VARARGS, doc_symbolic},
{"numeric", (PyCFunction) numeric, METH_VARARGS, doc_numeric},
{"solve", (PyCFunction) solve, METH_VARARGS|METH_KEYWORDS, doc_solve},
{NULL}  /* Sentinel */
};

PyMODINIT_FUNC initumfpack(void)
{
  PyObject *m;

  m = Py_InitModule3("cvxopt.umfpack", umfpack_functions,
      umfpack__doc__);

  if (import_cvxopt() < 0) return;
}
