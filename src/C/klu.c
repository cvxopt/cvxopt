/*
 * Copyright 2012-2014 M. Andersen and L. Vandenberghe.
 * Copyright 2010-2011 L. Vandenberghe.
 * Copyright 2004-2009 J. Dahl and L. Vandenberghe.
 *
 * This file is part of CVXOPT version 1.1.7.
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
#include "klu.h"
#include "misc.h"

#if (SIZEOF_INT < SIZEOF_LONG)
#define KLUD(name) klu_dl_ ## name
#define KLUZ(name) klu_zl_ ## name
#else
#define KLUD(name) klu_di_ ## name
#define KLUZ(name) klu_zi_ ## name
#endif

const int E_SIZE[] = {sizeof(int_t), sizeof(double), sizeof(double complex)};

static char klu_error[20];

PyDoc_STRVAR(klu__doc__,"Interface to the KLU library.\n\n"
    "Routines for symbolic and numeric LU factorization of sparse\n"
    "matrices and for solving sparse sets of linear equations.\n"
    "The default control settings of UMPFACK are used.\n\n"
    "See also http://www.cise.ufl.edu/research/sparse/klu.");

#if PY_MAJOR_VERSION >= 3
static void free_klu_d_symbolic(void *F)
{
    void *Fptr = PyCapsule_GetPointer(F, PyCapsule_GetName(F));
    KLUD(free_symbolic)(&Fptr);
}
#else
static void free_klu_d_symbolic(void *F, void *descr)
{
    KLUD(free_symbolic)(&F);
}
#endif


#if PY_MAJOR_VERSION >= 3
static void free_klu_d_numeric(void *F)
{
    void *Fptr = PyCapsule_GetPointer(F, PyCapsule_GetName(F));
    KLUD(free_numeric)(&Fptr);
}
#else
static void free_klu_d_numeric(void *F, void *descr)
{
    KLUD(free_numeric)(&F);
}
#endif




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
#if PY_MAJOR_VERSION >= 3
    int trans_ = 'N';
#endif
    char trans='N';
    int oB=0, n, nrhs=-1, ldB=0, k;
    klu_common *symbolic, *numeric, *solved;
    void *x;
    char *kwlist[] = {"A", "B", "trans", "nrhs", "ldB", "offsetB",
        NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Ciii", kwlist,
        &A, &B, &trans_, &nrhs, &ldB, &oB)) return NULL;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|ciii", kwlist,
        &A, &B, &trans, &nrhs, &ldB, &oB)) return NULL;
#endif

    if (!SpMatrix_Check(A) || SP_NROWS(A) != SP_NCOLS(A))
        PY_ERR_TYPE("A must be a square sparse matrix");
    n = SP_NROWS(A);

    if(!SP_ID(A)==DOUBLE)
        PY_ERR_TYPE("A must be a sparse real matrix")
    if (!Matrix_Check(B) || MAT_ID(B) != SP_ID(A))
        PY_ERR_TYPE("B must a dense matrix of the same numeric type "
            "as A");

    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("i", 0);
    if (ldB == 0) ldB = MAX(1,B->nrows);
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > MAT_LGT(B)) err_buf_len("B");


    KLUD(analyze)(n, SP_COL(A), SP_ROW(A), &symbolic);

    if (symbolic->status != KLU_OK){
        KLUD(free_symbolic)(&symbolic);

        if (symbolic->status == KLU_OUT_OF_MEMORY)
            return PyErr_NoMemory();
        else {
            snprintf(klu_error,20,"%s %i","KLU ERROR",
                (int) symbolic->status);
            PyErr_SetString(PyExc_ValueError, klu_error);
            return NULL;
        }
    }

    KLUD(factor)(SP_COL(A), SP_ROW(A), SP_VAL(A), &symbolic, &numeric);
    KLUD(free_symbolic)(&symbolic);

    if (numeric->status != KLU_OK){
        KLUD(free_numeric)(&numeric);
        if (numeric->status == KLU_OUT_OF_MEMORY)
            return PyErr_NoMemory();
        else {
            if (numeric->status == KLU_SINGULAR)
                PyErr_SetString(PyExc_ArithmeticError, "singular "
                    "matrix");
            else {
                snprintf(klu_error,20,"%s %i","KLU ERROR",
                    (int) numeric->status);
                PyErr_SetString(PyExc_ValueError, klu_error);
            }
            return NULL;
        }
    }

    if (!(x = malloc(n*E_SIZE[SP_ID(A)]))) {
        KLUD(free_numeric)(&numeric);
        return PyErr_NoMemory();
    }
    for (k=0; k<nrhs; k++){
        KLUD(solve)(symbolic, numeric, n, nrhs, x, &solved);

        if (solved -> status == KLU_OK)
            memcpy(B->buffer + (k*ldB + oB)*E_SIZE[SP_ID(A)], x,
                n*E_SIZE[SP_ID(A)]);
        else
	       break;
    }
    free(x);
    KLUD(free_numeric)(&numeric);

    if (solved->status != KLU_OK){
        if (solved->status == KLU_OUT_OF_MEMORY)
            return PyErr_NoMemory();
        else {
            if (solved->status == KLU_SINGULAR)
                PyErr_SetString(PyExc_ArithmeticError, "singular "
                    "matrix");
            else {
                snprintf(klu_error,20,"%s %i","KLU ERROR",
                    (int) solved->status);
                PyErr_SetString(PyExc_ValueError, klu_error);
            }
        return NULL;
        }
    }
    return Py_BuildValue("");
}

/*
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
    double info[KLU_INFO];
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
            KLUD(analyze)(SP_NROWS(A), SP_NCOLS(A), SP_COL(A),
                SP_ROW(A), SP_VAL(A), &symbolic, NULL, info);
            if (info[KLU_STATUS] == KLU_OK)
#if PY_MAJOR_VERSION >= 3
                return (PyObject *) PyCapsule_New( (void *) symbolic, 
                    "KLU SYM D FACTOR", 
                    (PyCapsule_Destructor) &free_klu_d_symbolic);  
#else
                return (PyObject *) PyCObject_FromVoidPtrAndDesc(
                    (void *) symbolic, "KLU SYM D FACTOR",
                    free_klu_d_symbolic);
#endif
            else
                KLUD(free_symbolic)(&symbolic);
            break;

        case COMPLEX:
            KLUZ(analyze)(SP_NROWS(A), SP_NCOLS(A), SP_COL(A),
                SP_ROW(A), SP_VAL(A), NULL, &symbolic, NULL, info);
            if (info[KLU_STATUS] == KLU_OK)
#if PY_MAJOR_VERSION >= 3
                return (PyObject *) PyCapsule_New(
                    (void *) symbolic, "KLU SYM Z FACTOR",
                    (PyCapsule_Destructor) &free_klu_z_symbolic);
#else
                return (PyObject *) PyCObject_FromVoidPtrAndDesc(
                    (void *) symbolic, "KLU SYM Z FACTOR",
                    free_klu_z_symbolic);
#endif
            else
                KLUZ(free_symbolic)(&symbolic);
            break;
    }

    if (info[KLU_STATUS] == KLU_ERROR_out_of_memory)
        return PyErr_NoMemory();
    else {
        snprintf(klu_error,20,"%s %i","KLU ERROR",
            (int) info[KLU_STATUS]);
        PyErr_SetString(PyExc_ValueError, klu_error);
        return NULL;
    }
}


static char doc_numeric[] =
    "Numeric LU factorization of a sparse matrix, given a symbolic\n"
    "factorization computed by klu.symbolic.  Raises an\n"
    "ArithmeticError if A is singular.\n\n"
    "Fn = numeric(A, Fs)\n\n"
    "ARGUMENTS\n"
    "A         sparse matrix; may be rectangular\n\n"
    "Fs        symbolic factorization of A, or a matrix with the same\n"
    "          sparsity pattern, dimensions, and typecode as A, \n"
    "          created by klu.symbolic\n\n"
    "Fn        the numeric factorization, as an opaque C object";

static PyObject* numeric(PyObject *self, PyObject *args)
{
    spmatrix *A;
    PyObject *Fs;
    double info[KLU_INFO];
    void *numeric;
#if PY_MAJOR_VERSION >= 3
    void *Fsptr;
    const char *descrd = "KLU SYM D FACTOR";
    const char *descrz = "KLU SYM Z FACTOR";
#endif

    if (!PyArg_ParseTuple(args, "OO", &A, &Fs)) return NULL;

    if (!SpMatrix_Check(A)) PY_ERR_TYPE("A must be a sparse matrix");
#if PY_MAJOR_VERSION >= 3
    if (!PyCapsule_CheckExact(Fs)) err_CO("Fs");
#else
    if (!PyCObject_Check(Fs)) err_CO("Fs");
#endif

    switch (SP_ID(A)) {
	case DOUBLE:
#if PY_MAJOR_VERSION >= 3
            TypeCheck_Capsule(Fs, descrd, "Fs is not the KLU symbolic "
                "factor of a 'd' matrix");
            if (!(Fsptr = (void *) PyCapsule_GetPointer(Fs, descrd)))
                err_CO("Fs");
            KLUD(factor)(SP_COL(A), SP_ROW(A), SP_VAL(A), Fsptr, &numeric,
                NULL, info);
#else
            TypeCheck_CObject(Fs, "KLU SYM D FACTOR", "Fs is not "
                "the KLU symbolic factor of a 'd' matrix");
            KLUD(factor)(SP_COL(A), SP_ROW(A), SP_VAL(A),
	        (void *) PyCObject_AsVoidPtr(Fs), &numeric, NULL, info);
#endif
            if (info[KLU_STATUS] == KLU_OK)
#if PY_MAJOR_VERSION >= 3
                return (PyObject *) PyCapsule_New(
                    (void *) numeric, "KLU NUM D FACTOR",
                    (PyCapsule_Destructor) &free_klu_d_numeric);
#else
                return (PyObject *) PyCObject_FromVoidPtrAndDesc(
                    (void *) numeric, "KLU NUM D FACTOR",
                    free_klu_d_numeric);
#endif
            else
                KLUD(free_numeric)(&numeric);
	    break;

        case COMPLEX:
#if PY_MAJOR_VERSION >= 3
            TypeCheck_Capsule(Fs, descrz, "Fs is not the KLU symbolic "
                "factor of a 'z' matrix");
            if (!(Fsptr = (void *) PyCapsule_GetPointer(Fs, descrz)))
                err_CO("Fs");
            KLUZ(factor)(SP_COL(A), SP_ROW(A), SP_VAL(A), NULL, Fsptr, 
                &numeric, NULL, info);
#else
            TypeCheck_CObject(Fs, "KLU SYM Z FACTOR", "Fs is not "
                "the KLU symbolic factor of a 'z' matrix");
            KLUZ(factor)(SP_COL(A), SP_ROW(A), SP_VAL(A), NULL,
	         (void *) PyCObject_AsVoidPtr(Fs), &numeric, NULL, info);
#endif
            if (info[KLU_STATUS] == KLU_OK)
#if PY_MAJOR_VERSION >= 3
                return (PyObject *) PyCapsule_New(
                    (void *) numeric, "KLU NUM Z FACTOR",
                    (PyCapsule_Destructor) &free_klu_z_numeric);
#else
                return (PyObject *) PyCObject_FromVoidPtrAndDesc(
                    (void *) numeric, "KLU NUM Z FACTOR",
                    free_klu_z_numeric);
#endif
	    else
                 KLUZ(free_numeric)(&numeric);
	    break;
    }

    if (info[KLU_STATUS] == KLU_ERROR_out_of_memory)
        return PyErr_NoMemory();
    else {
        if (info[KLU_STATUS] == KLU_WARNING_singular_matrix)
	    PyErr_SetString(PyExc_ArithmeticError, "singular matrix");
        else {
	    snprintf(klu_error,20,"%s %i","KLU ERROR",
	        (int) info[KLU_STATUS]);
	    PyErr_SetString(PyExc_ValueError, klu_error);
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
    "numeric factorization of A, computed by klu.numeric.\n"
    "On exit B is replaced by the solution.  A is not modified.\n\n"
    "ARGUMENTS\n"
    "A         square sparse matrix\n\n"
    "F         numeric factorization, as returned by klu.numeric\n"
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
#if PY_MAJOR_VERSION >= 3
    int trans_ = 'N';
    const char *descrd = "KLU NUM D FACTOR"; 
    const char *descrz = "KLU NUM Z FACTOR"; 
#endif
    char trans='N';
    double *x, info[KLU_INFO];
    int oB=0, n, ldB=0, nrhs=-1, k;
    char *kwlist[] = {"A", "F", "B", "trans", "nrhs", "ldB", "offsetB",
        NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|Ciii", kwlist,
        &A, &F, &B, &trans_, &nrhs, &ldB, &oB)) return NULL;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ciii", kwlist,
        &A, &F, &B, &trans, &nrhs, &ldB, &oB)) return NULL;
#endif

    if (!SpMatrix_Check(A) || SP_NROWS(A) != SP_NCOLS(A))
        PY_ERR_TYPE("A must a square sparse matrix");
    n = SP_NROWS(A);

#if PY_MAJOR_VERSION >= 3
    if (!PyCapsule_CheckExact(F)) err_CO("F");
    if (SP_ID(A) == DOUBLE) {
        TypeCheck_Capsule(F, descrd, "F is not the KLU numeric factor "
            "of a 'd' matrix");
    }
    else  {
        TypeCheck_Capsule(F, descrz, "F is not the KLU numeric factor "
            "of a 'z' matrix");
    }
#else
    if (!PyCObject_Check(F)) err_CO("F");
    if (SP_ID(A) == DOUBLE) {
        TypeCheck_CObject(F, "KLU NUM D FACTOR", "F is not the "
            "KLU numeric factor of a 'd' matrix");
    }
    else {
        TypeCheck_CObject(F, "KLU NUM Z FACTOR", "F is not the "
            "KLU numeric factor of a 'z' matrix");
    }
#endif

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
#if PY_MAJOR_VERSION >= 3
            KLUD(solve)(trans == 'N' ? KLU_A : KLU_Aat,
                SP_COL(A), SP_ROW(A), SP_VAL(A), x,
                MAT_BUFD(B) + k*ldB + oB,
                (void *) PyCapsule_GetPointer(F, descrd), NULL, info);
#else
            KLUD(solve)(trans == 'N' ? KLU_A : KLU_Aat,
                SP_COL(A), SP_ROW(A), SP_VAL(A), x,
                MAT_BUFD(B) + k*ldB + oB,
                (void *) PyCObject_AsVoidPtr(F), NULL, info);
#endif
        else
#if PY_MAJOR_VERSION >= 3
            KLUZ(solve)(trans == 'N' ? KLU_A : trans == 'C' ?
                KLU_At : KLU_Aat, SP_COL(A), SP_ROW(A),
                SP_VAL(A), NULL, x, NULL,
                (double *)(MAT_BUFZ(B) + k*ldB + oB), NULL,
                (void *) PyCapsule_GetPointer(F, descrz), NULL, info);
#else
            KLUZ(solve)(trans == 'N' ? KLU_A : trans == 'C' ?
                KLU_At : KLU_Aat, SP_COL(A), SP_ROW(A),
                SP_VAL(A), NULL, x, NULL,
                (double *)(MAT_BUFZ(B) + k*ldB + oB), NULL,
                (void *) PyCObject_AsVoidPtr(F), NULL, info);
#endif
        if (info[KLU_STATUS] == KLU_OK)
            memcpy(B->buffer + (k*ldB + oB)*E_SIZE[SP_ID(A)], x,
                n*E_SIZE[SP_ID(A)]);
        else
	    break;
    }
    free(x);

    if (info[KLU_STATUS] != KLU_OK){
        if (info[KLU_STATUS] == KLU_ERROR_out_of_memory)
            return PyErr_NoMemory();
        else {
            if (info[KLU_STATUS] == KLU_WARNING_singular_matrix)
                PyErr_SetString(PyExc_ArithmeticError,
                    "singular matrix");
            else {
                snprintf(klu_error,20,"%s %i","KLU ERROR",
                    (int) info[KLU_STATUS]);
                PyErr_SetString(PyExc_ValueError, klu_error);
            }
            return NULL;
        }
    }

    return Py_BuildValue("");
}
*/

static PyMethodDef klu_functions[] = {
    {"linsolve", (PyCFunction) linsolve, METH_VARARGS|METH_KEYWORDS,
        doc_linsolve},
    //{"symbolic", (PyCFunction) symbolic, METH_VARARGS, doc_symbolic},
    //{"numeric", (PyCFunction) numeric, METH_VARARGS, doc_numeric},
    //{"solve", (PyCFunction) solve, METH_VARARGS|METH_KEYWORDS, doc_solve},
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static PyModuleDef klu_module = {
    PyModuleDef_HEAD_INIT,
    "klu",
    klu__doc__,
    -1,
    klu_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_klu(void)
{
  PyObject *m;
  if (!(m = PyModule_Create(&klu_module))) return NULL;
  if (import_cvxopt() < 0) return NULL;
  return m;
}

#else

PyMODINIT_FUNC initklu(void)
{
  PyObject *m;
  m = Py_InitModule3("cvxopt.klu", klu_functions, klu__doc__);
  if (import_cvxopt() < 0) return;
}
#endif
