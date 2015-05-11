/*
* @Author: Uriel Sandoval
* @Date:   2015-04-28 18:56:49
* @Last Modified by:   Uriel Sandoval
* @Last Modified time: 2015-05-10 21:27:12
*/

#include "python.h"
#include "cvxopt.h"
#include "klu.h"
#include "misc.h"

// KLU functions
#if (SIZEOF_INT < SIZEOF_LONG)
#define KLUD(name) klu_l_ ## name
#define KLUZ(name) klu_zl_ ## name
#else
#define KLUD(name) klu_ ## name
#define KLUZ(name) klu_z_ ## name
#endif


// KLU types
#if (SIZEOF_INT < SIZEOF_LONG)
#define KLUS(name) klu_l_ ## name
#else
#define KLUS(name) klu_ ## name
#endif



const int E_SIZE[] = {sizeof(int_t), sizeof(double), sizeof(double complex)};


double klu_l_get_determinant(KLUS(numeric) *Numeric) {
    double det =  1;
    double *Udiag = Numeric->Udiag;
    int i, n =  Numeric->n;



    for (i = 0; i < n; i++)
        det *= Udiag[i];


    return det;
}


double complex klu_zl_get_determinant(KLUS(numeric) *Numeric) {
    double complex det =  1;
    double complex *Udiag = Numeric->Udiag;
    int i, n =  Numeric->n;



    for (i = 0; i < n; i++)
        det *= Udiag[i];


    return det;
}



static char klu_error[20];

PyDoc_STRVAR(klu__doc__, "Interface to the KLU library.\n\n"
             "Routines for symbolic and numeric LU factorization of sparse\n"
             "matrices and for solving sparse sets of linear equations.\n"
             "This library is well-suited for circuit simulation.\n\n"
             "The default control settings of KLU are used.\n\n"
             "See also http://faculty.cse.tamu.edu/davis/suitesparse.html");


#if PY_MAJOR_VERSION >=3
static void free_klu_d_symbolic(PyObject *F)
{
    KLUS(common) Common;
    KLUD(defaults)(&Common);
    KLUS(symbolic) *Fptr = PyCapsule_GetPointer(F, PyCapsule_GetName(F));
    KLUD(free_symbolic)(&Fptr, &Common);
}
static void free_klu_d_numeric(PyObject *F)
{
    KLUD(common) Common;
    KLUD(defaults)(&Common);
    KLUS(numeric) *Fptr = PyCapsule_GetPointer(F, PyCapsule_GetName(F));
    KLUD(free_numeric)(&Fptr, &Common);
}
#else
static void free_klu_d_symbolic(KLUS(symbolic) *F)
{
    KLUS(common) Common;
    KLUD(defaults)(&Common);
    KLUD(free_symbolic)(&F, &Common);
}
static void free_klu_d_numeric(KLUS(numeric) *F)
{

    KLUD(common) Common;
    KLUD(defaults)(&Common);
    KLUD(free_numeric)(&F, &Common);
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
    char trans = 'N';
    int oB = 0, n, nrhs = -1, ldB = 0, k;




    void *x;
    char *kwlist[] = {"A", "B", "trans", "nrhs", "ldB", "offsetB",
                      NULL
                     };

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

    if (!SP_ID(A) == DOUBLE)
        PY_ERR_TYPE("A must be a sparse real matrix")
        if (!Matrix_Check(B) || MAT_ID(B) != SP_ID(A))
            PY_ERR_TYPE("B must a dense matrix of the same numeric type "
                        "as A");

    KLUS(common) Common, CommonFree;
    KLUS(symbolic) *Symbolic;
    KLUS(numeric) *Numeric;

    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("i", 0);
    if (ldB == 0) ldB = MAX(1, B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs - 1)*ldB + n > MAT_LGT(B)) err_buf_len("B");


    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");

    KLUD(defaults)(&Common) ;
    KLUD(defaults)(&CommonFree) ;


    // Symbolic factorization

    Symbolic = KLUD(analyze)(n, SP_COL(A), SP_ROW(A), &Common);

    if (Common.status != KLU_OK) {
        KLUD(free_symbolic)(&Symbolic, &CommonFree);

        if (Common.status == KLU_OUT_OF_MEMORY)
            return PyErr_NoMemory();
        else {
            snprintf(klu_error, 20, "%s %i", "KLU ERROR",
                     (int) Common.status);
            PyErr_SetString(PyExc_ValueError, klu_error);
            return NULL;
        }
    }

    // Numeric factorization

    if (SP_ID(A) == DOUBLE)
        Numeric = KLUD(factor)(SP_COL(A), SP_ROW(A), SP_VAL(A), Symbolic, &Common);
    else
        Numeric = KLUD(factor)(SP_COL(A), SP_ROW(A), SP_VAL(A), Symbolic, &Common);

    if (Common.status != KLU_OK) {
        KLUD(free_numeric)(&Numeric, &CommonFree);
        KLUD(free_symbolic)(&Symbolic, &CommonFree);
        if (Common.status == KLU_OUT_OF_MEMORY)
            return PyErr_NoMemory();
        else {
            if (Common.status == KLU_SINGULAR)
                PyErr_SetString(PyExc_ArithmeticError, "singular "
                                "matrix");
            else {
                snprintf(klu_error, 20, "%s %i", "KLU ERROR",
                         (int) Common.status);
                PyErr_SetString(PyExc_ValueError, klu_error);
            }
            return NULL;
        }
    }


    if (!(x = malloc(n * E_SIZE[SP_ID(A)]))) {
        KLUD(free_symbolic)(&Symbolic, &CommonFree);
        if (SP_ID(A) == DOUBLE)
            KLUD(free_numeric)(&Numeric, &CommonFree);
        else
            KLUZ(free_numeric)(&Numeric, &CommonFree);

        return PyErr_NoMemory();
    }
    for (k = 0; k < nrhs; k++) {
        memcpy(x, B->buffer + (k * ldB + oB)*E_SIZE[SP_ID(A)],
               n * E_SIZE[SP_ID(A)]);
        if (SP_ID(A) == DOUBLE) {
            if (trans == 'N')
                KLUD(solve)(Symbolic, Numeric, n, nrhs, x, &Common);
            else
                KLUD(tsolve)(Symbolic, Numeric, n, nrhs, x, &Common);
        }
        else {
            if (trans == 'N')
                KLUZ(solve)(Symbolic, Numeric, n, nrhs, x, &Common);
            else
                KLUZ(tsolve)(Symbolic, Numeric, n, nrhs, x,
                             trans == 'C' ? 1 : 0, &Common);
        }


        if (Common.status == KLU_OK) {

            memcpy(B->buffer + (k * ldB + oB)*E_SIZE[SP_ID(A)], x,
                   n * E_SIZE[SP_ID(A)]);

        }
        else
            break;
    }
    free(x);


    if (Common.status != KLU_OK) {
        KLUD(free_symbolic)(&Symbolic, &CommonFree);
        if (SP_ID(A) == DOUBLE)
            KLUD(free_numeric)(&Numeric, &CommonFree);
        else
            KLUZ(free_numeric)(&Numeric, &CommonFree);

        if (Common.status == KLU_OUT_OF_MEMORY)
            return PyErr_NoMemory();
        else {
            if (Common.status == KLU_SINGULAR)
                PyErr_SetString(PyExc_ArithmeticError, "singular "
                                "matrix");
            else {
                snprintf(klu_error, 20, "%s %i", "KLU ERROR",
                         (int) Common.status);
                PyErr_SetString(PyExc_ValueError, klu_error);
            }
            return NULL;
        }
    }
    KLUD(free_symbolic)(&Symbolic, &CommonFree);
    if (SP_ID(A) == DOUBLE)
        KLUD(free_numeric)(&Numeric, &CommonFree);
    else
        KLUZ(free_numeric)(&Numeric, &CommonFree);
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
    int n;

    if (!PyArg_ParseTuple(args, "O", &A)) return NULL;

    if (!SpMatrix_Check(A) || SP_NROWS(A) != SP_NCOLS(A))
        PY_ERR_TYPE("A must be a square sparse matrix");
    n = SP_NROWS(A);

    if (!SpMatrix_Check(A)) PY_ERR_TYPE("A must be a sparse matrix");
    if (SP_NCOLS(A) == 0) {
        PyErr_SetString(PyExc_ValueError, "A must have at least one "
                        "row and column");
        return NULL;
    }
    KLUS(common) Common, CommonFree;
    KLUS(symbolic) *Symbolic;
    KLUD(defaults)(&Common);
    KLUD(defaults)(&CommonFree);

    Symbolic = KLUD(analyze)(n, SP_COL(A), SP_ROW(A), &Common);
    if (Common.status == KLU_OK) {
        if (SP_ID(A) == DOUBLE)
            return (PyObject *) PyCapsule_New(
                       (void *) Symbolic, "KLU SYM D FACTOR",
                       (PyCapsule_Destructor) &free_klu_d_symbolic);
        else
            return (PyObject *) PyCapsule_New(
                       (void *) Symbolic, "KLU SYM Z FACTOR",
                       (PyCapsule_Destructor) &free_klu_d_symbolic);
    }
    else
        KLUD(free_symbolic)(&Symbolic, &CommonFree);



    if (Common.status == KLU_OUT_OF_MEMORY)
        return PyErr_NoMemory();
    else {
        snprintf(klu_error, 20, "%s %i", "KLU ERROR",
                 (int) Common.status);
        PyErr_SetString(PyExc_ValueError, klu_error);
        return NULL;
    }
}




static char doc_numeric[] =
    "Numeric LU factorization of a sparse matrix, given a symbolic\n"
    "factorization computed by klu.symbolic. To speed-up de factorization\n"
    "a previous numeric factorization can be provided. In case that this \n"
    "refactorization leads to numerical issues a new (full) numeric factorization"
    "is returned.  Raises an ArithmeticError if A is singular.\n\n"
    "Fn = numeric(A, Fs, F)\n\n"
    "ARGUMENTS\n"
    "A         sparse matrix; may be rectangular\n\n"
    "Fs        symbolic factorization of A, or a matrix with the same\n"
    "          sparsity pattern, dimensions, and typecode as A, \n"
    "          created by klu.symbolic,\n"
    "F         numeric factorization of A, or  a matrix with the same\n"
    "          sparsity patthern, dimensions, and typecode as A.\n\n"
    "F         the numeric factorization, as an opaque C object";

static PyObject* numeric(PyObject *self, PyObject *args, PyObject *kwrds)
{
    spmatrix *A;
    PyObject *Fs;
    PyObject *F = NULL;
    void *Fsptr, *Fptr;
    int factorize = 0;
    const char *descrdFs = "KLU SYM D FACTOR";
    const char *descrzFs = "KLU SYM Z FACTOR";
    const char *descrdF = "KLU NUM D FACTOR";
    const char *descrzF = "KLU NUM Z FACTOR";


    if (!PyArg_ParseTuple(args, "OO|O", &A, &Fs, &F)) return NULL;

    if (!SpMatrix_Check(A)) PY_ERR_TYPE("A must be a sparse matrix");
    if (!PyCapsule_CheckExact(Fs)) err_CO("Fs");
    if (F != NULL)
        if (!PyCapsule_CheckExact(F)) err_CO("F");


    KLUS(common) Common, CommonFree;
    KLUD(defaults)(&Common);
    KLUD(defaults)(&CommonFree);
    KLUS(numeric) *Numeric;

    switch (SP_ID(A)) {
    case DOUBLE:
        TypeCheck_Capsule(Fs, descrdFs, "Fs is not the KLU symbolic "
                          "factor of a 'd' matrix");
        if (!(Fsptr = (void *) PyCapsule_GetPointer(Fs, descrdFs)))
            err_CO("Fs");


        if (F != NULL) {

            TypeCheck_Capsule(F, descrdF, "F is not the KLU numeric "
                              "factor of a 'd' matrix");
            if (!(Fptr = (KLUS(numeric) *) PyCapsule_GetPointer(F, descrdF)))
                err_CO("F");

            // F is a previous numeric factorization, hence, a "fast" computation
            // of the condition number of A is performed.

            KLUD(rcond)(Fsptr, Fptr, &Common);

            if (Common.rcond < 1E-20) {
                factorize =  1;

            }
            else {
                KLUD(refactor)(SP_COL(A), SP_ROW(A), SP_VAL(A), Fsptr,
                               Fptr, &Common);

                if (Common.status == KLU_OK)
                    Py_RETURN_NONE;
                else {
                    // Refactorization failed, tries to perform a full factorization.
                    factorize = 1;
                }

            }


        }

        if (F == NULL || factorize) {

            Numeric = KLUD(factor)(SP_COL(A), SP_ROW(A), SP_VAL(A), Fsptr,
                                   &Common);

            if (Common.status == KLU_OK)
                return (PyObject *) PyCapsule_New(
                           (void *) Numeric, "KLU NUM D FACTOR",
                           (PyCapsule_Destructor) &free_klu_d_numeric);

            else
                KLUD(free_numeric)(&Numeric, &CommonFree);
        }
        break;

    case COMPLEX:
        TypeCheck_Capsule(Fs, descrzFs, "Fs is not the KLU symbolic "
                          "factor of a 'd' matrix");
        if (!(Fsptr = (void *) PyCapsule_GetPointer(Fs, descrzFs)))
            err_CO("Fs");
        if (F != NULL) {
            TypeCheck_Capsule(F, descrzF, "F is not the KLU numeric "
                              "factor of a 'd' matrix");
            if (!(Fptr = (void *) PyCapsule_GetPointer(F, descrzF)))
                err_CO("F");

            KLUZ(rcond)(Fsptr, Fptr, &Common);

            if (Common.rcond < 1E-20) {
                factorize = 1;
            }
            else {
                KLUZ(refactor)(SP_COL(A), SP_ROW(A), SP_VAL(A), Fsptr,
                               Fptr, &Common);

                if (Common.status == KLU_OK)
                    Py_RETURN_NONE;
                else
                    factorize = 1;

            }

        }
        if (F == NULL || factorize) {

            Numeric = KLUZ(factor)(SP_COL(A), SP_ROW(A), SP_VAL(A), Fsptr,
                                   &Common);

            if (Common.status == KLU_OK)
                return (PyObject *) PyCapsule_New(
                           (void *) Numeric, "KLU NUM D FACTOR",
                           (PyCapsule_Destructor) &free_klu_d_numeric);

            else
                KLUZ(free_numeric)(&Numeric, &CommonFree);
        }
        break;
    }

    if (Common.status == KLU_OUT_OF_MEMORY)
        return PyErr_NoMemory();
    else {
        if (Common.status == KLU_SINGULAR)
            PyErr_SetString(PyExc_ArithmeticError, "singular matrix");
        else {
            snprintf(klu_error, 20, "%s %i", "KLU ERROR",
                     (int) Common.status);
            PyErr_SetString(PyExc_ValueError, klu_error);
        }
        return NULL;
    }
}



static char doc_solve[] =
    "Solves a factored set of linear equations.\n\n"
    "solve(A, Fs, F, B, trans='N', nrhs=B.size[1], ldB=max(1,B.size[0]),\n"
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
    "Fs        symbolic factorization, as returned by klu.symbolic\n"
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
    PyObject *F, *Fs;
    matrix *B;
#if PY_MAJOR_VERSION >= 3
    int trans_ = 'N';
#endif
    const char *descrdFs = "KLU SYM D FACTOR";
    const char *descrzFs = "KLU SYM Z FACTOR";
    const char *descrdF = "KLU NUM D FACTOR";
    const char *descrzF = "KLU NUM Z FACTOR";

    char trans = 'N';
    double *x;
    int oB = 0, n, ldB = 0, nrhs = -1, k;
    char *kwlist[] = {"A", "Fs", "F", "B", "trans", "nrhs", "ldB", "offsetB",
                      NULL
                     };

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOOO|Ciii", kwlist,
                                     &A, &Fs, &F, &B, &trans_, &nrhs, &ldB, &oB)) return NULL;
    trans = (char) trans_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOOO|ciii", kwlist,
                                     &A, &Fs, &F, &B, &trans, &nrhs, &ldB, &oB)) return NULL;
#endif

    if (!SpMatrix_Check(A) || SP_NROWS(A) != SP_NCOLS(A))
        PY_ERR_TYPE("A must a square sparse matrix");
    n = SP_NROWS(A);

    if (!PyCapsule_CheckExact(F)) err_CO("F");
    if (!PyCapsule_CheckExact(Fs)) err_CO("Fs");

    if (SP_ID(A) == DOUBLE) {
        TypeCheck_Capsule(Fs, descrdFs, "F is not the KLU symbolic factor "
                          "of a 'd' matrix");
        TypeCheck_Capsule(F, descrdF, "F is not the KLU numeric factor "
                          "of a 'd' matrix");
    }
    else  {
        TypeCheck_Capsule(Fs, descrzFs, "F is not the KLU symbolic factor "
                          "of a 'z' matrix");
        TypeCheck_Capsule(F, descrzF, "F is not the KLU numeric factor "
                          "of a 'z' matrix");
    }


    if (!Matrix_Check(B) || MAT_ID(B) != SP_ID(A))
        PY_ERR_TYPE("B must a dense matrix of the same numeric type "
                    "as A");
    if (nrhs < 0) nrhs = B->ncols;
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldB == 0) ldB = MAX(1, B->nrows);
    if (ldB < MAX(1, n)) err_ld("ldB");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs - 1)*ldB + n > MAT_LGT(B)) err_buf_len("B");

    if (trans != 'N' && trans != 'T' && trans != 'C')
        err_char("trans", "'N', 'T', 'C'");

    if (!(x = malloc(n * E_SIZE[SP_ID(A)]))) return PyErr_NoMemory();
    KLUS(common) Common, CommonFree;
    KLUD(defaults)(&Common);
    KLUD(defaults)(&CommonFree);
    for (k = 0; k < nrhs; k++) {

        memcpy(x, B->buffer + (k * ldB + oB)*E_SIZE[SP_ID(A)],
               n * E_SIZE[SP_ID(A)]);
        if (SP_ID(A) == DOUBLE)
            if (trans == 'N')
                KLUD(solve)(PyCapsule_GetPointer(Fs, descrdFs),
                            PyCapsule_GetPointer(F, descrdF)
                            , n, nrhs, x, &Common);
            else
                KLUD(tsolve)(PyCapsule_GetPointer(Fs, descrdFs),
                             PyCapsule_GetPointer(F, descrdF)
                             , n, nrhs, x, &Common);

        else if (trans == 'N')
            KLUZ(solve)(PyCapsule_GetPointer(Fs, descrdFs),
                        PyCapsule_GetPointer(F, descrdF)
                        , n, nrhs, x, &Common);
        else
            KLUZ(tsolve)(PyCapsule_GetPointer(Fs, descrdFs),
                         PyCapsule_GetPointer(F, descrdF)
                         , n, nrhs, x,  trans == 'C' ? 1 : 0,
                         &Common);


        if (Common.status == KLU_OK)
            memcpy(B->buffer + (k * ldB + oB)*E_SIZE[SP_ID(A)], x,
                   n * E_SIZE[SP_ID(A)]);
        else
            break;
    }
    free(x);

    if (Common.status != KLU_OK) {
        if (Common.status == KLU_OUT_OF_MEMORY)
            return PyErr_NoMemory();
        else {
            if (Common.status == KLU_SINGULAR)
                PyErr_SetString(PyExc_ArithmeticError,
                                "singular matrix");
            else {
                snprintf(klu_error, 20, "%s %i", "KLU ERROR",
                         (int) Common.status);
                PyErr_SetString(PyExc_ValueError, klu_error);
            }
            return NULL;
        }
    }

    return Py_BuildValue("");

}






static PyMethodDef klu_functions[] = {
    {   "linsolve", (PyCFunction) linsolve, METH_VARARGS | METH_KEYWORDS,
        doc_linsolve
    },
    {"symbolic", (PyCFunction) symbolic, METH_VARARGS, doc_symbolic},
    {"numeric", (PyCFunction) numeric, METH_VARARGS, doc_numeric},
    {"solve", (PyCFunction) solve, METH_VARARGS | METH_KEYWORDS, doc_solve},
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
    m = Py_InitModule3("klu", klu_functions, klu__doc__);
    if (import_cvxopt() < 0) return;
}
#endif
