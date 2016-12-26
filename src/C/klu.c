/*
* @Author: Uriel Sandoval
* @Date:   2015-04-28 18:56:49
* @Last Modified by:   Uriel Sandoval
* @Last Modified time: 2015-12-24 13:09:48
*/


#include "cvxopt.h"
#include "klu.h"
#include "misc.h"


// KLU functions
#if (SIZEOF_INT < SIZEOF_LONG) || defined(MS_WIN64)
#define KLUD(name) klu_l_ ## name
#define KLUZ(name) klu_zl_ ## name
#else
#define KLUD(name) klu_ ## name
#define KLUZ(name) klu_z_ ## name
#endif


// KLU types/structures
#if (SIZEOF_INT < SIZEOF_LONG) || defined(MS_WIN64)
#define KLUS(name) klu_l_ ## name
#else
#define KLUS(name) klu_ ## name
#endif



const int E_SIZE[] = {sizeof(int_t), sizeof(double), sizeof(double complex)};
const char *descrdFs = "KLU SYM D FACTOR";
const char *descrzFs = "KLU SYM Z FACTOR";
const char *descrdF = "KLU NUM D FACTOR";
const char *descrzF = "KLU NUM Z FACTOR";


static char klu_error[20];

PyDoc_STRVAR(klu__doc__, "Interface to the KLU library.\n\n"
             "Routines for symbolic and numeric LU factorization of sparse\n"
             "matrices and for solving sparse sets of linear equations.\n"
             "This library is well-suited for circuit simulation.\n\n"
             "The default control settings of KLU are used.\n\n"
             "See also http://faculty.cse.tamu.edu/davis/suitesparse.html");


static void free_klu_d_symbolic(PyObject *F)
{
    KLUS(common) Common;
    KLUD(defaults)(&Common);
    KLUS(symbolic) *Fptr = PyCapsule_GetPointer(F, PyCapsule_GetName(F));
    KLUD(free_symbolic)(&Fptr, &Common);
}
static void free_klu_d_numeric(PyObject *F)
{
    KLUS(common) Common;
    KLUD(defaults)(&Common);
    KLUD(numeric) *Fptr = PyCapsule_GetPointer(F, PyCapsule_GetName(F));
    if (Fptr != NULL)
        KLUD(free_numeric)(&Fptr, &Common);
}
static void free_klu_z_numeric(PyObject *F)
{
    KLUS(common) Common;
    KLUD(defaults)(&Common);
    KLUS(numeric) *Fptr = PyCapsule_GetPointer(F, PyCapsule_GetName(F));
    if (Fptr != NULL)
        KLUZ(free_numeric)(&Fptr, &Common);
}





static char doc_det[] = "Determinant of a KLU numeric object\n"
                        "d = det(A, F, N)\n\n"
                        "PURPOSE\n"
                        "A is a real sparse n by n matrix, F and N its symbolic and numeric \n"
                        "factorizations respectively.\n"
                        "On exit a float is returned with the value of the determinant.\n\n"
                        "ARGUMENTS\n"
                        "A         square sparse matrix\n\n"
                        "F         the symbolic factorization as an opaque C object\n\n"
                        "N         the numeric factorization as an opaque C object\n\n"
                        "d         the determinant value of the matrix\n\n";


static PyObject* det(PyObject *self, PyObject *args, PyObject *kwrds) {
    spmatrix *A;

    PyObject *F;
    PyObject *Fs;

    KLUS(common) Common;
    KLUS(symbolic) *Fsptr;
    KLUS(numeric) *Fptr;
    KLUD(defaults)(&Common) ;


    const char *descrdF = "KLU NUM D FACTOR";

    double *Udiag, *Rs;
    SuiteSparse_long *P, *Q, *Wi;
    double det = 1, d_sign;
    int n, npiv, itmp;

    if (!PyArg_ParseTuple(args, "OOO", &A, &Fs, &F))
        return NULL;

    if (!SpMatrix_Check(A)) PY_ERR_TYPE("A must be a sparse matrix");
    if (!PyCapsule_CheckExact(F)) err_CO("F");
    if (!PyCapsule_CheckExact(Fs)) err_CO("Fs");

    if (SP_ID(A) == COMPLEX)
        PY_ERR_TYPE("A must be a real sparse matrix");

    if (!(Fptr =  (KLUS(numeric) *) PyCapsule_GetPointer(F, descrdF)))
        err_CO("F");

    if (!(Fsptr =  (KLUS(symbolic) *) PyCapsule_GetPointer(Fs, descrdFs)))
        err_CO("Fs");


    /* This code is very similar to umfpack_get_determinan.c */

    Udiag = Fptr->Udiag;
    n =  Fptr->n;
    P = Fptr->Pnum;
    Q = Fsptr->Q;
    Rs =  Fptr->Rs;



    int i, k;
    for (k = 0; k < n; k++)
        det *= Udiag[k];


    for (k = 0; k < n; k++)
        det *= Rs[k];



    Wi = malloc(n * sizeof(SuiteSparse_long));

    /* ---------------------------------------------------------------------- */
    /* determine if P and Q are odd or even permutations */
    /* ---------------------------------------------------------------------- */

    npiv = 0 ;

    for (i = 0 ; i < n ; i++)
    {
        Wi [i] = P [i] ;
    }

    for (i = 0 ; i < n ; i++)
    {
        while (Wi [i] != i)
        {
            itmp = Wi [Wi [i]] ;
            Wi [Wi [i]] = Wi [i] ;
            Wi [i] = itmp ;
            npiv++ ;
        }
    }


    for (i = 0 ; i < n ; i++)
    {
        Wi [i] = Q [i] ;
    }

    for (i = 0 ; i < n ; i++)
    {
        while (Wi [i] != i)
        {
            itmp = Wi [Wi [i]] ;
            Wi [Wi [i]] = Wi [i] ;
            Wi [i] = itmp ;
            npiv++ ;
        }
    }

    /* if npiv is odd, the sign is -1.  if it is even, the sign is +1 */
    d_sign = (npiv % 2) ? -1. : 1. ;


    KLUD(free)(Wi, n, sizeof (SuiteSparse_long), &Common);

    return Py_BuildValue("d", det * d_sign);


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
#if PY_MAJOR_VERSION >= 3
    int trans_ = 'N';
#endif
    char trans = 'N';
    int oB = 0, n, nrhs = -1, ldB = 0;
    KLUS(common) Common, CommonFree;
    KLUS(symbolic) *Symbolic;
    KLUS(numeric) *Numeric;
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

    if (!Matrix_Check(B) || MAT_ID(B) != SP_ID(A))
        PY_ERR_TYPE("B must a dense matrix of the same numeric type "
                    "as A");



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
        Numeric = KLUZ(factor)(SP_COL(A), SP_ROW(A), SP_VAL(A), Symbolic, &Common);

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


    if (SP_ID(A) == DOUBLE) {
        if (trans == 'N')
            KLUD(solve)(Symbolic, Numeric, n, nrhs, MAT_BUFD(B), &Common);
        else
            KLUD(tsolve)(Symbolic, Numeric, n, nrhs, MAT_BUFD(B), &Common);
    }
    else {
        if (trans == 'N')
            KLUZ(solve)(Symbolic, Numeric, n, nrhs, (double *) MAT_BUFZ(B), &Common);
        else
            KLUZ(tsolve)(Symbolic, Numeric, n, nrhs, (double *) MAT_BUFZ(B),
                         trans == 'C' ? 1 : 0, &Common);
    }


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
    "N = numeric(A, F)\n\n"
    "ARGUMENTS\n"
    "A         sparse matrix; may be rectangular\n\n"
    "F         symbolic factorization of A, or a matrix with the same\n"
    "          sparsity pattern, dimensions, and typecode as A, \n"
    "          created by klu.symbolic,\n"
    "N         the numeric factorization, as an opaque C object";

static PyObject* numeric(PyObject *self, PyObject *args, PyObject *kwrds)
{
    spmatrix *A;
    PyObject *Fs;
    PyObject *F = NULL;
    KLUS(symbolic) *Fsptr;



    if (!PyArg_ParseTuple(args, "OO", &A, &Fs)) return NULL;

    if (!SpMatrix_Check(A)) PY_ERR_TYPE("A must be a sparse matrix");
    if (!PyCapsule_CheckExact(Fs)) err_CO("Fs");



    KLUS(common) Common, CommonFree;
    KLUD(defaults)(&Common);
    KLUD(defaults)(&CommonFree);
    KLUS(numeric) *Numeric;

    switch (SP_ID(A)) {
    case DOUBLE:
        TypeCheck_Capsule(Fs, descrdFs, "Fs is not the KLU symbolic "
                          "factor of a 'd' matrix");
        if (!(Fsptr = (KLUS(symbolic) *) PyCapsule_GetPointer(Fs, descrdFs)))
            err_CO("Fs");

        Numeric = KLUD(factor)(SP_COL(A), SP_ROW(A), SP_VAL(A), Fsptr,
                               &Common);

        if (Common.status == KLU_OK)
            return (PyObject *) PyCapsule_New(
                       (void *)Numeric, "KLU NUM D FACTOR",
                       (PyCapsule_Destructor) &free_klu_d_numeric);

        else
            KLUD(free_numeric)(&Numeric, &CommonFree);
        break;

    case COMPLEX:
        TypeCheck_Capsule(Fs, descrzFs, "Fs is not the KLU symbolic "
                          "factor of a 'z' matrix");

        if (!(Fsptr = (KLUS(symbolic) *) PyCapsule_GetPointer(Fs, descrzFs)))
            err_CO("Fs");


        Numeric = KLUZ(factor)(SP_COL(A), SP_ROW(A), SP_VAL(A), Fsptr,
                               &Common);

        if (Common.status == KLU_OK)
            return (PyObject *) PyCapsule_New(
                       (void *) Numeric, "KLU NUM Z FACTOR",
                       (PyCapsule_Destructor) &free_klu_z_numeric);

        else
            KLUZ(free_numeric)(&Numeric, &CommonFree);
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
    "solve(A, F, N, B, trans='N', nrhs=B.size[1], ldB=max(1,B.size[0]),\n"
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
    "F        symbolic factorization, as returned by klu.symbolic\n"
    "N         numeric factorization, as returned by klu.numeric\n"
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


    char trans = 'N';
    int oB = 0, n, ldB = 0, nrhs = -1;
    char *kwlist[] = {"A", "Fs", "F", "B", "trans", "nrhs", "ldB", "offsetB",
                      NULL
                     };
    KLUS(common) Common, CommonFree;


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

    KLUD(defaults)(&Common);
    KLUD(defaults)(&CommonFree);

    if (SP_ID(A) == DOUBLE) {
        if (trans == 'N')
            KLUD(solve)(PyCapsule_GetPointer(Fs, descrdFs),
                        PyCapsule_GetPointer(F, descrdF)
                        , n, nrhs, MAT_BUFD(B), &Common);
        else
            KLUD(tsolve)(PyCapsule_GetPointer(Fs, descrdFs),
                         PyCapsule_GetPointer(F, descrdF)
                         , n, nrhs, MAT_BUFD(B), &Common);
    }
    else {
        if (trans == 'N')
            KLUZ(solve)(PyCapsule_GetPointer(Fs, descrzFs),
                        PyCapsule_GetPointer(F, descrzF)
                        , n, nrhs, (double *) MAT_BUFZ(B), &Common);
        else
            KLUZ(tsolve)(PyCapsule_GetPointer(Fs, descrzFs),
                         PyCapsule_GetPointer(F, descrzF)
                         , n, nrhs, (double *) MAT_BUFZ(B),  trans == 'C' ? 1 : 0,
                         &Common);
    }


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
    {"det", (PyCFunction) det, METH_VARARGS | METH_KEYWORDS, doc_det},
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
