/*
 * Copyright 2020 Uriel Sandoval
 *
 * This file is part of KVXOPT.
 *
 * KVXOPT is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * KVXOPT is distributed in the hope that it will be useful,
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


// KLU functions and  types/structures
#if (SIZEOF_INT < SIZEOF_SIZE_T)
#define KLUD(name) klu_l_ ## name
#define KLUZ(name) klu_zl_ ## name
#define KLU(name) klu_l_ ## name
#else
#define KLUD(name) klu_ ## name
#define KLUZ(name) klu_z_ ## name
#define KLU(name) klu_ ## name
#endif

const char *descrdFs = "KLU SYM D FACTOR";
const char *descrzFs = "KLU SYM Z FACTOR";
const char *descrdFn = "KLU NUM D FACTOR";
const char *descrzFn = "KLU NUM Z FACTOR";

static char klu_error[20];

PyDoc_STRVAR(klu__doc__, "Interface to the KLU library.\n\n"
             "Routines for symbolic and numeric LU factorization of sparse\n"
             "matrices and for solving sparse sets of linear equations.\n"
             "This library is well-suited for circuit simulation.\n\n"
             "The default control settings of KLU are used.\n\n"
             "See also www.suitesparse.com.");


static void free_klu_symbolic(PyObject *F) {
    KLU(common) Common;
    KLUD(defaults)(&Common);
    KLU(symbolic) *Fptr = PyCapsule_GetPointer(F, PyCapsule_GetName(F));
    KLU(free_symbolic)(&Fptr, &Common);
}

static void free_klu_d_numeric(PyObject *F) {
    KLU(common) Common;
    KLUD(defaults)(&Common);
    KLUD(numeric) *Fptr = PyCapsule_GetPointer(F, PyCapsule_GetName(F));
    if (Fptr != NULL)
        KLUD(free_numeric)(&Fptr, &Common);
}

static void free_klu_z_numeric(PyObject *F) {
    KLU(common) Common;
    KLUD(defaults)(&Common);
    KLU(numeric) *Fptr = PyCapsule_GetPointer(F, PyCapsule_GetName(F));
    if (Fptr != NULL)
        KLUZ(free_numeric)(&Fptr, &Common);
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
    int oB = 0, nrhs = -1, ldB = 0;
    int_t n;
    KLU(common) Common, CommonFree;
    KLU(symbolic) *Symbolic;
    KLU(numeric) *Numeric;
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
                PyErr_SetString(PyExc_ArithmeticError, "singular matrix");
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
            KLUZ(solve)(Symbolic, Numeric, n, nrhs, (double *) MAT_BUFZ(B), 
                            &Common);
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
    int_t n;

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
    KLU(common) Common, CommonFree;
    KLU(symbolic) *Symbolic;
    KLUD(defaults)(&Common);
    KLUD(defaults)(&CommonFree);

    Symbolic = KLUD(analyze)(n, SP_COL(A), SP_ROW(A), &Common);
    if (Common.status == KLU_OK) {
        /* Symbolic is the same for DOUBLE and COMPLEX cases. Only make
         * difference in the Capsule descriptor to avoid user errors 
         */
        if (SP_ID(A) == DOUBLE)
            return (PyObject *) PyCapsule_New(
                       (void *) Symbolic, "KLU SYM D FACTOR",
                       (PyCapsule_Destructor) &free_klu_symbolic);
        else
            return (PyObject *) PyCapsule_New(
                       (void *) Symbolic, "KLU SYM Z FACTOR",
                       (PyCapsule_Destructor) &free_klu_symbolic);
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
    KLU(symbolic) *Fsptr;



    if (!PyArg_ParseTuple(args, "OO", &A, &Fs)) return NULL;

    if (!SpMatrix_Check(A)) PY_ERR_TYPE("A must be a sparse matrix");
    if (!PyCapsule_CheckExact(Fs)) err_CO("Fs");



    KLU(common) Common, CommonFree;
    KLUD(defaults)(&Common);
    KLUD(defaults)(&CommonFree);
    KLU(numeric) *Numeric;

    switch (SP_ID(A)) {
    case DOUBLE:
        TypeCheck_Capsule(Fs, descrdFs, "Fs is not the KLU symbolic "
                          "factor of a 'd' matrix");
        if (!(Fsptr = (KLU(symbolic) *) PyCapsule_GetPointer(Fs, descrdFs)))
            err_CO("Fs");

        Numeric = KLUD(factor)(SP_COL(A), SP_ROW(A), SP_VAL(A), Fsptr, &Common);

        if (Common.status == KLU_OK)
            return (PyObject *) PyCapsule_New(
                       (void *)Numeric, descrdFn,
                       (PyCapsule_Destructor) &free_klu_d_numeric);

        else
            KLUD(free_numeric)(&Numeric, &CommonFree);
        break;

    case COMPLEX:
        TypeCheck_Capsule(Fs, descrzFs, "Fs is not the KLU symbolic "
                          "factor of a 'z' matrix");
        if (!(Fsptr = (KLU(symbolic) *) PyCapsule_GetPointer(Fs, descrzFs)))
            err_CO("Fs");


        Numeric = KLUZ(factor)(SP_COL(A), SP_ROW(A), SP_VAL(A), Fsptr, &Common);

        if (Common.status == KLU_OK)
            return (PyObject *) PyCapsule_New(
                       (void *) Numeric, descrzFn,
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


static char doc_get_numeric[] =
    "This routine copies the LU factors and permutation vectors from the \n"
    "Numeric object into user-accessible arrays.  This routine is not \n"
    "needed to solve a linear system.\n\n"
    "L, U, P, Q, R, F, r = get_numeric(A, Fs, Fn)\n\n"
    "ARGUMENTS\n"
    "A         sparse matrix; must be square\n\n"
    "Fs        symbolic factorization, as returned by klu.symbolic\n\n"
    "Fn        numeric factorization, as returned by klu.numeric\n\n";

static PyObject* get_numeric(PyObject *self, PyObject *args, PyObject *kwrds)
{
    spmatrix *A, *L, *U, *P, *Q, *R, *F;
    PyObject *Fn, *Fs, *r;
    KLU(numeric) *numeric;
    KLU(symbolic) *symbolic;
    int_t i, nn, *rt, *Pt, *Qt, lnz, unz, fnz;
    int status;
    KLU(common) Common;
    double *Lx, *Lz, *Ux, *Uz, *Fx, *Fz;

    if (!PyArg_ParseTuple(args, "OOO", &A, &Fs, &Fn)) 
        return NULL;


    if (!SpMatrix_Check(A)) PY_ERR_TYPE("A must be a sparse matrix");
    

    nn = SP_NROWS(A);

    if (!PyCapsule_CheckExact(Fn)) err_CO("Fn");
    if (!PyCapsule_CheckExact(Fs)) err_CO("Fs");

    if (SP_ID(A) == DOUBLE) {
        TypeCheck_Capsule(Fs, descrdFs, "F is not the KLU symbolic factor "
                          "of a 'd' matrix");
        TypeCheck_Capsule(Fn, descrdFn, "F is not the KLU numeric factor "
                          "of a 'd' matrix");
    }
    else  {
        TypeCheck_Capsule(Fs, descrzFs, "F is not the KLU symbolic factor "
                          "of a 'z' matrix");
        TypeCheck_Capsule(Fn, descrzFn, "F is not the KLU numeric factor "
                          "of a 'z' matrix");
    }

    KLUD(defaults)(&Common);


    switch (SP_ID(A)) {
        case DOUBLE:
            symbolic = (KLU(symbolic) *) PyCapsule_GetPointer(Fs, descrdFs);
            numeric = (KLU(numeric) *) PyCapsule_GetPointer(Fn, descrdFn);
            break;

        case COMPLEX:
            symbolic = (KLU(symbolic) *) PyCapsule_GetPointer(Fs, descrzFs);
            numeric = (KLU(numeric) *) PyCapsule_GetPointer(Fn, descrzFn);
            break;

    }
    lnz = numeric->lnz;
    unz = numeric->unz;
    fnz = numeric->nzoff;
    L = SpMatrix_New(nn, nn, lnz, SP_ID(A));
    U = SpMatrix_New(nn, nn, unz, SP_ID(A));
    /* temporary space for the integer permutation vectors */
    Pt = (int_t *) malloc(nn * sizeof(int_t));
    Qt = (int_t *) malloc(nn * sizeof(int_t));
    R = SpMatrix_New(nn, nn, nn, DOUBLE);
    F = SpMatrix_New(nn, nn, fnz, SP_ID(A));
    /* We store r blocks as c array, then we can transform it into Python list */
    rt = malloc((symbolic->nblocks+1) * sizeof(int_t));

    /* KLU_extract does not handle packed complex arrays, thus we create 
     * two auxiliary arrays for L, U and F
     */
    switch (SP_ID(A)) {
        case DOUBLE:
            status = KLUD(extract)(numeric, symbolic, 
                                  SP_COL(L), SP_ROW(L), SP_VALD(L),
                                  SP_COL(U), SP_ROW(U), SP_VALD(U),
                                  SP_COL(F), SP_ROW(F), SP_VALD(F),
                                  Pt, Qt, SP_VALD(R), rt, &Common);
            break;

        case COMPLEX:
            /* KLU_extract does not handle packed complex arrays, thus we create 
             * two auxiliary arrays for L, U and F
             */
            Lx = malloc(lnz * sizeof(double));
            Lz = malloc(lnz * sizeof(double));
            Ux = malloc(unz * sizeof(double));
            Uz = malloc(unz * sizeof(double));
            Fx = malloc(fnz * sizeof(double));
            Fz = malloc(fnz * sizeof(double));

            status = KLUZ(extract)(numeric, symbolic, 
                                  SP_COL(L), SP_ROW(L), Lx, Lz,
                                  SP_COL(U), SP_ROW(U), Ux, Uz,
                                  SP_COL(F), SP_ROW(F), Fx, Fz,
                                  Pt, Qt, SP_VALD(R), rt, &Common);

            for(i = 0; i < lnz; i++){
                SP_VALD(L)[2*i] = Lx[i];
                SP_VALD(L)[2*i+1] = Lz[i];
            }
            for(i = 0; i < unz; i++){
                SP_VALD(U)[2*i] = Ux[i];
                SP_VALD(U)[2*i+1] = Uz[i];
            }
            for(i = 0; i < fnz; i++){
                SP_VALD(F)[2*i] = Fx[i];
                SP_VALD(F)[2*i+1] = Fz[i];
            }
            free(Lx);
            free(Lz);
            free(Ux);
            free(Uz);
            free(Fx);
            free(Fz);

            break;

    }   

    if (status != 1){
        snprintf(klu_error, 20, "%s %i", "KLU ERROR",
                     (int) Common.status);
        PyErr_SetString(PyExc_ValueError, klu_error);
        return NULL;
    }

    /* R is diagonal */
    for (i = 0; i < nn; i++){
        SP_COL(R)[i] = i;
        SP_ROW(R)[i] = i;
        /* Compute reciprocal to get R*P*A*Q instead of R\P*A*Q */
        SP_VALD(R)[i] = 1.0 / SP_VALD(R)[i];
    }
    SP_COL(R)[nn] = nn;


    /* create sparse permutation matrix for P */
    P = SpMatrix_New(nn, nn, nn, DOUBLE);
    for (i = 0; i < nn; i++){
        SP_COL(P)[i] = i;
        SP_ROW(P)[Pt[i]] = i;
        SP_VALD(P)[i] = 1;
    }
    SP_COL(P)[nn] = nn;

    /* create sparse permutation matrix for Q */
    Q = SpMatrix_New(nn, nn, nn, DOUBLE);
    for (i = 0; i < nn; i++){
        SP_COL(Q)[i] = i;
        SP_ROW(Q)[i] = Qt[i];
        SP_VALD(Q)[i] = 1;
    }
    SP_COL(Q)[nn] = nn;

    /* Create block list */
    r = PyList_New(symbolic->nblocks+1);
    for (i = 0; i < symbolic->nblocks+1; i++){
        PyList_SetItem(r, i, PyLong_FromLong((long ) rt[i]));
    }


    /* Free workspace */
    free(rt);
    free(Pt);
    free(Qt);



    return Py_BuildValue("OOOOOOO", L, U, P, Q, R, F, r);

      
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
    KLU(common) Common, CommonFree;


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
        TypeCheck_Capsule(F, descrdFn, "F is not the KLU numeric factor "
                          "of a 'd' matrix");
    }
    else  {
        TypeCheck_Capsule(Fs, descrzFs, "F is not the KLU symbolic factor "
                          "of a 'z' matrix");
        TypeCheck_Capsule(F, descrzFn, "F is not the KLU numeric factor "
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
                        PyCapsule_GetPointer(F, descrdFn)
                        , n, nrhs, MAT_BUFD(B), &Common);
        else
            KLUD(tsolve)(PyCapsule_GetPointer(Fs, descrdFs),
                         PyCapsule_GetPointer(F, descrdFn)
                         , n, nrhs, MAT_BUFD(B), &Common);
    }
    else {
        if (trans == 'N')
            KLUZ(solve)(PyCapsule_GetPointer(Fs, descrzFs),
                        PyCapsule_GetPointer(F, descrzFn)
                        , n, nrhs, (double *) MAT_BUFZ(B), &Common);
        else
            KLUZ(tsolve)(PyCapsule_GetPointer(Fs, descrzFs),
                         PyCapsule_GetPointer(F, descrzFn)
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


static char doc_get_det[] = 
    "Returns determinant of a KLU symbolic/numeric object\n"
    "d = get_det(A, Fs, Fn)\n\n"
    "PURPOSE\n"
    "A is a real sparse n by n matrix, F and N its symbolic and numeric \n"
    "factorizations respectively.\n"
    "On exit a double/complex is returned with the value of the determinant.\n\n"
    "ARGUMENTS\n"
    "A         square sparse matrix\n\n"
    "Fs        the symbolic factorization as an opaque C object\n\n"
    "Fn        the numeric factorization as an opaque C object\n\n"
    "d         the determinant value of the matrix\n\n";


static PyObject* get_det(PyObject *self, PyObject *args, PyObject *kwrds) {
    spmatrix *A;
    PyObject *Fn, *Fs;

    KLU(common) Common;
    KLU(symbolic) *Fsptr;
    KLU(numeric) *Fptr;
    KLUD(defaults)(&Common) ;

    double *Udiag, *Rs;

    double det = 1, d_sign;
#ifndef _MSC_VER
    double complex det_c = 1.0;
    double complex *Uzdiag;
#else
    _Dcomplex det_c = _Cbuild(1.0, 0.0);
    _Dcomplex *Uzdiag;
#endif

    int_t i, k, n, npiv, itmp, *P, *Q, *Wi;

    if (!PyArg_ParseTuple(args, "OOO", &A, &Fs, &Fn))
        return NULL;

    if (!SpMatrix_Check(A)) PY_ERR_TYPE("A must be a sparse matrix");
    if (!PyCapsule_CheckExact(Fn)) err_CO("F");
    if (!PyCapsule_CheckExact(Fs)) err_CO("Fs");


    if (SP_ID(A) == DOUBLE){
        TypeCheck_Capsule(Fs, descrdFs, "F is not the UMFPACK symbolic factor "
                          "of a 'd' matrix");
        TypeCheck_Capsule(Fn, descrdFn, "F is not the UMFPACK numeric factor "
                          "of a 'd' matrix");
        if (!(Fptr =  (KLU(numeric) *) PyCapsule_GetPointer(Fn, descrdFn)))
            err_CO("F");

        if (!(Fsptr =  (KLU(symbolic) *) PyCapsule_GetPointer(Fs, descrdFs)))
            err_CO("Fs");
        Udiag = Fptr->Udiag;
    }
    else{
        TypeCheck_Capsule(Fs, descrzFs, "F is not the UMFPACK symbolic factor "
                          "of a 'z' matrix");
        TypeCheck_Capsule(Fn, descrzFn, "F is not the UMFPACK numeric factor "
                          "of a 'z' matrix");
        if (!(Fptr =  (KLU(numeric) *) PyCapsule_GetPointer(Fn, descrzFn)))
            err_CO("F");

        if (!(Fsptr =  (KLU(symbolic) *) PyCapsule_GetPointer(Fs, descrzFs)))
            err_CO("Fs");
        Uzdiag = Fptr->Udiag;
    }

    /* This code is based on umfpack_get_determinant.c */

    n =  Fptr->n;
    P = Fptr->Pnum;
    Q = Fsptr->Q;
    Rs =  Fptr->Rs;

    if (SP_ID(A) == DOUBLE)    
        for (k = 0; k < n; k++)
            det *= (Udiag[k]*Rs[k]);
    else
        for (k = 0; k < n; k++)
#ifndef _MSC_VER
            det_c *= (Uzdiag[k]*Rs[k]);
#else
            det_c =  _Cmulcc(det_c, _Cmulcr(Uzdiag[k], Rs[k]));
#endif

    Wi = malloc(n * sizeof(int_t));

    /* ---------------------------------------------------------------------- */
    /* determine if P and Q are odd or even permutations */
    /* ---------------------------------------------------------------------- */

    npiv = 0 ;

    for (i = 0 ; i < n ; i++)
    {
        Wi [i] = P [i] ;
    }

    for (i = 0 ; i < n ; i++)
        while (Wi [i] != i){
            itmp = Wi [Wi [i]] ;
            Wi [Wi [i]] = Wi [i] ;
            Wi [i] = itmp ;
            npiv++ ;
        }

    for (i = 0 ; i < n ; i++)
        Wi [i] = Q [i] ;

    for (i = 0 ; i < n ; i++)
        while (Wi [i] != i){
            itmp = Wi [Wi [i]] ;
            Wi [Wi [i]] = Wi [i] ;
            Wi [i] = itmp ;
            npiv++ ;
        }

    /* if npiv is odd, the sign is -1.  if it is even, the sign is +1 */
    d_sign = (npiv % 2) ? -1. : 1. ;


    free(Wi);

    if (SP_ID(A) == DOUBLE)
        return Py_BuildValue("d", det * d_sign);
    else{
#ifndef _MSC_VER
        det_c *= d_sign;
#else
        det_c = _Cmulcr(det_c, d_sign);
#endif
        return PyComplex_FromDoubles(creal(det_c), cimag(det_c));
    }
}

static PyMethodDef klu_functions[] = {
    {"linsolve", (PyCFunction) linsolve, METH_VARARGS | METH_KEYWORDS,
        doc_linsolve},
    {"symbolic", (PyCFunction) symbolic, METH_VARARGS, doc_symbolic},
    {"numeric", (PyCFunction) numeric, METH_VARARGS, doc_numeric},
    {"get_numeric", (PyCFunction) get_numeric, METH_VARARGS | METH_KEYWORDS, doc_get_numeric},
    {"solve", (PyCFunction) solve, METH_VARARGS | METH_KEYWORDS, doc_solve},
    {"get_det", (PyCFunction) get_det, METH_VARARGS, doc_get_det},
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
    if (import_kvxopt() < 0) return NULL;
    return m;
}

#else

PyMODINIT_FUNC initklu(void)
{
    PyObject *m;
    m = Py_InitModule3("kvxopt.klu", klu_functions, klu__doc__);
    if (import_kvxopt() < 0) return;
}
#endif
