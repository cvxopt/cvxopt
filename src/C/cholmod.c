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

#define NO_ANSI99_COMPLEX

#include "cvxopt.h"
#include "misc.h"
#include "cholmod.h"
#include "complex.h"

const int E_SIZE[] = { sizeof(int_t), sizeof(double), sizeof(double complex) };

/* defined in pyconfig.h */
#if (SIZEOF_INT < SIZEOF_SIZE_T)
#define CHOL(name) cholmod_l_ ## name
#else
#define CHOL(name) cholmod_ ## name
#endif

PyDoc_STRVAR(cholmod__doc__, "Interface to the CHOLMOD library.\n\n"
"Routines for sparse Cholesky factorization.\n\n"
"The default values of the control parameters in the CHOLMOD 'Common'\n"
"object are used, except Common->print, which is set to 0, and \n"
"Common->supernodal, which is set to 2.\n"
"The parameters Common->supernodal, Common->print, Common->nmethods, \n"
"Common->postorder and Common->dbound can be modified by making an\n"
"entry in the dictionary cholmod.options, with key value 'supernodal', "
"\n'print', 'nmethods', 'postorder', or 'dbound', respectively.\n \n"
"These parameters have the following meaning.\n\n"
"options['supernodal']: If equal to 0, an LDL^T or LDL^H factorization\n"
"    is computed using a simplicial algorithm.  If equal to 2, an\n"
"    LL^T or LL^H factorization is computed using a supernodal\n"
"    algorithm.  If equal to 1, the most efficient of the two\n"
"    factorizations is selected, based on the sparsity pattern.\n\n"
"options['print']:  A nonnegative integer that controls the amount of\n"
"    output printed to the screen.\n\n"
"options['nmethods']: A nonnegative integer that specifies how many\n"
"    orderings are attempted prior to factorization.  If equal to 2,\n"
"    the user-provided ordering and the AMD ordering are compared, and \n"
"    the best ordering is used. (If the user does not provide an \n"
"    ordering the AMD ordering is used.)  If nmethods is equal to 1, \n"
"    the user-provided ordering is used. (In this case, the user must \n" 
"    provide an ordering.   If nmethods is equal to 0, the user-provided\n"
"    ordering (if any), the AMD ordering, and (if installed during the \n"
"    the CHOLMOD installation) a number of other orderings are compared,\n"
"    and the best ordering is used.  Default: 0.\n\n"
"noptions['postorder']: True or False.  If True the symbolic\n"
"    analysis is followed by a postordering.  Default: True.\n\n"
"options['dbound']: Smallest absolute value for the diagonal\n"
"    elements of D in an LDL^T factorization, or the diagonal\n"
"    elements of L in a Cholesky factorization.  Default: 0.0.\n\n"
"CHOLMOD is available from www.suitesparse.com.");

static PyObject *cholmod_module;
static cholmod_common Common;

static int set_options(void)
{
    int_t pos=0;
    PyObject *param, *key, *value;
    char err_str[100];
#if PY_MAJOR_VERSION < 3
    char *keystr; 
#endif

    CHOL(defaults)(&Common);
    Common.print = 0;
    Common.supernodal = 2;

    if (!(param = PyObject_GetAttrString(cholmod_module, "options")) ||
        ! PyDict_Check(param)) {
        PyErr_SetString(PyExc_AttributeError, "missing cholmod.options"
            "dictionary");
        return 0;
    }
    while (PyDict_Next(param, &pos, &key, &value))
#if PY_MAJOR_VERSION >= 3
        if (PyUnicode_Check(key)) {
            const char *keystr = _PyUnicode_AsString(key);
            if (!strcmp("supernodal", keystr) && PyLong_Check(value))
                Common.supernodal = (int) PyLong_AsLong(value);
            else if (!strcmp("print", keystr) && PyLong_Check(value))
                Common.print = (int) PyLong_AsLong(value);
            else if (!strcmp("nmethods", keystr) && PyLong_Check(value))
                Common.nmethods = (int) PyLong_AsLong(value);
            else if (!strcmp("postorder", keystr) &&
                PyBool_Check(value))
                Common.postorder = (int) PyLong_AsLong(value);
            else if (!strcmp("dbound", keystr) && PyFloat_Check(value))
                Common.dbound = (double) PyFloat_AsDouble(value);
            else {
                sprintf(err_str, "invalid value for CHOLMOD parameter:" \
                   " %-.20s", keystr);
                PyErr_SetString(PyExc_ValueError, err_str);
                Py_DECREF(param);
                return 0;
            }
        }
#else
        if ((keystr = PyString_AsString(key))) {
            if (!strcmp("supernodal", keystr) && PyInt_Check(value))
                Common.supernodal = (int) PyInt_AsLong(value);
            else if (!strcmp("print", keystr) && PyInt_Check(value))
                Common.print = (int) PyInt_AsLong(value);
            else if (!strcmp("nmethods", keystr) && PyInt_Check(value))
                Common.nmethods = (int) PyInt_AsLong(value);
            else if (!strcmp("postorder", keystr) &&
                PyBool_Check(value))
                Common.postorder = (int) PyInt_AsLong(value);
            else if (!strcmp("dbound", keystr) && PyFloat_Check(value))
                Common.dbound = (double) PyFloat_AsDouble(value);
            else {
                sprintf(err_str, "invalid value for CHOLMOD parameter:" \
                   " %-.20s", keystr);
                PyErr_SetString(PyExc_ValueError, err_str);
                Py_DECREF(param);
                return 0;
            }
        }
#endif
    Py_DECREF(param);
    return 1;
}


static cholmod_sparse *pack(spmatrix *A, char uplo)
{
    int j, k, n = SP_NROWS(A), nnz = 0, cnt = 0;
    cholmod_sparse *B;

    if (uplo == 'L'){
        for (j=0; j<n; j++){
            for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1] && SP_ROW(A)[k] < j;
                k++);
            nnz += SP_COL(A)[j+1] - k;
        }
        if (!(B = CHOL(allocate_sparse)(n, n, nnz, 1, 1, -1,
            (SP_ID(A) == DOUBLE ? CHOLMOD_REAL : CHOLMOD_COMPLEX),
            &Common))) return 0;
        for (j=0; j<n; j++){
            for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1] && SP_ROW(A)[k] < j;
                k++);
       	    for (; k<SP_COL(A)[j+1]; k++) {
                if (SP_ID(A) == DOUBLE)
                    ((double *)B->x)[cnt] = SP_VALD(A)[k];
                else
                    ((double complex *)B->x)[cnt] = SP_VALZ(A)[k];
                ((int_t *)B->p)[j+1]++;
                ((int_t *)B->i)[cnt++] = SP_ROW(A)[k];
	    }
        }
    }
    else {
        for (j=0; j<n; j++)
            for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1] && SP_ROW(A)[k] <= j;
                k++)
                nnz++;
        if (!(B = CHOL(allocate_sparse)(n, n, nnz, 1, 1, 1,
            (SP_ID(A) == DOUBLE ? CHOLMOD_REAL : CHOLMOD_COMPLEX),
            &Common))) return 0;

        for (j=0; j<n; j++)
            for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1] && SP_ROW(A)[k] <= j;
                k++) {
                if (SP_ID(A) == DOUBLE)
                    ((double *)B->x)[cnt] = SP_VALD(A)[k];
                else
                    ((double complex *)B->x)[cnt] = SP_VALZ(A)[k];
            ((int_t *)B->p)[j+1]++;
            ((int_t *)B->i)[cnt++] = SP_ROW(A)[k];
        }
    }
    for (j=0; j<n; j++) ((int_t *)B->p)[j+1] += ((int_t *)B->p)[j];
    return B;
}


static cholmod_sparse * create_matrix(spmatrix *A)
{
    cholmod_sparse *B;

    if (!(B = CHOL(allocate_sparse)(SP_NROWS(A), SP_NCOLS(A), 0,
        1, 0, 0, (SP_ID(A) == DOUBLE ? CHOLMOD_REAL : CHOLMOD_COMPLEX),
        &Common))) return NULL;

    int i;
    for (i=0; i<SP_NCOLS(A); i++)
        ((int_t *)B->nz)[i] = SP_COL(A)[i+1] - SP_COL(A)[i];
    B->x = SP_VAL(A);
    B->i = SP_ROW(A);
    B->nzmax = SP_NNZ(A);
    memcpy(B->p, SP_COL(A), (SP_NCOLS(A)+1)*sizeof(int_t));
    return B;
}


static void free_matrix(cholmod_sparse *A)
{
    A->x = NULL;
    A->i = NULL;
    CHOL(free_sparse)(&A, &Common);
}

#if PY_MAJOR_VERSION >= 3
static void cvxopt_free_cholmod_factor(void *L)
{
   void *Lptr = PyCapsule_GetPointer(L, PyCapsule_GetName(L));   
   CHOL(free_factor) ((cholmod_factor **) &Lptr, &Common);
}
#else
static void cvxopt_free_cholmod_factor(void *L, void *descr)
{
    CHOL(free_factor) ((cholmod_factor **) &L, &Common) ;
}
#endif


static char doc_symbolic[] =
    "Symbolic Cholesky factorization of a real symmetric or Hermitian\n"
    "sparse matrix.\n\n"
    "F = symbolic(A, p=None, uplo='L')\n\n"
    "PURPOSE\n"
    "If cholmod.options['supernodal'] = 2, factors A as\n"
    "P*A*P^T = L*L^T or P*A*P^T = L*L^H.  This is the default value.\n"
    "If cholmod.options['supernodal'] = 0, factors A as\n"
    "P*A*P^T = L*D*L^T or P*A*P^T = L*D*L^H with D diagonal and \n"
    "possibly nonpositive.\n"
    "If cholmod.options['supernodal'] = 1, the most efficient of the\n"
    "two factorizations is used.\n\n"
    "ARGUMENTS\n"
    "A         square sparse matrix of order n.\n\n"
    "p         None, or an 'i' matrix of length n that contains a \n"
    "          permutation vector.  If p is not provided, CHOLMOD\n"
    "          uses a permutation from the AMD library.\n\n"
    "uplo      'L' or 'U'.  If uplo is 'L', only the lower triangular\n"
    "          part of A is used and the upper triangular part is\n"
    "          ignored.  If uplo is 'U', only the upper triangular\n"
    "          part of A is used and the lower triangular part is\n"
    "          ignored.\n\n"
    "F         the symbolic factorization, including the permutation,\n"
    "          as an opaque C object that can be passed to\n"
    "          cholmod.numeric\n\n";

static PyObject* symbolic(PyObject *self, PyObject *args,
    PyObject *kwrds)
{
    spmatrix *A;
    cholmod_sparse *Ac = NULL;
    cholmod_factor *L;
    matrix *P=NULL;
#if PY_MAJOR_VERSION >= 3
    int uplo_='L';
#endif
    char uplo='L';
    int n;
    char *kwlist[] = {"A", "p", "uplo", NULL};

    if (!set_options()) return NULL;

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|OC", kwlist, &A,
        &P, &uplo_)) return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|Oc", kwlist, &A,
        &P, &uplo)) return NULL;
#endif
    if (!SpMatrix_Check(A) || SP_NROWS(A) != SP_NCOLS(A))
        PY_ERR_TYPE("A is not a square sparse matrix");
    n = SP_NROWS(A);

    if (P) {
        if (!Matrix_Check(P) || MAT_ID(P) != INT) err_int_mtrx("p");
        if (MAT_LGT(P) != n) err_buf_len("p");
        if (!CHOL(check_perm)(P->buffer, n, n, &Common))
            PY_ERR(PyExc_ValueError, "p is not a valid permutation");
    }
    if (uplo != 'U' && uplo != 'L') err_char("uplo", "'L', 'U'");
    if (!(Ac = pack(A, uplo))) return PyErr_NoMemory();
    L = CHOL(analyze_p)(Ac, P ? MAT_BUFI(P): NULL, NULL, 0, &Common);
    CHOL(free_sparse)(&Ac, &Common);

    if (Common.status != CHOLMOD_OK){
        if (Common.status == CHOLMOD_OUT_OF_MEMORY)
            return PyErr_NoMemory();
        else{
            PyErr_SetString(PyExc_ValueError, "symbolic factorization "
                "failed");
            return NULL;
        }
    }
#if PY_MAJOR_VERSION >= 3
    return (PyObject *) PyCapsule_New((void *) L, SP_ID(A)==DOUBLE ?  
        (uplo == 'L' ?  "CHOLMOD FACTOR D L" : "CHOLMOD FACTOR D U") :
        (uplo == 'L' ?  "CHOLMOD FACTOR Z L" : "CHOLMOD FACTOR Z U"),
        (PyCapsule_Destructor) &cvxopt_free_cholmod_factor); 
#else
    return (PyObject *) PyCObject_FromVoidPtrAndDesc((void *) L,
        SP_ID(A)==DOUBLE ?  
        (uplo == 'L' ?  "CHOLMOD FACTOR D L" : "CHOLMOD FACTOR D U") :
        (uplo == 'L' ?  "CHOLMOD FACTOR Z L" : "CHOLMOD FACTOR Z U"),
	cvxopt_free_cholmod_factor);
#endif
}


static char doc_numeric[] =
    "Numeric Cholesky factorization of a real symmetric or Hermitian\n"
    "sparse matrix.\n\n"
    "numeric(A, F)\n\n"
    "PURPOSE\n"
    "If cholmod.options['supernodal'] = 2, factors A as\n"
    "P*A*P^T = L*L^T or P*A*P^T = L*L^H.  This is the default value.\n"
    "If cholmod.options['supernodal'] = 0, factors A as\n"
    "P*A*P^T = L*D*L^T or P*A*P^T = L*D*L^H with D diagonal and \n"
    "possibly nonpositive.\n"
    "If cholmod.options['supernodal'] = 1, the most efficient of the\n"
    "two factorizations is used. \n"
    "On entry, F is the symbolic factorization computed by \n"
    "cholmod.symbolic.  On exit, F contains the numeric\n"
    "factorization.  If the matrix is singular, an ArithmeticError\n"
    "exception is raised, with as its first argument the index of the\n"
    "column at which the factorization failed.\n\n"
    "ARGUMENTS\n"
    "A         square sparse matrix.  If F was created by calling\n"
    "          cholmod.symbolic with uplo='L', then only the lower\n"
    "          triangular part of A is used.  If F was created with\n"
    "          uplo='U', then only the upper triangular part is used.\n"
    "\n"
    "F         symbolic factorization computed by cholmod.symbolic\n"
    "          applied to a matrix with the same sparsity patter and\n"
    "          type as A.  After a successful call, F contains the\n"
    "          numeric factorization.";

static PyObject* numeric(PyObject *self, PyObject *args)
{
    spmatrix *A;
    PyObject *F;
    cholmod_factor *Lc;
    cholmod_sparse *Ac = NULL;
    char uplo;
#if PY_MAJOR_VERSION >= 3
    const char *descr; 
#else
    char *descr; 
#endif

    if (!set_options()) return NULL;

    if (!PyArg_ParseTuple(args, "OO", &A, &F)) return NULL;

    if (!SpMatrix_Check(A) || SP_NROWS(A) != SP_NCOLS(A))
        PY_ERR_TYPE("A is not a sparse matrix");

#if PY_MAJOR_VERSION >= 3
    if (!PyCapsule_CheckExact(F) || !(descr = PyCapsule_GetName(F)))
        err_CO("F");
#else
    if (!PyCObject_Check(F)) err_CO("F");
    descr = PyCObject_GetDesc(F);
    if (!descr) PY_ERR_TYPE("F is not a CHOLMOD factor");
#endif
    if (SP_ID(A) == DOUBLE){
        if (!strcmp(descr, "CHOLMOD FACTOR D L"))
	    uplo = 'L';
	else if (!strcmp(descr, "CHOLMOD FACTOR D U"))
	    uplo = 'U';
        else
	    PY_ERR_TYPE("F is not the CHOLMOD factor of a 'd' matrix");
    } else {
        if (!strcmp(descr, "CHOLMOD FACTOR Z L"))
	    uplo = 'L';
	else if (!strcmp(descr, "CHOLMOD FACTOR Z U"))
	    uplo = 'U';
        else
	    PY_ERR_TYPE("F is not the CHOLMOD factor of a 'z' matrix");
    }
#if PY_MAJOR_VERSION >= 3
    Lc = (cholmod_factor *) PyCapsule_GetPointer(F, descr);
#else
    Lc = (cholmod_factor *) PyCObject_AsVoidPtr(F);
#endif
    if (!(Ac = pack(A, uplo))) return PyErr_NoMemory();
    CHOL(factorize) (Ac, Lc, &Common);
    CHOL(free_sparse)(&Ac, &Common);

    if (Common.status < 0) switch (Common.status) {
        case CHOLMOD_OUT_OF_MEMORY:
            return PyErr_NoMemory();

        default:
            PyErr_SetString(PyExc_ValueError, "factorization failed");
            return NULL;
    }

    if (Common.status > 0) switch (Common.status) {
        case CHOLMOD_NOT_POSDEF:
            PyErr_SetObject(PyExc_ArithmeticError, Py_BuildValue("i",
                Lc->minor));
            return NULL;
            break;

        case CHOLMOD_DSMALL:
            /* This never happens unless we change the default value
             * of Common.dbound (0.0).  */
            if (Lc->is_ll)
                PyErr_Warn(PyExc_RuntimeWarning, "tiny diagonal "\
                    "elements in L");
            else
                PyErr_Warn(PyExc_RuntimeWarning, "tiny diagonal "\
                    "elements in D");
            break;

        default:
            PyErr_Warn(PyExc_UserWarning, "");
    }

    return Py_BuildValue("");
}


static char doc_solve[] =
    "Solves a sparse set of linear equations with a factored\n"
    "coefficient matrix and an dense matrix as right-hand side.\n\n"
    "solve(F, B, sys=0, nrhs=B.size[1], ldB=max(1,B.size[0], "
    "offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves one of the following systems using the factorization\n"
    "computed by cholmod.numeric:\n\n"
    "   sys   System              sys   System\n"
    "   0     A*X = B             5     L'*X = B\n"
    "   1     L*D*L'*X = B        6     D*X = B\n"
    "   2     L*D*X = B           7     P'*X = B\n"
    "   3     D*L'*X = B          8     P*X = B\n"
    "   4     L*X = B\n\n"
    "If A was factored as P*A*P' = L*L', then D = I in this table.\n"
    "B is stored using the LAPACK and BLAS conventions.  On exit it\n"
    "is overwritten with the solution.\n\n"
    "ARGUMENTS\n"
    "F         the factorization object of A computed by\n"
    "          cholmod.numeric\n\n"
    "B         dense 'd' or 'z' matrix.  Must have the same type\n"
    "          as A.\n\n"
    "sys       integer (0 <= sys <= 8)\n\n"
    "nrhs      integer.  If negative, the default value is used.\n\n"
    "ldB       nonnegative integer.  ldB >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetB   nonnegative integer";

static PyObject* solve(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *B;
    PyObject *F;
    int i, n, oB=0, ldB=0, nrhs=-1, sys=0;
#if PY_MAJOR_VERSION >= 3
    const char *descr;
#else
    char *descr;
#endif
    char *kwlist[] = {"F", "B", "sys", "nrhs", "ldB", "offsetB", NULL};
    int sysvalues[] = { CHOLMOD_A, CHOLMOD_LDLt, CHOLMOD_LD,
        CHOLMOD_DLt, CHOLMOD_L, CHOLMOD_Lt, CHOLMOD_D, CHOLMOD_P,
        CHOLMOD_Pt };

    if (!set_options()) return NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|iiii", kwlist,
        &F, &B, &sys, &nrhs, &ldB, &oB)) return NULL;

#if PY_MAJOR_VERSION >= 3
    if (!PyCapsule_CheckExact(F) || !(descr = PyCapsule_GetName(F)))
        err_CO("F");
    if (strncmp(descr, "CHOLMOD FACTOR", 14))
        PY_ERR_TYPE("F is not a CHOLMOD factor");
    cholmod_factor *L = (cholmod_factor *) PyCapsule_GetPointer(F, descr);
#else
    if (!PyCObject_Check(F)) err_CO("F");
    descr = PyCObject_GetDesc(F);
    if (!descr || strncmp(descr, "CHOLMOD FACTOR", 14))
        PY_ERR_TYPE("F is not a CHOLMOD factor");
    cholmod_factor *L = (cholmod_factor *) PyCObject_AsVoidPtr(F);
#endif
    if (L->xtype == CHOLMOD_PATTERN)
        PY_ERR(PyExc_ValueError, "called with symbolic factor");

    n = L->n;
    if (L->minor<n) PY_ERR(PyExc_ArithmeticError, "singular matrix");

    if (sys < 0 || sys > 8)
         PY_ERR(PyExc_ValueError, "invalid value for sys");

    if (!Matrix_Check(B) || MAT_ID(B) == INT ||
        (MAT_ID(B) == DOUBLE && L->xtype == CHOLMOD_COMPLEX) ||
        (MAT_ID(B) == COMPLEX && L->xtype == CHOLMOD_REAL))
            PY_ERR_TYPE("B must a dense matrix of the same numerical "
                "type as F");

    if (nrhs < 0) nrhs = MAT_NCOLS(B);
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldB == 0) ldB = MAX(1,MAT_NROWS(B));
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > MAT_LGT(B)) err_buf_len("B");

    cholmod_dense *x;
    cholmod_dense *b = CHOL(allocate_dense)(n, 1, n,
        (MAT_ID(B) == DOUBLE ? CHOLMOD_REAL : CHOLMOD_COMPLEX),
        &Common);
    if (Common.status == CHOLMOD_OUT_OF_MEMORY) return PyErr_NoMemory();

    void *b_old = b->x;
    for (i=0; i<nrhs; i++){
        b->x = (unsigned char*)MAT_BUF(B) + (i*ldB + oB)*E_SIZE[MAT_ID(B)];
        x = CHOL(solve) (sysvalues[sys], L, b, &Common);
        if (Common.status != CHOLMOD_OK){
            PyErr_SetString(PyExc_ValueError, "solve step failed");
            CHOL(free_dense)(&x, &Common);
            CHOL(free_dense)(&b, &Common);
	    return NULL;
	}
	memcpy(b->x, x->x, n*E_SIZE[MAT_ID(B)]);
        CHOL(free_dense)(&x, &Common);
    }
    b->x = b_old;
    CHOL(free_dense)(&b, &Common);

    return Py_BuildValue("");
}


static char doc_spsolve[] =
    "Solves a sparse set of linear equations with a factored\n"
    "coefficient matrix and sparse righthand side.\n\n"
    "X = spsolve(F, B, sys=0)\n\n"
    "PURPOSE\n"
    "Solves one of the following systems using the factorization F\n"
    "computed by cholmod.numeric:\n\n"
    "   sys   System              sys   System\n"
    "   0     A*X = B             5     L'*X = B\n"
    "   1     L*D*L'*X = B        6     D*X = B\n"
    "   2     L*D*X = B           7     P'*X = B\n"
    "   3     D*L'*X = B          8     P*X = B\n"
    "   4     L*X = B\n\n"
    "If A was factored as P*A*P^T = L*L^T, then D = I in this table.\n"
    "On exit B is overwritten with the solution.\n\n"
    "ARGUMENTS\n"
    "F         the factorization object of A computed by\n"
    "          cholmod.numeric\n\n"
    "B         sparse unsymmetric matrix\n\n"
    "sys       integer (0 <= sys <= 8)\n\n"
    "X         sparse unsymmetric matrix";

static PyObject* spsolve(PyObject *self, PyObject *args,
    PyObject *kwrds)
{
    spmatrix *B, *X=NULL;
    cholmod_sparse *Bc=NULL, *Xc=NULL;
    PyObject *F;
    cholmod_factor *L;
    int n, sys=0;
#if PY_MAJOR_VERSION >= 3
    const char *descr;
#else
    char *descr;
#endif
    char *kwlist[] = {"F", "B", "sys", NULL};
    int sysvalues[] = {CHOLMOD_A, CHOLMOD_LDLt, CHOLMOD_LD,
        CHOLMOD_DLt, CHOLMOD_L, CHOLMOD_Lt, CHOLMOD_D, CHOLMOD_P,
        CHOLMOD_Pt };

    if (!set_options()) return NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|i", kwlist, &F,
        &B, &sys)) return NULL;

#if PY_MAJOR_VERSION >= 3
    if (!PyCapsule_CheckExact(F) || !(descr = PyCapsule_GetName(F)))
        err_CO("F");
    if (strncmp(descr, "CHOLMOD FACTOR", 14))
        PY_ERR_TYPE("F is not a CHOLMOD factor");
    L = (cholmod_factor *) PyCapsule_GetPointer(F, descr);
#else
    if (!PyCObject_Check(F)) err_CO("F");
    descr = PyCObject_GetDesc(F);
    if (!descr || strncmp(descr, "CHOLMOD FACTOR", 14))
        PY_ERR_TYPE("F is not a CHOLMOD factor");
    L = (cholmod_factor *) PyCObject_AsVoidPtr(F);
#endif
    if (L->xtype == CHOLMOD_PATTERN)
        PY_ERR(PyExc_ValueError, "called with symbolic factor");
    n = L->n;
    if (L->minor<n) PY_ERR(PyExc_ArithmeticError, "singular matrix");

    if (sys < 0 || sys > 8)
         PY_ERR(PyExc_ValueError, "invalid value for sys");

    if (!SpMatrix_Check(B) ||
        (SP_ID(B) == DOUBLE  && L->xtype == CHOLMOD_COMPLEX) ||
        (SP_ID(B) == COMPLEX && L->xtype == CHOLMOD_REAL))
            PY_ERR_TYPE("B must a sparse matrix of the same "
                "numerical type as F");
    if (SP_NROWS(B) != n)
        PY_ERR(PyExc_ValueError, "incompatible dimensions for B");

    if (!(Bc = create_matrix(B))) return PyErr_NoMemory();
    Xc = CHOL(spsolve)(sysvalues[sys], L, Bc, &Common);
    free_matrix(Bc);
    if (Common.status == CHOLMOD_OUT_OF_MEMORY) return PyErr_NoMemory();
    if (Common.status != CHOLMOD_OK)
        PY_ERR(PyExc_ValueError, "solve step failed");

    if (!(X = SpMatrix_New(Xc->nrow, Xc->ncol,
        ((int_t*)Xc->p)[Xc->ncol], (L->xtype == CHOLMOD_REAL ? DOUBLE :
        COMPLEX)))) {
        CHOL(free_sparse)(&Xc, &Common);
        return PyErr_NoMemory();
    }
    memcpy(SP_COL(X), Xc->p, (Xc->ncol+1)*sizeof(int_t));
    memcpy(SP_ROW(X), Xc->i, ((int_t *)Xc->p)[Xc->ncol]*sizeof(int_t));
    memcpy(SP_VAL(X), Xc->x,
        ((int_t *) Xc->p)[Xc->ncol]*E_SIZE[SP_ID(X)]);
    CHOL(free_sparse)(&Xc, &Common);
    return (PyObject *) X;
}


static char doc_linsolve[] =
    "Solves a sparse positive definite set of linear equations.\n\n"
    "linsolve(A, B, p=None, uplo='L', nrhs=B.size[1], \n"
    "         ldB=max(1,B.size[0]), offsetB=0)\n\n"
    "PURPOSE\n"
    "Solves A*X = B using a sparse Cholesky factorization.  On exit\n"
    "B is overwritten with the solution.  The argument p specifies\n"
    "an optional permutation matrix.  If it is not provided, CHOLMOD\n"
    "uses a permutation from the AMD library.  An ArithmeticError\n"
    "exception is raised if the matrix is singular, with as its first\n"
    "argument the index of the column at which the factorization\n"
    "failed.\n\n"
    "ARGUMENTS\n"
    "A         square sparse matrix\n\n"
    "B         dense 'd' or 'z' matrix.  Must have the same type\n"
    "          as A.\n\n"
    "p         None, or an 'i' matrix of length n that contains\n"
    "          a permutation vector\n\n"
    "uplo      'L' or 'U'.  If uplo is 'L', only the lower triangular\n"
    "          part of A is used and the upper triangular part is\n"
    "          ignored.  If uplo is 'U', only the upper triangular\n"
    "          part is used, and the lower triangular part is ignored."
    "\n\n"
    "nrhs      integer.  If negative, the default value is used.\n\n"
    "ldB       nonnegative integer.  ldB >= max(1,n).  If zero, the\n"
    "          default value is used.\n\n"
    "offsetB   nonnegative integer";

static PyObject* linsolve(PyObject *self, PyObject *args,
    PyObject *kwrds)
{
    spmatrix *A;
    matrix *B, *P=NULL;
    int i, n, nnz, oB=0, ldB=0, nrhs=-1;
    cholmod_sparse *Ac=NULL;
    cholmod_factor *L=NULL;
    cholmod_dense *x=NULL, *b=NULL;
    void *b_old;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo='L';
    char *kwlist[] = {"A", "B", "p", "uplo", "nrhs", "ldB", "offsetB",
        NULL};

    if (!set_options()) return NULL;
#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|OCiii", kwlist,
        &A,  &B, &P, &uplo_, &nrhs, &ldB, &oB)) return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Ociii", kwlist,
        &A,  &B, &P, &uplo, &nrhs, &ldB, &oB)) return NULL;
#endif

    if (!SpMatrix_Check(A) || SP_NROWS(A) != SP_NCOLS(A))
        PY_ERR_TYPE("A is not a sparse matrix");
    n = SP_NROWS(A);
    nnz = SP_NNZ(A);

    if (!Matrix_Check(B) || MAT_ID(B) != SP_ID(A))
        PY_ERR_TYPE("B must be a dense matrix of the same numerical "
            "type as A");
    if (nrhs < 0) nrhs = MAT_NCOLS(B);
    if (n == 0 || nrhs == 0) return Py_BuildValue("");
    if (ldB == 0) ldB = MAX(1,MAT_NROWS(B));
    if (ldB < MAX(1,n)) err_ld("ldB");
    if (oB < 0) err_nn_int("offsetB");
    if (oB + (nrhs-1)*ldB + n > MAT_LGT(B)) err_buf_len("B");

    if (P) {
        if (!Matrix_Check(P) || MAT_ID(P) != INT) err_int_mtrx("p");
        if (MAT_LGT(P) != n) err_buf_len("p");
        if (!CHOL(check_perm)(P->buffer, n, n, &Common))
            PY_ERR(PyExc_ValueError, "not a valid permutation");
    }
    if (uplo != 'U' && uplo != 'L') err_char("uplo", "'L', 'U'");

    if (!(Ac = pack(A, uplo))) return PyErr_NoMemory();
    L = CHOL(analyze_p)(Ac, P ? MAT_BUFI(P): NULL, NULL, 0, &Common);
    if (Common.status != CHOLMOD_OK){
        free_matrix(Ac);
        CHOL(free_sparse)(&Ac, &Common);
        CHOL(free_factor)(&L, &Common);
        if (Common.status == CHOLMOD_OUT_OF_MEMORY)
            return PyErr_NoMemory();
        else {
            PyErr_SetString(PyExc_ValueError, "symbolic factorization "
                "failed");
            return NULL;
        }
    }

    CHOL(factorize) (Ac, L, &Common);
    CHOL(free_sparse)(&Ac, &Common);
    if (Common.status < 0) {
        CHOL(free_factor)(&L, &Common);
        switch (Common.status) {
            case CHOLMOD_OUT_OF_MEMORY:
                return PyErr_NoMemory();

            default:
                PyErr_SetString(PyExc_ValueError, "factorization "
                    "failed");
                return NULL;
        }
    }
    if (Common.status > 0) switch (Common.status) {
        case CHOLMOD_NOT_POSDEF:
            PyErr_SetObject(PyExc_ArithmeticError,
                Py_BuildValue("i", L->minor));
            CHOL(free_factor)(&L, &Common);
            return NULL;
            break;

        case CHOLMOD_DSMALL:
            /* This never happens unless we change the default value
             * of Common.dbound (0.0).  */
            if (L->is_ll)
                PyErr_Warn(PyExc_RuntimeWarning, "tiny diagonal "
                    "elements in L");
            else
                PyErr_Warn(PyExc_RuntimeWarning, "tiny diagonal "
                    "elements in D");
            break;

        default:
            PyErr_Warn(PyExc_UserWarning, "");
    }

    if (L->minor<n) {
        CHOL(free_factor)(&L, &Common);
        PY_ERR(PyExc_ArithmeticError, "singular matrix");
    }
    b = CHOL(allocate_dense)(n, 1, n, (MAT_ID(B) == DOUBLE ?
        CHOLMOD_REAL : CHOLMOD_COMPLEX) , &Common);
    if (Common.status == CHOLMOD_OUT_OF_MEMORY) {
        CHOL(free_factor)(&L, &Common);
        CHOL(free_dense)(&b, &Common);
        return PyErr_NoMemory();
    }
    b_old = b->x;
    for (i=0; i<nrhs; i++) {
        b->x = (unsigned char*)MAT_BUF(B) + (i*ldB + oB)*E_SIZE[MAT_ID(B)];
        x = CHOL(solve) (CHOLMOD_A, L, b, &Common);
        if (Common.status != CHOLMOD_OK){
            PyErr_SetString(PyExc_ValueError, "solve step failed");
            CHOL(free_factor)(&L, &Common);
            b->x = b_old;
            CHOL(free_dense)(&b, &Common);
            CHOL(free_dense)(&x, &Common);
            return NULL;
        }
        memcpy(b->x, x->x, SP_NROWS(A)*E_SIZE[MAT_ID(B)]);
        CHOL(free_dense)(&x, &Common);
    }
    b->x = b_old;
    CHOL(free_dense)(&b, &Common);
    CHOL(free_factor)(&L, &Common);
    return Py_BuildValue("");
}



static char doc_splinsolve[] =
    "Solves a sparse positive definite set of linear equations with\n"
    "sparse righthand side.\n\n"
    "X = splinsolve(A, B, p=None, uplo='L')\n\n"
    "PURPOSE\n"
    "Solves A*X = B using a sparse Cholesky factorization.\n"
    "The argument p specifies an optional permutation matrix.  If it\n"
    "is not provided, CHOLMOD uses a permutation from the AMD\n"
    "library.  An ArithmeticError exception is raised if the\n"
    "factorization does not exist.\n\n"
    "ARGUMENTS\n"
    "A         square sparse matrix.  Only the lower triangular part\n"
    "          of A is used; the upper triangular part is ignored.\n\n"
    "B         sparse matrix of the same type as A\n\n"
    "p         None, or an 'i' matrix of length n that contains\n"
    "          a permutation vector";

static PyObject* splinsolve(PyObject *self, PyObject *args,
    PyObject *kwrds)
{
    spmatrix *A, *B, *X;
    matrix *P=NULL;
    int n, nnz;
    cholmod_sparse *Ac=NULL, *Bc=NULL, *Xc=NULL;
    cholmod_factor *L=NULL;
#if PY_MAJOR_VERSION >= 3
    int uplo_='L';
#endif
    char uplo='L';
    char *kwlist[] = {"A", "B", "p", "uplo", NULL};

    if (!set_options()) return NULL;
#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|OC", kwlist, &A,
        &B, &P, &uplo_)) return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|Oc", kwlist, &A,
        &B, &P, &uplo)) return NULL;
#endif

    if (!SpMatrix_Check(A) || SP_NROWS(A) != SP_NCOLS(A))
        PY_ERR_TYPE("A is not a square sparse matrix");
    n = SP_NROWS(A);
    nnz = SP_NNZ(A);

    if (!SpMatrix_Check(B) || SP_ID(A) != SP_ID(B))
        PY_ERR_TYPE("B must be a sparse matrix of the same type as A");
    if (SP_NROWS(B) != n)
        PY_ERR(PyExc_ValueError, "incompatible dimensions for B");

    if (P) {
        if (!Matrix_Check(P) || MAT_ID(P) != INT) err_int_mtrx("p");
        if (MAT_LGT(P) != n) err_buf_len("p");
        if (!CHOL(check_perm)(P->buffer, n, n, &Common))
            PY_ERR(PyExc_ValueError, "not a valid permutation");
    }

    if (uplo != 'U' && uplo != 'L') err_char("uplo", "'L', 'U'");
    if (!(Ac = pack(A, uplo))) return PyErr_NoMemory();

    L = CHOL(analyze_p) (Ac, P ? MAT_BUFI(P): NULL, NULL, 0, &Common);
    if (Common.status != CHOLMOD_OK){
        CHOL(free_factor)(&L, &Common);
        CHOL(free_sparse)(&Ac, &Common);
        if (Common.status == CHOLMOD_OUT_OF_MEMORY)
            return PyErr_NoMemory();
        else {
            PyErr_SetString(PyExc_ValueError, "symbolic factorization "
                "failed");
            return NULL;
        }
    }

    CHOL(factorize) (Ac, L, &Common);
    CHOL(free_sparse)(&Ac, &Common);
    if (Common.status > 0) switch (Common.status) {
        case CHOLMOD_NOT_POSDEF:
            PyErr_SetObject(PyExc_ArithmeticError, Py_BuildValue("i",
                L->minor));
            CHOL(free_factor)(&L, &Common);
            return NULL;
            break;

        case CHOLMOD_DSMALL:
            /* This never happens unless we change the default value
             * of Common.dbound (0.0).  */
            if (L->is_ll)
                PyErr_Warn(PyExc_RuntimeWarning, "tiny diagonal "
                    "elements in L");
            else
                PyErr_Warn(PyExc_RuntimeWarning, "tiny diagonal "
                    "elements in D");
            break;

        default:
            PyErr_Warn(PyExc_UserWarning, "");
    }

    if (L->minor<n) {
        CHOL(free_factor)(&L, &Common);
        PY_ERR(PyExc_ArithmeticError, "singular matrix");
    }
    if (!(Bc = create_matrix(B))) {
      CHOL(free_factor)(&L, &Common);
      return PyErr_NoMemory();
    }

    Xc = CHOL(spsolve)(0, L, Bc, &Common);
    free_matrix(Bc);
    CHOL(free_factor)(&L, &Common);
    if (Common.status != CHOLMOD_OK){
        CHOL(free_sparse)(&Xc, &Common);
        if (Common.status == CHOLMOD_OUT_OF_MEMORY)
            return PyErr_NoMemory();
        else
            PY_ERR(PyExc_ValueError, "solve step failed");
    }

    if (!(X = SpMatrix_New(Xc->nrow, Xc->ncol,
        ((int_t*)Xc->p)[Xc->ncol], SP_ID(A)))) {
        CHOL(free_sparse)(&Xc, &Common);
        return PyErr_NoMemory();
    }
    memcpy(SP_COL(X), (int_t *) Xc->p, (Xc->ncol+1)*sizeof(int_t));
    memcpy(SP_ROW(X), (int_t *) Xc->i,
        ((int_t *) Xc->p)[Xc->ncol]*sizeof(int_t));
    memcpy(SP_VAL(X), (double *) Xc->x,
        ((int_t *) Xc->p)[Xc->ncol]*E_SIZE[SP_ID(X)]);
    CHOL(free_sparse)(&Xc, &Common);
    return (PyObject *) X;
}


static char doc_diag[] =
    "Returns the diagonal of a Cholesky factor.\n\n"
    "D = diag(F)\n\n"
    "PURPOSE\n"
    "Returns the diagonal of the Cholesky factor L in a \n"
    "factorization P*A*P^T = L*L^T or P*A*P^T = L*L^H.\n\n"
    "ARGUMENTS\n"
    "D         an nx1 'd' matrix with the diagonal elements of L.\n"
    "          Note that the permutation P is not returned, so the\n"
    "          order of the diagonal elements is unknown.\n\n"
    "F         a numeric Cholesky factor obtained by a call to\n"
    "          cholmod.numeric computed with options['supernodal'] = 2";

extern void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
extern void zcopy_(int *n, double complex *x, int *incx, double complex *y, int *incy);

static PyObject* diag(PyObject *self, PyObject *args)
{
    PyObject *F;
    matrix *d=NULL;
    cholmod_factor *L;
#if PY_MAJOR_VERSION >= 3
    const char *descr;
#else
    char *descr;
#endif
    int k, strt, incx=1, incy, nrows, ncols;

    if (!set_options()) return NULL;
    if (!PyArg_ParseTuple(args, "O", &F)) return NULL;

#if PY_MAJOR_VERSION >= 3
    if (!PyCapsule_CheckExact(F) || !(descr = PyCapsule_GetName(F)))
        err_CO("F");
    if (strncmp(descr, "CHOLMOD FACTOR", 14))
        PY_ERR_TYPE("F is not a CHOLMOD factor");
    L = (cholmod_factor *) PyCapsule_GetPointer(F, descr);
#else
    if (!PyCObject_Check(F)) err_CO("F");
    descr = PyCObject_GetDesc(F);
    if (!descr || strncmp(descr, "CHOLMOD FACTOR", 14))
        PY_ERR_TYPE("F is not a CHOLMOD factor");
    L = (cholmod_factor *) PyCObject_AsVoidPtr(F);
#endif

    /* Check factorization */
    if (L->xtype == CHOLMOD_PATTERN  || L->minor<L->n || !L->is_ll
        || !L->is_super)
        PY_ERR(PyExc_ValueError, "F must be a nonsingular supernodal "
            "Cholesky factor");
    if (!(d = Matrix_New(L->n,1,L->xtype == CHOLMOD_REAL ? DOUBLE :
        COMPLEX))) return PyErr_NoMemory();

    strt = 0;
    for (k=0; k<L->nsuper; k++){
	/* x[L->px[k], .... ,L->px[k+1]-1] is a dense lower-triangular
	 * nrowx times ncols matrix.  We copy its diagonal to
	 * d[strt, ..., strt+ncols-1] */

        ncols = (int)((int_t *) L->super)[k+1] -
            ((int_t *) L->super)[k];
        nrows = (int)((int_t *) L->pi)[k+1] - ((int_t *) L->pi)[k];
        incy = nrows+1;
        if (MAT_ID(d) == DOUBLE)
	    dcopy_(&ncols, ((double *) L->x) + ((int_t *) L->px)[k],
                &incy, MAT_BUFD(d)+strt, &incx);
        else
	    zcopy_(&ncols, ((double complex *) L->x) + ((int_t *) L->px)[k],
                &incy, MAT_BUFZ(d)+strt, &incx);
        strt += ncols;
    }
    return (PyObject *)d;
}


static PyObject* getfactor(PyObject *self, PyObject *args)
{
    PyObject *F;
    cholmod_factor *Lf;
    cholmod_sparse *Ls;
#if PY_MAJOR_VERSION >= 3
    const char *descr;
#else
    char *descr;
#endif

    if (!set_options()) return NULL;
    if (!PyArg_ParseTuple(args, "O", &F)) return NULL;

#if PY_MAJOR_VERSION >= 3
    if (!PyCapsule_CheckExact(F) || !(descr = PyCapsule_GetName(F)))
        err_CO("F");
    if (strncmp(descr, "CHOLMOD FACTOR", 14))
        PY_ERR_TYPE("F is not a CHOLMOD factor");
    Lf = (cholmod_factor *) PyCapsule_GetPointer(F, descr);
#else
    if (!PyCObject_Check(F)) err_CO("F");
    descr = PyCObject_GetDesc(F);
    if (!descr || strncmp(descr, "CHOLMOD FACTOR", 14))
        PY_ERR_TYPE("F is not a CHOLMOD factor");
    Lf = (cholmod_factor *) PyCObject_AsVoidPtr(F);
#endif

    /* Check factorization */
    if (Lf->xtype == CHOLMOD_PATTERN)
        PY_ERR(PyExc_ValueError, "F must be a numeric Cholesky factor");

    if (!(Ls = CHOL(factor_to_sparse)(Lf, &Common)))
        return PyErr_NoMemory();

    spmatrix *ret = SpMatrix_New(Ls->nrow, Ls->ncol, Ls->nzmax,
       (Ls->xtype == CHOLMOD_REAL ? DOUBLE : COMPLEX));
    if (!ret) {
        CHOL(free_sparse)(&Ls, &Common);
        return PyErr_NoMemory();
    }

    memcpy(SP_COL(ret), Ls->p, (Ls->ncol+1)*sizeof(int_t));
    memcpy(SP_ROW(ret), Ls->i, (Ls->nzmax)*sizeof(int_t));
    memcpy(SP_VAL(ret), Ls->x, (Ls->nzmax)*E_SIZE[SP_ID(ret)]);
    CHOL(free_sparse)(&Ls, &Common);

    return (PyObject *)ret;
}


static PyMethodDef cholmod_functions[] = {
  {"symbolic", (PyCFunction) symbolic, METH_VARARGS|METH_KEYWORDS,
   doc_symbolic},
  {"numeric", (PyCFunction) numeric, METH_VARARGS|METH_KEYWORDS,
   doc_numeric},
  {"solve", (PyCFunction) solve, METH_VARARGS|METH_KEYWORDS,
   doc_solve},
  {"spsolve", (PyCFunction) spsolve, METH_VARARGS|METH_KEYWORDS,
   doc_spsolve},
  {"linsolve", (PyCFunction) linsolve, METH_VARARGS|METH_KEYWORDS,
   doc_linsolve},
  {"splinsolve", (PyCFunction) splinsolve, METH_VARARGS|METH_KEYWORDS,
   doc_splinsolve},
  {"diag", (PyCFunction) diag, METH_VARARGS|METH_KEYWORDS, doc_diag},
  {"getfactor", (PyCFunction) getfactor, METH_VARARGS|METH_KEYWORDS,
   ""},
  {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static PyModuleDef cholmod_module_def = {
    PyModuleDef_HEAD_INIT,
    "cholmod",
    cholmod__doc__,
    -1,
    cholmod_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_cholmod(void)
{
    CHOL(start) (&Common);
    if (!(cholmod_module = PyModule_Create(&cholmod_module_def)))
        return NULL;
    PyModule_AddObject(cholmod_module, "options", PyDict_New());
    if (import_cvxopt() < 0) return NULL;
    return cholmod_module;
}

#else 

PyMODINIT_FUNC initcholmod(void)
{
    CHOL(start) (&Common);
    cholmod_module = Py_InitModule3("cvxopt.cholmod", cholmod_functions,
        cholmod__doc__);
    PyModule_AddObject(cholmod_module, "options", PyDict_New());
    if (import_cvxopt() < 0) return;
}
#endif
