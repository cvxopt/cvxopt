/*
 * Copyright 2012-2020 M. Andersen and L. Vandenberghe.
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
#include "amd.h"
#include "misc.h"

/* defined in pyconfig.h */
#if (SIZEOF_INT < SIZEOF_SIZE_T)
#define amd_order amd_l_order
#define amd_defaults amd_l_defaults
#endif


PyDoc_STRVAR(amd__doc__,"Interface to the AMD library.\n\n"
"Approximate minimum degree ordering of sparse matrices.\n\n"
"The default values of the control parameterse in the AMD 'Control' "
"array\n"
"described in the AMD User Guide are used.  The values can be modified "
"by\n"
"making an entry in the dictionary amd.options, with possible key "
"values\n"
"'AMD_DENSE' and 'AMD_AGGRESSIVE'.\n\n"
"AMD is available from www.suitesparse.com.");


static  PyObject *amd_module;

typedef struct {
    char  name[20];
    int   idx;
}   param_tuple;

static const param_tuple AMD_PARAM_LIST[] = {
    {"AMD_DENSE", AMD_DENSE},
    {"AMD_AGGRESSIVE", AMD_AGGRESSIVE}
}; /* 2 parameters */

#if PY_MAJOR_VERSION >= 3
static int get_param_idx(const char *str, int *idx)
#else
static int get_param_idx(char *str, int *idx)
#endif
{
    int i;

    for (i=0; i<2; i++) if (!strcmp(AMD_PARAM_LIST[i].name, str)) {
        *idx =  AMD_PARAM_LIST[i].idx;
        return 1;
    }
    return 0;
}

static int set_defaults(double *control)
{
    int_t pos=0;
    int param_id;
    PyObject *param, *key, *value;
#if PY_MAJOR_VERSION < 3
    char *keystr; 
#endif
    char err_str[100];

    amd_defaults(control);

    if (!(param = PyObject_GetAttrString(amd_module, "options")) ||
        !PyDict_Check(param)){
        PyErr_SetString(PyExc_AttributeError, "missing amd.options"
            "dictionary");
        return 0;
    }
    while (PyDict_Next(param, &pos, &key, &value))
#if PY_MAJOR_VERSION >= 3
        if ((PyUnicode_Check(key)) && 
            get_param_idx(_PyUnicode_AsString(key),&param_id)) {
            if (!PyLong_Check(value) && !PyFloat_Check(value)){
                sprintf(err_str, "invalid value for AMD parameter: %-.20s",
                    _PyUnicode_AsString(key));
#else
        if ((keystr = PyString_AsString(key)) && get_param_idx(keystr,
            &param_id)) {
            if (!PyInt_Check(value) && !PyFloat_Check(value)){
                sprintf(err_str, "invalid value for AMD parameter: "
                    "%-.20s", keystr);
#endif
                PyErr_SetString(PyExc_ValueError, err_str);
                Py_DECREF(param);
                return 0;
            }
            control[param_id] = PyFloat_AsDouble(value);
        }
    Py_DECREF(param);
    return 1;
}


static char doc_order[] =
    "Computes the approximate minimum degree ordering of a square "
    "matrix.\n\n"
    "p = order(A, uplo='L')\n\n"
    "PURPOSE\n"
    "Computes a permutation p that reduces fill-in in the Cholesky\n"
    "factorization of A[p,p].\n\n"
    "ARGUMENTS\n"
    "A         square sparse matrix\n\n"
    "uplo      'L' or 'U'.  If uplo is 'L', the lower triangular part\n"
    "          of A is used and the upper triangular is ignored.  If\n"
    "          uplo is 'U', the upper triangular part is used and the\n"
    "          lower triangular part is ignored.\n\n"
    "p         'i' matrix of length equal to the order of A";


static PyObject* order_c(PyObject *self, PyObject *args, PyObject *kwrds)
{
    spmatrix *A;
    matrix *perm;
#if PY_MAJOR_VERSION >= 3
    int uplo_ = 'L';
#endif
    char uplo = 'L';
    int j, k, n, nnz, alloc=0, info;
    int_t *rowind=NULL, *colptr=NULL;
    double control[AMD_CONTROL];
    char *kwlist[] = {"A", "uplo", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|C", kwlist, &A,
        &uplo_)) return NULL;
    uplo = (char) uplo_;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|c", kwlist, &A,
        &uplo)) return NULL;
#endif
    if (!set_defaults(control)) return NULL;
    if (!SpMatrix_Check(A) || SP_NROWS(A) != SP_NCOLS(A)){
        PyErr_SetString(PyExc_TypeError, "A must be a square sparse "
            "matrix");
        return NULL;
    }
    if (uplo != 'U' && uplo != 'L') err_char("uplo", "'L', 'U'");
    if (!(perm = (matrix *) Matrix_New((int)SP_NROWS(A),1,INT)))
      return NULL;
    n = SP_NROWS(A);
    for (nnz=0, j=0; j<n; j++) {
        if (uplo == 'L'){
            for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1] && SP_ROW(A)[k]<j; k++);
            nnz += SP_COL(A)[j+1] - k;
        }
        else {
            for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1] && SP_ROW(A)[k] <= j;
                k++);
            nnz += k - SP_COL(A)[j];
        }
    }
    if (nnz == SP_NNZ(A)){
        colptr = (int_t *) SP_COL(A);
        rowind = (int_t *) SP_ROW(A);
    }
    else {
        alloc = 1;
        colptr = (int_t *) calloc(n+1, sizeof(int_t));
        rowind = (int_t *) calloc(nnz, sizeof(int_t));
        if (!colptr || !rowind) {
            Py_XDECREF(perm);  free(colptr);  free(rowind);
            return PyErr_NoMemory();
        }
        colptr[0] = 0;
        for (j=0; j<n; j++) {
            if (uplo == 'L'){
                for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1] && SP_ROW(A)[k] < j; 
                    k++);
                nnz = SP_COL(A)[j+1] - k;
                colptr[j+1] = colptr[j] + nnz;
                memcpy(rowind + colptr[j], (int_t *) SP_ROW(A) + k,
                    nnz*sizeof(int_t));
            }
            else {
                for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1] && SP_ROW(A)[k] <= j;
                    k++);
                nnz = k - SP_COL(A)[j];
                colptr[j+1] = colptr[j] + nnz;
                memcpy(rowind + colptr[j], (int_t *) (SP_ROW(A) +
                    SP_COL(A)[j]), nnz*sizeof(int_t));
            }
        }
    }
    info = amd_order(n, colptr, rowind, MAT_BUFI(perm), control, NULL);
    if (alloc){
        free(colptr);
        free(rowind);
    }
    switch (info) {
        case AMD_OUT_OF_MEMORY:
            Py_XDECREF(perm);
            return PyErr_NoMemory();

        case AMD_INVALID:
            Py_XDECREF(perm);
            return Py_BuildValue("");

        case AMD_OK:
            return (PyObject *) perm;
    }
    return Py_BuildValue("");
}

static PyMethodDef amd_functions[] = {
    {"order", (PyCFunction) order_c, METH_VARARGS|METH_KEYWORDS, doc_order},
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static PyModuleDef amd_module_def = {
    PyModuleDef_HEAD_INIT,
    "amd",
    amd__doc__,
    -1,
    amd_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_amd(void)
{
    if (!(amd_module = PyModule_Create(&amd_module_def))) return NULL;
    PyModule_AddObject(amd_module, "options", PyDict_New());
    if (import_cvxopt() < 0) return NULL;
    return amd_module;
}

#else
PyMODINIT_FUNC initamd(void)
{
    amd_module = Py_InitModule3("cvxopt.amd", amd_functions, amd__doc__);
    PyModule_AddObject(amd_module, "options", PyDict_New());
    if (import_cvxopt() < 0) return;
}
#endif
