/*
 * Copyright 2010 L. Vandenberghe.
 * Copyright 2004-2009 J. Dahl and L. Vandenberghe.
 *
 * This file is part of CVXOPT version 1.1.3.
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
#include "misc.h"
#include "glpk.h"

PyDoc_STRVAR(glpk__doc__,
    "Interface to the simplex algorithm in GLPK.\n\n"
    "The GLPK control parameters have the default values listed in \n"
    "the GLPK documentation, except for 'LPX_K_PRESOL', which is set\n"
    "to 1 and cannot be modified.  The other parameters can be\n"
    "modified by making an entry in the dictionary glpk.options.\n"
    "For example, the command glpk.options['LPX_K_MSGLEV'] = 0 turns\n"
    "off the printed output during execution of glpk.simplex().\n"
    "See the documentation at www.gnu.org/software/glpk/glpk.html for\n"
    "the list of GLPK control parameters and their default values.");

static PyObject *glpk_module;

typedef struct {
    char  name[20];
    int   idx;
    char  type;
}   param_tuple;

static const param_tuple GLPK_PARAM_LIST[] = {
    {"LPX_K_MSGLEV", 300, 'i'},
    {"LPX_K_SCALE",  301, 'i'},
    {"LPX_K_DUAL",   302, 'i'},
    {"LPX_K_PRICE",  303, 'i'},
    {"LPX_K_RELAX",  304, 'f'},
    {"LPX_K_TOLBND", 305, 'f'},
    {"LPX_K_TOLDJ",  306, 'f'},
    {"LPX_K_TOLPIV", 307, 'f'},
    {"LPX_K_ROUND",  308, 'i'},
    {"LPX_K_OBJLL",  309, 'f'},
    {"LPX_K_OBJUL",  310, 'f'},
    {"LPX_K_ITLIM",  311, 'i'},
    {"LPX_K_ITCNT",  312, 'i'},
    {"LPX_K_TMLIM",  313, 'f'},
    {"LPX_K_OUTFRQ", 314, 'i'},
    {"LPX_K_OUTDLY", 315, 'f'},
    {"LPX_K_BRANCH", 316, 'i'},
    {"LPX_K_BTRACK", 317, 'i'},
    {"LPX_K_TOLINT", 318, 'f'},
    {"LPX_K_TOLOBJ", 319, 'f'},
    {"LPX_K_MPSINFO",320, 'i'},
    {"LPX_K_MPSOBJ", 321, 'i'},
    {"LPX_K_MPSORIG",322, 'i'},
    {"LPX_K_MPSWIDE",323, 'i'},
    {"LPX_K_MPSFREE",324, 'i'},
    {"LPX_K_MPSSKIP",325, 'i'},
    {"LPX_K_LPTORIG",326, 'i'},
    {"LPX_K_PRESOL", 327, 'i'}
}; /* 28 paramaters */


static int get_param_idx(char *str, int *idx, char *type)
{
    int i;

    for (i=0; i<28; i++) {
        if (!strcmp(GLPK_PARAM_LIST[i].name, str)) {
            *idx =  GLPK_PARAM_LIST[i].idx;
            *type = GLPK_PARAM_LIST[i].type;
            return 1;
        }
    }
    return 0;
}


static char doc_simplex[] =
    "Solves a linear program using GLPK.\n\n"
    "(status, x, z, y) = lp(c, G, h, A, b)\n"
    "(status, x, z) = lp(c, G, h)\n\n"
    "PURPOSE\n"
    "(status, x, z, y) = lp(c, G, h, A, b) solves the pair\n"
    "of primal and dual LPs\n\n"
    "    minimize    c'*x            maximize    -h'*z + b'*y\n"
    "    subject to  G*x <= h        subject to  G'*z + A'*y + c = 0\n"
    "                A*x = b                     z >= 0.\n\n"
    "(status, x, z) = lp(c, G, h) solves the pair of primal\n"
    "and dual LPs\n\n"
    "    minimize    c'*x            maximize    -h'*z \n"
    "    subject to  G*x <= h        subject to  G'*z + c = 0\n"
    "                                            z >= 0.\n\n"
    "ARGUMENTS\n"
    "c            nx1 dense 'd' matrix with n>=1\n\n"
    "G            mxn dense or sparse 'd' matrix with m>=1\n\n"
    "h            mx1 dense 'd' matrix\n\n"
    "A            pxn dense or sparse 'd' matrix with p>=0\n\n"
    "b            px1 dnese 'd' matrix\n\n"
    "status       'optimal', 'primal infeasible', 'dual infeasible' \n"
    "             or 'unknown'\n\n"
    "x            if status is 'optimal', a primal optimal solution;\n"
    "             None otherwise\n\n"
    "z,y          if status is 'optimal', the dual optimal solution;\n"
    "             None otherwise";


static PyObject *simplex(PyObject *self, PyObject *args,
    PyObject *kwrds)
{
    matrix *c, *h, *b=NULL, *x=NULL, *z=NULL, *y=NULL;
    PyObject *G, *A=NULL, *t=NULL, *param, *key, *value;
    LPX *lp;
    int m, n, p, i, j, k, nnz, nnzmax, *rn=NULL, *cn=NULL, param_id;
    int_t pos=0;
    double *a=NULL, val;
    char param_type, err_str[100], *keystr;
    char *kwlist[] = {"c", "G", "h", "A", "b", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|OO", kwlist, &c,
        &G, &h, &A, &b)) return NULL;

    if ((Matrix_Check(G) && MAT_ID(G) != DOUBLE) ||
        (SpMatrix_Check(G) && SP_ID(G) != DOUBLE) ||
        (!Matrix_Check(G) && !SpMatrix_Check(G))){
        PyErr_SetString(PyExc_TypeError, "G must be a 'd' matrix");
        return NULL;
    }
    if ((m = Matrix_Check(G) ? MAT_NROWS(G) : SP_NROWS(G)) <= 0)
        err_p_int("m");
    if ((n = Matrix_Check(G) ? MAT_NCOLS(G) : SP_NCOLS(G)) <= 0)
        err_p_int("n");

    if (!Matrix_Check(h) || h->id != DOUBLE) err_dbl_mtrx("h");
    if (h->nrows != m || h->ncols != 1){
        PyErr_SetString(PyExc_ValueError, "incompatible dimensions");
        return NULL;
    }

    if (A){
        if ((Matrix_Check(A) && MAT_ID(A) != DOUBLE) ||
            (SpMatrix_Check(A) && SP_ID(A) != DOUBLE) ||
            (!Matrix_Check(A) && !SpMatrix_Check(A))){
                PyErr_SetString(PyExc_ValueError, "A must be a dense "
                    "'d' matrix or a general sparse matrix");
                return NULL;
	}
        if ((p = Matrix_Check(A) ? MAT_NROWS(A) : SP_NROWS(A)) < 0)
            err_p_int("p");
        if ((Matrix_Check(A) ? MAT_NCOLS(A) : SP_NCOLS(A)) != n){
            PyErr_SetString(PyExc_ValueError, "incompatible "
                "dimensions");
            return NULL;
	}
    }
    else p = 0;

    if (b && (!Matrix_Check(b) || b->id != DOUBLE)) err_dbl_mtrx("b");
    if ((b && (b->nrows != p || b->ncols != 1)) || (!b && p !=0 )){
        PyErr_SetString(PyExc_ValueError, "incompatible dimensions");
        return NULL;
    }

    lp = lpx_create_prob();
    lpx_add_rows(lp, m+p);
    lpx_add_cols(lp, n);

    for (i=0; i<n; i++){
        lpx_set_obj_coef(lp, i+1, MAT_BUFD(c)[i]);
        lpx_set_col_bnds(lp, i+1, LPX_FR, 0.0, 0.0);
    }
    for (i=0; i<m; i++)
        lpx_set_row_bnds(lp, i+1, LPX_UP, 0.0, MAT_BUFD(h)[i]);
    for (i=0; i<p; i++)
        lpx_set_row_bnds(lp, i+m+1, LPX_FX, MAT_BUFD(b)[i],
            MAT_BUFD(b)[i]);

    nnzmax = (SpMatrix_Check(G) ? SP_NNZ(G) : m*n ) +
        ((A && SpMatrix_Check(A)) ? SP_NNZ(A) : p*n);
    a = (double *) calloc(nnzmax+1, sizeof(double));
    rn = (int *) calloc(nnzmax+1, sizeof(int));
    cn = (int *) calloc(nnzmax+1, sizeof(int));
    if (!a || !rn || !cn){
        free(a);  free(rn);  free(cn);  lpx_delete_prob(lp);
        return PyErr_NoMemory();
    }

    nnz = 0;
    if (SpMatrix_Check(G)) {
        for (j=0; j<n; j++) for (k=SP_COL(G)[j]; k<SP_COL(G)[j+1]; k++)
            if ((val = SP_VALD(G)[k]) != 0.0){
                a[1+nnz] = val;
                rn[1+nnz] = SP_ROW(G)[k]+1;
                cn[1+nnz] = j+1;
                nnz++;
            }
    }
    else for (j=0; j<n; j++) for (i=0; i<m; i++)
        if ((val = MAT_BUFD(G)[i+j*m]) != 0.0){
            a[1+nnz] = val;
            rn[1+nnz] = i+1;
            cn[1+nnz] = j+1;
            nnz++;
        }

    if (A && SpMatrix_Check(A)){
        for (j=0; j<n; j++) for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1]; k++)
            if ((val = SP_VALD(A)[k]) != 0.0){
                a[1+nnz] = val;
                rn[1+nnz] = m+SP_ROW(A)[k]+1;
                cn[1+nnz] = j+1;
                nnz++;
            }
    }
    else for (j=0; j<n; j++) for (i=0; i<p; i++)
        if ((val = MAT_BUFD(A)[i+j*p]) != 0.0){
            a[1+nnz] = val;
            rn[1+nnz] = m+i+1;
            cn[1+nnz] = j+1;
            nnz++;
        }

    lpx_load_matrix(lp, nnz, rn, cn, a);
    free(rn);  free(cn);  free(a);

    if (!(t = PyTuple_New(A ? 4 : 3))){
        lpx_delete_prob(lp);
        return PyErr_NoMemory();
    }

    if (!(param = PyObject_GetAttrString(glpk_module, "options"))
        || !PyDict_Check(param)){
            lpx_delete_prob(lp);
            PyErr_SetString(PyExc_AttributeError,
                "missing glpk.options dictionary");
            return NULL;
    }

    while (PyDict_Next(param, &pos, &key, &value))
        if ((keystr = PyString_AsString(key)) && get_param_idx(keystr,
            &param_id, &param_type)){
	    if (param_type == 'i'){
	        if (!PyInt_Check(value)){
                    sprintf(err_str, "invalid value for integer "
                        "GLPK parameter: %-.20s", keystr);
                    PyErr_SetString(PyExc_ValueError, err_str);
	            lpx_delete_prob(lp);
	            Py_DECREF(param);
                    return NULL;
	        }
                if (!strcmp("LPX_K_PRESOL", keystr) &&
                    PyInt_AS_LONG(value) != 1){
                    PyErr_Warn(PyExc_UserWarning, "ignoring value of "
                        "GLPK parameter 'LPX_K_PRESOL'");
                }
                else lpx_set_int_parm(lp, param_id,
                    PyInt_AS_LONG(value));
	    }
	    else {
	        if (!PyInt_Check(value) && !PyFloat_Check(value)){
                    sprintf(err_str, "invalid value for floating point "
                        "GLPK parameter: %-.20s", keystr);
                    PyErr_SetString(PyExc_ValueError, err_str);
	            lpx_delete_prob(lp);
	            Py_DECREF(param);
                    return NULL;
	        }
	        lpx_set_real_parm(lp, param_id,
                    PyFloat_AsDouble(value));
	    }
    }
    lpx_set_int_parm(lp, LPX_K_PRESOL, 1);
    Py_DECREF(param);

    switch (lpx_simplex(lp)){

        case LPX_E_OK:

            x = (matrix *) Matrix_New(n,1,DOUBLE);
            z = (matrix *) Matrix_New(m,1,DOUBLE);
            if (A) y = (matrix *) Matrix_New(p,1,DOUBLE);
            if (!x || !z || (A && !y)){
                Py_XDECREF(x);
                Py_XDECREF(z);
                Py_XDECREF(y);
                Py_XDECREF(t);
                lpx_delete_prob(lp);
                return PyErr_NoMemory();
            }

            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PyString_FromString("optimal"));

            for (i=0; i<n; i++)
                MAT_BUFD(x)[i] = lpx_get_col_prim(lp, i+1);
            PyTuple_SET_ITEM(t, 1, (PyObject *) x);

            for (i=0; i<m; i++)
                MAT_BUFD(z)[i] = -lpx_get_row_dual(lp, i+1);
            PyTuple_SET_ITEM(t, 2, (PyObject *) z);

            if (A){
                for (i=0; i<p; i++)
                    MAT_BUFD(y)[i] = -lpx_get_row_dual(lp, m+i+1);
                PyTuple_SET_ITEM(t, 3, (PyObject *) y);
            }

            lpx_delete_prob(lp);
            return (PyObject *) t;

        case LPX_E_NOPFS:

            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PyString_FromString("primal infeasible"));
            break;

        case LPX_E_NODFS:

            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PyString_FromString("dual infeasible"));
            break;

        default:

            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PyString_FromString("unknown"));
    }

    lpx_delete_prob(lp);

    PyTuple_SET_ITEM(t, 1, Py_BuildValue(""));
    PyTuple_SET_ITEM(t, 2, Py_BuildValue(""));
    if (A) PyTuple_SET_ITEM(t, 3, Py_BuildValue(""));

    return (PyObject *) t;
}



static char doc_integer[] =
    "Solves a mixed integer linear program using GLPK.\n\n"
    "(status, x) = ilp(c, G, h, A, b, I, B)\n\n"
    "PURPOSE\n"
    "Solves the mixed integer linear programming problem\n\n"
    "    minimize    c'*x\n"
    "    subject to  G*x <= h\n"
    "                A*x = b\n"
    "                x[I] are all integer\n"
    "                x[B] are all binary\n\n"
    "ARGUMENTS\n"
    "c            nx1 dense 'd' matrix with n>=1\n\n"
    "G            mxn dense or sparse 'd' matrix with m>=1\n\n"
    "h            mx1 dense 'd' matrix\n\n"
    "A            pxn dense or sparse 'd' matrix with p>=0\n\n"
    "b            px1 dense 'd' matrix\n\n"
    "I            set with indices of integer variables\n\n"
    "B            set with indices of binary variables\n\n"
    "status       'optimal', 'primal infeasible', 'dual infeasible', \n"
    "             'invalid MIP formulation', 'maxiters exceeded', \n"
    "             'time limit exceeded', 'unknown'\n\n"
    "x            an optimal solution if status is 'optimal';\n"
    "             None otherwise";

static PyObject *integer(PyObject *self, PyObject *args,
    PyObject *kwrds)
{
    matrix *c, *h, *b=NULL, *x=NULL;
    PyObject *G, *A=NULL, *IntSet=NULL, *BinSet = NULL;
    PyObject *t=NULL, *param, *key, *value;
    LPX *lp;
    int m, n, p, i, j, k, nnz, nnzmax, *rn=NULL, *cn=NULL, param_id;
    int_t pos=0;
    double *a=NULL, val;
    char param_type, err_str[100], *keystr;
    char *kwlist[] = {"c", "G", "h", "A", "b", "I", "B", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|OOOO", kwlist, &c,
	    &G, &h, &A, &b, &IntSet, &BinSet)) return NULL;

    if ((Matrix_Check(G) && MAT_ID(G) != DOUBLE) ||
        (SpMatrix_Check(G) && SP_ID(G) != DOUBLE) ||
        (!Matrix_Check(G) && !SpMatrix_Check(G))){
        PyErr_SetString(PyExc_TypeError, "G must be a 'd' matrix");
        return NULL;
    }
    if ((m = Matrix_Check(G) ? MAT_NROWS(G) : SP_NROWS(G)) <= 0)
        err_p_int("m");
    if ((n = Matrix_Check(G) ? MAT_NCOLS(G) : SP_NCOLS(G)) <= 0)
        err_p_int("n");

    if (!Matrix_Check(h) || h->id != DOUBLE) err_dbl_mtrx("h");
    if (h->nrows != m || h->ncols != 1){
        PyErr_SetString(PyExc_ValueError, "incompatible dimensions");
        return NULL;
    }

    if (A){
        if ((Matrix_Check(A) && MAT_ID(A) != DOUBLE) ||
            (SpMatrix_Check(A) && SP_ID(A) != DOUBLE) ||
            (!Matrix_Check(A) && !SpMatrix_Check(A))){
                PyErr_SetString(PyExc_ValueError, "A must be a dense "
                    "'d' matrix or a general sparse matrix");
                return NULL;
	}
        if ((p = Matrix_Check(A) ? MAT_NROWS(A) : SP_NROWS(A)) < 0)
            err_p_int("p");
        if ((Matrix_Check(A) ? MAT_NCOLS(A) : SP_NCOLS(A)) != n){
            PyErr_SetString(PyExc_ValueError, "incompatible "
                "dimensions");
            return NULL;
	}
    }
    else p = 0;

    if (b && (!Matrix_Check(b) || b->id != DOUBLE)) err_dbl_mtrx("b");
    if ((b && (b->nrows != p || b->ncols != 1)) || (!b && p !=0 )){
        PyErr_SetString(PyExc_ValueError, "incompatible dimensions");
        return NULL;
    }

    if ((IntSet) && (!PyAnySet_Check(IntSet)))
      PY_ERR_TYPE("invalid integer index set");

    if ((BinSet) && (!PyAnySet_Check(BinSet)))
      PY_ERR_TYPE("invalid binary index set");

    lp = lpx_create_prob();
    lpx_add_rows(lp, m+p);
    lpx_add_cols(lp, n);

    for (i=0; i<n; i++){
        lpx_set_obj_coef(lp, i+1, MAT_BUFD(c)[i]);
        lpx_set_col_bnds(lp, i+1, LPX_FR, 0.0, 0.0);
    }
    for (i=0; i<m; i++)
        lpx_set_row_bnds(lp, i+1, LPX_UP, 0.0, MAT_BUFD(h)[i]);
    for (i=0; i<p; i++)
        lpx_set_row_bnds(lp, i+m+1, LPX_FX, MAT_BUFD(b)[i],
            MAT_BUFD(b)[i]);

    nnzmax = (SpMatrix_Check(G) ? SP_NNZ(G) : m*n ) +
        ((A && SpMatrix_Check(A)) ? SP_NNZ(A) : p*n);
    a = (double *) calloc(nnzmax+1, sizeof(double));
    rn = (int *) calloc(nnzmax+1, sizeof(int));
    cn = (int *) calloc(nnzmax+1, sizeof(int));
    if (!a || !rn || !cn){
        free(a);  free(rn);  free(cn);  lpx_delete_prob(lp);
        return PyErr_NoMemory();
    }

    nnz = 0;
    if (SpMatrix_Check(G)) {
        for (j=0; j<n; j++) for (k=SP_COL(G)[j]; k<SP_COL(G)[j+1]; k++)
            if ((val = SP_VALD(G)[k]) != 0.0){
                a[1+nnz] = val;
                rn[1+nnz] = SP_ROW(G)[k]+1;
                cn[1+nnz] = j+1;
                nnz++;
            }
    }
    else for (j=0; j<n; j++) for (i=0; i<m; i++)
        if ((val = MAT_BUFD(G)[i+j*m]) != 0.0){
            a[1+nnz] = val;
            rn[1+nnz] = i+1;
            cn[1+nnz] = j+1;
            nnz++;
        }

    if (A && SpMatrix_Check(A)){
        for (j=0; j<n; j++) for (k=SP_COL(A)[j]; k<SP_COL(A)[j+1]; k++)
            if ((val = SP_VALD(A)[k]) != 0.0){
                a[1+nnz] = val;
                rn[1+nnz] = m+SP_ROW(A)[k]+1;
                cn[1+nnz] = j+1;
                nnz++;
            }
    }
    else for (j=0; j<n; j++) for (i=0; i<p; i++)
        if ((val = MAT_BUFD(A)[i+j*p]) != 0.0){
            a[1+nnz] = val;
            rn[1+nnz] = m+i+1;
            cn[1+nnz] = j+1;
            nnz++;
        }

    lpx_load_matrix(lp, nnz, rn, cn, a);
    free(rn);  free(cn);  free(a);

    if (!(t = PyTuple_New(2))) {
        lpx_delete_prob(lp);
        return PyErr_NoMemory();
    }

    if (!(param = PyObject_GetAttrString(glpk_module, "options"))
        || !PyDict_Check(param)){
            lpx_delete_prob(lp);
            PyErr_SetString(PyExc_AttributeError,
                "missing glpk.options dictionary");
            return NULL;
    }

    while (PyDict_Next(param, &pos, &key, &value))
        if ((keystr = PyString_AsString(key)) && get_param_idx(keystr,
            &param_id, &param_type)){
	    if (param_type == 'i'){
	        if (!PyInt_Check(value)){
                    sprintf(err_str, "invalid value for integer "
                        "GLPK parameter: %-.20s", keystr);
                    PyErr_SetString(PyExc_ValueError, err_str);
	            lpx_delete_prob(lp);
	            Py_DECREF(param);
                    return NULL;
	        }
                if (!strcmp("LPX_K_PRESOL", keystr) &&
                    PyInt_AS_LONG(value) != 1){
                    PyErr_Warn(PyExc_UserWarning, "ignoring value of "
                        "GLPK parameter 'LPX_K_PRESOL'");
                }
                else lpx_set_int_parm(lp, param_id,
                    PyInt_AS_LONG(value));
	    }
	    else {
	        if (!PyInt_Check(value) && !PyFloat_Check(value)){
                    sprintf(err_str, "invalid value for floating point "
                        "GLPK parameter: %-.20s", keystr);
                    PyErr_SetString(PyExc_ValueError, err_str);
	            lpx_delete_prob(lp);
	            Py_DECREF(param);
                    return NULL;
	        }
	        lpx_set_real_parm(lp, param_id,
                    PyFloat_AsDouble(value));
	    }
    }
    lpx_set_int_parm(lp, LPX_K_PRESOL, 1);
    Py_DECREF(param);

    if (IntSet) {
      PyObject *iter = PySequence_Fast(IntSet, "Critical error: not sequence");

      for (i=0; i<PySet_GET_SIZE(IntSet); i++) {

	PyObject *tmp = PySequence_Fast_GET_ITEM(iter, i);
	if (!PyInt_Check(tmp)) {
	  lpx_delete_prob(lp);
	  Py_DECREF(iter);
	  PY_ERR_TYPE("non-integer element in I");
	}
	int k = PyInt_AS_LONG(tmp);
	if ((k < 0) || (k >= n)) {
	  lpx_delete_prob(lp);
	  Py_DECREF(iter);
	  PY_ERR(PyExc_IndexError, "index element out of range in I");
	}
	glp_set_col_kind(lp, k+1, GLP_IV);
      }

      Py_DECREF(iter);
    }

    if (BinSet) {
      PyObject *iter = PySequence_Fast(BinSet, "Critical error: not sequence");

      for (i=0; i<PySet_GET_SIZE(BinSet); i++) {

	PyObject *tmp = PySequence_Fast_GET_ITEM(iter, i);
	if (!PyInt_Check(tmp)) {
	  lpx_delete_prob(lp);
	  Py_DECREF(iter);
	  PY_ERR_TYPE("non-binary element in I");
	}
	int k = PyInt_AS_LONG(tmp);
	if ((k < 0) || (k >= n)) {
	  lpx_delete_prob(lp);
	  Py_DECREF(iter);
	  PY_ERR(PyExc_IndexError, "index element out of range in B");
	}
	glp_set_col_kind(lp, k+1, GLP_BV);
      }

      Py_DECREF(iter);

    }



    switch (lpx_intopt(lp)){

        case LPX_E_OK:

            x = (matrix *) Matrix_New(n,1,DOUBLE);
            if (!x) {
                Py_XDECREF(t);
                lpx_delete_prob(lp);
                return PyErr_NoMemory();
            }
            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PyString_FromString("optimal"));

            for (i=0; i<n; i++)
                MAT_BUFD(x)[i] = lpx_mip_col_val(lp, i+1);
            PyTuple_SET_ITEM(t, 1, (PyObject *) x);

            lpx_delete_prob(lp);
            return (PyObject *) t;

        case LPX_E_FAULT:
            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PyString_FromString("invalid MIP formulation"));

	case LPX_E_NOPFS:
            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PyString_FromString("primal infeasible"));

	case LPX_E_NODFS:

            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PyString_FromString("dual infeasible"));

        case LPX_E_ITLIM:

            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PyString_FromString("maxiters exceeded"));

        case LPX_E_TMLIM:

            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PyString_FromString("time limit exceeded"));

	case LPX_E_SING:

            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PyString_FromString("singular or ill-conditioned basis"));

        default:

            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PyString_FromString("unknown"));
    }

    lpx_delete_prob(lp);

    PyTuple_SET_ITEM(t, 1, Py_BuildValue(""));
    return (PyObject *) t;
}


static PyMethodDef glpk_functions[] = {
    {"lp", (PyCFunction) simplex, METH_VARARGS|METH_KEYWORDS,
        doc_simplex},
    {"ilp", (PyCFunction) integer, METH_VARARGS|METH_KEYWORDS,
        doc_integer},
    {NULL}  /* Sentinel */
};


PyMODINIT_FUNC initglpk(void)
{
    glpk_module = Py_InitModule3("cvxopt.glpk", glpk_functions,
        glpk__doc__);

    PyModule_AddObject(glpk_module, "options", PyDict_New());

    if (import_cvxopt() < 0) return;
}
