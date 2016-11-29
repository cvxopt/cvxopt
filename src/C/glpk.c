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

#include "cvxopt.h"
#include "misc.h"
#include <glpk.h>
#include <float.h>
#include <limits.h>

PyDoc_STRVAR(glpk__doc__,
    "Interface to the simplex and mixed integer LP algorithms in GLPK.\n\n"
    "The GLPK control parameters have the default values listed in \n"
    "the GLPK documentation (section 2.8.1 for the simplex solver and \n"
    "section 2.10.5 for the MILP solver).  The control  parameters can \n"
    "be modified by making an entry in the dictionary glpk.options.\n"
    "For example, glpk.options['msg_lev'] = 'GLP_MSG_OFF' turns off the \n"
    "printed output will be turned off during execution of glpk.lp().\n"  
    "Setting glpk.options['it_lim'] = 10 sets the simplex iteration \n"
    "limit to 10.  Unrecognized entries in glpk.options are ignored.");

static PyObject *glpk_module;

#if PY_MAJOR_VERSION >= 3
#define PYINT_CHECK(value) PyLong_Check(value)
#define PYINT_AS_LONG(value) PyLong_AS_LONG(value)
#define PYSTRING_FROMSTRING(str) PyUnicode_FromString(str)
#define PYSTRING_CHECK(a) PyUnicode_Check(a)
#define PYSTRING_COMPARE(a,b) PyUnicode_CompareWithASCIIString(a, b)
#else
#define PYINT_CHECK(value) PyInt_Check(value)
#define PYINT_AS_LONG(value) PyInt_AS_LONG(value)
#define PYSTRING_FROMSTRING(str) PyString_FromString(str)
#define PYSTRING_CHECK(a) PyString_Check(a)
#define PYSTRING_COMPARE(a,b) strcmp(PyString_AsString(a), b)
#endif


static char doc_simplex[] =
    "Solves a linear program using GLPK.\n\n"
    "(status, x, z, y) = lp(c, G, h, A, b)\n"
    "(status, x, z) = lp(c, G, h)\n\n"
    "PURPOSE\n"
    "(status, x, z, y) = lp(c, G, h, A, b) solves the pair of primal and\n"
    "dual LPs\n\n"
    "    minimize    c'*x            maximize    -h'*z + b'*y\n"
    "    subject to  G*x <= h        subject to  G'*z + A'*y + c = 0\n"
    "                A*x = b                     z >= 0.\n\n"
    "(status, x, z) = lp(c, G, h) solves the pair of primal and dual LPs"
    "\n\n"
    "    minimize    c'*x            maximize    -h'*z \n"
    "    subject to  G*x <= h        subject to  G'*z + c = 0\n"
    "                                            z >= 0.\n\n"
    "ARGUMENTS\n"
    "c            nx1 dense 'd' matrix with n>=1\n\n"
    "G            mxn dense or sparse 'd' matrix with m>=1\n\n"
    "h            mx1 dense 'd' matrix\n\n"
    "A            pxn dense or sparse 'd' matrix with p>=0\n\n"
    "b            px1 dense 'd' matrix\n\n"
    "status       'optimal', 'primal infeasible', 'dual infeasible' \n"
    "             or 'unknown'\n\n"
    "x            if status is 'optimal', a primal optimal solution;\n"
    "             None otherwise\n\n"
    "z,y          if status is 'optimal', the dual optimal solution;\n"
    "             None otherwise";


static PyObject *simplex(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *c, *h, *b=NULL, *x=NULL, *z=NULL, *y=NULL;
    PyObject *G, *A=NULL, *t=NULL, *param, *key, *value;
    glp_prob *lp;
    glp_smcp smcp;
    int m, n, p, i, j, k, nnz, nnzmax, *rn=NULL, *cn=NULL;
    int_t pos=0;
    double *a=NULL, val;
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

    lp = glp_create_prob();
    glp_add_rows(lp, m+p);
    glp_add_cols(lp, n);

    for (i=0; i<n; i++){
        glp_set_obj_coef(lp, i+1, MAT_BUFD(c)[i]);
        glp_set_col_bnds(lp, i+1, GLP_FR, 0.0, 0.0);
    }
    for (i=0; i<m; i++)
        glp_set_row_bnds(lp, i+1, GLP_UP, 0.0, MAT_BUFD(h)[i]);
    for (i=0; i<p; i++)
        glp_set_row_bnds(lp, i+m+1, GLP_FX, MAT_BUFD(b)[i],
            MAT_BUFD(b)[i]);

    nnzmax = (SpMatrix_Check(G) ? SP_NNZ(G) : m*n ) +
        ((A && SpMatrix_Check(A)) ? SP_NNZ(A) : p*n);
    a = (double *) calloc(nnzmax+1, sizeof(double));
    rn = (int *) calloc(nnzmax+1, sizeof(int));
    cn = (int *) calloc(nnzmax+1, sizeof(int));
    if (!a || !rn || !cn){
        free(a);  free(rn);  free(cn);  
        glp_delete_prob(lp);
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

    glp_load_matrix(lp, nnz, rn, cn, a);
    free(rn);  free(cn);  free(a);

    if (!(t = PyTuple_New(A ? 4 : 3))){
        glp_delete_prob(lp);
        return PyErr_NoMemory();
    }

    if (!(param = PyObject_GetAttrString(glpk_module, "options"))
        || !PyDict_Check(param)){
            glp_delete_prob(lp);
            PyErr_SetString(PyExc_AttributeError,
                "missing glpk.options dictionary");
            return NULL;
    }

    glp_init_smcp(&smcp);

    while (PyDict_Next(param, &pos, &key, &value))
        if (PYSTRING_CHECK(key)){
            if (!PYSTRING_COMPARE(key, "msg_lev"))
                if (PYSTRING_CHECK(value)){
                    if (!PYSTRING_COMPARE(value, "GLP_MSG_OFF"))
                        smcp.msg_lev = GLP_MSG_OFF;
                    else if (!PYSTRING_COMPARE(value, "GLP_MSG_ERR")) 
                        smcp.msg_lev = GLP_MSG_ERR;
                    else if (!PYSTRING_COMPARE(value, "GLP_MSG_ON")) 
                        smcp.msg_lev = GLP_MSG_ON;
                    else if (!PYSTRING_COMPARE(value, "GLP_MSG_ALL")) 
                        smcp.msg_lev = GLP_MSG_ALL;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['msg_lev'] with default value", 
                            1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['msg_lev'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "meth")) 
                if (PYSTRING_CHECK(value)){
                    if (!PYSTRING_COMPARE(value, "GLP_PRIMAL"))
                        smcp.meth = GLP_PRIMAL;
                    else if (!PYSTRING_COMPARE(value, "GLP_DUAL")) 
                        smcp.meth = GLP_DUAL;
                    else if (!PYSTRING_COMPARE(value, "GLP_DUALP")) 
                        smcp.meth = GLP_DUALP;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['meth'] with default value", 1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['meth'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "pricing"))
                if (PYSTRING_CHECK(value)){
                    if (!PYSTRING_COMPARE(value, "GLP_PT_STD"))
                        smcp.pricing = GLP_PT_STD;
                    else if (!PYSTRING_COMPARE(value, "GLP_PT_PSE")) 
                        smcp.pricing = GLP_PT_PSE;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['pricing'] with default value",
                            1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['pricing'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "r_test"))
                if (PYSTRING_CHECK(value)){
                    if (!PYSTRING_COMPARE(value, "GLP_RT_STD"))
                        smcp.r_test = GLP_RT_STD;
                    else if (!PYSTRING_COMPARE(value, "GLP_RT_HAR")) 
                        smcp.r_test = GLP_RT_HAR;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['r_test'] with default value",
                            1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['r_test'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "tol_bnd"))
                if (PyFloat_Check(value))
                    smcp.tol_bnd = PyFloat_AsDouble(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['tol_bnd'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "tol_dj"))
                if (PyFloat_Check(value))
                    smcp.tol_dj = PyFloat_AsDouble(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['tol_dj'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "tol_piv")) 
                if (PyFloat_Check(value))
                    smcp.tol_piv = PyFloat_AsDouble(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['tol_piv'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "obj_ll"))
                if (PyFloat_Check(value))
                    smcp.obj_ll = PyFloat_AsDouble(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['obj_ll'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "obj_ul"))
                if (PyFloat_Check(value))
                    smcp.obj_ul = PyFloat_AsDouble(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['obj_ul'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "it_lim"))
                if (PYINT_CHECK(value)) 
                    smcp.it_lim = PYINT_AS_LONG(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['it_lim'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "tm_lim"))
                if (PYINT_CHECK(value)) 
                    smcp.tm_lim = PYINT_AS_LONG(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['tm_lim'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "out_frq"))
                if (PYINT_CHECK(value)) 
                    smcp.out_frq = PYINT_AS_LONG(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['out_frq'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "out_dly"))
                if (PYINT_CHECK(value)) 
                    smcp.out_dly = PYINT_AS_LONG(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['out_dly'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "presolve")){
                if (PYSTRING_CHECK(value)) {
                    if (!PYSTRING_COMPARE(value, "GLP_ON"))
                        smcp.presolve = GLP_ON;
                    else if (!PYSTRING_COMPARE(value, "GLP_OFF")) 
                        smcp.presolve = GLP_OFF;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['presolve'] with default value",
                            1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing "
                        "glpk.options['presolve'] with default value", 1);
            }
        }    

    Py_DECREF(param);

    switch (glp_simplex(lp, &smcp)){

        case 0:
            switch(glp_get_status(lp)){
                case GLP_OPT:
                    x = (matrix *) Matrix_New(n,1,DOUBLE);
                    z = (matrix *) Matrix_New(m,1,DOUBLE);
                    if (A) y = (matrix *) Matrix_New(p,1,DOUBLE);
                    if (!x || !z || (A && !y)){
                        Py_XDECREF(x);
                        Py_XDECREF(z);
                        Py_XDECREF(y);
                        Py_XDECREF(t);
                        glp_delete_prob(lp);
                        return PyErr_NoMemory();
                    }

                    PyTuple_SET_ITEM(t, 0, (PyObject *)
                        PYSTRING_FROMSTRING("optimal"));

                    for (i=0; i<n; i++)
                        MAT_BUFD(x)[i] = glp_get_col_prim(lp, i+1);
                    PyTuple_SET_ITEM(t, 1, (PyObject *) x);

                    for (i=0; i<m; i++)
                        MAT_BUFD(z)[i] = -glp_get_row_dual(lp, i+1);
                    PyTuple_SET_ITEM(t, 2, (PyObject *) z);

                    if (A){
                        for (i=0; i<p; i++)
                            MAT_BUFD(y)[i] = -glp_get_row_dual(lp, m+i+1);
                        PyTuple_SET_ITEM(t, 3, (PyObject *) y);
                    }
                    glp_delete_prob(lp);
                    return (PyObject *) t;
                    break;

                case GLP_NOFEAS:
                    PyTuple_SET_ITEM(t, 0, (PyObject *)
                        PYSTRING_FROMSTRING("primal infeasible"));
                    break;

                case GLP_UNBND:
                    PyTuple_SET_ITEM(t, 0, (PyObject *)
                        PYSTRING_FROMSTRING("dual infeasible"));
                    break;

                default: 
                    PyTuple_SET_ITEM(t, 0, (PyObject *)
                        PYSTRING_FROMSTRING("unknown"));
            }
            break;

        case GLP_ENOPFS:
            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PYSTRING_FROMSTRING("primal infeasible"));
            break;

        case GLP_ENODFS:
            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PYSTRING_FROMSTRING("dual infeasible"));
            break;

        default:
            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PYSTRING_FROMSTRING("unknown"));
    }

    glp_delete_prob(lp);
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
    "                x[k] is integer for k in I\n"
    "                x[k] is binary for k in B\n\n"
    "ARGUMENTS\n"
    "c            nx1 dense 'd' matrix with n>=1\n\n"
    "G            mxn dense or sparse 'd' matrix with m>=1\n\n"
    "h            mx1 dense 'd' matrix\n\n"
    "A            pxn dense or sparse 'd' matrix with p>=0\n\n"
    "b            px1 dense 'd' matrix\n\n"
    "I            set of indices of integer variables\n\n"
    "B            set of indices of binary variables\n\n"
    "status       if status is 'optimal', 'feasible', or 'undefined',\n"
    "             a value of x is returned and the status string \n"
    "             gives the status of x.  Other possible values of "
    "             status are:  'invalid formulation', \n"
    "             'infeasible problem', 'LP relaxation is primal \n"
    "             infeasible', 'LP relaxation is dual infeasible', \n"
    "             'unknown'.\n\n"
    "x            a (sub-)optimal solution if status is 'optimal', \n"
    "             'feasible', or 'undefined'.  None otherwise";

static PyObject *integer(PyObject *self, PyObject *args,
    PyObject *kwrds)
{
    matrix *c, *h, *b=NULL, *x=NULL;
    PyObject *G, *A=NULL, *IntSet=NULL, *BinSet = NULL;
    PyObject *t=NULL, *param, *key, *value;
    glp_prob *lp;
    glp_iocp iocp;
    int m, n, p, i, j, k, nnz, nnzmax, *rn=NULL, *cn=NULL, info, status;
    int_t pos=0;
    double *a=NULL, val;
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

    lp = glp_create_prob();
    glp_add_rows(lp, m+p);
    glp_add_cols(lp, n);

    for (i=0; i<n; i++){
        glp_set_obj_coef(lp, i+1, MAT_BUFD(c)[i]);
        glp_set_col_bnds(lp, i+1, GLP_FR, 0.0, 0.0);
    }
    for (i=0; i<m; i++)
        glp_set_row_bnds(lp, i+1, GLP_UP, 0.0, MAT_BUFD(h)[i]);
    for (i=0; i<p; i++)
        glp_set_row_bnds(lp, i+m+1, GLP_FX, MAT_BUFD(b)[i],
            MAT_BUFD(b)[i]);

    nnzmax = (SpMatrix_Check(G) ? SP_NNZ(G) : m*n ) +
        ((A && SpMatrix_Check(A)) ? SP_NNZ(A) : p*n);
    a = (double *) calloc(nnzmax+1, sizeof(double));
    rn = (int *) calloc(nnzmax+1, sizeof(int));
    cn = (int *) calloc(nnzmax+1, sizeof(int));
    if (!a || !rn || !cn){
        free(a);  free(rn);  free(cn);  glp_delete_prob(lp);
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

    glp_load_matrix(lp, nnz, rn, cn, a);
    free(rn);  free(cn);  free(a);

    if (!(t = PyTuple_New(2))) {
        glp_delete_prob(lp);
        return PyErr_NoMemory();
    }

    if (!(param = PyObject_GetAttrString(glpk_module, "options"))
        || !PyDict_Check(param)){
            glp_delete_prob(lp);
            PyErr_SetString(PyExc_AttributeError,
                "missing glpk.options dictionary");
            return NULL;
    }

    glp_init_iocp(&iocp);

    while (PyDict_Next(param, &pos, &key, &value))
        if (PYSTRING_CHECK(key)){
            if (!PYSTRING_COMPARE(key, "msg_lev"))
                if (PYSTRING_CHECK(value)){
                    if (!PYSTRING_COMPARE(value, "GLP_MSG_OFF"))
                        iocp.msg_lev = GLP_MSG_OFF;
                    else if (!PYSTRING_COMPARE(value, "GLP_MSG_ERR")) 
                        iocp.msg_lev = GLP_MSG_ERR;
                    else if (!PYSTRING_COMPARE(value, "GLP_MSG_ON")) 
                        iocp.msg_lev = GLP_MSG_ON;
                    else if (!PYSTRING_COMPARE(value, "GLP_MSG_ALL")) 
                        iocp.msg_lev = GLP_MSG_ALL;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['msg_lev'] with default value", 
                            1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['msg_lev'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "br_tech")) 
                if (PYSTRING_CHECK(value)){
                    if (!PYSTRING_COMPARE(value, "GLP_BR_FFV"))
                        iocp.br_tech= GLP_BR_FFV;
                    else if (!PYSTRING_COMPARE(value, "GLP_BR_LFV")) 
                        iocp.br_tech = GLP_BR_LFV;
                    else if (!PYSTRING_COMPARE(value, "GLP_BR_MFV")) 
                        iocp.br_tech = GLP_BR_MFV;
                    else if (!PYSTRING_COMPARE(value, "GLP_BR_DTH")) 
                        iocp.br_tech = GLP_BR_DTH;
                    else if (!PYSTRING_COMPARE(value, "GLP_BR_PCH")) 
                        iocp.br_tech = GLP_BR_PCH;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['br_tech'] with default value", 
                             1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['br_tech'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "bt_tech"))
                if (PYSTRING_CHECK(value)){
                    if (!PYSTRING_COMPARE(value, "GLP_BT_DFS"))
                        iocp.bt_tech = GLP_BT_DFS;
                    else if (!PYSTRING_COMPARE(value, "GLP_BT_BFS")) 
                        iocp.bt_tech = GLP_BT_BFS;
                    else if (!PYSTRING_COMPARE(value, "GLP_BT_BLB")) 
                        iocp.bt_tech = GLP_BT_BLB;
                    else if (!PYSTRING_COMPARE(value, "GLP_BT_BPH")) 
                        iocp.bt_tech = GLP_BT_BPH;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['bt_tech'] with default value",
                            1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['bt_tech'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "pp_tech"))
                if (PYSTRING_CHECK(value)){
                    if (!PYSTRING_COMPARE(value, "GLP_PP_NONE"))
                        iocp.pp_tech = GLP_PP_NONE;
                    else if (!PYSTRING_COMPARE(value, "GLP_PP_ROOT")) 
                        iocp.pp_tech = GLP_PP_ROOT;
                    else if (!PYSTRING_COMPARE(value, "GLP_PP_ALL")) 
                        iocp.pp_tech = GLP_PP_ALL;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['pp_tech'] with default value",
                            1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['pp_tech'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "fp_heur"))
                if (PYSTRING_CHECK(value)){
                    if (!PYSTRING_COMPARE(value, "GLP_ON"))
                        iocp.fp_heur = GLP_ON;
                    else if (!PYSTRING_COMPARE(value, "GLP_OFF")) 
                        iocp.fp_heur = GLP_OFF;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['fp_heur'] with default value",
                            1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['fp_heur'] "
                        "with default value", 1);
#if 0
            else if (!PYSTRING_COMPARE(key, "ps_heur"))
                if (PYSTRING_CHECK(value)){
                    if (!PYSTRING_COMPARE(value, "GLP_ON"))
                        iocp.ps_heur = GLP_ON;
                    else if (!PYSTRING_COMPARE(value, "GLP_OFF")) 
                        iocp.ps_heur = GLP_OFF;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['ps_heur'] with default value",
                            1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['ps_heur'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "ps_tm_lim"))
                if (PYINT_CHECK(value)) 
                    iocp.ps_tm_lim = PYINT_AS_LONG(value);
                else 
                    PyErr_WarnEx(NULL, "replacing "
                        "glpk.options['ps_tm_lim'] with default value", 1);
#endif
            else if (!PYSTRING_COMPARE(key, "gmi_cuts"))
                if (PYSTRING_CHECK(value)){
                    if (!PYSTRING_COMPARE(value, "GLP_ON"))
                        iocp.gmi_cuts = GLP_ON;
                    else if (!PYSTRING_COMPARE(value, "GLP_OFF")) 
                        iocp.gmi_cuts = GLP_OFF;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['gmi_cuts'] with default value",
                            1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing "
                        "glpk.options['gmi_cuts'] with default value", 1);
            else if (!PYSTRING_COMPARE(key, "mir_cuts"))
                if (PYSTRING_CHECK(value)){
                    if (!PYSTRING_COMPARE(value, "GLP_ON"))
                        iocp.mir_cuts = GLP_ON;
                    else if (!PYSTRING_COMPARE(value, "GLP_OFF")) 
                        iocp.mir_cuts = GLP_OFF;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['mir_cuts'] with default value",
                            1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing "
                        "glpk.options['mir_cuts'] with default value", 1);
            else if (!PYSTRING_COMPARE(key, "cov_cuts"))
                if (PYSTRING_CHECK(value)){
                    if (!PYSTRING_COMPARE(value, "GLP_ON"))
                        iocp.cov_cuts = GLP_ON;
                    else if (!PYSTRING_COMPARE(value, "GLP_OFF")) 
                        iocp.cov_cuts = GLP_OFF;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['cov_cuts'] with default value",
                            1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing "
                        "glpk.options['cov_cuts'] with default value", 1);
            else if (!PYSTRING_COMPARE(key, "clq_cuts"))
                if (PYSTRING_CHECK(value)){
                    if (!PYSTRING_COMPARE(value, "GLP_ON"))
                        iocp.clq_cuts = GLP_ON;
                    else if (!PYSTRING_COMPARE(value, "GLP_OFF")) 
                        iocp.clq_cuts = GLP_OFF;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['clq_cuts'] with default value",
                            1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing "
                        "glpk.options['clq_cuts'] with default value", 1);
            else if (!PYSTRING_COMPARE(key, "tol_int"))
                if (PyFloat_Check(value))
                    iocp.tol_int = PyFloat_AsDouble(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['tol_int'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "tol_obj"))
                if (PyFloat_Check(value))
                    iocp.tol_obj = PyFloat_AsDouble(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['tol_obj'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "mip_gap"))
                if (PyFloat_Check(value))
                    iocp.mip_gap = PyFloat_AsDouble(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['mip_gap'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "tm_lim"))
                if (PYINT_CHECK(value)) 
                    iocp.tm_lim = PYINT_AS_LONG(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['tm_lim'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "out_frq"))
                if (PYINT_CHECK(value)) 
                    iocp.out_frq = PYINT_AS_LONG(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['out_frq'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "out_dly"))
                if (PYINT_CHECK(value)) 
                    iocp.out_dly = PYINT_AS_LONG(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['out_dly'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "glp_tree"))
                PyErr_WarnEx(NULL, "replacing glpk.options['glp_tree'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "cb_info"))
                PyErr_WarnEx(NULL, "replacing glpk.options['cb_info'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "cb_size"))
                if (PYINT_CHECK(value)) 
                    iocp.cb_size = PYINT_AS_LONG(value);
                else 
                    PyErr_WarnEx(NULL, "replacing glpk.options['cb_size'] "
                        "with default value", 1);
            else if (!PYSTRING_COMPARE(key, "presolve"))
                if (PYSTRING_CHECK(value)) {
                    if (!PYSTRING_COMPARE(value, "GLP_ON"))
                        iocp.presolve = GLP_ON;
                    else if (!PYSTRING_COMPARE(value, "GLP_OFF")){
                        iocp.presolve = GLP_ON;
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['presolve'] with GLP_ON", 1);
                    }
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['presolve'] with GLP_ON", 1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing "
                        "glpk.options['presolve'] with GLP_ON", 1);
            else if (!PYSTRING_COMPARE(key, "binarize")) {
                if (PYSTRING_CHECK(value)) {
                    if (!PYSTRING_COMPARE(value, "GLP_ON"))
                        iocp.binarize = GLP_ON;
                    else if (!PYSTRING_COMPARE(value, "GLP_OFF")) 
                        iocp.binarize = GLP_OFF;
                    else 
                        PyErr_WarnEx(NULL, "replacing "
                            "glpk.options['binarize'] with default "
                            "value", 1);
                }
                else 
                    PyErr_WarnEx(NULL, "replacing "
                        "glpk.options['binarize'] with default value", 1);
            }
        }

    Py_DECREF(param);
    iocp.presolve = GLP_ON;

    if (IntSet) {
        PyObject *iter = PySequence_Fast(IntSet, 
            "Critical error: not sequence");
        for (i=0; i<PySet_GET_SIZE(IntSet); i++) {
            PyObject *tmp = PySequence_Fast_GET_ITEM(iter, i);
            if (!PYINT_CHECK(tmp)) {
                glp_delete_prob(lp);
                Py_DECREF(iter);
                PY_ERR_TYPE("non-integer element in I");
            }
            int k = PYINT_AS_LONG(tmp);
            if ((k < 0) || (k >= n)) {
                 glp_delete_prob(lp);
                 Py_DECREF(iter);
                 PY_ERR(PyExc_IndexError, "index element out of range "
                     "in I");
            }
            glp_set_col_kind(lp, k+1, GLP_IV);
        }
        Py_DECREF(iter);
    }

    if (BinSet){
        PyObject *iter = PySequence_Fast(BinSet, 
            "Critical error: not sequence");
        for (i=0; i<PySet_GET_SIZE(BinSet); i++) {
            PyObject *tmp = PySequence_Fast_GET_ITEM(iter, i);
            if (!PYINT_CHECK(tmp)) {
                glp_delete_prob(lp);
                Py_DECREF(iter);
                PY_ERR_TYPE("non-binary element in I");
            }
            int k = PYINT_AS_LONG(tmp);
            if ((k < 0) || (k >= n)) {
                glp_delete_prob(lp);
                Py_DECREF(iter);
                PY_ERR(PyExc_IndexError, 
                    "index element out of range in B");
	    }
	    glp_set_col_kind(lp, k+1, GLP_BV);
        }
        Py_DECREF(iter);
    }

    info = glp_intopt(lp, &iocp);
    status = glp_mip_status(lp);

    switch (info){

        case 0:
        case GLP_EMIPGAP:
        case GLP_ETMLIM:
            switch(status){
                case GLP_OPT:     /* x is optimal */
                case GLP_FEAS:    /* x is integer feasible */
                case GLP_UNDEF:   /* x is undefined */
                    x = (matrix *) Matrix_New(n,1,DOUBLE);
                    if (!x) {
                        Py_XDECREF(t);
                        glp_delete_prob(lp);
                        return PyErr_NoMemory();
                    }
                    if (status == GLP_OPT)
                        PyTuple_SET_ITEM(t, 0, 
                            (PyObject *) PYSTRING_FROMSTRING("optimal"));
                    else if (status == GLP_FEAS)
                        PyTuple_SET_ITEM(t, 0, 
                           (PyObject *)PYSTRING_FROMSTRING("feasible"));
                    else 
                        PyTuple_SET_ITEM(t, 0, 
                           (PyObject *)PYSTRING_FROMSTRING("undefined"));
                    for (i=0; i<n; i++)
                        MAT_BUFD(x)[i] = glp_mip_col_val(lp, i+1);
                    PyTuple_SET_ITEM(t, 1, (PyObject *) x);
                    glp_delete_prob(lp);
                    return (PyObject *) t;
                    break;

                case GLP_NOFEAS:
                    PyTuple_SET_ITEM(t, 0, (PyObject *)
                        PYSTRING_FROMSTRING("infeasible problem"));
                    PyTuple_SET_ITEM(t, 1, Py_BuildValue(""));
                    break;

                default: 
                    PyTuple_SET_ITEM(t, 1, Py_BuildValue(""));
                    PyTuple_SET_ITEM(t, 0, (PyObject *)
                        PYSTRING_FROMSTRING("unknown"));
            }
            break;

#if 0
        case GLP_EBADB:

        case GLP_ECOND:
#endif

        case GLP_EBOUND:
            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PYSTRING_FROMSTRING("invalid MIP formulation"));
            break;

        case GLP_ENOPFS:
            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PYSTRING_FROMSTRING("LP relaxation is primal infeasible"));
            break;

	case GLP_ENODFS:
            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PYSTRING_FROMSTRING("LP relaxation is dual infeasible"));
            break;

	case GLP_EFAIL:
            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PYSTRING_FROMSTRING("solver failure"));
            break;

        case GLP_EROOT: /* only occurs if presolver is off */
        case GLP_ESTOP: /* only occurs when advanced interface is used */
        default:
            PyTuple_SET_ITEM(t, 0, (PyObject *)
                PYSTRING_FROMSTRING("unknown"));
    }

    glp_delete_prob(lp);
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

#if PY_MAJOR_VERSION >= 3

static PyModuleDef glpk_module_def = {
    PyModuleDef_HEAD_INIT,
    "glpk",
    glpk__doc__,
    -1,
    glpk_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_glpk(void)
{
  if (!(glpk_module = PyModule_Create(&glpk_module_def))) return NULL;
  PyModule_AddObject(glpk_module, "options", PyDict_New());
  if (import_cvxopt() < 0) return NULL;
  return glpk_module;
}

#else

PyMODINIT_FUNC initglpk(void)
{
    glpk_module = Py_InitModule3("cvxopt.glpk", glpk_functions, 
        glpk__doc__);
    PyModule_AddObject(glpk_module, "options", PyDict_New());
    if (import_cvxopt() < 0) return;
}

#endif
