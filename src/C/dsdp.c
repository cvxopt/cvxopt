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
#include "misc.h"
#include "dsdp5.h"
#include "math.h"

PyDoc_STRVAR(dsdp__doc__,"Interface to DSDP version 5.8.\n\n"
    "Software for Semidefinite Programming.\n\n"
    "Three control parameters can be modified by making an entry in \n"
    "dictionary dsdp.options:\n"
    "    options['DSDP_Monitor']: set to k in order to show \n"
    "        progress after every kth iteration (default: 0). \n"
    "    options['DSDP_MaxIts']:  maximum number of iterations\n"
    "    options['DSDP_GapTolerance']: the relative tolerance used\n"
    "        in the exit condition (default: 1e-5).\n\n"
    "DSDSP is available from www-unix.mcs.anl.gov/DSDP.");

static  PyObject *dsdp_module;

static char doc_dsdp[] =
    "Solves a semidefinite program using DSDP.\n\n"
    "(status, x, r, zl, zs) = sdp(c, Gl=None, hl=None, Gs=None, hs=None"
    ",\n"
    "                             gamma=1e8, beta=1e7)"
    "\n\n"
    "PURPOSE\n"
    "Solves the SDP\n\n"
    "    minimize    c'*x + gamma*r\n"
    "    subject to  Gl*x <= hl + r\n"
    "                mat(Gs[k]*x) <= hs[k] + r*I, k=1,...,L\n"
    "                -beta <= x <= beta,  r >= 0\n\n"
    "and its dual\n\n"
    "    maximize    -hl'*zl - sum_k tr(hs[k]*zs[k]) - beta*||zb||_1\n"
    "    subject to  Gl'*zl + sum_k Gs[k]'*vec(zs[k]) + zb + c = 0\n"
    "                sum(zl) + sum_k tr(zs[k]) <= gamma \n"
    "                zl >= 0,  zs[k] >=0, k=1,...,L. \n\n"
    "For an mxm matrix y, vec(y) denotes the m^2-vector with the\n"
    "entries of y stored columnwise.   mat(y) is the inverse\n"
    "operation.\n\n"
    "ARGUMENTS\n"
    "c         n by 1 dense 'd' matrix\n\n"
    "Gl        ml by n dense or sparse 'd' matrixi.  The default\n"
    "          value is a matrix with zero rows.\n\n"
    "hl        ml by 1 dense 'd' matrix.  The default value is a\n"
    "          vector of length zero.\n\n"
    "Gs        list of L dense or sparse 'd' matrices.  If the kth\n"
    "          linear matrix inequality has size mk, then Gs[k] is a\n"
    "          matrix of size mk**2 by n, and mat(Gs[k][:,i]) is the\n"
    "          coefficient of the ith variable i in inequality k.\n"
    "          Only the lower triangular entries in mat(Gs[k][:,i])\n"
    "          are accessed.  The default value of Gs is an empty list."
    "\n\n"
    "hs        list of L square dense 'd' matrices.  hs[k] is the\n"
    "          righthand side in the kth linear matrix inequality.\n"
    "          Only the lower triangular entries of hs[k] are\n"
    "          accessed.  The default value of hs is an empty list.\n\n"
    "beta      positive double\n\n"
    "gamma     positive double\n\n"
    "status    the DSDP solution status: 'DSDP_PDFEASIBLE', \n"
    "          'DSDP_UNBOUNDED', 'DSDP_INFEASIBLE', or\n"
    "          'DSDP_UNKNOWN'.\n\n"
    "x         the primal solution, as a dense 'd' matrix of size\n"
    "          n by 1\n\n"
    "r         the optimal value of the variable r\n\n"
    "zl        the dual solution as a dense 'd' matrix of size\n"
    "          ml by 1\n\n"
    "zs        the dual solution as a list of L square dense 'd'\n"
    "          matrices.  Each matrix represents a symmetric matrix\n"
    "          in unpacked lower triangular format.";


typedef struct {     /* symmetric matrix X in DSDP packed storage */
    int n;           /* order of X */
    char issparse;
    double *val;     /* lower triangular nonzeros of X.  Stored in row
                      * major order as an n*(n+1)/2-array if X is dense,
                      * and in arbitrary order if X is sparse.*/
    int *ind;        /* NULL if X is dense; otherwise, the indices of
                      *	the elements of val in the n*(n+1)/2 array of
                      *	the lower triangular entries of X stored
                      *	rowwise. */
    int nnz;         /* length of val */
} dsdp_matrix;

extern void dcopy_(int *n, double *x, int *incx, double *y, int *incy);

static PyObject* solvesdp(PyObject *self, PyObject *args,
    PyObject *kwrds)
{
    matrix *c, *hl=NULL, *hk, *x=NULL, *zl=NULL, *zsk=NULL;
    PyObject *Gl=NULL, *Gs=NULL, *hs=NULL, *Gk, *t=NULL, *zs=NULL, *opts=NULL,
        *param, *key, *value;
    int i, j, k, n, ml, l, mk, nnz, *lp_colptr=NULL, *lp_rowind=NULL,
      incx, incy, lngth, maxm;
    int_t pos=0;
    double *lp_values=NULL, *zlvals=NULL, *zk=NULL, r, beta=-1.0,
        gamma=-1.0, tol;
    dsdp_matrix **lmis=NULL;
    div_t qr;
    DSDP sdp;
    LPCone lpcone;
    SDPCone sdpcone;
    DSDPTerminationReason info;
    DSDPSolutionType status;
    char err_str[100];
#if PY_MAJOR_VERSION >= 3
    const char *keystr;
#else
    char *keystr;
#endif
    char *kwlist[] = {"c", "Gl", "hl", "Gs", "hs", "gamma", "beta", "options",
        NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "O|OOOOddO", kwlist,
        &c, &Gl, &hl, &Gs, &hs, &gamma, &beta, &opts)) return NULL;

    if (!Matrix_Check(c) || MAT_NCOLS(c) != 1 || MAT_ID(c) != DOUBLE)
        PY_ERR_TYPE("c must be a dense 'd' matrix with one column");
    n = MAT_NROWS(c);

    if (Gl == Py_None) Gl = NULL;
    if (Gl && ((!Matrix_Check(Gl) && !SpMatrix_Check(Gl)) ||
        X_NCOLS(Gl) != n || X_ID(Gl) != DOUBLE))
        PY_ERR_TYPE("invalid type or dimensions for Gl");
    ml = Gl ? X_NROWS(Gl) : 0;

    if ((PyObject *)hl == Py_None) hl = NULL;
    if ((!hl && ml) || (hl && (!Matrix_Check(hl) || MAT_NCOLS(hl) != 1
        || MAT_NROWS(hl) != ml || MAT_ID(hl) != DOUBLE)))
        PY_ERR_TYPE("invalid type or dimensions for hl");

    if (Gs && !PyList_Check(Gs)) PY_ERR_TYPE("Gs must be a list");
    l = Gs ? PyList_Size(Gs) : 0;
    if (hs && !PyList_Check(hs)) PY_ERR_TYPE("hs must be a list");
    if ((!hs && l) || (hs && PyList_Size(hs) != l))
        PY_ERR_TYPE("Gs and hs must be lists of equal length");
    for (maxm=0, k=0; k<l; k++) {
        Gk = PyList_GetItem(Gs, k);
        hk = (matrix *) PyList_GetItem(hs, k);
        if ((!Matrix_Check(Gk) && !SpMatrix_Check(Gk)) ||
            X_ID(Gk) != DOUBLE)
            PY_ERR_TYPE("Gs must be a list of 'd' matrices with n "
                "columns");
        if (!Matrix_Check(hk) || MAT_ID(hk) != DOUBLE ||
            ((mk = MAT_NCOLS(hk)) != MAT_NROWS(hk)))
            PY_ERR_TYPE("hs must be a list of square dense 'd' "
                "matrices");
        if (X_NROWS(Gk) != mk*mk || X_NCOLS(Gk) != n)
            PY_ERR_TYPE("incompatible dimensions for elements of Gs");
        maxm = MAX(mk, maxm);
    }

    if (DSDPCreate(n, &sdp) || DSDPCreateLPCone(sdp, &lpcone) ||
        DSDPCreateSDPCone(sdp, l, &sdpcone)){
        t = PyErr_NoMemory();
        goto done;
    }

    if (opts && PyDict_Check(opts))
        param = opts;
    else
        param = PyObject_GetAttrString(dsdp_module, "options");
    if (!param || !PyDict_Check(param)){
        PyErr_SetString(PyExc_AttributeError, "missing dsdp.options "
            " dictionary");
        t = NULL;
        goto done;
    }
    while (PyDict_Next(param, &pos, &key, &value))
#if PY_MAJOR_VERSION >= 3
	if (PyUnicode_Check(key)) {
	    keystr = _PyUnicode_AsString(key);
#else
        if ((keystr = PyString_AsString(key))){
#endif
            if (!strcmp(keystr, "DSDP_Monitor")){
#if PY_MAJOR_VERSION >= 3
                if (!PyLong_Check(value)) {
#else
                if (!PyInt_Check(value)) {
#endif
                    sprintf(err_str, "invalid value for integer "
                        "DSDP parameter: DSDP_Monitor");
                    PyErr_SetString(PyExc_ValueError, err_str);
                    t = NULL;
                    Py_DECREF(param);
                    goto done;
                }
#if PY_MAJOR_VERSION >= 3
                else DSDPSetStandardMonitor(sdp, PyLong_AsLong(value));
#else
                else DSDPSetStandardMonitor(sdp, PyInt_AsLong(value));
#endif
            }
            if (!strcmp(keystr, "DSDP_MaxIts")){
#if PY_MAJOR_VERSION >= 3
                if (!PyLong_Check(value) || (k = PyLong_AsLong(value)) < 0){
#else
                if (!PyInt_Check(value) || (k = PyInt_AsLong(value)) < 0){
#endif
                    sprintf(err_str, "invalid value for nonnegative "
                        "integer DSDP parameter: DSDP_MaxIts");
                    PyErr_SetString(PyExc_ValueError, err_str);
                    t = NULL;
                    Py_DECREF(param);
                    goto done;
                }
		else DSDPSetMaxIts(sdp, k);
            }
            if (!strcmp(keystr, "DSDP_GapTolerance")){
#if PY_MAJOR_VERSION >= 3
                if ((!PyFloat_Check(value) && !PyLong_Check(value)) ||
#else
                if ((!PyFloat_Check(value) && !PyInt_Check(value)) ||
#endif
                    (tol = PyFloat_AsDouble(value)) <= 0.0) {
                    sprintf(err_str, "invalid value for float "
                        "DSDP parameter: DSDP_GapTolerance");
                    PyErr_SetString(PyExc_ValueError, err_str);
                    t = NULL;
                    Py_DECREF(param);
                    goto done;
                }
                else DSDPSetGapTolerance(sdp, tol);
            }
        }
    Py_DECREF(param);

    if (gamma > 0) DSDPSetPenaltyParameter(sdp, gamma);
    if (beta > 0) DSDPSetYBounds(sdp, -beta, beta);

    /* cost function */
    for (k=0; k<n; k++) DSDPSetDualObjective(sdp, k+1, -MAT_BUFD(c)[k]);

    /* linear inequalities: store [Gl, hl] in CCS format */
    nnz = ml ? ml + (Matrix_Check(Gl) ? ml*n : SP_NNZ(Gl)) : 0;
    if (!(lp_colptr = (int *) calloc(n+2, sizeof(int))) ||
        !(lp_rowind = (int *) calloc(nnz, sizeof(int))) ||
        !(lp_values = (double *) calloc(nnz, sizeof(double)))){
        t = PyErr_NoMemory();
        goto done;
    }
    lp_colptr[0] = 0;
    if (ml){
        if (Matrix_Check(Gl)){
            memcpy(lp_values, MAT_BUFD(Gl), ml*n*sizeof(double));
            for (k=0; k<n; k++){
                for (j=0; j<ml; j++) lp_rowind[ml*k+j] = j;
                lp_colptr[k+1] = lp_colptr[k] + ml;
            }
        }
        else {
            memcpy(lp_values, SP_VALD(Gl), SP_NNZ(Gl)*sizeof(double));
            for (k=0; k<n; k++){
                for (j=SP_COL(Gl)[k]; j<SP_COL(Gl)[k+1]; j++)
                    lp_rowind[j] = (int) SP_ROW(Gl)[j];
                lp_colptr[k+1] = lp_colptr[k] + (int) (SP_COL(Gl)[k+1] -
                    SP_COL(Gl)[k]);
            }
        }
        memcpy(lp_values+lp_colptr[n], MAT_BUFD(hl), ml*sizeof(double));
        for (k=0; k<ml; k++) lp_rowind[lp_colptr[n]+k] = k;
        lp_colptr[n+1] = lp_colptr[n] + ml;
    }
    if (LPConeSetData2(lpcone, ml, lp_colptr, lp_rowind, lp_values)){
        t = PyErr_NoMemory();
        goto done;
    }
    /* LPConeView(lpcone); */

    /* linear matrix inequalities: store mat(hs[k]), mat(Gs[k][:,i])
     * as an lx(n+1) array of dsdp matrices. */
    if (!(lmis = (dsdp_matrix **) calloc(l, sizeof(dsdp_matrix *)))){
        t = PyErr_NoMemory();
        goto done;
    }
    for (k=0; k<l; k++) lmis[k] = NULL;
    for (k=0; k<l; k++){
        Gk = PyList_GetItem(Gs, k);
        hk = (matrix *) PyList_GetItem(hs, k);
        if (!(lmis[k] = (dsdp_matrix *) calloc(n+1,
            sizeof(dsdp_matrix)))){
            t = PyErr_NoMemory();
            goto done;
        }

	/* lmis[k][0] is hs[k] as a dsdp matrix */
        mk = MAT_NROWS(hk);
        lmis[k][0].n = mk;
        lmis[k][0].issparse = 0;
        if (!(lmis[k][0].val = (double *) calloc(mk*(mk+1)/2,
            sizeof(double)))){
            t = PyErr_NoMemory();
            goto done;
        }
        lmis[k][0].ind = NULL;
        lmis[k][0].nnz = mk*(mk+1)/2;
        for (j=0; j<mk; j++){
            lngth = j+1;  incx = mk;  incy = 1;
            dcopy_(&lngth, MAT_BUFD(hk)+j, &incx,
                lmis[k][0].val+j*(j+1)/2, &incy);
        }

        /* lmis[k][i+1] is mat(Gs[k][i]) as a dsdp matrix */
        for (i=0; i<n; i++){
            lmis[k][i+1].n = mk;
            if (Matrix_Check(Gk)){
                lmis[k][i+1].issparse = 0;
                if (!(lmis[k][i+1].val = (double *) calloc(mk*(mk+1)/2,
                    sizeof(double)))){
                    t = PyErr_NoMemory();
                    goto done;
                }
                lmis[k][i+1].ind = NULL;
                lmis[k][i+1].nnz = mk*(mk+1)/2;
                for (j=0; j<mk; j++){
                    lngth = j+1;  incx = mk;  incy = 1;
                    dcopy_(&lngth, MAT_BUFD(Gk)+i*mk*mk+j, &incx,
                        lmis[k][i+1].val+j*(j+1)/2, &incy);
                }
            } else {
                lmis[k][i+1].issparse = 1;
                /* nnz is number of lower triangular nonzeros in
                 * Gk[:,i] */
                for (nnz=0, j=SP_COL(Gk)[i]; j<SP_COL(Gk)[i+1]; j++){
                    qr = div(SP_ROW(Gk)[j], mk);
                    if (qr.quot <= qr.rem) nnz++;
                }
                lmis[k][i+1].nnz = nnz;
                if (!(lmis[k][i+1].val = (double *) calloc(nnz,
                    sizeof(double))) || !(lmis[k][i+1].ind = (int *)
                    calloc(nnz, sizeof(int)))){
                    t = PyErr_NoMemory();
                    goto done;
                }
                /* lmis[k][i+1].val, lmis[k][i+1].ind are the lower
                 * triangular nonzeros/indices of Gk[:,i].  The indices
		 * refer to the postions in the lower triangular part
		 * stored in row major order as an mk*(mk+1)/2 array. */
                for (nnz=0, j=SP_COL(Gk)[i]; j<SP_COL(Gk)[i+1]; j++){
                    qr = div(SP_ROW(Gk)[j], mk);
                    if (qr.quot <= qr.rem){
                        lmis[k][i+1].val[nnz] = SP_VALD(Gk)[j];
                        lmis[k][i+1].ind[nnz] = qr.rem*(qr.rem+1)/2 +
                            qr.quot;
                        nnz++;
                    }
                }
            }
        }
    }
    for (k=0; k<l; k++) for(i=0; i<n+1; i++){
        if (lmis[k][i].issparse){
            SDPConeSetASparseVecMat(sdpcone, k, i, lmis[k][i].n, 1.0, 0,
                lmis[k][i].ind, lmis[k][i].val, lmis[k][i].nnz);
        }
        else {
            SDPConeSetADenseVecMat(sdpcone, k, i, lmis[k][i].n, 1.0,
                lmis[k][i].val, lmis[k][i].nnz);
        }
        /* SDPConeViewDataMatrix(sdpcone, k, i);  */
    }

    DSDPSetup(sdp);
    if (DSDPSolve(sdp)){
        PyErr_SetString(PyExc_ArithmeticError, "DSDP error");
        t = NULL;
        goto done;
    }
    DSDPStopReason(sdp, &info);
    if (info != DSDP_CONVERGED && info != DSDP_SMALL_STEPS &&
        info != DSDP_INDEFINITE_SCHUR_MATRIX && info != DSDP_MAX_IT
        && info != DSDP_NUMERICAL_ERROR && info != DSDP_UPPERBOUND ){
        PyErr_SetObject(PyExc_ArithmeticError, Py_BuildValue("i",info));
        t = NULL;
        goto done;
    }

    if (!(zs = PyList_New(l)) || !(x = (matrix *) Matrix_New(n, 1,
        DOUBLE)) || !(zl = (matrix *) Matrix_New(ml, 1, DOUBLE)) ||
        !(zk = (double *) calloc(maxm*(maxm+1)/2, sizeof(double))) ||
        !(t = PyTuple_New(5))) {
        Py_XDECREF(x);  Py_XDECREF(zl);  Py_XDECREF(zs);  Py_XDECREF(t);
        t = PyErr_NoMemory();
        goto done;
    }
    DSDPGetSolutionType(sdp, &status);

    if (info == DSDP_CONVERGED) {
        switch (status){
            case DSDP_PDFEASIBLE:
#if PY_MAJOR_VERSION >= 3
                PyTuple_SET_ITEM(t, 0, (PyObject *)PyUnicode_FromString("DSDP_PDFEASIBLE"));
#else
                PyTuple_SET_ITEM(t, 0, (PyObject *)PyString_FromString("DSDP_PDFEASIBLE"));
#endif
                break;
            case DSDP_UNBOUNDED:
#if PY_MAJOR_VERSION >= 3
                PyTuple_SET_ITEM(t, 0, (PyObject *)PyUnicode_FromString("DSDP_UNBOUNDED"));
#else
                PyTuple_SET_ITEM(t, 0, (PyObject *)PyString_FromString("DSDP_UNBOUNDED"));
#endif
                break;
            case DSDP_INFEASIBLE:
#if PY_MAJOR_VERSION >= 3
                PyTuple_SET_ITEM(t, 0, (PyObject *)PyUnicode_FromString("DSDP_INFEASIBLE"));
#else
                PyTuple_SET_ITEM(t, 0, (PyObject *)PyString_FromString("DSDP_INFEASIBLE"));
#endif
                break;
            case DSDP_PDUNKNOWN:
#if PY_MAJOR_VERSION >= 3
                PyTuple_SET_ITEM(t, 0, (PyObject *)PyUnicode_FromString("DSDP_UNKNOWN"));
#else
                PyTuple_SET_ITEM(t, 0, (PyObject *)PyString_FromString("DSDP_UNKNOWN"));
#endif
                break;
        }
    } else {
#if PY_MAJOR_VERSION >= 3
        PyTuple_SET_ITEM(t, 0, (PyObject *)PyUnicode_FromString("DSDP_UNKNOWN"));
#else
        PyTuple_SET_ITEM(t, 0, (PyObject *)PyString_FromString("DSDP_UNKNOWN"));
#endif
    }

    DSDPGetY(sdp, MAT_BUFD(x), n);
    PyTuple_SET_ITEM(t, 1, (PyObject *) x);

    DSDPGetR(sdp, &r);
    PyTuple_SET_ITEM(t, 2, Py_BuildValue("d", r));

    DSDPComputeX(sdp);
    LPConeGetXArray(lpcone, &zlvals, &k);
    memcpy(MAT_BUFD(zl), zlvals, ml*sizeof(double));
    PyTuple_SET_ITEM(t, 3, (PyObject *) zl);

    for (k=0; k<l; k++){
        hk = (matrix *) PyList_GetItem(hs, k);
        mk = MAT_NROWS(hk);
        if (!(zsk = (matrix *) Matrix_New(mk, mk, DOUBLE))){
            Py_XDECREF(x);  Py_XDECREF(zl);  Py_XDECREF(zs);
            Py_XDECREF(t);
            t = PyErr_NoMemory();
            goto done;
        }
        SDPConeComputeX(sdpcone, k, mk, zk, maxm*(maxm+1)/2);
        for (j=0; j<mk; j++){
            lngth=j+1;  incx=1;  incy=mk;
            dcopy_(&lngth, zk+j*(j+1)/2, &incx, MAT_BUFD(zsk)+j, &incy);
        }
        PyList_SetItem(zs, k, (PyObject *) zsk);
    }
    PyTuple_SET_ITEM(t, 4, (PyObject *) zs);

    done:
        free(lp_colptr);  free(lp_rowind);  free(lp_values);  free(zk);
        DSDPDestroy(sdp);
        if (lmis) for (k=0; k<l; k++){
            if (lmis[k]) for (i=0; i<n+1; i++){
                if (lmis[k][i].issparse) free(lmis[k][i].ind);
                free(lmis[k][i].val);
            }
        }
        free(lmis);
        return t;
}

static PyMethodDef dsdp_functions[] = {
    {"sdp", (PyCFunction) solvesdp, METH_VARARGS|METH_KEYWORDS, doc_dsdp},
    {NULL}  /* Sentinel */
};


#if PY_MAJOR_VERSION >= 3

static PyModuleDef dsdp_module_def = {
    PyModuleDef_HEAD_INIT,
    "dsdp",
    dsdp__doc__,
    -1,
    dsdp_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_dsdp(void)
{
  if (!(dsdp_module = PyModule_Create(&dsdp_module_def))) return NULL;
  PyModule_AddObject(dsdp_module, "options", PyDict_New());
  if (import_cvxopt() < 0) return NULL;
  return dsdp_module;
}

#else

PyMODINIT_FUNC initdsdp(void)
{
    dsdp_module = Py_InitModule3("cvxopt.dsdp", dsdp_functions,
        dsdp__doc__);
    PyModule_AddObject(dsdp_module, "options", PyDict_New());
    if (import_cvxopt() < 0) return;
}

#endif
