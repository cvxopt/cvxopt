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

#include "Python.h"
#include "cvxopt.h"
#include "misc.h"
#include "math.h"
#include "float.h"

PyDoc_STRVAR(misc_solvers__doc__, "Miscellaneous functions used by the "
    "CVXOPT solvers.");

extern void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
extern double dnrm2_(int *n, double *x, int *incx);
extern double ddot_(int *n, double *x, int *incx, double *y, int *incy);
extern void dscal_(int *n, double *alpha, double *x, int *incx);
extern void daxpy_(int *n, double *alpha, double *x, int *incx, double *y,
    int *incy);
extern void dtbmv_(char *uplo, char *trans, char *diag, int *n, int *k,
    double *A, int *lda, double *x, int *incx);
extern void dtbsv_(char *uplo, char *trans, char *diag, int *n, int *k,
    double *A, int *lda, double *x, int *incx);
extern void dgemv_(char* trans, int *m, int *n, double *alpha, double *A,
    int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void dger_(int *m, int *n, double *alpha, double *x, int *incx,
    double *y, int *incy, double *A, int *lda);
extern void dtrmm_(char *side, char *uplo, char *transa, char *diag,
    int *m, int *n, double *alpha, double *A, int *lda, double *B,
    int *ldb);
extern void dsyr2k_(char *uplo, char *trans, int *n, int *k, double *alpha,
    double *A, int *lda, double *B, int *ldb, double *beta, double *C,
    int *ldc);
extern void dlacpy_(char *uplo, int *m, int *n, double *A, int *lda,
    double *B, int *ldb);
extern void dsyevr_(char *jobz, char *range, char *uplo, int *n, double *A,
    int *ldA, double *vl, double *vu, int *il, int *iu, double *abstol,
    int *m, double *W, double *Z, int *ldZ, int *isuppz, double *work,
    int *lwork, int *iwork, int *liwork, int *info);
extern void dsyevd_(char *jobz, char *uplo, int *n, double *A, int *ldA,
    double *W, double *work, int *lwork, int *iwork, int *liwork,
    int *info);


static char doc_scale[] =
    "Applies Nesterov-Todd scaling or its inverse.\n\n"
    "scale(x, W, trans = 'N', inverse = 'N')\n\n"
    "Computes\n\n"
    "    x := W*x        (trans is 'N', inverse = 'N')\n"
    "    x := W^T*x      (trans is 'T', inverse = 'N')\n"
    "    x := W^{-1}*x   (trans is 'N', inverse = 'I')\n"
    "    x := W^{-T}*x   (trans is 'T', inverse = 'I').\n\n"
    "x is a dense 'd' matrix.\n\n"
    "W is a dictionary with entries:\n\n"
    "- W['dnl']: positive vector\n"
    "- W['dnli']: componentwise inverse of W['dnl']\n"
    "- W['d']: positive vector\n"
    "- W['di']: componentwise inverse of W['d']\n"
    "- W['v']: lists of 2nd order cone vectors with unit hyperbolic \n"
    "  norms\n"
    "- W['beta']: list of positive numbers\n"
    "- W['r']: list of square matrices\n"
    "- W['rti']: list of square matrices.  rti[k] is the inverse\n"
    "  transpose of r[k]. \n\n"
    "The 'dnl' and 'dnli' entries are optional, and only present when \n"
    "the function is called from the nonlinear solver.";

static PyObject* scale(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x, *d, *vk, *rk;
    PyObject *W, *v, *beta, *r, *betak;
#if PY_MAJOR_VERSION >= 3
    int trans = 'N', inverse = 'N';
#else
    char trans = 'N', inverse = 'N';
#endif
    int m, n, xr, xc, ind = 0, int0 = 0, int1 = 1, i, k, inc, len, ld, 
        maxn, N;
    double b, dbl0 = 0.0, dbl1 = 1.0, dblm1 = -1.0, dbl2 = 2.0, dbl5 = 0.5,
        *wrk;
    char *kwlist[] = {"x", "W", "trans", "inverse", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|CC", kwlist,
        &x, &W, &trans, &inverse)) return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|cc", kwlist,
        &x, &W, &trans, &inverse)) return NULL;
#endif

    xr = x->nrows;
    xc = x->ncols;


    /*
     * Scaling for nonlinear component xk is xk := dnl .* xk; inverse is
     * xk ./ dnl = dnli .* xk, where dnl = W['dnl'], dnli = W['dnli'].
     */

    if ((d = (inverse == 'N') ? (matrix *) PyDict_GetItemString(W, "dnl") :
        (matrix *) PyDict_GetItemString(W, "dnli"))){
        m = len(d);
        for (i = 0; i < xc; i++)
            dtbmv_("L", "N", "N", &m, &int0, MAT_BUFD(d), &int1,
                MAT_BUFD(x) + i*xr, &int1);
        ind += m;
    }


    /*
     * Scaling for 'l' component xk is xk := d .* xk; inverse scaling is
     * xk ./ d = di .* xk, where d = W['d'], di = W['di'].
     */

    if (!(d = (inverse == 'N') ? (matrix *) PyDict_GetItemString(W, "d") :
        (matrix *) PyDict_GetItemString(W, "di"))){
        PyErr_SetString(PyExc_KeyError, "missing item W['d'] or W['di']");
        return NULL;
    }
    m = len(d);
    for (i = 0; i < xc; i++)
        dtbmv_("L", "N", "N", &m, &int0, MAT_BUFD(d), &int1, MAT_BUFD(x)
            + i*xr + ind, &int1);
    ind += m;


    /*
     * Scaling for 'q' component is
     *
     *     xk := beta * (2*v*v' - J) * xk
     *         = beta * (2*v*(xk'*v)' - J*xk)
     *
     * where beta = W['beta'][k], v = W['v'][k], J = [1, 0; 0, -I].
     *
     * Inverse scaling is
     *
     *     xk := 1/beta * (2*J*v*v'*J - J) * xk
     *         = 1/beta * (-J) * (2*v*((-J*xk)'*v)' + xk).
     */

    v = PyDict_GetItemString(W, "v");
    beta = PyDict_GetItemString(W, "beta");
    N = (int) PyList_Size(v);
    if (!(wrk = (double *) calloc(xc, sizeof(double))))
        return PyErr_NoMemory();
    for (k = 0; k < N; k++){
        vk = (matrix *) PyList_GetItem(v, (Py_ssize_t) k);
        m = vk->nrows;
        if (inverse == 'I')
            dscal_(&xc, &dblm1, MAT_BUFD(x) + ind, &xr);
        ld = MAX(xr, 1);
        dgemv_("T", &m, &xc, &dbl1, MAT_BUFD(x) + ind, &ld, MAT_BUFD(vk), 
            &int1, &dbl0, wrk, &int1);
        dscal_(&xc, &dblm1, MAT_BUFD(x) + ind, &xr);
        dger_(&m, &xc, &dbl2, MAT_BUFD(vk), &int1, wrk, &int1,
            MAT_BUFD(x) + ind, &ld);
        if (inverse == 'I')
            dscal_(&xc, &dblm1, MAT_BUFD(x) + ind, &xr);

        betak = PyList_GetItem(beta, (Py_ssize_t) k);
        b = PyFloat_AS_DOUBLE(betak);
        if (inverse == 'I') b = 1.0 / b;
        for (i = 0; i < xc; i++)
            dscal_(&m, &b, MAT_BUFD(x) + ind + i*xr, &int1);
        ind += m;
    }
    free(wrk);


    /*
     * Scaling for 's' component xk is
     *
     *     xk := vec( r' * mat(xk) * r )  if trans = 'N'
     *     xk := vec( r * mat(xk) * r' )  if trans = 'T'.
     *
     * r is kth element of W['r'].
     *
     * Inverse scaling is
     *
     *     xk := vec( rti * mat(xk) * rti' )  if trans = 'N'
     *     xk := vec( rti' * mat(xk) * rti )  if trans = 'T'.
     *
     * rti is kth element of W['rti'].
     */

    r = (inverse == 'N') ? PyDict_GetItemString(W, "r") :
        PyDict_GetItemString(W, "rti");
    N = (int) PyList_Size(r);
    for (k = 0, maxn = 0; k < N; k++){
        rk = (matrix *) PyList_GetItem(r, (Py_ssize_t) k);
        maxn = MAX(maxn, rk->nrows);
    }
    if (!(wrk = (double *) calloc(maxn*maxn, sizeof(double))))
        return PyErr_NoMemory();
    for (k = 0; k < N; k++){
        rk = (matrix *) PyList_GetItem(r, (Py_ssize_t) k);
        n = rk->nrows;
        for (i = 0; i < xc; i++){

            /* scale diagonal of rk by 0.5 */
            inc = n + 1;
            dscal_(&n, &dbl5, MAT_BUFD(x) + ind + i*xr, &inc);

            /* wrk = r*tril(x) if inverse is 'N' and trans is 'T' or
             *                 inverse is 'I' and trans is 'N'
             * wrk = tril(x)*r otherwise. */
            len = n*n;
            dcopy_(&len, MAT_BUFD(rk), &int1, wrk, &int1);
            ld = MAX(1, n);
            dtrmm_( (( inverse == 'N' && trans == 'T') || ( inverse == 'I'
                && trans == 'N')) ? "R" : "L", "L", "N", "N", &n, &n,
                &dbl1, MAT_BUFD(x) + ind + i*xr, &ld, wrk, &ld);

            /* x := (r*wrk' + wrk*r') if inverse is 'N' and trans is 'T'
             *                        or inverse is 'I' and trans is 'N'
             * x := (r'*wrk + wrk'*r) otherwise. */
            dsyr2k_("L", ((inverse == 'N' && trans == 'T') ||
                (inverse == 'I' && trans == 'N')) ? "N" : "T", &n, &n,
                &dbl1, MAT_BUFD(rk), &ld, wrk, &ld, &dbl0, MAT_BUFD(x) +
                ind + i*xr, &ld);
        }
        ind += n*n;
    }
    free(wrk);

    return Py_BuildValue("");
}


static char doc_scale2[] =
    "Multiplication with square root of the Hessian.\n\n"
    "scale2(lmbda, x, dims, mnl = 0, inverse = 'N')\n\n"
    "Computes\n\n"
    "Evaluates\n\n"
    "    x := H(lambda^{1/2}) * x   (inverse is 'N')\n"
    "    x := H(lambda^{-1/2}) * x  (inverse is 'I').\n\n"
    "H is the Hessian of the logarithmic barrier.";

static PyObject* scale2(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *lmbda, *x;
    PyObject *dims, *O, *Ok;
#if PY_MAJOR_VERSION >= 3
    int inverse = 'N';
#else
    char inverse = 'N';
#endif
    double a, lx, x0, b, *c = NULL, *sql = NULL;
    int m = 0, mk, i, j, len, int0 = 0, int1 = 1, maxn = 0, ind2;
    char *kwlist[] = {"lmbda", "x", "dims", "mnl", "inverse", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|iC", kwlist, &lmbda,
        &x, &dims, &m, &inverse)) return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ic", kwlist, &lmbda,
        &x, &dims, &m, &inverse)) return NULL;
#endif


    /*
     * For nonlinear and 'l' blocks:
     *
     *     xk := xk ./ l  (invers is 'N')
     *     xk := xk .* l  (invers is 'I')
     *
     * where l is the first mnl + dims['l'] components of lmbda.
     */

    O = PyDict_GetItemString(dims, "l");
#if PY_MAJOR_VERSION >= 3
    m += (int) PyLong_AsLong(O);
#else
    m += (int) PyInt_AsLong(O);
#endif
    if (inverse == 'N')
        dtbsv_("L", "N", "N", &m, &int0, MAT_BUFD(lmbda), &int1,
             MAT_BUFD(x), &int1);
    else
        dtbmv_("L", "N", "N", &m, &int0, MAT_BUFD(lmbda), &int1,
             MAT_BUFD(x), &int1);


    /*
     * For 'q' blocks, if inverse is 'N',
     *
     *     xk := 1/a * [ l'*J*xk;
     *         xk[1:] - (xk[0] + l'*J*xk) / (l[0] + 1) * l[1:] ].
     *
     *  If inverse is 'I',
     *
     *     xk := a * [ l'*xk;
     *         xk[1:] + (xk[0] + l'*xk) / (l[0] + 1) * l[1:] ].
     *
     * a = sqrt(lambda_k' * J * lambda_k), l = lambda_k / a.
     */

    O = PyDict_GetItemString(dims, "q");
    for (i = 0; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        mk = (int) PyLong_AsLong(Ok);
#else
        mk = (int) PyInt_AsLong(Ok);
#endif
        len = mk - 1;
        a = dnrm2_(&len, MAT_BUFD(lmbda) + m + 1, &int1);
        a = sqrt(MAT_BUFD(lmbda)[m] + a) * sqrt(MAT_BUFD(lmbda)[m] - a);
        if (inverse == 'N')
            lx = ( MAT_BUFD(lmbda)[m] * MAT_BUFD(x)[m] -
                ddot_(&len, MAT_BUFD(lmbda) + m + 1, &int1, MAT_BUFD(x) + m
                    + 1, &int1) ) / a;
        else
            lx = ddot_(&mk, MAT_BUFD(lmbda) + m, &int1, MAT_BUFD(x) + m,
                &int1) / a;
        x0 = MAT_BUFD(x)[m];
        MAT_BUFD(x)[m] = lx;
        b = (x0 + lx) / (MAT_BUFD(lmbda)[m]/a + 1.0) / a;
        if (inverse == 'N')  b *= -1.0;
        daxpy_(&len, &b, MAT_BUFD(lmbda) + m + 1, &int1,
            MAT_BUFD(x) + m + 1, &int1);
        if (inverse == 'N')  a = 1.0 / a;
        dscal_(&mk, &a, MAT_BUFD(x) + m, &int1);
        m += mk;
    }


    /*
     *  For the 's' blocks, if inverse is 'N',
     *
     *      xk := vec( diag(l)^{-1/2} * mat(xk) * diag(k)^{-1/2}).
     *
     *  If inverse is 'I',
     *
     *     xk := vec( diag(l)^{1/2} * mat(xk) * diag(k)^{1/2}).
     *
     * where l is kth block of lambda.
     *
     * We scale upper and lower triangular part of mat(xk) because the
     * inverse operation will be applied to nonsymmetric matrices.
     */

    O = PyDict_GetItemString(dims, "s");
    for (i = 0; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        maxn = MAX(maxn, (int) PyLong_AsLong(Ok));
#else
        maxn = MAX(maxn, (int) PyInt_AsLong(Ok));
#endif
    }
    if (!(c = (double *) calloc(maxn, sizeof(double))) ||
        !(sql = (double *) calloc(maxn, sizeof(double)))){
        free(c); free(sql);
        return PyErr_NoMemory();
    }
    ind2 = m;
    for (i = 0; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        mk = (int) PyLong_AsLong(Ok);
#else
        mk = (int) PyInt_AsLong(Ok);
#endif
        for (j = 0; j < mk; j++)
            sql[j] = sqrt(MAT_BUFD(lmbda)[ind2 + j]);
        for (j = 0; j < mk; j++){
            dcopy_(&mk, sql, &int1, c, &int1);
            b = sqrt(MAT_BUFD(lmbda)[ind2 + j]);
            dscal_(&mk, &b, c, &int1);
            if (inverse == 'N')
                dtbsv_("L", "N", "N", &mk, &int0, c, &int1, MAT_BUFD(x) +
                    m + j*mk, &int1);
            else
                dtbmv_("L", "N", "N", &mk, &int0, c, &int1, MAT_BUFD(x) +
                    m + j*mk, &int1);
        }
        m += mk*mk;
        ind2 += mk;
    }
    free(c); free(sql);

    return Py_BuildValue("");
}


static char doc_pack[] =
    "Copy x to y using packed storage.\n\n"
    "pack(x, y, dims, mnl = 0, offsetx = 0, offsety = 0)\n\n"
    "The vector x is an element of S, with the 's' components stored in\n"
    "unpacked storage.  On return, x is copied to y with the 's' \n"
    "components stored in packed storage and the off-diagonal entries \n"
    "scaled by sqrt(2).";

static PyObject* pack(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x, *y;
    PyObject *O, *Ok, *dims;
    double a;
    int i, k, nlq = 0, ox = 0, oy = 0, np, iu, ip, int1 = 1, len, n;
    char *kwlist[] = {"x", "y", "dims", "mnl", "offsetx", "offsety", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|iii", kwlist, &x,
        &y, &dims, &nlq, &ox, &oy)) return NULL;

    O = PyDict_GetItemString(dims, "l");
#if PY_MAJOR_VERSION >= 3
    nlq += (int) PyLong_AsLong(O);
#else
    nlq += (int) PyInt_AsLong(O);
#endif

    O = PyDict_GetItemString(dims, "q");
    for (i = 0; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        nlq += (int) PyLong_AsLong(Ok);
#else
        nlq += (int) PyInt_AsLong(Ok);
#endif
    }
    dcopy_(&nlq, MAT_BUFD(x) + ox, &int1, MAT_BUFD(y) + oy, &int1);

    O = PyDict_GetItemString(dims, "s");
    for (i = 0, np = 0, iu = ox + nlq, ip = oy + nlq; i < (int)
        PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        n = (int) PyLong_AsLong(Ok);
#else
        n = (int) PyInt_AsLong(Ok);
#endif
        for (k = 0; k < n; k++){
            len = n-k;
            dcopy_(&len, MAT_BUFD(x) + iu + k*(n+1), &int1,  MAT_BUFD(y) +
                ip, &int1);
            MAT_BUFD(y)[ip] /= sqrt(2.0);
            ip += len;
        }
        np += n*(n+1)/2;
        iu += n*n;
    }

    a = sqrt(2.0);
    dscal_(&np, &a, MAT_BUFD(y) + oy + nlq, &int1);

    return Py_BuildValue("");
}


static char doc_pack2[] =
    "In-place version of pack().\n\n"
    "pack2(x, dims, mnl = 0)\n\n"
    "In-place version of pack(), which also accepts matrix arguments x.\n"
    "The columns of x are elements of S, with the 's' components stored\n"
    "in unpacked storage.  On return, the 's' components are stored in\n"
    "packed storage and the off-diagonal entries are scaled by sqrt(2).";

static PyObject* pack2(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x;
    PyObject *O, *Ok, *dims;
    double a = sqrt(2.0), *wrk;
    int i, j, k, nlq = 0, iu, ip, len, n, maxn, xr, xc;
    char *kwlist[] = {"x", "dims", "mnl", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|i", kwlist, &x,
        &dims, &nlq)) return NULL;

    xr = x->nrows;
    xc = x->ncols;

    O = PyDict_GetItemString(dims, "l");
#if PY_MAJOR_VERSION >= 3
    nlq += (int) PyLong_AsLong(O);
#else
    nlq += (int) PyInt_AsLong(O);
#endif

    O = PyDict_GetItemString(dims, "q");
    for (i = 0; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        nlq += (int) PyLong_AsLong(Ok);
#else
        nlq += (int) PyInt_AsLong(Ok);
#endif
    }

    O = PyDict_GetItemString(dims, "s");
    for (i = 0, maxn = 0; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        maxn = MAX(maxn, (int) PyLong_AsLong(Ok));
#else
        maxn = MAX(maxn, (int) PyInt_AsLong(Ok));
#endif
    }
    if (!maxn) return Py_BuildValue("");
    if (!(wrk = (double *) calloc(maxn * xc, sizeof(double))))
        return PyErr_NoMemory();

    for (i = 0, iu = nlq, ip = nlq; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        n = (int) PyLong_AsLong(Ok);
#else
        n = (int) PyInt_AsLong(Ok);
#endif
        for (k = 0; k < n; k++){
            len = n-k;
            dlacpy_(" ", &len, &xc, MAT_BUFD(x) + iu + k*(n+1), &xr, wrk, 
                &maxn);
            for (j = 1; j < len; j++)
                dscal_(&xc, &a, wrk + j, &maxn);
            dlacpy_(" ", &len, &xc, wrk, &maxn, MAT_BUFD(x) + ip, &xr);
            ip += len;
        }
        iu += n*n;
    }

    free(wrk);
    return Py_BuildValue("");
}


static char doc_unpack[] =
    "Unpacks x into y.\n\n"
    "unpack(x, y, dims, mnl = 0, offsetx = 0, offsety = 0)\n\n"
    "The vector x is an element of S, with the 's' components stored in\n"
    "unpacked storage and off-diagonal entries scaled by sqrt(2).\n"
    "On return, x is copied to y with the 's' components stored in\n"
    "unpacked storage.";

static PyObject* unpack(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x, *y;
    PyObject *O, *Ok, *dims;
    double a = 1.0 / sqrt(2.0);
    int m = 0, ox = 0, oy = 0, int1 = 1, iu, ip, len, i, k, n;
    char *kwlist[] = {"x", "y", "dims", "mnl", "offsetx", "offsety", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|iii", kwlist, &x,
        &y, &dims, &m, &ox, &oy)) return NULL;

    O = PyDict_GetItemString(dims, "l");
#if PY_MAJOR_VERSION >= 3
    m += (int) PyLong_AsLong(O);
#else
    m += (int) PyInt_AsLong(O);
#endif

    O = PyDict_GetItemString(dims, "q");
    for (i = 0; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        m += (int) PyLong_AsLong(Ok);
#else
        m += (int) PyInt_AsLong(Ok);
#endif
    }
    dcopy_(&m, MAT_BUFD(x) + ox, &int1, MAT_BUFD(y) + oy, &int1);

    O = PyDict_GetItemString(dims, "s");
    for (i = 0, ip = ox + m, iu = oy + m; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        n = (int) PyLong_AsLong(Ok);
#else
        n = (int) PyInt_AsLong(Ok);
#endif
        for (k = 0; k < n; k++){
            len = n-k;
            dcopy_(&len, MAT_BUFD(x) + ip, &int1, MAT_BUFD(y) + iu +
                k*(n+1), &int1);
            ip += len;
            len -= 1;
            dscal_(&len, &a, MAT_BUFD(y) + iu + k*(n+1) + 1, &int1);
        }
        iu += n*n;
    }

    return Py_BuildValue("");
}


static char doc_symm[] =
    "Converts lower triangular matrix to symmetric.\n\n"
    "symm(x, n, offset = 0)\n\n"
    "Fills in the upper triangular part of the symmetric matrix stored\n"
    "in x[offset : offset+n*n] using 'L' storage.";

static PyObject* symm(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x;
    int n, ox = 0, k, len, int1 = 1;
    char *kwlist[] = {"x", "n", "offset", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "Oi|i", kwlist, &x, &n,
        &ox)) return NULL;

    if (n > 1) for (k = 0; k < n; k++){
        len = n-k-1;
        dcopy_(&len, MAT_BUFD(x) + ox + k*(n+1) + 1, &int1, MAT_BUFD(x) +
            ox + (k+1)*(n+1)-1, &n);
    }
    return Py_BuildValue("");
}


static char doc_sprod[] =
    "The product x := (y o x).\n\n"
    "sprod(x, y, dims, mnl = 0, diag = 'N')\n\n"
    "If diag is 'D', the 's' part of y is diagonal and only the diagonal\n"
    "is stored.";

static PyObject* sprod(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x, *y;
    PyObject *dims, *O, *Ok;
    int i, j, k, mk, len, maxn, ind = 0, ind2, int0 = 0, int1 = 1, ld;
    double a, *A = NULL, dbl2 = 0.5, dbl0 = 0.0;
#if PY_MAJOR_VERSION >= 3
    int diag = 'N';
#else
    char diag = 'N';
#endif
    char *kwlist[] = {"x", "y", "dims", "mnl", "diag", NULL};

#if PY_MAJOR_VERSION >= 3
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|iC", kwlist, &x, &y,
        &dims, &ind, &diag)) return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|ic", kwlist, &x, &y,
        &dims, &ind, &diag)) return NULL;
#endif


    /*
     * For nonlinear and 'l' blocks:
     *
     *     yk o xk = yk .* xk
     */

    O = PyDict_GetItemString(dims, "l");
#if PY_MAJOR_VERSION >= 3
    ind += (int) PyLong_AsLong(O);
#else
    ind += (int) PyInt_AsLong(O);
#endif
    dtbmv_("L", "N", "N", &ind, &int0, MAT_BUFD(y), &int1, MAT_BUFD(x),
        &int1);


    /*
     * For 'q' blocks:
     *
     *                [ l0   l1'  ]
     *     yk o xk =  [           ] * xk
     *                [ l1   l0*I ]
     *
     * where yk = (l0, l1).
     */

    O = PyDict_GetItemString(dims, "q");
    for (i = 0; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        mk = (int) PyLong_AsLong(Ok);
#else
        mk = (int) PyInt_AsLong(Ok);
#endif
        a = ddot_(&mk, MAT_BUFD(y) + ind, &int1, MAT_BUFD(x) + ind, &int1);
        len = mk - 1;
        dscal_(&len, MAT_BUFD(y) + ind, MAT_BUFD(x) + ind + 1, &int1);
        daxpy_(&len, MAT_BUFD(x) + ind, MAT_BUFD(y) + ind + 1, &int1,
            MAT_BUFD(x) + ind + 1, &int1);
        MAT_BUFD(x)[ind] = a;
        ind += mk;
    }


    /*
     * For the 's' blocks:
     *
     *    yk o sk = .5 * ( Yk * mat(xk) + mat(xk) * Yk )
     *
     * where Yk = mat(yk) if diag is 'N' and Yk = diag(yk) if diag is 'D'.
     */

    O = PyDict_GetItemString(dims, "s");
    for (i = 0, maxn = 0; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        maxn = MAX(maxn, (int) PyLong_AsLong(Ok));
#else
        maxn = MAX(maxn, (int) PyInt_AsLong(Ok));
#endif
    }
    if (diag == 'N'){
        if (!(A = (double *) calloc(maxn * maxn, sizeof(double))))
            return PyErr_NoMemory();
        for (i = 0; i < (int) PyList_Size(O); ind += mk*mk, i++){
            Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
            mk = (int) PyLong_AsLong(Ok);
#else
            mk = (int) PyInt_AsLong(Ok);
#endif
            len = mk*mk;
            dcopy_(&len, MAT_BUFD(x) + ind, &int1, A, &int1);

            if (mk > 1) for (k = 0; k < mk; k++){
                len = mk - k - 1;
                dcopy_(&len, A + k*(mk+1) + 1, &int1, A + (k+1)*(mk+1)-1,
                    &mk);
                dcopy_(&len, MAT_BUFD(y) + ind + k*(mk+1) + 1, &int1,
                    MAT_BUFD(y) + ind + (k+1)*(mk+1)-1, &mk);
            }

            ld = MAX(1, mk);
            dsyr2k_("L", "N", &mk, &mk, &dbl2, A, &ld, MAT_BUFD(y) + ind,
                &ld, &dbl0, MAT_BUFD(x) + ind, &ld);
        }
    }
    else {
        if (!(A = (double *) calloc(maxn, sizeof(double))))
            return PyErr_NoMemory();
        for (i = 0, ind2 = ind; i < (int) PyList_Size(O); ind += mk*mk,
            ind2 += mk, i++){
            Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
            mk = (int) PyLong_AsLong(Ok);
#else
            mk = (int) PyInt_AsLong(Ok);
#endif
            for (k = 0; k < mk; k++){
                len = mk - k;
                dcopy_(&len, MAT_BUFD(y) + ind2 + k, &int1, A, &int1);
                for (j = 0; j < len; j++) A[j] += MAT_BUFD(y)[ind2 + k];
                dscal_(&len, &dbl2, A, &int1);
                dtbmv_("L", "N", "N", &len, &int0, A, &int1, MAT_BUFD(x) +
                    ind + k * (mk+1), &int1);
            }
        }
    }

    free(A);
    return Py_BuildValue("");
}


static char doc_sinv[] =
    "The inverse of the product x := (y o x) when the 's' components of \n"
    "y are diagonal.\n\n"
    "sinv(x, y, dims, mnl = 0)";

static PyObject* sinv(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x, *y;
    PyObject *dims, *O, *Ok;
    int i, j, k, mk, len, maxn, ind = 0, ind2, int0 = 0, int1 = 1;
    double a, c, d, alpha, *A = NULL, dbl2 = 0.5;
    char *kwlist[] = {"x", "y", "dims", "mnl", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|i", kwlist, &x, &y,
        &dims, &ind)) return NULL;


    /*
     * For nonlinear and 'l' blocks:
     *
     *     yk o\ xk = yk .\ xk
     */

    O = PyDict_GetItemString(dims, "l");
#if PY_MAJOR_VERSION >= 3
    ind += (int) PyLong_AsLong(O);
#else
    ind += (int) PyInt_AsLong(O);
#endif
    dtbsv_("L", "N", "N", &ind, &int0, MAT_BUFD(y), &int1, MAT_BUFD(x),
        &int1);


    /*
     * For 'q' blocks:
     *
     *                        [  l0   -l1'               ]
     *     yk o\ xk = 1/a^2 * [                          ] * xk
     *                        [ -l1    (a*I + l1*l1')/l0 ]
     *
     * where yk = (l0, l1) and a = l0^2 - l1'*l1.
     */

    O = PyDict_GetItemString(dims, "q");
    for (i = 0; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        mk = (int) PyLong_AsLong(Ok);
#else
        mk = (int) PyInt_AsLong(Ok);
#endif
        len = mk - 1;
        a = dnrm2_(&len, MAT_BUFD(y) + ind + 1, &int1);
        a = (MAT_BUFD(y)[ind] + a) * (MAT_BUFD(y)[ind] - a);
        c = MAT_BUFD(x)[ind];
        d = ddot_(&len, MAT_BUFD(x) + ind + 1, &int1,
            MAT_BUFD(y) + ind + 1, &int1);
        MAT_BUFD(x)[ind] = c * MAT_BUFD(y)[ind] - d;
        alpha = a / MAT_BUFD(y)[ind];
        dscal_(&len, &alpha, MAT_BUFD(x) + ind + 1, &int1);
        alpha = d / MAT_BUFD(y)[ind] - c;
        daxpy_(&len, &alpha, MAT_BUFD(y) + ind + 1, &int1, MAT_BUFD(x) +
            ind + 1, &int1);
        alpha = 1.0 / a;
        dscal_(&mk, &alpha, MAT_BUFD(x) + ind, &int1);
        ind += mk;
    }


    /*
     * For the 's' blocks:
     *
     *    yk o\ sk = xk ./ gamma
     *
     * where  gammaij = .5 * (yk_i + yk_j).
     */

    O = PyDict_GetItemString(dims, "s");
    for (i = 0, maxn = 0; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        maxn = MAX(maxn, (int) PyLong_AsLong(Ok));
#else
        maxn = MAX(maxn, (int) PyInt_AsLong(Ok));
#endif
    }
    if (!(A = (double *) calloc(maxn, sizeof(double))))
        return PyErr_NoMemory();
    for (i = 0, ind2 = ind; i < (int) PyList_Size(O); ind += mk*mk,
        ind2 += mk, i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        mk = (int) PyLong_AsLong(Ok);
#else
        mk = (int) PyInt_AsLong(Ok);
#endif
        for (k = 0; k < mk; k++){
            len = mk - k;
            dcopy_(&len, MAT_BUFD(y) + ind2 + k, &int1, A, &int1);
            for (j = 0; j < len; j++) A[j] += MAT_BUFD(y)[ind2 + k];
            dscal_(&len, &dbl2, A, &int1);
            dtbsv_("L", "N", "N", &len, &int0, A, &int1, MAT_BUFD(x) + ind
                + k * (mk+1), &int1);
        }
    }

    free(A);
    return Py_BuildValue("");
}



static char doc_trisc[] =
    "Sets the upper triangular part of the 's' components of x equal to\n"
    "zero and scales the strictly lower triangular part\n\n"
    "trisc(x, dims, offset = 0)";

static PyObject* trisc(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x;
    double dbl0 = 0.0, dbl2 = 2.0;
    int ox = 0, i, k, nk, len, int1 = 1;
    PyObject *dims, *O, *Ok;
    char *kwlist[] = {"x", "dims", "offset", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|i", kwlist, &x,
        &dims, &ox)) return NULL;

    O = PyDict_GetItemString(dims, "l");
#if PY_MAJOR_VERSION >= 3
    ox += (int) PyLong_AsLong(O);
#else
    ox += (int) PyInt_AsLong(O);
#endif

    O = PyDict_GetItemString(dims, "q");
    for (i = 0; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        ox += (int) PyLong_AsLong(Ok);
#else
        ox += (int) PyInt_AsLong(Ok);
#endif
    }

    O = PyDict_GetItemString(dims, "s");
    for (k = 0; k < (int) PyList_Size(O); k++){
        Ok = PyList_GetItem(O, (Py_ssize_t) k);
#if PY_MAJOR_VERSION >= 3
        nk = (int) PyLong_AsLong(Ok);
#else
        nk = (int) PyInt_AsLong(Ok);
#endif
        for (i = 1; i < nk; i++){
            len = nk - i;
            dscal_(&len, &dbl0, MAT_BUFD(x) + ox + i*(nk+1) - 1, &nk);
            dscal_(&len, &dbl2, MAT_BUFD(x) + ox + nk*(i-1) + i, &int1);
        }
        ox += nk*nk;
    }

    return Py_BuildValue("");
}


static char doc_triusc[] =
    "Scales the strictly lower triangular part of the 's' components of\n"
    "x by 0.5.\n\n"
    "triusc(x, dims, offset = 0)";

static PyObject* triusc(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x;
    double dbl5 = 0.5;
    int ox = 0, i, k, nk, len, int1 = 1;
    PyObject *dims, *O, *Ok;
    char *kwlist[] = {"x", "dims", "offset", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|i", kwlist, &x,
        &dims, &ox)) return NULL;

    O = PyDict_GetItemString(dims, "l");
#if PY_MAJOR_VERSION >= 3
    ox += (int) PyLong_AsLong(O);
#else
    ox += (int) PyInt_AsLong(O);
#endif

    O = PyDict_GetItemString(dims, "q");
    for (i = 0; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        ox += (int) PyLong_AsLong(Ok);
#else
        ox += (int) PyInt_AsLong(Ok);
#endif
    }

    O = PyDict_GetItemString(dims, "s");
    for (k = 0; k < (int) PyList_Size(O); k++){
        Ok = PyList_GetItem(O, (Py_ssize_t) k);
#if PY_MAJOR_VERSION >= 3
        nk = (int) PyLong_AsLong(Ok);
#else
        nk = (int) PyInt_AsLong(Ok);
#endif
        for (i = 1; i < nk; i++){
            len = nk - i;
            dscal_(&len, &dbl5, MAT_BUFD(x) + ox + nk*(i-1) + i, &int1);
        }
        ox += nk*nk;
    }

    return Py_BuildValue("");
}


static char doc_sdot[] =
    "Inner product of two vectors in S.\n\n"
    "sdot(x, y, dims, mnl= 0)";

static PyObject* sdot(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x, *y;
    int m = 0, int1 = 1, i, k, nk, inc, len;
    double a;
    PyObject *dims, *O, *Ok;
    char *kwlist[] = {"x", "y", "dims", "mnl", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|i", kwlist, &x, &y,
        &dims, &m)) return NULL;

    O = PyDict_GetItemString(dims, "l");
#if PY_MAJOR_VERSION >= 3
    m += (int) PyLong_AsLong(O);
#else
    m += (int) PyInt_AsLong(O);
#endif

    O = PyDict_GetItemString(dims, "q");
    for (i = 0; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        m += (int) PyLong_AsLong(Ok);
#else
        m += (int) PyInt_AsLong(Ok);
#endif
    }
    a = ddot_(&m, MAT_BUFD(x), &int1, MAT_BUFD(y), &int1);

    O = PyDict_GetItemString(dims, "s");
    for (k = 0; k < (int) PyList_Size(O); k++){
        Ok = PyList_GetItem(O, (Py_ssize_t) k);
#if PY_MAJOR_VERSION >= 3
        nk = (int) PyLong_AsLong(Ok);
#else
        nk = (int) PyInt_AsLong(Ok);
#endif
        inc = nk+1;
        a += ddot_(&nk, MAT_BUFD(x) + m, &inc, MAT_BUFD(y) + m, &inc);
        for (i = 1; i < nk; i++){
            len = nk - i;
            a += 2.0 * ddot_(&len, MAT_BUFD(x) + m + i, &inc,
                MAT_BUFD(y) + m + i, &inc);
        }
        m += nk*nk;
    }

    return Py_BuildValue("d", a);
}


static char doc_max_step[] =
    "Returns min {t | x + t*e >= 0}\n\n."
    "max_step(x, dims, mnl = 0, sigma = None)\n\n"
    "e is defined as follows\n\n"
    "- For the nonlinear and 'l' blocks: e is the vector of ones.\n"
    "- For the 'q' blocks: e is the first unit vector.\n"
    "- For the 's' blocks: e is the identity matrix.\n\n"
    "When called with the argument sigma, also returns the eigenvalues\n"
    "(in sigma) and the eigenvectors (in x) of the 's' components of x.\n";

static PyObject* max_step(PyObject *self, PyObject *args, PyObject *kwrds)
{
    matrix *x, *sigma = NULL;
    PyObject *dims, *O, *Ok;
    int i, mk, len, maxn, ind = 0, ind2, int1 = 1, ld, Ns = 0, info, lwork,
        *iwork = NULL, liwork, iwl, m;
    double t = -FLT_MAX, dbl0 = 0.0, *work = NULL, wl, *Q = NULL,
        *w = NULL;
    char *kwlist[] = {"x", "dims", "mnl", "sigma", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OO|iO", kwlist, &x,
        &dims, &ind, &sigma)) return NULL;

    O = PyDict_GetItemString(dims, "l");
#if PY_MAJOR_VERSION >= 3
    ind += (int) PyLong_AsLong(O);
#else
    ind += (int) PyInt_AsLong(O);
#endif
    for (i = 0; i < ind; i++) t = MAX(t, -MAT_BUFD(x)[i]);

    O = PyDict_GetItemString(dims, "q");
    for (i = 0; i < (int) PyList_Size(O); i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        mk = (int) PyLong_AsLong(Ok);
#else
        mk = (int) PyInt_AsLong(Ok);
#endif
        len = mk - 1;
        t = MAX(t, dnrm2_(&len, MAT_BUFD(x) + ind + 1, &int1) -
            MAT_BUFD(x)[ind]);
        ind += mk;
    }

    O = PyDict_GetItemString(dims, "s");
    Ns = (int) PyList_Size(O);
    for (i = 0, maxn = 0; i < Ns; i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        maxn = MAX(maxn, (int) PyLong_AsLong(Ok));
#else
        maxn = MAX(maxn, (int) PyInt_AsLong(Ok));
#endif
    }
    if (!maxn) return Py_BuildValue("d", (ind) ? t : 0.0);

    lwork = -1;
    liwork = -1;
    ld = MAX(1, maxn);
    if (sigma){
        dsyevd_("V", "L", &maxn, NULL, &ld, NULL, &wl, &lwork, &iwl,
            &liwork, &info);
    }
    else {
        if (!(Q = (double *) calloc(maxn * maxn, sizeof(double))) ||
            !(w = (double *) calloc(maxn, sizeof(double)))){
            free(Q); free(w);
            return PyErr_NoMemory();
        }
        dsyevr_("N", "I", "L", &maxn, NULL, &ld, &dbl0, &dbl0, &int1,
            &int1, &dbl0, &maxn, NULL, NULL, &int1, NULL, &wl, &lwork,
            &iwl, &liwork, &info);
    }
    lwork = (int) wl;
    liwork = iwl;
    if (!(work = (double *) calloc(lwork, sizeof(double))) ||
        (!(iwork = (int *) calloc(liwork, sizeof(int))))){
        free(Q);  free(w);  free(work); free(iwork);
        return PyErr_NoMemory();
    }
    for (i = 0, ind2 = 0; i < Ns; i++){
        Ok = PyList_GetItem(O, (Py_ssize_t) i);
#if PY_MAJOR_VERSION >= 3
        mk = (int) PyLong_AsLong(Ok);
#else
        mk = (int) PyInt_AsLong(Ok);
#endif
        if (mk){
            if (sigma){
                dsyevd_("V", "L", &mk, MAT_BUFD(x) + ind, &mk,
                    MAT_BUFD(sigma) + ind2, work, &lwork, iwork, &liwork,
                    &info);
                t = MAX(t, -MAT_BUFD(sigma)[ind2]);
            }
            else {
                len = mk*mk;
                dcopy_(&len, MAT_BUFD(x) + ind, &int1, Q, &int1);
                ld = MAX(1, mk);
                dsyevr_("N", "I", "L", &mk, Q, &mk, &dbl0, &dbl0, &int1,
                    &int1, &dbl0, &m, w, NULL, &int1, NULL, work, &lwork,
                    iwork, &liwork, &info);
                t = MAX(t, -w[0]);
            }
        }
        ind += mk*mk;
        ind2 += mk;
    }
    free(work);  free(iwork);  free(Q);  free(w);

    return Py_BuildValue("d", (ind) ? t : 0.0);
}

static PyMethodDef misc_solvers_functions[] = {
    {"scale", (PyCFunction) scale, METH_VARARGS|METH_KEYWORDS, doc_scale},
    {"scale2", (PyCFunction) scale2, METH_VARARGS|METH_KEYWORDS,
        doc_scale2},
    {"pack", (PyCFunction) pack, METH_VARARGS|METH_KEYWORDS, doc_pack},
    {"pack2", (PyCFunction) pack2, METH_VARARGS|METH_KEYWORDS, doc_pack2},
    {"unpack", (PyCFunction) unpack, METH_VARARGS|METH_KEYWORDS,
        doc_unpack},
    {"symm", (PyCFunction) symm, METH_VARARGS|METH_KEYWORDS, doc_symm},
    {"trisc", (PyCFunction) trisc, METH_VARARGS|METH_KEYWORDS, doc_trisc},
    {"triusc", (PyCFunction) triusc, METH_VARARGS|METH_KEYWORDS,
        doc_triusc},
    {"sdot", (PyCFunction) sdot, METH_VARARGS|METH_KEYWORDS, doc_sdot},
    {"sprod", (PyCFunction) sprod, METH_VARARGS|METH_KEYWORDS, doc_sprod},
    {"sinv", (PyCFunction) sinv, METH_VARARGS|METH_KEYWORDS, doc_sinv},
    {"max_step", (PyCFunction) max_step, METH_VARARGS|METH_KEYWORDS,
        doc_max_step},
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static PyModuleDef misc_solvers_module = {
    PyModuleDef_HEAD_INIT,
    "misc_solvers",
    misc_solvers__doc__,
    -1,
    misc_solvers_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_misc_solvers(void)
{
  PyObject *m;
  if (!(m = PyModule_Create(&misc_solvers_module))) return NULL;
  if (import_cvxopt() < 0) return NULL;
  return m;
}

#else

PyMODINIT_FUNC initmisc_solvers(void)
{
  PyObject *m;
  m = Py_InitModule3("cvxopt.misc_solvers", misc_solvers_functions,
      misc_solvers__doc__);
  if (import_cvxopt() < 0) return;
}

#endif
