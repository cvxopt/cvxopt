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

#ifndef __MISC__
#define __MISC__

#if PY_MAJOR_VERSION >= 3
#define PY_NUMBER(O) (PyLong_Check(O) || PyFloat_Check(O) || PyComplex_Check(O))
#else
#define PY_NUMBER(O) (PyInt_Check(O) || PyFloat_Check(O) || PyComplex_Check(O))
#endif

#ifndef NO_ANSI99_COMPLEX
typedef union {
  double d;
  int_t i;
#ifndef _MSC_VER
  double complex z;
#else
  _Dcomplex z;
#endif
} number;
#endif

#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

#define PY_ERR(E,str) { PyErr_SetString(E, str); return NULL; }
#define PY_ERR_INT(E,str) { PyErr_SetString(E, str); return -1; }
#define PY_ERR_TYPE(str) PY_ERR(PyExc_TypeError, str)

/* Python style cyclic wrap-around for indices*/
#define CWRAP(i,m) (i >= 0 ? i : m+i)
#define OUT_RNG(i, dim) (i < -dim || i >= dim)


#define VALID_TC_MAT(t) (t=='i' || t=='d' || t=='z')
#define VALID_TC_SP(t)  (t=='d' || t=='z')
#define TC2ID(c) (c=='i' ? 0 : (c=='d' ? 1 : 2))

#define X_ID(O)    (Matrix_Check(O) ? MAT_ID(O)    : SP_ID(O))
#define X_NROWS(O) (Matrix_Check(O) ? MAT_NROWS(O) : SP_NROWS(O))
#define X_NCOLS(O) (Matrix_Check(O) ? MAT_NCOLS(O) : SP_NCOLS(O))
#define X_Matrix_Check(O) (Matrix_Check(O) || SpMatrix_Check(O))

#if PY_MAJOR_VERSION >= 3
#define TypeCheck_Capsule(O,str,errstr) { \
    if (!PyCapsule_CheckExact(O)) PY_ERR(PyExc_TypeError, errstr); \
    const char *descr = PyCapsule_GetName(O);  \
    if (!descr || strcmp(descr,str)) PY_ERR(PyExc_TypeError,errstr); }
#else
#define TypeCheck_CObject(O,str,errstr) { \
    char *descr = PyCObject_GetDesc(O);   \
    if (!descr || strcmp(descr,str)) PY_ERR(PyExc_TypeError,errstr); }
#endif


#define len(x) (Matrix_Check(x) ? MAT_LGT(x) : SP_LGT(x))

#define err_mtrx(s) PY_ERR_TYPE(s " must be a matrix")

#define err_bool(s) PY_ERR_TYPE(s " must be True or False")

#define err_conflicting_ids { \
    PY_ERR_TYPE("conflicting types for matrix arguments"); }

#define err_invalid_id {PY_ERR_TYPE( \
	"matrix arguments must have type 'd' or 'z'") }

#define err_nz_int(s) PY_ERR_TYPE(s " must be a nonzero integer")

#define err_nn_int(s) PY_ERR_TYPE(s " must be a nonnegative integer")

#define err_buf_len(s) PY_ERR_TYPE("length of " s " is too small")

#define err_type(s) PY_ERR_TYPE("incompatible type for " s)

#define err_p_int(s) { \
    PY_ERR(PyExc_ValueError, s " must be a positive integer") }

#define err_char(s1,s2) { \
    PY_ERR(PyExc_ValueError, "possible values of " s1 " are: " s2) }

#define err_ld(s) { \
    PY_ERR(PyExc_ValueError, "illegal value of " s) }

#define err_int_mtrx(s) { \
    PY_ERR_TYPE(s " must be a matrix with typecode 'i'") }

#define err_dbl_mtrx(s) { \
    PY_ERR_TYPE(s " must be a matrix with typecode 'd'") }

#if PY_MAJOR_VERSION >= 3
#define err_CO(s) PY_ERR_TYPE(s " is not a Capsule")
#else
#define err_CO(s) PY_ERR_TYPE(s " is not a CObject")
#endif


#define err_msk_noparam "missing options dictionary"


#endif
