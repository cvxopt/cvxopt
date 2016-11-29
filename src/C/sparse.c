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

#define BASE_MODULE

#include "Python.h"
#include "cvxopt.h"
#include "misc.h"

#include <complexobject.h>

#define CONJ(flag, val) (flag == 'C' ? conj(val) : val)

extern const int  E_SIZE[];
extern const char TC_CHAR[][2];
extern number One[3], MinusOne[3], Zero[3];
extern int intOne;

extern void (*write_num[])(void *, int, void *, int) ;
extern int (*convert_num[])(void *, void *, int, int_t) ;
extern PyObject * (*num2PyObject[])(void *, int) ;
extern void (*scal[])(int *, number *, void *, int *) ;
extern void (*axpy[])(int *, void *, void *, int *, void *, int *) ;
extern void (*gemm[])(char *, char *, int *, int *, int *, void *,
    void *, int *, void *, int *, void *, void *, int *) ;
extern int (*div_array[])(void *, number, int) ;
extern int get_id(void *, int ) ;

extern PyTypeObject matrix_tp ;
extern matrix * Matrix_NewFromMatrix(matrix *, int) ;
extern matrix * Matrix_NewFromSequence(PyObject *, int) ;
extern matrix * Matrix_NewFromPyBuffer(PyObject *, int, int *) ;
extern matrix * Matrix_NewFromNumber(int , int , int , void *, int ) ;
extern matrix * create_indexlist(int, PyObject *) ;
extern matrix * Matrix_New(int, int, int) ;
extern matrix * dense(spmatrix *) ;
extern PyObject * matrix_add(PyObject *, PyObject *) ;
extern PyObject * matrix_sub(PyObject *, PyObject *) ;
extern void * convert_mtx_alloc(matrix *, int) ;

PyTypeObject spmatrix_tp ;
spmatrix * SpMatrix_New(int_t, int_t, int_t, int ) ;

extern void (*scal[])(int *, number *, void *, int *) ;
extern void (*gemm[])(char *, char *, int *, int *, int *, void *, void *,
    int *, void *, int *, void *, void *, int *) ;
extern void (*syrk[])(char *, char *, int *, int *, void *, void *,
    int *, void *, void *, int *) ;

static int sp_daxpy(number, void *, void *, int, int, int, void **) ;
static int sp_zaxpy(number, void *, void *, int, int, int, void **) ;
int (*sp_axpy[])(number, void *, void *, int, int, int, void **)
= { NULL, sp_daxpy, sp_zaxpy };

static int sp_dgemm(char, char, number, void *, void *, number, void *,
    int, int, int, int, void **, int, int, int) ;
static int sp_zgemm(char, char, number, void *, void *, number, void *,
    int, int, int, int, void **, int, int, int) ;
int (*sp_gemm[])(char, char, number, void *, void *, number, void *,
    int, int, int, int, void **, int, int, int)
    = { NULL, sp_dgemm, sp_zgemm };

static int sp_dgemv(char, int, int, number, void *, int,
    void *, int, number, void *, int) ;
static int sp_zgemv(char, int, int, number, void *, int,
    void *, int, number, void *, int) ;
int (*sp_gemv[])(char, int, int, number, void *, int, void *, int,
    number, void *, int) = { NULL, sp_dgemv, sp_zgemv } ;

static int sp_dsymv(char, int, number, ccs *, int, void *, int,
    number, void *, int) ;
static int sp_zsymv(char, int, number, ccs *, int, void *, int,
    number, void *, int) ;
int (*sp_symv[])(char, int, number, ccs *, int, void *, int,
    number, void *, int) = { NULL, sp_dsymv, sp_zsymv } ;

static int sp_dsyrk(char, char, number, void *, number,
    void *, int, int, int, int, void **) ;
int (*sp_syrk[])(char, char, number, void *, number,
    void *, int, int, int, int, void **) = { NULL, sp_dsyrk, NULL };

typedef struct {
  int_t key, value;
} int_list;

static int comp_int(const void *x, const void *y) {
  if (((int_list *)x)->key == ((int_list *)y)->key)
    return 0;
  else
    return (((int_list *)x)->key > ((int_list *)y)->key ? 1 : -1);
}

typedef struct {
  int_t key; double value;
} double_list;

static int comp_double(const void *x, const void *y) {
  if (((double_list *)x)->key == ((double_list *)y)->key)
    return 0;
  else
    return (((double_list *)x)->key > ((double_list *)y)->key ? 1 : -1);
}

typedef struct {
  int_t key; double complex value;
} complex_list;

static int comp_complex(const void *x, const void *y) {
  if (((complex_list *)x)->key == ((complex_list *)y)->key)
    return 0;
  else
    return (((complex_list *)x)->key > ((complex_list *)y)->key ? 1 : -1);
}

#define spmatrix_getitem_i(O,i,v) \
    spmatrix_getitem_ij(O,i%SP_NROWS(O),i/SP_NROWS(O),v)
#define spmatrix_setitem_i(O,i,v) \
    spmatrix_setitem_ij(O,i%SP_NROWS(O),i/SP_NROWS(O),v)

#define free_lists_exit(argI,argJ,I,J,ret) { \
    if (argI && !Matrix_Check(argI)) { Py_XDECREF(I); } \
    if (argJ && !Matrix_Check(argJ)) { Py_XDECREF(J); } \
    return ret; }

#define INCR_SP_ROW_IDX(O, j, k) {		\
    while (((ccs *)O)->colptr[j+1] == k+1) j++;	\
    k++; }

int
convert_array(void *dest, void *src, int dest_id, int src_id, int n) {

  if (dest_id != MAX(dest_id,src_id))
    return -1;

  int i;
  if (dest_id == src_id)
    memcpy(dest, src, n*E_SIZE[dest_id]);
  else if (dest_id == DOUBLE) {
    for (i=0; i<n; i++)
      ((double *)dest)[i] = ((int *)src)[i];
  } else {
    if (src_id == INT) {
      for (i=0; i<n; i++)
        ((double complex *)dest)[i] = ((int *)src)[i];
    } else {
      for (i=0; i<n; i++)
        ((double complex *)dest)[i] = ((double *)src)[i];
    }
  }
  return 0;
}

ccs * alloc_ccs(int_t nrows, int_t ncols, int_t nnz, int id)
{
  ccs *obj = malloc(sizeof(ccs));
  if (!obj) return NULL;

  obj->nrows = nrows;
  obj->ncols = ncols;
  obj->id = id;

  obj->values = malloc(E_SIZE[id]*nnz);
  obj->colptr = calloc(ncols+1,sizeof(int_t));
  obj->rowind = malloc(sizeof(int_t)*nnz);

  if (!obj->values || !obj->colptr || !obj->rowind) {
    free(obj->values); free(obj->colptr); free(obj->rowind); free(obj);
    return NULL;
  }

  return obj;
}

void free_ccs(ccs *obj) {
  free(obj->values);
  free(obj->rowind);
  free(obj->colptr);
  free(obj);
}

static int
realloc_ccs(ccs *obj, int_t nnz) {

  int_t *rowind;
  void *values;

  if ((rowind = realloc(obj->rowind, nnz*sizeof(int_t))))
    obj->rowind = rowind;
  else
    return 0;

  if ((values = realloc(obj->values, nnz*E_SIZE[obj->id])))
    obj->values = values;
  else
    return 0;

  return 1;
}

static ccs * convert_ccs(ccs *src, int id) {

  if (src->id == id) return src;

  if (id != MAX(id,src->id)) PY_ERR_TYPE("incompatible matrix types");

  ccs *ret = alloc_ccs(src->nrows,src->ncols,CCS_NNZ(src),id);
  if (!ret) return (ccs *)PyErr_NoMemory();

  convert_array(ret->values, src->values, id, src->id, CCS_NNZ(src));
  memcpy(ret->rowind, src->rowind, CCS_NNZ(src)*sizeof(int_t));
  memcpy(ret->colptr, src->colptr, (src->ncols+1)*sizeof(int_t));
  return ret;
}

spmatrix *SpMatrix_NewFromMatrix(matrix *src, int id)
{
  spmatrix *A;
  int_t nnz = 0, i, j;

  if (id < MAT_ID(src)) PY_ERR_TYPE("illegal type conversion");

  for (j=0; j<MAT_NCOLS(src); j++) {
    for (i=0; i<MAT_NROWS(src); i++) {

      number *a = (number*)((unsigned char*)MAT_BUF(src) + (i+j*MAT_NROWS(src))*E_SIZE[MAT_ID(src)]);
      if (((MAT_ID(src) == INT) && (a->i != Zero[INT].i)) ||
          ((MAT_ID(src) == DOUBLE) && (a->d != Zero[DOUBLE].d)) ||
          ((MAT_ID(src) == COMPLEX) && (a->z != Zero[COMPLEX].z)))
        nnz++;
    }
  }

  if (!(A = SpMatrix_New(MAT_NROWS(src), MAT_NCOLS(src), (int_t)nnz, id)))
    return (spmatrix *)PyErr_NoMemory();

  int cnt = 0;
  for (j=0; j<MAT_NCOLS(src); j++) {
    for (i=0; i<MAT_NROWS(src); i++) {

      number a;
      convert_num[id](&a, src, 0, i+j*MAT_NROWS(src));
      if (((id == INT) && (a.i != Zero[INT].i)) ||
          ((id == DOUBLE) && (a.d != Zero[DOUBLE].d)) ||
          ((id == COMPLEX) && (a.z != Zero[COMPLEX].z))) {
        write_num[id](SP_VAL(A), cnt, &a, 0);
        SP_ROW(A)[cnt++] = i;
        SP_COL(A)[j+1]++;
      }
    }
  }

  for (i=0; i<SP_NCOLS(A); i++)
    SP_COL(A)[i+1] += SP_COL(A)[i];

  return A;
}

spmatrix * sparse_concat(PyObject *L, int id_arg)
{
  int id=0;
  int_t m=0, n=0, mk=0, nk=0, i=0, j, nnz=0;
  PyObject *col;

  int_t single_col = (PyList_GET_SIZE(L) > 0 &&
      !PyList_Check(PyList_GET_ITEM(L, 0)));

  for (j=0; j<(single_col ? 1 : PyList_GET_SIZE(L)); j++) {

    col = (single_col ? L : PyList_GET_ITEM(L, j));
    if (!PyList_Check(col))
      PY_ERR_TYPE("L must be a list of lists with matrices");

    mk = 0;
    for (i=0; i<PyList_GET_SIZE(col); i++) {
      PyObject *Lij = PyList_GET_ITEM(col, i);
      if (!Matrix_Check(Lij) && !SpMatrix_Check(Lij) && !PY_NUMBER(Lij))
        PY_ERR_TYPE("invalid type in list");

      int_t blk_nrows, blk_ncols;
      if (Matrix_Check(Lij) || SpMatrix_Check(Lij)) {
        blk_nrows = X_NROWS(Lij); blk_ncols = X_NCOLS(Lij);
        id = MAX(id, X_ID(Lij));
      } else {
        blk_nrows = 1; blk_ncols = 1;
        id = MAX(id, get_id(Lij,1));
      }

      if (Matrix_Check(Lij)) {

        int_t ik, jk;
        for (jk=0; jk<MAT_NCOLS(Lij); jk++) {
          for (ik=0; ik<MAT_NROWS(Lij); ik++) {

            number *a = (number*)((unsigned char*)MAT_BUF(Lij) +
				  (ik+jk*MAT_NROWS(Lij))*E_SIZE[MAT_ID(Lij)]);

            if (((MAT_ID(Lij) == INT) && (a->i != Zero[INT].i)) ||
                ((MAT_ID(Lij) == DOUBLE) && (a->d != Zero[DOUBLE].d)) ||
                ((MAT_ID(Lij) == COMPLEX) && (a->z != Zero[COMPLEX].z)))
              nnz++;
          }
        }
      } else if (SpMatrix_Check(Lij)) {
        int_t ik, jk;
        for (jk=0; jk<SP_NCOLS(Lij); jk++) {

          for (ik=SP_COL(Lij)[jk]; ik<SP_COL(Lij)[jk+1]; ik++) {
            if (((SP_ID(Lij) == DOUBLE) && (SP_VALD(Lij)[ik] != 0.0)) ||
                ((SP_ID(Lij) == COMPLEX) && (SP_VALZ(Lij)[ik] != 0.0)))
              nnz++;
          }
        }
      }
      else nnz += 1;

      if (i==0) {
        nk = blk_ncols; n += nk;
        mk = blk_nrows;
      } else {
        if (blk_ncols != nk)
          PY_ERR_TYPE("incompatible dimensions of subblocks");
        mk += blk_nrows;
      }
    }
    if (j==0)
      m = mk;
    else if (m != mk) PY_ERR_TYPE("incompatible dimensions of subblocks");
  }

  if ((id_arg >= 0) && (id_arg < id))
    PY_ERR_TYPE("illegal type conversion");

  id = MAX(DOUBLE, MAX(id, id_arg));
  spmatrix *A = SpMatrix_New(m, n, nnz, id);
  if (!A) return (spmatrix *)PyErr_NoMemory();

  int_t ik = 0, jk, cnt = 0;
  nk = 0;
  for (j=0; j<(single_col ? 1 : PyList_GET_SIZE(L)); j++) {
    col = (single_col ? L : PyList_GET_ITEM(L, j));

    if (PyList_GET_SIZE(col) > 0) {

      int_t tmp = (PY_NUMBER(PyList_GET_ITEM(col, 0)) ? 1 :
      X_NCOLS(PyList_GET_ITEM(col, 0)));

      for (jk=0; jk<tmp; jk++) {

        mk = 0;
        int_t blk_nrows = 0, blk_ncols = 0;
        for (i=0; i<PyList_GET_SIZE(col); i++) {

          PyObject *Lij = PyList_GET_ITEM(col, i);

          if (Matrix_Check(Lij) || SpMatrix_Check(Lij)) {
            blk_nrows = X_NROWS(Lij); blk_ncols = X_NCOLS(Lij);
          } else {
            blk_nrows = 1; blk_ncols = 1;
          }

          if (Matrix_Check(Lij)) {
            for (ik=0; ik<MAT_NROWS(Lij); ik++) {

              number a;
              convert_num[id](&a, Lij, 0, ik + jk*MAT_NROWS(Lij));

              if (((id == DOUBLE) && (a.d != Zero[DOUBLE].d)) ||
                  ((id == COMPLEX) && (a.z != Zero[COMPLEX].z))) {

                write_num[id](SP_VAL(A), cnt, &a, 0);
                SP_ROW(A)[cnt++] = mk + ik;
                SP_COL(A)[nk+1]++;
              }
            }
          } else if SpMatrix_Check(Lij) {

            int_t ik;
            for (ik=SP_COL(Lij)[jk]; ik<SP_COL(Lij)[jk+1]; ik++) {
              if ((SP_ID(Lij) == DOUBLE) && (SP_VALD(Lij)[ik] != 0.0)) {
                if (id == DOUBLE)
                  SP_VALD(A)[cnt] = SP_VALD(Lij)[ik];
                else
                  SP_VALZ(A)[cnt] = SP_VALD(Lij)[ik];

                SP_ROW(A)[cnt++] = mk + SP_ROW(Lij)[ik];
                SP_COL(A)[nk+1]++;
                nnz++;
              }
              else if ((SP_ID(Lij) == COMPLEX) && (SP_VALZ(Lij)[ik] != 0.0)) {

                SP_VALZ(A)[cnt] = SP_VALZ(Lij)[ik];
                SP_ROW(A)[cnt++] = mk + SP_ROW(Lij)[ik];
                SP_COL(A)[nk+1]++;
                nnz++;
              }
            }
          } else {

            number a;
            convert_num[id](&a,	Lij, 1, 0);

            if (((id == DOUBLE) && (a.d != Zero[DOUBLE].d)) ||
                ((id == COMPLEX) && (a.z != Zero[COMPLEX].z))) {

              write_num[id](SP_VAL(A), cnt, &a, 0);
              SP_ROW(A)[cnt++] = mk;
              SP_COL(A)[nk+1]++;
            }
          }
          mk += blk_nrows;
        }
        nk++;
      }
    }
  }

  for (i=0; i<n; i++)
    SP_COL(A)[i+1] += SP_COL(A)[i];

  return A;
}


static ccs * transpose(ccs *A, int conjugate) {

  ccs *B = alloc_ccs(A->ncols, A->nrows, CCS_NNZ(A), A->id);
  if (!B) return NULL;

  int_t i, j, *buf = calloc(A->nrows,sizeof(int_t));
  if (!buf) { free_ccs(B); return NULL; }

  /* Run through matrix and count number of elms in each row */
  for (i=0; i<A->colptr[A->ncols]; i++) buf[ A->rowind[i] ]++;

  /* generate new colptr */
  for (i=0; i<B->ncols; i++)
    B->colptr[i+1] = B->colptr[i] + buf[i];

  /* fill in rowind and values */
  for (i=0; i<A->nrows; i++) buf[i] = 0;
  for (i=0; i<A->ncols; i++) {
    if (A->id == DOUBLE)
      for (j=A->colptr[i]; j<A->colptr[i+1]; j++) {
        B->rowind[ B->colptr[A->rowind[j]] + buf[A->rowind[j]] ] = i;
        ((double *)B->values)[B->colptr[A->rowind[j]] + buf[A->rowind[j]]++] =
            ((double *)A->values)[j];
      }
    else
      for (j=A->colptr[i]; j<A->colptr[i+1]; j++) {
        B->rowind[ B->colptr[A->rowind[j]] + buf[A->rowind[j]] ] = i;
        ((double complex *)B->values)[B->colptr[A->rowind[j]]+buf[A->rowind[j]]++] =
            (conjugate ? conj(((double complex *)A->values)[j]) :
              ((double complex *)A->values)[j]);
      }
  }
  free(buf);
  return B;
}

static int sort_ccs(ccs *A) {

  ccs *t = transpose(A, 0);
  if (!t) return -1;

  ccs *t2 = transpose(t, 0);
  if (!t2) {
    free_ccs(t); return -1;
  }

  free(A->colptr); free(A->rowind); free(A->values);
  A->colptr = t2->colptr; A->rowind = t2->rowind; A->values = t2->values;

  free(t2);
  free_ccs(t);

  return 0;
}

/*
   Sparse accumulator (spa) - dense representation of a sparse vector
 */
typedef struct {
  void *val;
  char *nz;
  int *idx;
  int nnz, n, id;
} spa;

static spa * alloc_spa(int_t n, int id) {

  spa *s = malloc(sizeof(spa));

  if (s) {
    s->val = malloc( E_SIZE[id]*n );
    s->nz  = malloc( n*sizeof(char) );
    s->idx = malloc( n*sizeof(int) );
    s->nnz = 0;
    s->n = n;
    s->id = id;
  }

  if (!s || !s->val || !s->nz || !s->idx) {
    free(s->val); free(s->nz); free(s->idx); free(s);
    return NULL;
  }

  int_t i;
  for (i=0; i<n; i++) s->nz[i] = 0;

  return s;
}

static void free_spa(spa *s) {
  if (s) {
    free(s->val); free(s->nz); free(s->idx); free(s);
  }
}

static void init_spa(spa *s, ccs *X, int col) {
  int_t i;
  for (i=0; i<s->nnz; i++)
    s->nz[s->idx[i]] = 0;

  s->nnz = 0;

  if (X && X->id == DOUBLE) {
    for (i=X->colptr[col]; i<X->colptr[col+1]; i++) {
      s->nz[X->rowind[i]] = 1;
      ((double *)s->val)[X->rowind[i]] = ((double *)X->values)[i];
      s->idx[s->nnz++] = X->rowind[i];
    }
  } else if (X && X->id == COMPLEX) {
    for (i=X->colptr[col]; i<X->colptr[col+1]; i++) {
      s->nz[X->rowind[i]] = 1;
      ((double complex *)s->val)[X->rowind[i]] = ((double complex *)X->values)[i];
      s->idx[s->nnz++] = X->rowind[i];
    }
  }
}

static inline void
spa_daxpy_partial (double a, ccs *X, int col, spa *y) {
  int i;

  for (i=X->colptr[col]; i<X->colptr[col+1]; i++) {
    if (y->nz[X->rowind[i]]) {
      ((double *)y->val)[X->rowind[i]] += a*((double *)X->values)[i];
    }
  }
}

static inline void
spa_zaxpy_partial (double complex a, ccs *X, int col, spa *y) {
  int i;

  for (i=X->colptr[col]; i<X->colptr[col+1]; i++) {
    if (y->nz[X->rowind[i]]) {
      ((double complex *)y->val)[X->rowind[i]] += a*((double complex *)X->values)[i];
    }
  }
}

static inline void spa_daxpy (double a, ccs *X, int col, spa *y) {
  int i;

  for (i=X->colptr[col]; i<X->colptr[col+1]; i++) {
    if (y->nz[X->rowind[i]]) {
      ((double *)y->val)[X->rowind[i]] += a*((double *)X->values)[i];
    }
    else {
      ((double *)y->val)[X->rowind[i]] = a*((double *)X->values)[i];
      y->nz[X->rowind[i]] = 1;
      y->idx[y->nnz++] = X->rowind[i];
    }
  }
}

static inline void spa_zaxpy (double complex a, ccs *X, char conjx, int col, spa *y)
{
  int i;

  for (i=X->colptr[col]; i<X->colptr[col+1]; i++) {
    if (y->nz[X->rowind[i]]) {
      ((double complex *)y->val)[X->rowind[i]] +=
          a*CONJ(conjx,((double complex *)X->values)[i]);
    }
    else {
      ((double complex *)y->val)[X->rowind[i]] =
          a*CONJ(conjx,((double complex *)X->values)[i]);
      y->nz[X->rowind[i]] = 1;
      y->idx[y->nnz++] = X->rowind[i];
    }
  }
}

static inline void
spa_daxpy_uplo (double a, ccs *X, int col, spa *y, int j, char uplo) {
  int i;

  for (i=X->colptr[col]; i<X->colptr[col+1]; i++) {
    if ((uplo == 'U' && X->rowind[i] <= j) ||
        (uplo == 'L' && X->rowind[i] >= j)) {
      if (y->nz[X->rowind[i]]) {
        ((double *)y->val)[X->rowind[i]] += a*((double *)X->values)[i];
      }
      else {
        ((double *)y->val)[X->rowind[i]] = a*((double *)X->values)[i];
        y->nz[X->rowind[i]] = 1;
        y->idx[y->nnz++] = X->rowind[i];
      }
    }
  }
}

/* Unused function
static inline void
spa_zaxpy_uplo (double complex a, ccs *X, int col, spa *y, int j, char uplo) {
  int i;

  for (i=X->colptr[col]; i<X->colptr[col+1]; i++) {
    if ((uplo == 'U' && X->rowind[i] <= j) ||
        (uplo == 'L' && X->rowind[i] >= j)) {
      if (y->nz[X->rowind[i]]) {
        ((double complex *)y->val)[X->rowind[i]] += a*((double complex *)X->values)[i];
      }
      else {
        ((double complex *)y->val)[X->rowind[i]] = a*((double complex *)X->values)[i];
        y->nz[X->rowind[i]] = 1;
        y->idx[y->nnz++] = X->rowind[i];
      }
    }
  }
}
*/

static inline void spa_symb_axpy (ccs *X, int col, spa *y) {
  int i;
  for (i=X->colptr[col]; i<X->colptr[col+1]; i++)
    if (!y->nz[X->rowind[i]]) {
      y->nz[X->rowind[i]] = 1;
      y->idx[y->nnz++] = X->rowind[i];
    }
}

static inline void
spa_symb_axpy_uplo (ccs *X, int col, spa *y, int j, char uplo) {
  int i;
  for (i=X->colptr[col]; i<X->colptr[col+1]; i++)
    if ((uplo == 'U' && X->rowind[i] <= j) ||
        (uplo == 'L' && X->rowind[i] >= j)) {

      if (!y->nz[X->rowind[i]]) {
        y->nz[X->rowind[i]] = 1;
        y->idx[y->nnz++] = X->rowind[i];
      }
    }
}

static inline double spa_ddot (ccs *X, int col, spa *y) {
  int i;
  double a = 0;
  for (i=X->colptr[col]; i<X->colptr[col+1]; i++)
    if (y->nz[X->rowind[i]])
      a += ((double *)X->values)[i]*((double *)y->val)[X->rowind[i]];

  return a;
}

static inline
double complex spa_zdot (ccs *X, int col, spa *y, char conjx, char conjy) {
  int i;
  double complex a = 0;
  for (i=X->colptr[col]; i<X->colptr[col+1]; i++)
    if (y->nz[X->rowind[i]])
      a += CONJ(conjx, ((double complex *)X->values)[i])*
      CONJ(conjy,((double complex *)y->val)[X->rowind[i]]);

  return a;
}

static void spa2compressed(spa *s, ccs *A, int col) {
  int i, k=0;

  switch (A->id) {
  case (DOUBLE):
    for (i=A->colptr[col]; i<A->colptr[col+1]; i++) {
      A->rowind[i] = s->idx[k];
      ((double *)A->values)[i] = ((double *)s->val)[s->idx[k++]];
    }
  break;
  case COMPLEX:
    for (i=A->colptr[col]; i<A->colptr[col+1]; i++) {
      A->rowind[i] = s->idx[k];
      ((double complex *)A->values)[i] = ((double complex *)s->val)[s->idx[k++]];
    }
    break;
  }
}

static int sp_daxpy(number a, void *x, void *y, int sp_x, int sp_y,
    int partial, void **z)
{
  int j, k;
  if (sp_x && !sp_y) {

    ccs *X = x;
    double *Y = y;

    for (j=0; j<X->ncols; j++) {
      for (k=X->colptr[j]; k<X->colptr[j+1]; k++)
        Y[X->rowind[k] + j*X->nrows] += a.d*(((double *)X->values)[k]);
    }
  }
  else if (sp_x && sp_y && partial) {

    ccs *X = x, *Y = y;
    spa *s = alloc_spa(X->nrows, DOUBLE);

    int n = X->ncols;

    for (j=0; j<n; j++) {

      init_spa(s, Y, j);
      spa_daxpy_partial(a.d, X, j, s);
      spa2compressed(s, Y, j);

    }
    free_spa(s);

  }
  else if (sp_x && sp_y && !partial) {

    ccs *X = x, *Y = y;
    spa *s = alloc_spa(X->nrows, DOUBLE);

    int m = X->nrows, n = X->ncols;
    ccs *Z = alloc_ccs(m, n, X->colptr[n]+Y->colptr[n], DOUBLE);
    if (!Z) return -1;

    for (j=0; j<n; j++) {

      init_spa(s, Y, j);
      spa_daxpy(a.d, X, j, s);
      Z->colptr[j+1] = Z->colptr[j] + s->nnz;
      spa2compressed(s, Z, j);

    }
    free_spa(s);

    Z->rowind = realloc(Z->rowind, Z->colptr[n]*sizeof(int_t));
    Z->values = realloc(Z->values, Z->colptr[n]*sizeof(double));

    ccs *Zt = transpose(Z, 0);
    free_ccs(Z);
    if (!Zt) return -1;

    *z = transpose(Zt, 0);
    free_ccs(Zt);
    if (!(*z)) return -1;

  }
  else if (!sp_x && sp_y && partial) {

    double *X = x;
    ccs *Y = y;
    int kY, jY;

    for (jY=0; jY<Y->ncols; jY++) {
      for (kY=Y->colptr[jY]; kY<Y->colptr[jY+1]; kY++)
        ((double *)Y->values)[kY] += a.d*X[jY*Y->nrows + Y->rowind[kY]];
    }
  }
  else { // if (!sp_x && !sp_y) {

    double *X = x;
    ccs *Y = y;

    int_t mn = Y->nrows*Y->ncols;
    ccs *Z = alloc_ccs(Y->nrows, Y->ncols, mn, Y->id);
    if (!Z) return -1;

    memcpy(Z->values, X, sizeof(double)*mn);

    int mn_int = (int) mn; 
    scal[Y->id](&mn_int, &a, Z->values, &intOne);

    int j, k;
    for (j=0; j<Y->ncols; j++) {

      Z->colptr[j+1] = Z->colptr[j] + Y->nrows;

      for (k=0; k<Y->nrows; k++)
        Z->rowind[j*Y->nrows+k] = k;

      for (k=Y->colptr[j]; k<Y->colptr[j+1]; k++)
        ((double *)Z->values)[j*Y->nrows + Y->rowind[k]] +=
            ((double *)Y->values)[k];
    }
    *z = Z;
  }
  return 0;
}

static int sp_zaxpy(number a, void *x, void *y, int sp_x, int sp_y,
    int partial, void **z)
{
  int j, k;
  if (sp_x && !sp_y) {

    ccs *X = x;
    double complex *Y = y;

    for (j=0; j<X->ncols; j++) {
      for (k=X->colptr[j]; k<X->colptr[j+1]; k++)
        Y[X->rowind[k] + j*X->nrows] += a.z*(((double complex *)X->values)[k]);
    }
  }
  else if (sp_x && sp_y && partial) {

    ccs *X = x, *Y = y;
    spa *s = alloc_spa(X->nrows, COMPLEX);

    int n = X->ncols;

    for (j=0; j<n; j++) {

      init_spa(s, Y, j);
      spa_zaxpy_partial(a.z, X, j, s);
      spa2compressed(s, Y, j);

    }
    free_spa(s);
  }
  else if (sp_x && sp_y && !partial) {

    ccs *X = x, *Y = y;
    spa *s = alloc_spa(X->nrows, COMPLEX);

    int m = X->nrows, n = X->ncols;
    ccs *Z = alloc_ccs(m, n, X->colptr[n]+Y->colptr[n], COMPLEX);
    if (!Z) return -1;

    for (j=0; j<n; j++) {

      init_spa(s, Y, j);
      spa_zaxpy(a.z, X, 'N', j, s);
      Z->colptr[j+1] = Z->colptr[j] + s->nnz;
      spa2compressed(s, Z, j);

    }
    free_spa(s);

    Z->rowind = realloc(Z->rowind, Z->colptr[n]*sizeof(int_t));
    Z->values = realloc(Z->values, Z->colptr[n]*sizeof(double complex));

    ccs *Zt = transpose(Z, 0);
    free_ccs(Z);
    if (!Zt) return -1;

    *z = transpose(Zt, 0);
    free_ccs(Zt);
    if (!(*z)) return -1;

  }
  else if (!sp_x && sp_y && partial)
    {
    double complex *X = x;
    ccs *Y = y;
    int kY, jY;

    for (jY=0; jY<Y->ncols; jY++) {
      for (kY=Y->colptr[jY]; kY<Y->colptr[jY+1]; kY++)
        ((double complex *)Y->values)[kY] += a.z*X[jY*Y->nrows + Y->rowind[kY]];
    }
    }
  else { // if (!p_x && !sp_y) {

    double complex *X = x;
    ccs *Y = y;

    int_t mn = Y->nrows*Y->ncols;
    ccs *Z = alloc_ccs(Y->nrows, Y->ncols, mn, Y->id);
    if (!Z) return -1;

    memcpy(Z->values, X, sizeof(double complex)*mn);

    int mn_int = (int) mn;
    scal[Y->id](&mn_int, &a, Z->values, &intOne);

    int j, k;
    for (j=0; j<Y->ncols; j++) {

      Z->colptr[j+1] = Z->colptr[j] + Y->nrows;

      for (k=0; k<Y->nrows; k++)
        Z->rowind[j*Y->nrows+k] = k;

      for (k=Y->colptr[j]; k<Y->colptr[j+1]; k++)
        ((double complex *)Z->values)[j*Y->nrows + Y->rowind[k]] +=
            ((double complex *)Y->values)[k];
    }
    *z = Z;
  }
  return 0;
}

static int sp_dgemv(char tA, int m, int n, number alpha, void *a, int oA,
    void *x, int ix, number beta, void *y, int iy)
{
  ccs *A = a;
  double *X = x, *Y = y;

  scal[A->id]((tA == 'N' ? &m : &n), &beta, Y, &iy);

  if (!m) return 0;
  int i, j, k, oi = oA % A->nrows, oj = oA / A->nrows;

  if (tA == 'N') {
    for (j=oj; j<n+oj; j++) {
      for (k=A->colptr[j]; k<A->colptr[j+1]; k++)
        if ((A->rowind[k] >= oi) && (A->rowind[k] < oi+m))
          Y[iy*(A->rowind[k]-oi + (iy > 0 ? 0 : 1 - m))] +=
              alpha.d*((double *)A->values)[k]*
              X[ix*(j-oj + (ix > 0 ? 0 : 1 - n))];

    }
  } else {
    for (i=oj; i<oj+n; i++) {
      for (k=A->colptr[i]; k<A->colptr[i+1]; k++) {
        if ((A->rowind[k] >= oi) && (A->rowind[k] < oi+m))
          Y[iy*(i-oj + (iy > 0 ? 0 : 1 - n))] += alpha.d*
          ((double *)A->values)[k]*
          X[ix*(A->rowind[k]-oi + (ix > 0 ? 0 : 1 - m))];
      }
    }
  }
  return 0;
}

static int sp_zgemv(char tA, int m, int n, number alpha, void *a, int oA,
    void *x, int ix, number beta, void *y, int iy)
{
  ccs *A = a;
  double complex *X = x, *Y = y;

  scal[A->id]((tA == 'N' ? &m : &n), &beta, Y, &iy);

  if (!m) return 0;
  int i, j, k, oi = oA % A->nrows, oj = oA / A->nrows;

  if (tA == 'N') {
    for (j=oj; j<n+oj; j++) {
      for (k=A->colptr[j]; k<A->colptr[j+1]; k++)
        if ((A->rowind[k] >= oi) && (A->rowind[k] < oi+m))
          Y[iy*(A->rowind[k]-oi + (iy > 0 ? 0 : 1 - m))] +=
              alpha.z*((double complex *)A->values)[k]*
              X[ix*(j-oj + (ix > 0 ? 0 : 1 - n))];
    }
  } else {
    for (i=oj; i<oj+n; i++) {
      for (k=A->colptr[i]; k<A->colptr[i+1]; k++) {
        if ((A->rowind[k] >= oi) && (A->rowind[k] < oi+m))
          Y[iy*(i-oj + (iy > 0 ? 0 : 1 - n))] += alpha.z*
          CONJ(tA, ((double complex *)A->values)[k])*
          X[ix*(A->rowind[k]-oi + (ix > 0 ? 0 : 1 - m))];
      }
    }
  }
  return 0;
}


int sp_dsymv(char uplo, int n, number alpha, ccs *A, int oA, void *x, int ix,
    number beta, void *y, int iy)
{
  double *X = x, *Y = y;
  scal[A->id](&n, &beta, y, &iy);

  if (!n) return 0;
  int i, j, k, oi = oA % A->nrows, oj = oA / A->nrows;
  for (j=0; j<n; j++) {

    for (k = A->colptr[j+oj]; k < A->colptr[j+oj+1]; k++) {
      i = A->rowind[k] - oi;

      if ((i >= 0) && (i < n)) {
        if ((uplo == 'U') && (i > j))
          break;
        if ((uplo == 'U') && (i <= j)) {
          Y[iy*(i + (iy > 0 ? 0 : 1 - n))] += alpha.d*((double *)A->values)[k]*
              X[ix*(j + (ix > 0 ? 0 : 1-n))];
          if (i != j)
            Y[iy*(j + (iy > 0 ? 0 : 1-n))] += alpha.d*((double *)A->values)[k]*
            X[ix*(i + (ix > 0 ? 0 : 1-n))];
        }
        else if ((uplo == 'L') && (i >= j)) {
          Y[iy*(i + (iy > 0 ? 0 : 1-n))] += alpha.d*((double *)A->values)[k]*
              X[ix*(j + (ix > 0 ? 0 : 1-n))];
          if (i != j)
            Y[iy*(j + (iy > 0 ? 0 : 1-n))] += alpha.d*((double *)A->values)[k]*
            X[ix*(i + (ix > 0 ? 0 : 1-n))];
        }
      }
    }
  }
  return 0;
}

int sp_zsymv(char uplo, int n, number alpha, ccs *A, int oA, void *x, int ix,
    number beta, void *y, int iy)
{
  double complex *X = x, *Y = y;
  scal[A->id](&n, &beta, y, &iy);

  if (!n) return 0;
  int i, j, k, oi = oA % A->nrows, oj = oA / A->nrows;
  for (j=0; j<n; j++) {

    for (k = A->colptr[j+oj]; k < A->colptr[j+oj+1]; k++) {
      i = A->rowind[k] - oi;

      if ((i >= 0) && (i < n)) {
        if ((uplo == 'U') && (i > j))
          break;
        if ((uplo == 'U') && (i <= j)) {
          Y[iy*(i + (iy > 0 ? 0 : 1-n))] += alpha.z*((double complex *)A->values)[k]*
              X[ix*(j + (ix > 0 ? 0 : 1-n))];
          if (i != j)
            Y[iy*(j+(iy>0 ? 0 : 1-n))] += alpha.z*((double complex *)A->values)[k]*
            X[ix*(i + (ix > 0 ? 0 : 1-n))];
        }
        else if ((uplo == 'L') && (i >= j)) {
          Y[iy*(i + (iy > 0 ? 0 : 1-n))] += alpha.z*((double complex *)A->values)[k]*
              X[ix*(j + (ix > 0 ? 0 : 1-n))];
          if (i != j)
            Y[iy*(j+(iy > 0 ? 0 : 1-n))] += alpha.z*((double complex *)A->values)[k]*
            X[ix*(i + (ix > 0 ? 0 : 1-n))];
        }
      }
    }
  }
  return 0;
}

static int sp_dgemm(char tA, char tB, number alpha, void *a, void *b,
    number beta, void *c, int sp_a, int sp_b, int sp_c, int partial, 
    void **z, int m, int n, int k)
{

  if (sp_a && sp_b && sp_c && partial) {

    ccs *A = (tA == 'T' ? a : transpose(a, 0));
    ccs *B = (tB == 'N' ? b : transpose(b, 0));
    ccs *C = c;
    int j, l;

    spa *s = alloc_spa(A->nrows, A->id);
    if (!s) {
      if (A != a) free_ccs(A);
      return -1;
    }

    for (j=0; j<n; j++) {
      for (l=C->colptr[j]; l<C->colptr[j+1]; l++) {
        init_spa(s, A, C->rowind[l]);
        ((double *)C->values)[l] = alpha.d*spa_ddot (B, j, s) +
            beta.d*((double *)C->values)[l];
      }
    }
    free_spa(s);
    if (A != a) free_ccs(A);
    if (B != b) free_ccs(B);
  }

  else if (sp_a && sp_b && sp_c && !partial) {

    ccs *A = (tA == 'N' ? a : transpose(a, 0));
    ccs *B = (tB == 'N' ? b : transpose(b, 0));
    ccs *C = c;
    int_t *colptr_new = calloc(C->ncols+1,sizeof(int_t));

    spa *s = alloc_spa(A->nrows, A->id);
    if (!s || !colptr_new) {
      free(colptr_new);
      if (A != a) free_ccs(A);
      if (B != b) free_ccs(B);
      return -1;
    }

    int j, l;
    for (j=0; j<n; j++) {
      init_spa(s, NULL, 0);
      if (beta.d != 0.0)
        spa_symb_axpy (C, j, s);

      for (l=B->colptr[j]; l<B->colptr[j+1]; l++)
        spa_symb_axpy (A, B->rowind[l], s);

      colptr_new[j+1] = colptr_new[j] + s->nnz;
    }

    int_t nnz = colptr_new[n];
    ccs *Z = alloc_ccs(m, n, MAX(nnz,CCS_NNZ(C)), C->id);
    if (!Z) {
      if (A != a) free_ccs(A);
      if (B != b) free_ccs(B);
      free(colptr_new); free_spa(s);
      return -1;
    }
    free(Z->colptr); Z->colptr = colptr_new;

    for (j=0; j<n; j++) {
      init_spa(s, NULL, 0);
      if (beta.d != 0.0)
        spa_daxpy (beta.d, C, j, s);

      for (l=B->colptr[j]; l<B->colptr[j+1]; l++)
        spa_daxpy (alpha.d*((double *)B->values)[l], A, B->rowind[l], s);

      spa2compressed(s, Z, j);
    }

    free_spa(s);

    if (A != a) free_ccs(A);
    if (B != b) free_ccs(B);

    if (sort_ccs(Z)) {
      free_ccs(Z); return -1;
    }
    *z = Z;
  }
  else if (sp_a && sp_b && !sp_c) {

    ccs *A = (tA == 'N' ? a : transpose(a, 0));
    ccs *B = (tB == 'N' ? b : transpose(b, 0));
    double *C = c;

    spa *s = alloc_spa(A->nrows, A->id);
    if (!s) {
      if (A != a) free_ccs(A);
      if (B != b) free_ccs(B);
      return -1;
    }

    int mn = m*n;
    scal[A->id](&mn, &beta, C, &intOne);

    int j, l;
    for (j=0; j<n; j++) {
      init_spa(s, NULL, 0);

      for (l=B->colptr[j]; l<B->colptr[j+1]; l++)
        spa_daxpy (((double *)B->values)[l], A, B->rowind[l], s);

      for (l=0; l<s->nnz; l++)
        C[j*A->nrows + s->idx[l]] += alpha.d*((double *)s->val)[s->idx[l]];
    }
    free_spa(s);

    if (A != a) free_ccs(A);
    if (B != b) free_ccs(B);
  }

  else if (!sp_a && sp_b && !sp_c) {

    double *A = a, *C = c;
    ccs *B = (tB == 'N' ? b : transpose(b, 0));

    int j, l, mn_ = m*n;
    scal[DOUBLE](&mn_, &beta, C, &intOne);

    for (j=0; j<n; j++) {
      for (l=B->colptr[j]; l<B->colptr[j+1]; l++) {
        double a_ = alpha.d*((double *)B->values)[l];
        axpy[DOUBLE](&m, &a_, A + (tA=='N' ? B->rowind[l]*m : B->rowind[l]),
            (tA=='N' ? &intOne : &k), C + j*m, &intOne);
      }
    }
    if (B != b) free_ccs(B);
  }

  else if (sp_a && !sp_b && !sp_c) {

    ccs *A = (tA == 'N' ? a : transpose(a, 0));
    double *B = b, *C = c;

    int j, l, mn_ = m*n, ib = (tB == 'N' ? k : 1);
    scal[DOUBLE](&mn_, &beta, C, &intOne);

    for (j=0; j<A->ncols; j++) {
      for (l=A->colptr[j]; l<A->colptr[j+1]; l++) {

        double a_ = alpha.d*((double *)A->values)[l];
        axpy[DOUBLE](&n, &a_, B + (tB == 'N' ? j : j*n), &ib,
            C + A->rowind[l], &m);
      }
    }
    if (A != a) free_ccs(A);
  }

  else if (!sp_a && sp_b && sp_c && partial) {

    double *A = a, val;
    ccs *B = (tB == 'N' ? b : transpose(b, 0)), *C = c;
    int j, l, o;

    for (j=0; j<n; j++) {
      for (o=C->colptr[j]; o<C->colptr[j+1]; o++) {

        val = 0;
        for (l=B->colptr[j]; l < B->colptr[j+1]; l++)
          val += A[tA == 'N' ? C->rowind[o] + B->rowind[l]*m :
          B->rowind[l] + C->rowind[o]*B->nrows]*
          ((double *)B->values)[l];

        ((double *)C->values)[o] = alpha.d*val +
            beta.d*((double *)C->values)[o];
      }
    }
    if (B != b) free_ccs(B);
  }
  else if (!sp_a && sp_b && sp_c && !partial) {

    double *A = a;
    ccs *B = (tB == 'N' ? b : transpose(b,0)), *C = c;
    int_t *colptr_new = calloc(C->ncols+1,sizeof(int_t));

    if (!colptr_new) {
      if (B != b) free_ccs(B);
      free(colptr_new);
      return -1;
    }

    int j, l;
    for (j=0; j<n; j++)
      colptr_new[j+1] = colptr_new[j] +
      MAX(((B->colptr[j+1]-B->colptr[j])>0)*m, C->colptr[j+1]-C->colptr[j]);

    int_t nnz = colptr_new[n];
    ccs *Z = alloc_ccs(m, n, nnz, C->id);
    if (!Z) {
      if (B != b) free_ccs(B);
      free(colptr_new);
      return -1;
    }
    free(Z->colptr); Z->colptr = colptr_new;
    for (l=0; l<nnz; l++) ((double *)Z->values)[l] = 0;

    for (j=0; j<C->ncols; j++) {

      if (B->colptr[j+1]-B->colptr[j])
        for (l=0; l<m; l++)
          Z->rowind[Z->colptr[j]+l] = l;

      for (k=B->colptr[j]; k<B->colptr[j+1]; k++) {

        double a_ = alpha.d*((double *)B->values)[k];
        axpy[DOUBLE](&m, &a_, A +
            (tA=='N' ? B->rowind[k]*m : B->rowind[k]),
            (tA=='N' ? &intOne : &k),
            (double *)Z->values + Z->colptr[j], &intOne);
      }

      if (beta.d != 0.0) {
        if (Z->colptr[j+1]-Z->colptr[j] == m) {
          for (l=C->colptr[j]; l<C->colptr[j+1]; l++) {
            ((double *)Z->values)[Z->colptr[j]+C->rowind[l]] +=
                beta.d*((double *)C->values)[l];
          }
        }
        else {
          for (l=C->colptr[j]; l<C->colptr[j+1]; l++) {
            ((double *)Z->values)[Z->colptr[j]+l-C->colptr[j]] =
                beta.d*((double *)C->values)[l];
            Z->rowind[Z->colptr[j]+l-C->colptr[j]] = C->rowind[l];
          }
        }
      }
    }
    if (B != b) free_ccs(B);
    *z = Z;
  }
  else if (sp_a && !sp_b && sp_c && partial) {

    ccs *A = (tA == 'N' ? transpose(a,0) : a), *C = c;
    double *B = b, val;

    int j, l, o;
    for (j=0; j<n; j++) {
      for (o=C->colptr[j]; o<C->colptr[j+1]; o++) {

        val = 0;
        for (l=A->colptr[C->rowind[o]]; l < A->colptr[C->rowind[o]+1]; l++)
          val += ((double *)A->values)[l]*
          B[tB == 'N' ? j*A->nrows + A->rowind[l] :
          A->rowind[l]*C->ncols + j];

        ((double *)C->values)[o] = alpha.d*val +
            beta.d*((double *)C->values)[o];
      }
    }
    if (A != a) free_ccs(A);
  }
  else if (sp_a && !sp_b && sp_c && !partial) {

    ccs *A = (tA == 'N' ? a : transpose(a,0)), *C = c;
    double *B = b;

    spa *s = alloc_spa(A->nrows, A->id);
    int_t *colptr_new = calloc(n+1,sizeof(int_t));
    if (!s || !colptr_new) {
      free(s); free(colptr_new);
      if (A != a) free_ccs(A);
      return -1;
    }

    int j, l;
    for (j=0; j<n; j++) {
      init_spa(s, NULL, 0);

      for (l=0; l<A->ncols; l++)
        spa_symb_axpy(A, l, s);

      if (beta.d != 0.0) spa_symb_axpy(C, j, s);
      colptr_new[j+1] = colptr_new[j] + s->nnz;
    }

    int_t nnz = colptr_new[n];
    ccs *Z = alloc_ccs(m, n, nnz, C->id);
    if (!Z) {
      if (A != a) free_ccs(A);
      free_spa(s); free(colptr_new);
      return -1;
    }
    free(Z->colptr); Z->colptr = colptr_new;

    for (j=0; j<n; j++) {
      init_spa(s, NULL, 0);

      for (l=0; l<k; l++) {
        spa_daxpy(alpha.d*B[tB == 'N' ? l + j*k : j + l*n], A, l, s);
      }

      if (beta.d != 0.0) spa_daxpy(beta.d, C, j, s);
      spa2compressed(s, Z, j);
    }
    free_spa(s);
    if (A != a) free_ccs(A);
    if (sort_ccs(Z)) {
      free_ccs(Z); return -1;
    }
    *z = Z;
  }
  else if (!sp_a && !sp_b && sp_c && partial) {
    ccs *C = c;
    double *A = a, *B = b, val;

    int j, l, o;
    for (j=0; j<C->ncols; j++) {
      for (o=C->colptr[j]; o<C->colptr[j+1]; o++) {

        val = 0;
        for (l=0; l<k; l++)
          val += A[tA == 'N' ? m*l + C->rowind[o] : l + C->rowind[o]*k]*
          B[tB == 'N' ? j*k + l : l*n + j];

        ((double *)C->values)[o] = alpha.d*val +
            beta.d*((double *)C->values)[o];
      }
    }
  }
  else if (!sp_a && !sp_b && sp_c && !partial) {

    double *A = a, *B = b;
    ccs *C = c;

    ccs *Z = alloc_ccs(m, n, m*n, C->id);
    if (!Z) return -1;

    int j, l;
    for (j=0; j<m*n; j++) ((double *)Z->values)[j] = 0.0;

    int ldA = MAX(1, (tA == 'N' ? m : k));
    int ldB = MAX(1, (tB == 'N' ? k : n));
    int ldC = MAX(1, m);

    gemm[DOUBLE](&tA, &tB, &m, &n, &k, &alpha, A, &ldA,
        B, &ldB, &Zero[DOUBLE], Z->values, &ldC);

    for (j=0; j<n; j++) {
      for (l=0; l<m; l++)
        Z->rowind[j*m + l] = l;

      Z->colptr[j+1] = Z->colptr[j] + m;
    }

    if (beta.d != 0.0) {
      for (j=0; j<n; j++) {
        for (l=C->colptr[j]; l<C->colptr[j+1]; l++) {
          ((double *)Z->values)[j*m + C->rowind[l]] +=
              beta.d*((double *)C->values)[l];
        }
      }
    }
    *z = Z;
  }
  return 0;
}

static int sp_zgemm(char tA, char tB, number alpha, void *a, void *b,
    number beta, void *c, int sp_a, int sp_b, int sp_c, int partial, void **z,
    int m, int n, int k)
{

  if (sp_a && sp_b && sp_c && partial) {

    ccs *A = (tA == 'N' ? transpose(a, 0) : a);
    ccs *B = (tB == 'N' ? b : transpose(b, 0));
    ccs *C = c;
    int j, l;

    spa *s = alloc_spa(A->nrows, A->id);
    if (!s) {
      if (A != a) free_ccs(A);
      return -1;
    }

    for (j=0; j<n; j++) {
      for (l=C->colptr[j]; l<C->colptr[j+1]; l++) {
        init_spa(s, A, C->rowind[l]);
        ((double complex *)C->values)[l] = alpha.z*spa_zdot(B, j, s, tB, tA) +
            beta.z*((double complex *)C->values)[l];
      }
    }
    free_spa(s);
    if (A != a) free_ccs(A);
    if (B != b) free_ccs(B);
  }

  else if (sp_a && sp_b && sp_c && !partial) {

    ccs *A = (tA == 'N' ? a : transpose(a, 0));
    ccs *B = (tB == 'N' ? b : transpose(b, 0));
    ccs *C = c;
    int_t *colptr_new = calloc(C->ncols+1,sizeof(int_t));

    spa *s = alloc_spa(A->nrows, A->id);
    if (!s || !colptr_new) {
      free(colptr_new);
      if (A != a) free_ccs(A);
      if (B != b) free_ccs(B);
      return -1;
    }

    int j, l;
    for (j=0; j<n; j++) {
      init_spa(s, NULL, 0);
      if (beta.z != 0.0)
        spa_symb_axpy (C, j, s);

      for (l=B->colptr[j]; l<B->colptr[j+1]; l++)
        spa_symb_axpy (A, B->rowind[l], s);

      colptr_new[j+1] = colptr_new[j] + s->nnz;
    }

    int_t nnz = colptr_new[n];
    ccs *Z = alloc_ccs(m, n, nnz, A->id);
    if (!Z) {
      if (A != a) free_ccs(A);
      if (B != b) free_ccs(B);
      free(colptr_new); free_spa(s);
      return -1;
    }
    free(Z->colptr); Z->colptr = colptr_new;

    for (j=0; j<n; j++) {
      init_spa(s, NULL, 0);
      if (beta.z != 0.0)
        spa_zaxpy (beta.z, C, 'N', j, s);

      for (l=B->colptr[j]; l<B->colptr[j+1]; l++)
        spa_zaxpy(alpha.z*CONJ(tB, ((double complex *)B->values)[l]), A, tA, B->rowind[l], s);

      spa2compressed(s, Z, j);
    }

    free_spa(s);

    if (A != a) free_ccs(A);
    if (B != b) free_ccs(B);

    if (sort_ccs(Z)) {
      free_ccs(Z); return -1;
    }
    *z = Z;
  }
  else if (sp_a && sp_b && !sp_c) {

    ccs *A = (tA == 'N' ? a : transpose(a, 0));
    ccs *B = (tB == 'N' ? b : transpose(b, 0));
    double complex *C = c;

    spa *s = alloc_spa(A->nrows, A->id);
    if (!s) {
      if (A != a) free_ccs(A);
      if (B != b) free_ccs(B);
      return -1;
    }

    int mn = m*n;
    scal[A->id](&mn, &beta, C, &intOne);

    int j, l;
    for (j=0; j<n; j++) {
      init_spa(s, NULL, 0);

      for (l=B->colptr[j]; l<B->colptr[j+1]; l++)
        spa_zaxpy (CONJ(tB,((double complex *)B->values)[l]), A, tA, B->rowind[l], s);

      for (l=0; l<s->nnz; l++)
        C[j*A->nrows + s->idx[l]] += alpha.z*((double complex *)s->val)[s->idx[l]];
    }
    free_spa(s);

    if (A != a) free_ccs(A);
    if (B != b) free_ccs(B);
  }

  else if (!sp_a && sp_b && !sp_c) {

    double complex *A = a, *C = c;
    ccs *B = (tB == 'N' ? b : transpose(b, 0));

    int i, j, l, mn_ = m*n;
    scal[COMPLEX](&mn_, &beta, C, &intOne);

    for (j=0; j<n; j++) {
      for (l=B->colptr[j]; l<B->colptr[j+1]; l++) {

        for (i=0; i<m; i++)
          C[i+j*m] += alpha.z*CONJ(tA,A[tA=='N' ? i+B->rowind[l]*m :
          B->rowind[l]+i*k])*
          CONJ(tB,((double complex *)B->values)[l]);
      }
    }
    if (B != b) free_ccs(B);
  }

  else if (sp_a && !sp_b && !sp_c) {

    ccs *A = (tA == 'N' ? a : transpose(a, 0));
    double complex *B = b, *C = c;

    int i, j, l, mn_ = m*n;
    scal[COMPLEX](&mn_, &beta, C, &intOne);

    for (j=0; j<A->ncols; j++) {
      for (l=A->colptr[j]; l<A->colptr[j+1]; l++) {

        for (i=0; i<n; i++)
          C[A->rowind[l]+i*m] += alpha.z*CONJ(tA,((double complex *)A->values)[l])*
          CONJ(tB,B[tB=='N' ? j+i*k : i+j*n]);
      }
    }
    if (A != a) free_ccs(A);
  }

  else if (!sp_a && sp_b && sp_c && partial) {

    double complex *A = a, val;
    ccs *B = (tB == 'N' ? b : transpose(b, 0)), *C = c;
    int j, l, o;

    for (j=0; j<n; j++) {
      for (o=C->colptr[j]; o<C->colptr[j+1]; o++) {

        val = 0;
        for (l=B->colptr[j]; l < B->colptr[j+1]; l++)
          val += CONJ(tA,A[tA == 'N' ? C->rowind[o] + B->rowind[l]*m :
          B->rowind[l] + C->rowind[o]*B->nrows])*
          CONJ(tB,((double complex *)B->values)[l]);

        ((double complex *)C->values)[o] = alpha.z*val +
            beta.z*((double complex *)C->values)[o];
      }
    }
    if (B != b) free_ccs(B);
  }
  else if (!sp_a && sp_b && sp_c && !partial) {

    double complex *A = a;
    ccs *B = (tB == 'N' ? b : transpose(b,0)), *C = c;
    int_t *colptr_new = calloc(C->ncols+1,sizeof(int_t));

    if (!colptr_new) {
      if (B != b) free_ccs(B);
      free(colptr_new);
      return -1;
    }

    int i, j, l;
    for (j=0; j<n; j++)
      colptr_new[j+1] = colptr_new[j] +
      MAX(((B->colptr[j+1]-B->colptr[j])>0)*m, C->colptr[j+1]-C->colptr[j]);

    int_t nnz = colptr_new[n];
    ccs *Z = alloc_ccs(m, n, nnz, C->id);
    if (!Z) {
      if (B != b) free_ccs(B);
      free(colptr_new);
      return -1;
    }
    free(Z->colptr); Z->colptr = colptr_new;
    for (l=0; l<nnz; l++) ((double complex *)Z->values)[l] = 0;

    for (j=0; j<C->ncols; j++) {

      if (B->colptr[j+1]-B->colptr[j])
        for (l=0; l<m; l++)
          Z->rowind[Z->colptr[j]+l] = l;

      for (l=B->colptr[j]; l<B->colptr[j+1]; l++) {

        for (i=0; i<m; i++) {
          ((double complex*)Z->values)[Z->colptr[j]+i] += alpha.z*
              CONJ(tA,A[tA=='N' ? B->rowind[l]*m+i : B->rowind[l]+i*k])*
              CONJ(tB,((double complex *)B->values)[l]);

        }
      }

      if (beta.z != 0.0) {
        if (Z->colptr[j+1]-Z->colptr[j] == m) {
          for (l=C->colptr[j]; l<C->colptr[j+1]; l++) {
            ((double complex *)Z->values)[Z->colptr[j]+C->rowind[l]] +=
                beta.z*((double complex *)C->values)[l];
          }
        }
        else {
          for (l=C->colptr[j]; l<C->colptr[j+1]; l++) {
            ((double complex *)Z->values)[Z->colptr[j]+l-C->colptr[j]] =
                beta.z*((double complex *)C->values)[l];
            Z->rowind[Z->colptr[j]+l-C->colptr[j]] = C->rowind[l];
          }
        }
      }
    }
    if (B != b) free_ccs(B);
    *z = Z;
  }
  else if (sp_a && !sp_b && sp_c && partial) {

    ccs *A = (tA == 'N' ? transpose(a,0) : a), *C = c;
    double complex *B = b, val;

    int j, l, o;
    for (j=0; j<n; j++) {
      for (o=C->colptr[j]; o<C->colptr[j+1]; o++) {

        val = 0;
        for (l=A->colptr[C->rowind[o]]; l < A->colptr[C->rowind[o]+1]; l++)
          val += CONJ(tA, ((double complex *)A->values)[l])*
          CONJ(tB, B[tB == 'N' ? j*A->nrows + A->rowind[l] :
          A->rowind[l]*C->ncols + j]);

        ((double complex *)C->values)[o] = alpha.z*val +
            beta.z*((double complex *)C->values)[o];
      }
    }
    if (A != a) free_ccs(A);
  }
  else if (sp_a && !sp_b && sp_c && !partial) {

    ccs *A = (tA == 'N' ? a : transpose(a,0)), *C = c;
    double complex *B = b;

    spa *s = alloc_spa(A->nrows, A->id);
    int_t *colptr_new = calloc(n+1,sizeof(int_t));
    if (!s || !colptr_new) {
      free(s); free(colptr_new);
      if (A != a) free_ccs(A);
      return -1;
    }

    int j, l;
    for (j=0; j<n; j++) {
      init_spa(s, NULL, 0);

      for (l=0; l<A->ncols; l++)
        spa_symb_axpy(A, l, s);

      if (beta.z != 0.0) spa_symb_axpy(C, j, s);
      colptr_new[j+1] = colptr_new[j] + s->nnz;
    }

    int_t nnz = colptr_new[n];
    ccs *Z = alloc_ccs(m, n, nnz, C->id);
    if (!Z) {
      if (A != a) free_ccs(A);
      free_spa(s); free(colptr_new);
      return -1;
    }
    free(Z->colptr); Z->colptr = colptr_new;

    for (j=0; j<n; j++) {
      init_spa(s, NULL, 0);

      for (l=0; l<k; l++) {
        spa_zaxpy(alpha.z*CONJ(tB, B[tB == 'N' ? l + j*k : j + l*n]),
            A, tA, l, s);
      }

      if (beta.z != 0.0) spa_zaxpy(beta.z, C, 'N', j, s);
      spa2compressed(s, Z, j);
    }
    free_spa(s);
    if (A != a) free_ccs(A);
    if (sort_ccs(Z)) {
      free_ccs(Z); return -1;
    }
    *z = Z;
  }
  else if (!sp_a && !sp_b && sp_c && partial) {
    ccs *C = c;
    double complex *A = a, *B = b, val;

    int j, l, o;
    for (j=0; j<C->ncols; j++) {
      for (o=C->colptr[j]; o<C->colptr[j+1]; o++) {

        val = 0;
        for (l=0; l<k; l++)
          val += CONJ(tA, A[tA=='N' ? m*l+C->rowind[o] : l+C->rowind[o]*k])*
          CONJ(tB, B[tB == 'N' ? j*k + l : l*n + j]);

        ((double complex *)C->values)[o] = alpha.z*val +
            beta.z*((double complex *)C->values)[o];
      }
    }
  }
  else if (!sp_a && !sp_b && sp_c && !partial) {

    double complex *A = a, *B = b;
    ccs *C = c;

    ccs *Z = alloc_ccs(m, n, m*n, C->id);
    if (!Z) return -1;

    int j, l;
    for (j=0; j<m*n; j++) ((double complex *)Z->values)[j] = 0.0;

    int ldA = MAX(1, (tA == 'N' ? m : k));
    int ldB = MAX(1, (tB == 'N' ? k : n));
    int ldC = MAX(1, m);

    gemm[COMPLEX](&tA, &tB, &m, &n, &k, &alpha, A, &ldA,
        B, &ldB, &Zero[COMPLEX], Z->values, &ldC);

    for (j=0; j<n; j++) {
      for (l=0; l<m; l++)
        Z->rowind[j*m + l] = l;

      Z->colptr[j+1] = Z->colptr[j] + m;
    }

    if (beta.z != 0.0) {
      for (j=0; j<n; j++) {
        for (l=C->colptr[j]; l<C->colptr[j+1]; l++) {
          ((double complex *)Z->values)[j*m + C->rowind[l]] +=
              beta.z*((double complex *)C->values)[l];
        }
      }
    }
    *z = Z;
  }
  return 0;
}

static int sp_dsyrk(char uplo, char trans, number alpha, void *a,
    number beta, void *c, int sp_a, int sp_c, int partial, int k, void **z)
{
  if (sp_a && sp_c && partial) {

    ccs *A = (trans == 'N' ?  transpose(a, 0) : a), *C = c;
    spa *s = alloc_spa(A->nrows, C->id);
    if (!A || !s) {
      if (A != a) free_ccs(A);
      free_spa(s);
      return -1;
    }
    int j, k;
    for (j=0; j<C->ncols; j++) {
      for (k=C->colptr[j]; k<C->colptr[j+1]; k++) {
        if ((uplo == 'L' && C->rowind[k] >= j) ||
            (uplo == 'U' && C->rowind[k] <= j)) {
          init_spa(s, A, C->rowind[k]);
          ((double *)C->values)[k] = alpha.d*spa_ddot(A, j, s) +
              beta.d*((double *)C->values)[k];
        }
      }
    }
    free_spa(s);
    if (A != a) free_ccs(A);
  }
  else if (sp_a && sp_c && !partial) {

    ccs *A = (trans == 'N' ? a : transpose(a, 0));
    ccs *B = (trans == 'N' ? transpose(a, 0) : a);
    ccs *C = c;
    spa *s = alloc_spa(C->nrows, C->id);
    int_t *colptr_new = calloc(C->ncols+1,sizeof(int_t));

    if (!A || !B || !s || !colptr_new) {
      if (A != a) free_ccs(A);
      if (B != a) free_ccs(B);
      free_spa(s); free(colptr_new);
      return -1;
    }

    int j, k;
    for (j=0; j<B->ncols; j++) {
      init_spa(s, NULL, 0);
      if (beta.d != 0.0)
        spa_symb_axpy_uplo(C, j, s, j, uplo);

      for (k=B->colptr[j]; k<B->colptr[j+1]; k++)
        spa_symb_axpy_uplo(A, B->rowind[k], s, j, uplo);

      colptr_new[j+1] = colptr_new[j] + s->nnz;
    }

    int_t nnz = colptr_new[C->ncols];
    ccs *Z = alloc_ccs(C->nrows, C->ncols, nnz, C->id);
    if (!Z) {
      if (A != a) free_ccs(A);
      if (B != a) free_ccs(B);
      free_spa(s); free(colptr_new);
      return -1;
    }
    free(Z->colptr); Z->colptr = colptr_new;

    for (j=0; j<B->ncols; j++) {
      init_spa(s, NULL, 0);
      if (beta.d != 0.0)
        spa_daxpy_uplo(beta.d, C, j, s, j, uplo);

      for (k=B->colptr[j]; k<B->colptr[j+1]; k++) {
        spa_daxpy_uplo(alpha.d*((double *)B->values)[k], A, B->rowind[k],
            s, j, uplo);
      }
      spa2compressed(s, Z, j);
    }

    if (A != a) free_ccs(A);
    if (B != a) free_ccs(B);
    free_spa(s);

    if (sort_ccs(Z)) {
      free_ccs(Z); return -1;
    }
    *z = Z;
  }
  else if (sp_a && !sp_c) {

    int n  = (trans == 'N' ? ((ccs *)a)->nrows : ((ccs *)a)->ncols);
    ccs *A = (trans == 'N' ? a : transpose(a, 0));
    ccs *B = (trans == 'N' ? transpose(a, 0) : a);
    double *C = c;

    spa *s = alloc_spa(n, A->id);
    if (!A || !B || !s) {
      if (A != a) free_ccs(A);
      if (B != a) free_ccs(B);
      free_spa(s);
      return -1;
    }

    int j, k;
    for (j=0; j<B->ncols; j++) {
      init_spa(s, NULL, 0);

      for (k=B->colptr[j]; k<B->colptr[j+1]; k++) {
        spa_daxpy_uplo(alpha.d*((double *)B->values)[k], A, B->rowind[k],
            s, j, uplo);
      }

      if (uplo == 'U') {
        int m = j+1;
        scal[DOUBLE](&m, &beta, C + j*n, &intOne);

        for (k=0; k<s->nnz; k++) {
          if (s->idx[k] <= j)
            C[j*n + s->idx[k]] += alpha.d*((double *)s->val)[s->idx[k]];
        }
      } else {
        int m = n-j;
        scal[DOUBLE](&m, &beta, C + j*(n+1), &intOne);

        for (k=0; k<s->nnz; k++) {
          if (s->idx[k] >= j)
            C[j*n + s->idx[k]] += alpha.d*((double *)s->val)[s->idx[k]];
        }
      }
    }
    if (A != a) free_ccs(A);
    if (B != a) free_ccs(B);
    free_spa(s);
  }
  else if (!sp_a && sp_c && partial) {

    ccs *C = c;
    double *A = a;

    int j, i, l, n=C->nrows;
    for (j=0; j<n; j++) {
      for (i=C->colptr[j]; i<C->colptr[j+1]; i++) {
        if ((uplo == 'L' && C->rowind[i] >= j) ||
            (uplo == 'U' && C->rowind[i] <= j)) {

          ((double *)C->values)[i] *= beta.d;

          for (l=0; l<k; l++)
            ((double *)C->values)[i] +=
                alpha.d*A[trans == 'N' ? C->rowind[i]+l*n : l+C->rowind[i]*k]*
                A[trans == 'N' ? j+l*n : l+j*k];
        }
      }
    }
  }
  else if (!sp_a && sp_c) {

    ccs *C = c;
    double *A = a, *C_ = malloc( C->nrows*C->nrows*sizeof(double) );
    int_t *colptr_new = calloc(C->ncols+1,sizeof(int_t));

    if (!C_ || !colptr_new) {
      free(C_); free(colptr_new); return -1;
    }
    int j, i, n=C->nrows;
    for (j=0; j<n; j++)
      colptr_new[j+1] = colptr_new[j] + (uplo == 'U' ? j+1 : n-j);

    int_t nnz = colptr_new[n];
    ccs *Z = alloc_ccs(n, n, nnz, C->id);
    if (!Z) {
      free(C_); free(colptr_new);
      return -1;
    }
    free(Z->colptr); Z->colptr = colptr_new;

    syrk[DOUBLE](&uplo, &trans, &n, &k, &alpha, A,
        (trans == 'N' ? &n : &k), &Zero[DOUBLE], C_, &n);

    for (j=0; j<n; j++) {
      for (i=Z->colptr[j]; i<Z->colptr[j+1]; i++) {
        if (uplo == 'U') {
          ((double *)Z->values)[i] = C_[j*n + i-Z->colptr[j]];
          Z->rowind[i] = i-Z->colptr[j];
        } else {
          ((double *)Z->values)[i] = C_[j*(n+1) + i-Z->colptr[j]];
          Z->rowind[i] = j+i-Z->colptr[j];
        }
      }

      for (i=C->colptr[j]; i<C->colptr[j+1]; i++) {
        if (uplo == 'U' && C->rowind[i] <= j)
          ((double *)Z->values)[Z->colptr[j]+C->rowind[i]] +=
              beta.d*((double *)C->values)[i];
        else if (uplo == 'L' && C->rowind[i] >= j)
          ((double *)Z->values)[Z->colptr[j]+C->rowind[i]-j] +=
              beta.d*((double *)C->values)[i];
      }
    }
    free(C_);
    *z = Z;
  }

  return 0;
}

/* No error checking: Il and Jl must be valid indexlist,
   and Il,Jl,V must be of same length (V can also be NULL) */
static spmatrix *
triplet2dccs(matrix *Il, matrix *Jl, matrix *V,
    int_t nrows, int_t ncols)
    {
  spmatrix *ret = SpMatrix_New(nrows,ncols, MAT_LGT(Il), DOUBLE);
  double_list *dlist = malloc(MAT_LGT(Jl)*sizeof(double_list));
  int_t *colcnt = calloc(ncols,sizeof(int_t));

  if (!ret || !dlist || !colcnt) {
    Py_XDECREF(ret); free(dlist); free(colcnt);
    return (spmatrix *)PyErr_NoMemory();
  }

  /* build colptr */
  int_t i,j,k,l;
  for (j=0; j<ncols+1; j++) SP_COL(ret)[j] = 0;
  for (j=0; j<MAT_LGT(Jl); j++) {
    SP_COL(ret)[1+MAT_BUFI(Jl)[j]]++;
    dlist[j].key = -1;
  }

  for (j=0; j<ncols; j++) SP_COL(ret)[j+1] += SP_COL(ret)[j];

  /* build rowind and values */
  for (k=0; k<MAT_LGT(Il); k++) {
    i = MAT_BUFI(Il)[k], j = MAT_BUFI(Jl)[k];

    for (l=SP_COL(ret)[j]; l<SP_COL(ret)[j+1]; l++)
      if (dlist[l].key == i) {

        number n;
        if (V) {
          convert_num[DOUBLE](&n, V, 0, k);
          dlist[l].value += n.d;
        }

        goto skip;
      }

    if (V)
      convert_num[DOUBLE](&dlist[SP_COL(ret)[j]+colcnt[j]].value, V, 0, k);

    dlist[SP_COL(ret)[j] + colcnt[j]++].key = i;

    skip:
    ;
  }

  for (j=0; j<ncols; j++)
    qsort(&dlist[SP_COL(ret)[j]],colcnt[j],sizeof(double_list),&comp_double);

  int_t cnt = 0;
  for (j=0; j<ncols; j++) {
    for (i=0; i<colcnt[j]; i++) {

      SP_ROW(ret)[cnt] = dlist[i + SP_COL(ret)[j]].key;
      SP_VALD(ret)[cnt++] = dlist[i + SP_COL(ret)[j]].value;

    }
  }

  for (j=0; j<ncols; j++)
    SP_COL(ret)[j+1] = colcnt[j] + SP_COL(ret)[j];

  free(dlist); free(colcnt);
  return ret;
    }

static spmatrix *
triplet2zccs(matrix *Il, matrix *Jl, matrix *V,
    int_t nrows, int_t ncols)
    {
  spmatrix *ret = SpMatrix_New(nrows,ncols, MAT_LGT(Il), COMPLEX);
  complex_list *zlist = malloc(MAT_LGT(Jl)*sizeof(complex_list));
  int_t *colcnt = calloc(ncols,sizeof(int_t));

  if (!ret || !zlist || !colcnt) {
    Py_XDECREF(ret); free(zlist); free(colcnt);
    return (spmatrix *)PyErr_NoMemory();
  }

  /* build colptr */
  int_t i,j,k,l;
  for (j=0; j<ncols+1; j++) SP_COL(ret)[j] = 0;
  for (j=0; j<MAT_LGT(Jl); j++) {
    SP_COL(ret)[1+MAT_BUFI(Jl)[j]]++;
    zlist[j].key = -1;
  }

  for (j=0; j<ncols; j++) SP_COL(ret)[j+1] += SP_COL(ret)[j];

  /* build rowind and values */
  for (k=0; k<MAT_LGT(Il); k++) {
    i = MAT_BUFI(Il)[k], j = MAT_BUFI(Jl)[k];

    for (l=SP_COL(ret)[j]; l<SP_COL(ret)[j+1]; l++)
      if (zlist[l].key == i) {

        number n;
        if (V) {
          convert_num[COMPLEX](&n, V, 0, k);
          zlist[l].value += n.z;
        }

        goto skip;
      }

    if (V)
      convert_num[COMPLEX](&zlist[SP_COL(ret)[j]+colcnt[j]].value, V, 0, k);

    zlist[SP_COL(ret)[j] + colcnt[j]++].key = i;

    skip:
    ;
  }

  for (j=0; j<ncols; j++)
    qsort(&zlist[SP_COL(ret)[j]],colcnt[j],sizeof(complex_list),&comp_complex);

  int_t cnt = 0;
  for (j=0; j<ncols; j++) {
    for (i=0; i<colcnt[j]; i++) {

      SP_ROW(ret)[cnt] = zlist[i + SP_COL(ret)[j]].key;
      SP_VALZ(ret)[cnt++] = zlist[i + SP_COL(ret)[j]].value;

    }
  }

  for (j=0; j<ncols; j++)
    SP_COL(ret)[j+1] = colcnt[j] + SP_COL(ret)[j];

  free(zlist); free(colcnt);
  return ret;
    }

/*
  SpMatrix_New. In API.

  Returns an uninitialized spmatrix object.

  Arguments:
  mrows,nrows  : Dimension of spmatrix.
  nnz          : Number of nonzero elements.
  id           : DOUBLE/COMPLEX
 */
spmatrix *
SpMatrix_New(int_t nrows, int_t ncols, int_t nnz, int id)
{
  spmatrix *ret;
  if (!(ret = (spmatrix *)spmatrix_tp.tp_alloc(&spmatrix_tp, 0)))
    return (spmatrix *)PyErr_NoMemory();

  ret->obj = alloc_ccs(nrows, ncols, nnz, id);
  if (!ret->obj) { Py_DECREF(ret); return (spmatrix *)PyErr_NoMemory(); }

  return ret;
}

static spmatrix * SpMatrix_NewFromCCS(ccs *x)
{
  spmatrix *ret;
  if (!(ret = (spmatrix *)spmatrix_tp.tp_alloc(&spmatrix_tp, 0)))
    return (spmatrix *)PyErr_NoMemory();

  ret->obj = x;
  return ret;
}

/*
  SpMatrix_NewFromSpMatrix. In API.

  Returns a copy of an spmatrix object.

  Arguments:
  A            : spmatrix object
  id           : DOUBLE/COMPLEX
 */
spmatrix * SpMatrix_NewFromSpMatrix(spmatrix *A, int id)
{
  if ((id == DOUBLE) && (SP_ID(A) == COMPLEX))
    PY_ERR_TYPE("cannot convert complex to double");

  spmatrix *ret = SpMatrix_New
      (SP_NROWS(A), SP_NCOLS(A), SP_NNZ(A), id);

  if (!ret) return (spmatrix *)PyErr_NoMemory();

  convert_array(SP_VAL(ret), SP_VAL(A), id, SP_ID(A), SP_NNZ(A));
  memcpy(SP_COL(ret), SP_COL(A), (SP_NCOLS(A)+1)*sizeof(int_t));
  memcpy(SP_ROW(ret), SP_ROW(A), SP_NNZ(A)*sizeof(int_t));

  return ret;
}

/*
  SpMatrix_NewFromIJV.

  Returns a spmatrix object from a triplet description.

  Arguments:
  Il,Jl,V : (INT,INT,DOUBLE/COMPLEX) triplet description
  m,n     : Dimension of spmatrix. If either m==0 or n==0, then
            the dimension is taken as MAX(I) x MAX(Jl).
  id      : DOUBLE, COMPLEX
 */
spmatrix * SpMatrix_NewFromIJV(matrix *Il, matrix *Jl, matrix *V,
    int_t m, int_t n, int id)
    {

  if (MAT_ID(Il) != INT || MAT_ID(Jl) != INT)
    PY_ERR_TYPE("index sets I and J must be integer");

  if (MAT_LGT(Il) != MAT_LGT(Jl))
    PY_ERR_TYPE("index sets I and J must be of same length");

  if (V && !Matrix_Check(V)) PY_ERR_TYPE("invalid V argument");

  if (V && Matrix_Check(V) && (MAX(id,MAT_ID(V)) != id))
    PY_ERR_TYPE("matrix V has invalid type");

  if (V && (MAT_LGT(V) != MAT_LGT(Il)))
    PY_ERR_TYPE("I, J and V must have same length");

  if (!Il || !Jl) return SpMatrix_New(0,0,0,id);

  int_t k, Imax=-1, Jmax=-1;
  for (k=0; k<MAT_LGT(Il); k++) {
    if (MAT_BUFI(Il)[k]>Imax) Imax = MAT_BUFI(Il)[k];
    if (MAT_BUFI(Jl)[k]>Jmax) Jmax = MAT_BUFI(Jl)[k];
  }

  if ((m<0) || (n<0)) { m = MAX(Imax+1,m); n = MAX(Jmax+1,n);}

  if (m < Imax+1 || n < Jmax+1) PY_ERR_TYPE("dimension too small");

  for (k=0; k<MAT_LGT(Il); k++)
    if ((MAT_BUFI(Il)[k] < 0) || (MAT_BUFI(Il)[k] >= m) ||
        (MAT_BUFI(Jl)[k] < 0) || (MAT_BUFI(Jl)[k] >= n) )
      PY_ERR_TYPE("index out of range");

  return (id == DOUBLE ? triplet2dccs(Il,Jl,V,m,n) :
    triplet2zccs(Il,Jl,V,m,n));
    }

static void spmatrix_dealloc(spmatrix* self)
{
  free(self->obj->values);
  free(self->obj->colptr);
  free(self->obj->rowind);
  free(self->obj);
#if PY_MAJOR_VERSION >= 3
  Py_TYPE(self)->tp_free((PyObject*)self);
#else
  self->ob_type->tp_free((PyObject*)self);
#endif
}

static PyObject *
spmatrix_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  PyObject *size = NULL;
  matrix *Il=NULL, *Jl=NULL, *V=NULL;
  int_t nrows = -1, ncols = -1;

  static char *kwlist[] = { "V", "I", "J", "size","tc", NULL};

#if PY_MAJOR_VERSION >= 3
  int tc = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|OC:spmatrix", kwlist,
      &V, &Il, &Jl, &size, &tc))
#else
  char tc = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|Oc:spmatrix", kwlist,
      &V, &Il, &Jl, &size, &tc))
#endif
    return NULL;

  if (!(PySequence_Check((PyObject *)V) || Matrix_Check(V) || PY_NUMBER(V))) {
    PY_ERR_TYPE("V must be either a sequence type, a matrix, or a number");
  }

  if (size && !PyArg_ParseTuple(size, "nn", &nrows, &ncols))
    PY_ERR_TYPE("invalid dimension tuple");

  if (size && (nrows < 0 || ncols < 0))
    PY_ERR_TYPE("dimensions must be non-negative");

  if (tc && !(VALID_TC_SP(tc))) PY_ERR_TYPE("tc must be 'd' or 'z'");
  int id = (tc ? TC2ID(tc) : -1);


  int ndim = 0;

  /* convert lists to matrices */
  if (Matrix_Check(Il))
    Py_INCREF(Il);

  else if (PyObject_CheckBuffer((PyObject *)Il)) {
    if (!(Il = Matrix_NewFromPyBuffer((PyObject *)Il, INT, &ndim))) {
      return NULL;
    }
  }
  else if (PySequence_Check((PyObject *)Il)) {
    if (!(Il = Matrix_NewFromSequence((PyObject *)Il, INT)))
      return NULL;
  }
  else PY_ERR_TYPE("invalid type for I");

  if (Matrix_Check(Jl))
    Py_INCREF(Jl);

  else if (PyObject_CheckBuffer((PyObject *)Jl)) {
    if (!(Jl = Matrix_NewFromPyBuffer((PyObject *)Jl, INT, &ndim))) {
      Py_DECREF(Il);
      return NULL;
    }
  }

  else if (PySequence_Check((PyObject *)Jl)) {
    if (!(Jl = Matrix_NewFromSequence((PyObject *)Jl, INT))) {
      Py_DECREF(Il);
      return NULL;
    }
  }
  else {
    Py_DECREF(Il);
    PY_ERR_TYPE("invalid type for J");
  }

  if (Matrix_Check(V))
    Py_INCREF(V);

  else if (PyObject_CheckBuffer((PyObject *)V)) {
    int ndim = 0;
    if (!(V = Matrix_NewFromPyBuffer((PyObject *)V, id, &ndim))) {
      Py_DECREF(Il);
      Py_DECREF(Jl);
      return NULL;
    }
  }

  else if (PySequence_Check((PyObject *)V))
    {
    if (!(V = Matrix_NewFromSequence((PyObject *)V, id))) {
      Py_DECREF(Il);
      Py_DECREF(Jl);
      return NULL;
    }
    }
  else if (PY_NUMBER(V))
    {
    if (!(V = Matrix_NewFromNumber(MAT_LGT(Il), 1, get_id(V, 1), V, 1))) {
      Py_DECREF(Il);
      Py_DECREF(Jl);
      return PyErr_NoMemory();
    }
    }
  else {
    Py_DECREF(Il);
    Py_DECREF(Jl);
    PY_ERR_TYPE("invalid type for V");
  }

  id = (id == -1 ? MAX(get_id(V, !Matrix_Check(V)), DOUBLE) : id);

  spmatrix *ret = SpMatrix_NewFromIJV(Il, Jl, V, nrows, ncols,
      id == -1 ? MAX(MAT_ID(V),DOUBLE) : id);

  Py_DECREF(Il);
  Py_DECREF(Jl);
  Py_DECREF(V);

  return (PyObject *)ret;
}

static PyObject *spmatrix_str(matrix *self) {

  PyObject *cvxopt = PyImport_ImportModule("cvxopt");
  PyObject *str, *ret;

  if (!(str = PyObject_GetAttrString(cvxopt, "spmatrix_str"))) {
    Py_DECREF(cvxopt);
    PY_ERR(PyExc_KeyError, "missing 'spmatrix_str' in 'cvxopt'");
  }

  Py_DECREF(cvxopt);
  if (!PyCallable_Check(str)) PY_ERR_TYPE("'spmatrix_str' is not callable");

  ret = PyObject_CallFunctionObjArgs(str, (PyObject *)self, NULL);
  Py_DECREF(str);

  return ret;
}

static PyObject *
spmatrix_repr(matrix *self) {

  PyObject *cvxopt = PyImport_ImportModule("cvxopt");
  PyObject *repr, *ret;

  if (!(repr = PyObject_GetAttrString(cvxopt, "spmatrix_repr"))) {
    Py_DECREF(cvxopt);
    PY_ERR(PyExc_KeyError, "missing 'spmatrix_repr' in 'cvxopt'");
  }

  Py_DECREF(cvxopt);
  if (!PyCallable_Check(repr)) PY_ERR_TYPE("'spmatrix_repr' is not callable");

  ret = PyObject_CallFunctionObjArgs(repr, (PyObject *)self, NULL);
  Py_DECREF(repr);

  return ret;
}

static PyObject *
spmatrix_richcompare(PyObject *self, PyObject *other, int op) {
  Py_INCREF(Py_NotImplemented);
  return Py_NotImplemented;
}

int * spmatrix_compare(PyObject *self, PyObject *other) {
  PyErr_SetString(PyExc_NotImplementedError, "matrix comparison not implemented"); return 0;
}

static PyObject * spmatrix_get_size(spmatrix *self, void *closure)
{
  PyObject *t = PyTuple_New(2);

#if PY_MAJOR_VERSION >= 3
  PyTuple_SET_ITEM(t, 0, PyLong_FromLong(SP_NROWS(self)));
  PyTuple_SET_ITEM(t, 1, PyLong_FromLong(SP_NCOLS(self)));
#else
  PyTuple_SET_ITEM(t, 0, PyInt_FromLong(SP_NROWS(self)));
  PyTuple_SET_ITEM(t, 1, PyInt_FromLong(SP_NCOLS(self)));
#endif

  return t;
}

static int spmatrix_set_size(spmatrix *self, PyObject *value, void *closure)
{
  if (!value) PY_ERR_INT(PyExc_TypeError,"size attribute cannot be deleted");

  if (!PyTuple_Check(value) || PyTuple_Size(value) != 2)
    PY_ERR_INT(PyExc_TypeError, "can only assign a 2-tuple to size");

#if PY_MAJOR_VERSION >= 3
  if (!PyLong_Check(PyTuple_GET_ITEM(value, 0)) ||
      !PyLong_Check(PyTuple_GET_ITEM(value, 1)))
#else
  if (!PyInt_Check(PyTuple_GET_ITEM(value, 0)) ||
      !PyInt_Check(PyTuple_GET_ITEM(value, 1)))
#endif
    PY_ERR_INT(PyExc_TypeError, "invalid size tuple");

#if PY_MAJOR_VERSION >= 3
  int m = PyLong_AS_LONG(PyTuple_GET_ITEM(value, 0));
  int n = PyLong_AS_LONG(PyTuple_GET_ITEM(value, 1));
#else
  int m = PyInt_AS_LONG(PyTuple_GET_ITEM(value, 0));
  int n = PyInt_AS_LONG(PyTuple_GET_ITEM(value, 1));
#endif

  if (m<0 || n<0)
    PY_ERR_INT(PyExc_TypeError, "dimensions must be non-negative");

  if (m*n != SP_NROWS(self)*SP_NCOLS(self))
    PY_ERR_INT(PyExc_TypeError, "number of elements in matrix cannot change");

  int_t *colptr = calloc((n+1),sizeof(int_t));
  if (!colptr) PY_ERR_INT(PyExc_MemoryError, "insufficient memory");

  int j, k, in, jn;
  for (j=0; j<SP_NCOLS(self); j++) {
    for (k=SP_COL(self)[j]; k<SP_COL(self)[j+1]; k++) {
      jn = (SP_ROW(self)[k] + j*SP_NROWS(self)) / m;
      in = (SP_ROW(self)[k] + j*SP_NROWS(self)) % m;
      colptr[jn+1]++;
      SP_ROW(self)[k] = in;
    }
  }

  for (j=1; j<n+1; j++) colptr[j] += colptr[j-1];

  free(SP_COL(self));
  SP_COL(self) = colptr;
  SP_NROWS(self) = m;
  SP_NCOLS(self) = n;

  return 0;
}

static PyObject * spmatrix_get_typecode(matrix *self, void *closure)
{
#if PY_MAJOR_VERSION >= 3
  return PyUnicode_FromStringAndSize(TC_CHAR[SP_ID(self)], 1);
#else
  return PyString_FromStringAndSize(TC_CHAR[SP_ID(self)], 1);
#endif
}

static PyObject *
spmatrix_get_V(spmatrix *self, void *closure)
{
  matrix *ret = Matrix_New(SP_NNZ(self), 1, SP_ID(self));
  if (!ret) return PyErr_NoMemory();

  memcpy(MAT_BUF(ret), SP_VAL(self), SP_NNZ(self)*E_SIZE[SP_ID(self)]);
  return (PyObject *)ret;
}

static int
spmatrix_set_V(spmatrix *self, PyObject *value, void *closure)
{
  if (!value) PY_ERR_INT(PyExc_AttributeError, "attribute cannot be deleted");

  if (PY_NUMBER(value)) {
    number val;
    if (convert_num[SP_ID(self)](&val, value, 1, 0))
      PY_ERR_INT(PyExc_TypeError, "invalid type in assignment");

    int i;
    for (i=0; i<SP_NNZ(self); i++)
      write_num[SP_ID(self)](SP_VAL(self),i,&val,0);

    return 0;
  }
  else if (Matrix_Check(value) && MAT_ID(value) == SP_ID(self) &&
      MAT_LGT(value) == SP_NNZ(self) && MAT_NCOLS(value) == 1) {

    memcpy(SP_VAL(self), MAT_BUF(value), MAT_LGT(value)*E_SIZE[SP_ID(self)]);
    return 0;
  } else PY_ERR_INT(PyExc_TypeError, "invalid assignment for V attribute");
}

static PyObject *spmatrix_get_I(spmatrix *self, void *closure)
{
  matrix *A = Matrix_New( SP_NNZ(self), 1, INT);
  if (!A) return PyErr_NoMemory();

  memcpy(MAT_BUF(A), SP_ROW(self), SP_NNZ(self)*sizeof(int_t));
  return (PyObject *)A;
}

static PyObject * spmatrix_get_J(spmatrix *self, void *closure)
{
  matrix *A = Matrix_New( SP_NNZ(self), 1, INT);
  if (!A) return PyErr_NoMemory();

  int_t k, j;
  for (k=0; k<SP_NCOLS(self); k++)
    for (j=SP_COL(self)[k]; j<SP_COL(self)[k+1]; j++)
      MAT_BUFI(A)[j] = k;

  return (PyObject *)A;
}

static PyObject *spmatrix_get_CCS(spmatrix *self, void *closure)
{
  matrix *colptr = Matrix_New( SP_NCOLS(self)+1, 1, INT);
  matrix *rowind = Matrix_New( SP_NNZ(self), 1, INT);
  matrix *val    = Matrix_New( SP_NNZ(self), 1, SP_ID(self));
  PyObject *ret  = PyTuple_New(3);

  if (!colptr || !rowind || !val || !ret) {
    Py_XDECREF(colptr);
    Py_XDECREF(rowind);
    Py_XDECREF(val);
    Py_XDECREF(ret);
    return PyErr_NoMemory();
  }

  memcpy(MAT_BUF(colptr), SP_COL(self), (SP_NCOLS(self)+1)*sizeof(int_t));
  memcpy(MAT_BUF(rowind), SP_ROW(self), SP_NNZ(self)*sizeof(int_t));
  memcpy(MAT_BUF(val),    SP_VAL(self), SP_NNZ(self)*E_SIZE[SP_ID(self)]);

  PyTuple_SET_ITEM(ret, 0, (PyObject *)colptr);
  PyTuple_SET_ITEM(ret, 1, (PyObject *)rowind);
  PyTuple_SET_ITEM(ret, 2, (PyObject *)val);

  return ret;
}

static spmatrix * spmatrix_get_T(spmatrix *self, void *closure)
{
  return SpMatrix_NewFromCCS(transpose(((spmatrix *)self)->obj,0));
}

static spmatrix * spmatrix_get_H(spmatrix *self, void *closure)
{
  return SpMatrix_NewFromCCS(transpose(((spmatrix *)self)->obj,1));
}


static PyGetSetDef spmatrix_getsets[] = {
    {"size", (getter) spmatrix_get_size, (setter) spmatrix_set_size,
        "matrix dimensions"},
        {"typecode", (getter) spmatrix_get_typecode, NULL, "type character"},
        {"V", (getter) spmatrix_get_V, (setter) spmatrix_set_V,
            "the value list of the matrix in triplet form"},
            {"I", (getter) spmatrix_get_I, NULL,
                "the I (row) list of the matrix in triplet form"},
                {"J", (getter) spmatrix_get_J, NULL,
                    "the J (column) list of the matrix in triplet form"},
                    {"T", (getter) spmatrix_get_T, NULL, "transpose"},
                    {"H", (getter) spmatrix_get_H, NULL, "conjugate transpose"},
                    {"CCS", (getter) spmatrix_get_CCS, NULL, "CCS representation"},
                    {NULL}  /* Sentinel */
};

static PyObject *
spmatrix_getstate(spmatrix *self)
{
  PyObject *Il = spmatrix_get_I(self, NULL);
  PyObject *Jl = spmatrix_get_J(self, NULL);
  PyObject *V  = spmatrix_get_V(self, NULL);
  PyObject *size = PyTuple_New(2);
  if (!Il || !Jl || !V || !size) {
    Py_XDECREF(Il); Py_XDECREF(Jl); Py_XDECREF(V); Py_XDECREF(size);
    return NULL;
  }

#if PY_MAJOR_VERSION >= 3
  PyTuple_SET_ITEM(size, 0, PyLong_FromLong(SP_NROWS(self)));
  PyTuple_SET_ITEM(size, 1, PyLong_FromLong(SP_NCOLS(self)));
#else
  PyTuple_SET_ITEM(size, 0, PyInt_FromLong(SP_NROWS(self)));
  PyTuple_SET_ITEM(size, 1, PyInt_FromLong(SP_NCOLS(self)));
#endif

  return Py_BuildValue("NNNNs", V, Il, Jl, size, TC_CHAR[SP_ID(self)]);

  return NULL;
}

static PyObject * spmatrix_trans(spmatrix *self) {

  return (PyObject *)SpMatrix_NewFromCCS(transpose(((spmatrix *)self)->obj,0));

}

static PyObject * spmatrix_ctrans(spmatrix *self) {

  return (PyObject *)SpMatrix_NewFromCCS(transpose(((spmatrix *)self)->obj,1));

}

static PyObject * spmatrix_real(spmatrix *self) {

  if (SP_ID(self) != COMPLEX)
    return (PyObject *)SpMatrix_NewFromSpMatrix(self, SP_ID(self));

  spmatrix *ret = SpMatrix_New(SP_NROWS(self), SP_NCOLS(self),
      SP_NNZ(self), DOUBLE);
  if (!ret) return PyErr_NoMemory();

  int i;
  for (i=0; i < SP_NNZ(self); i++)
    SP_VALD(ret)[i] = creal(SP_VALZ(self)[i]);

  memcpy(SP_COL(ret), SP_COL(self), (SP_NCOLS(self)+1)*sizeof(int_t));
  memcpy(SP_ROW(ret), SP_ROW(self), SP_NNZ(self)*sizeof(int_t));
  return (PyObject *)ret;
}

static PyObject * spmatrix_imag(spmatrix *self) {

  if (SP_ID(self) != COMPLEX)
    return (PyObject *)SpMatrix_NewFromSpMatrix(self, SP_ID(self));

  spmatrix *ret = SpMatrix_New(SP_NROWS(self), SP_NCOLS(self),
      SP_NNZ(self), DOUBLE);
  if (!ret) return PyErr_NoMemory();

  int i;
  for (i=0; i < SP_NNZ(self); i++)
    SP_VALD(ret)[i] = cimag(SP_VALZ(self)[i]);

  memcpy(SP_COL(ret), SP_COL(self), (SP_NCOLS(self)+1)*sizeof(int_t));
  memcpy(SP_ROW(ret), SP_ROW(self), SP_NNZ(self)*sizeof(int_t));
  return (PyObject *)ret;
}

static PyObject *
spmatrix_reduce(spmatrix* self)
{
#if PY_MAJOR_VERSION >= 3
  return Py_BuildValue("ON", Py_TYPE(self), spmatrix_getstate(self));
#else
  return Py_BuildValue("ON", self->ob_type, spmatrix_getstate(self));
#endif
}

static PyMethodDef spmatrix_methods[] = {
    {"real", (PyCFunction)spmatrix_real, METH_NOARGS,
        "Returns real part of sparse matrix"},
    {"imag", (PyCFunction)spmatrix_imag, METH_NOARGS,
        "Returns imaginary part of sparse matrix"},
    {"trans", (PyCFunction)spmatrix_trans, METH_NOARGS,
        "Returns the matrix transpose"},
    {"ctrans", (PyCFunction)spmatrix_ctrans, METH_NOARGS,
        "Returns the matrix conjugate transpose"},
    {"__reduce__", (PyCFunction)spmatrix_reduce, METH_NOARGS,
        "__reduce__() -> (cls, state)"},
    {NULL}  /* Sentinel */
};


static int
bsearch_int(int_t *lower, int_t *upper, int_t key, int_t *k) {

  if (lower>upper) { *k = 0; return 0; }

  int_t *mid, *start = lower;

  while (upper - lower > 1) {
    mid = lower+((upper-lower)>>1);
    if (*mid > key)
      upper = mid;
    else if (*mid < key)
      lower = mid;
    else {
      *k = mid - start;
      return 1;
    }
  }

  if (*upper == key) {
    *k = upper - start; return 1;
  }
  else if (*lower == key) {
    *k = lower - start; return 1;
  }
  else {
    if (*lower > key)
      *k = lower - start;
    else if (*upper < key)
      *k = upper - start + 1;
    else
      *k = upper - start;
    return 0;
  }
}

int spmatrix_getitem_ij(spmatrix *A, int_t i, int_t j, number *value)
{
  int_t k;

  if (SP_NNZ(A) && bsearch_int(&(SP_ROW(A)[SP_COL(A)[j]]),&
      (SP_ROW(A)[SP_COL(A)[j+1]-1]), i, &k)) {

    write_num[SP_ID(A)](value, 0, SP_VAL(A), SP_COL(A)[j]+k);
    return 1;

  } else {

    write_num[SP_ID(A)](value, 0, &Zero, 0);
    return 0;
  }
}

static void
spmatrix_setitem_ij(spmatrix *A, int_t i, int_t j, number *value) {

  int_t k, l;

  if (bsearch_int(&(SP_ROW(A)[SP_COL(A)[j]]),
      &(SP_ROW(A)[SP_COL(A)[j+1]-1]),i, &k)) {

    write_num[SP_ID(A)](SP_VAL(A), SP_COL(A)[j] + k, value, 0);
    return;
  }
  k += SP_COL(A)[j];

  for (l=j+1; l<SP_NCOLS(A)+1; l++) SP_COL(A)[l]++;

  /* split rowind and value lists at position 'k' and insert element */
  for (l=SP_NNZ(A)-1; l>k; l--) {
    SP_ROW(A)[l] = SP_ROW(A)[l-1];
    write_num[SP_ID(A)](SP_VAL(A),l,SP_VAL(A),l-1);
  }

  SP_ROW(A)[k] = i;
  write_num[SP_ID(A)](SP_VAL(A), k, value, 0);
}

static int
spmatrix_length(spmatrix *self)
{
  return SP_NNZ(self);
}

static PyObject*
spmatrix_subscr(spmatrix* self, PyObject* args)
{
  int_t i = 0, j = 0, k;
  number val;
  matrix *Il = NULL, *Jl = NULL;

  /* single integer */
#if PY_MAJOR_VERSION >= 3
  if (PyLong_Check(args)) {
    i = PyLong_AS_LONG(args);
#else
  if (PyInt_Check(args)) {
    i = PyInt_AS_LONG(args);
#endif
    if ( i<-SP_LGT(self) || i >= SP_LGT(self) )
      PY_ERR(PyExc_IndexError, "index out of range");

    spmatrix_getitem_i(self, CWRAP(i,SP_LGT(self)), &val);

    return num2PyObject[SP_ID(self)](&val, 0);
  }

  else if (PyList_Check(args) || Matrix_Check(args) || PySlice_Check(args)) {

    if (!(Il = create_indexlist(SP_LGT(self), args))) return NULL;

    int_t i, idx, lgt = MAT_LGT(Il), nnz = 0, k = 0;
    /* count # elements in index list */
    for (i=0; i<lgt; i++) {
      idx = MAT_BUFI(Il)[i];
      if (idx < -SP_LGT(self) || idx >= SP_LGT(self)) {
        Py_DECREF(Il);
        PY_ERR(PyExc_IndexError, "index out of range");
      }
      nnz += spmatrix_getitem_i(self, CWRAP(idx,SP_LGT(self)), &val);
    }

    spmatrix *B = SpMatrix_New(lgt,1,nnz,SP_ID(self));
    if (!B) { Py_DECREF(Il); return PyErr_NoMemory(); }

    SP_COL(B)[1] = nnz;
    /* fill up rowind and values */
    for (i=0; i<lgt; i++) {
      idx = MAT_BUFI(Il)[i];
      if (spmatrix_getitem_i(self, CWRAP(idx,SP_LGT(self)), &val)) {
        SP_ROW(B)[k] = i;
        write_num[SP_ID(B)](SP_VAL(B), k++, &val, 0);
      }
    }
    free_lists_exit(args,(PyObject *)NULL,Il,(PyObject *)NULL,(PyObject *)B);
  }

  /* remainding cases are different two argument indexing */
  PyObject *argI = NULL, *argJ = NULL;
  if (!PyArg_ParseTuple(args, "OO", &argI, &argJ))
    PY_ERR(PyExc_TypeError, "invalid index sets I or J");

  /* two integers, subscript form, handle separately */
#if PY_MAJOR_VERSION >= 3
  if (PyLong_Check(argI) && PyLong_Check(argJ)) {
    i = PyLong_AS_LONG(argI); j = PyLong_AS_LONG(argJ);
#else
  if (PyInt_Check(argI) && PyInt_Check(argJ)) {
    i = PyInt_AS_LONG(argI); j = PyInt_AS_LONG(argJ);
#endif
    if ( OUT_RNG(i, SP_NROWS(self)) || OUT_RNG(j, SP_NCOLS(self)) )
      PY_ERR(PyExc_IndexError, "index out of range");

    spmatrix_getitem_ij(self,CWRAP(i,SP_NROWS(self)),
        CWRAP(j,SP_NCOLS(self)), &val);

    return num2PyObject[SP_ID(self)](&val,0);
  }

  if (PySlice_Check(argI)) {
    int_t rowstart, rowstop, rowstep, rowlgt, rowcnt;
   
#if PY_MAJOR_VERSION >= 3
    if (PySlice_GetIndicesEx(argI, SP_NROWS(self), &rowstart, &rowstop, 
        &rowstep, &rowlgt) < 0) return NULL;
#else
    if (PySlice_GetIndicesEx((PySliceObject*)argI, SP_NROWS(self),
        &rowstart, &rowstop, &rowstep, &rowlgt) < 0) return NULL;
#endif

    int_t colstart, colstop, colstep, collgt, colcnt;
    if (PySlice_Check(argJ)) {
#if PY_MAJOR_VERSION >= 3
      if (PySlice_GetIndicesEx(argJ, SP_NCOLS(self), &colstart, &colstop, 
          &colstep, &collgt) < 0) return NULL;
#else
      if (PySlice_GetIndicesEx((PySliceObject*)argJ, SP_NCOLS(self),
          &colstart, &colstop, &colstep, &collgt) < 0) return NULL;
#endif
    }
#if PY_MAJOR_VERSION >= 3
    else if (PyLong_Check(argJ)){
      j = PyLong_AS_LONG(argJ);
#else
    else if (PyInt_Check(argJ)){
      j = PyInt_AS_LONG(argJ);
#endif
      if ( OUT_RNG(j, SP_NCOLS(self)) )
          PY_ERR(PyExc_IndexError, "index out of range");
      colstart = CWRAP(j,SP_NCOLS(self)); 
      colstop = colstart; 
      collgt = 1; 
      colstep = 1;
    }
    else if (PyList_Check(argJ) || Matrix_Check(argJ)) {
      if (!(Jl = create_indexlist(SP_NCOLS(self), argJ))) 
        return NULL;
      colstart = 0; 
      colstop = MAT_LGT(Jl)-1; 
      collgt = MAT_LGT(Jl); 
      colstep = 1;
    }
    else PY_ERR_TYPE("invalid index argument");

    int_t *colptr = calloc(collgt+1, sizeof(int_t));
    if (!colptr) {
      if (Jl && !Matrix_Check(argJ)) { Py_DECREF(Jl); }
      return PyErr_NoMemory();
    }

    for (colcnt=0; colcnt<collgt; colcnt++) {
      j = (Jl ? MAT_BUFI(Jl)[colcnt] : colstart + colcnt*colstep);

      if (rowstart == 0 && rowstop == SP_NROWS(self) && rowstep == 1) {
        /* copy entire column */
        colptr[colcnt+1] = colptr[colcnt] + SP_COL(self)[j+1] - SP_COL(self)[j];
      } 
      else if (rowstart >= 0 && rowstart < rowstop && rowstop <= SP_NROWS(self) && rowstep == 1) {
	colptr[colcnt+1] = colptr[colcnt];
	for (k = SP_COL(self)[j]; k < SP_COL(self)[j+1]; k++) {
	  if (SP_ROW(self)[k] >= rowstart && SP_ROW(self)[k] < rowstop) 
	    colptr[colcnt+1]++;
	} 
      }
      else {
        colptr[colcnt+1] += colptr[colcnt];
        rowcnt = 0;
        if (rowstep > 0) {
          for (k=SP_COL(self)[j]; k<SP_COL(self)[j+1]; k++) {

            while (rowstart + rowcnt*rowstep < SP_ROW(self)[k] && rowcnt < rowlgt)
              rowcnt++;

            if (rowcnt == rowlgt) break;

            if (rowstart + rowcnt*rowstep == SP_ROW(self)[k]) {
              colptr[colcnt+1]++;
              rowcnt++;
            }
          }
        } 
	else {
          for (k=SP_COL(self)[j+1]-1; k>=SP_COL(self)[j]; k--) {

            while (rowstart + rowcnt*rowstep > SP_ROW(self)[k] && rowcnt < rowlgt)
              rowcnt++;

            if (rowcnt == rowlgt) break;

            if (rowstart + rowcnt*rowstep == SP_ROW(self)[k]) {
              colptr[colcnt+1]++;
              rowcnt++;
            }
          }
        }
      }
    }

    ccs *A;
    if (!(A = alloc_ccs(rowlgt, collgt, colptr[collgt], SP_ID(self)))) {
      free(colptr);
      if (Jl && !Matrix_Check(argJ)) { Py_DECREF(Jl); }
      return PyErr_NoMemory();
    }

    free(A->colptr);
    A->colptr = colptr;

    for (colcnt=0; colcnt<collgt; colcnt++) {
      j = (Jl ? MAT_BUFI(Jl)[colcnt] : colstart + colcnt*colstep);

      if (rowstart == 0 && rowstop == SP_NROWS(self) && rowstep == 1) {
        /* copy entire column */
        rowcnt = 0;
        for (k = SP_COL(self)[j]; k < SP_COL(self)[j+1]; k++) {
          A->rowind[A->colptr[colcnt] + rowcnt] = SP_ROW(self)[k];
          if (SP_ID(self) == DOUBLE)
            ((double *)A->values)[colptr[colcnt] + rowcnt] = SP_VALD(self)[k];
          else
            ((double complex *)A->values)[colptr[colcnt] + rowcnt] = SP_VALZ(self)[k];

          rowcnt++;
        }
      }
      else if (rowstart >= 0 && rowstart < rowstop && rowstop<=SP_NROWS(self) && rowstep == 1) {
	rowcnt = 0;
	for (k = SP_COL(self)[j]; k < SP_COL(self)[j+1]; k++) {
	  if (SP_ROW(self)[k] >= rowstart && SP_ROW(self)[k] < rowstop) {
	    A->rowind[A->colptr[colcnt] + rowcnt] = SP_ROW(self)[k] - rowstart;
	    if (SP_ID(self) == DOUBLE) 
	      ((double *)A->values)[colptr[colcnt] + rowcnt] = SP_VALD(self)[k];
	    else
	      ((double complex *)A->values)[colptr[colcnt] + rowcnt] = SP_VALZ(self)[k];

	    rowcnt++;
	  }
	}
      }
      else {

        rowcnt = 0; i = 0;
        if (rowstep > 0) {
          for (k=SP_COL(self)[j]; k<SP_COL(self)[j+1]; k++) {

            while (rowstart + rowcnt*rowstep < SP_ROW(self)[k] && rowcnt < rowlgt)
              rowcnt++;

            if (rowcnt == rowlgt) break;

            if (rowstart + rowcnt*rowstep == SP_ROW(self)[k]) {

              A->rowind[colptr[colcnt] + i] = rowcnt;
              if (SP_ID(self) == DOUBLE)
                ((double *)A->values)[colptr[colcnt] + i] = SP_VALD(self)[k];
              else
                ((double complex *)A->values)[colptr[colcnt] + i] = SP_VALZ(self)[k];

              rowcnt++;
              i++;
            }
          }
        } else {
          for (k=SP_COL(self)[j+1]-1; k>=SP_COL(self)[j]; k--) {

            while (rowstart + rowcnt*rowstep > SP_ROW(self)[k] && rowcnt < rowlgt)
              rowcnt++;

            if (rowcnt == rowlgt) break;

            if (rowstart + rowcnt*rowstep == SP_ROW(self)[k]) {

              A->rowind[colptr[colcnt] + i] = rowcnt;
              if (SP_ID(self) == DOUBLE)
                ((double *)A->values)[colptr[colcnt] + i] = SP_VALD(self)[k];
              else
                ((double complex *)A->values)[colptr[colcnt] + i] = SP_VALZ(self)[k];

              rowcnt++;
              i++;
            }
          }
        }
      }
    }

    if (Jl && !Matrix_Check(argJ)) { Py_DECREF(Jl); }

    spmatrix *B = SpMatrix_New(A->nrows, A->ncols, 0, A->id);
    free_ccs(B->obj);
    B->obj = A;
    return (PyObject *)B;
  }

  if (!(Il = create_indexlist(SP_NROWS(self), argI)) ||
      !(Jl = create_indexlist(SP_NCOLS(self), argJ))) {
    free_lists_exit(argI, argJ, Il, Jl, NULL);
  }

  int lgt_row = MAT_LGT(Il), lgt_col = MAT_LGT(Jl), nnz = 0;
  ccs *A = self->obj;
  spa *s = alloc_spa(A->nrows, A->id);
  if (!s) {
    PyErr_SetString(PyExc_MemoryError, "insufficient memory");
    free_lists_exit(argI, argJ, Il, Jl, NULL);
  }

  for (j=0; j<lgt_col; j++) {
    init_spa(s, A, CWRAP(MAT_BUFI(Jl)[j], SP_NCOLS(self)));

    for (k=0; k<lgt_row; k++)
      nnz += s->nz[CWRAP(MAT_BUFI(Il)[k], SP_NROWS(self))];
  }

  spmatrix *B = SpMatrix_New(lgt_row, lgt_col,nnz,A->id);
  if (!B) {
    free_spa(s);
    PyErr_SetNone(PyExc_MemoryError);
    free_lists_exit(argI, argJ, Il, Jl, NULL);
  }

  nnz = 0;
  for (j=0; j<lgt_col; j++) {
    init_spa(s, A, CWRAP(MAT_BUFI(Jl)[j],SP_NCOLS(self)));

    for (k=0; k<lgt_row; k++) {
      if (s->nz[ CWRAP(MAT_BUFI(Il)[k], SP_NROWS(self))]) {
        if (A->id == DOUBLE)
          SP_VALD(B)[nnz]   = ((double *)s->val)
          [CWRAP(MAT_BUFI(Il)[k],SP_NROWS(self))];
        else
          SP_VALZ(B)[nnz]   = ((double complex *)s->val)
          [CWRAP(MAT_BUFI(Il)[k],SP_NROWS(self))];
        SP_ROW(B) [nnz++] = k;
        SP_COL(B)[j+1]++;
      }
    }
    SP_COL(B)[j+1] += SP_COL(B)[j];
  }
  free_spa(s);
  free_lists_exit(argI, argJ, Il, Jl, (PyObject *)B);
}


static int
spmatrix_ass_subscr(spmatrix* self, PyObject* args, PyObject* value)
{
  int_t i = 0, j = 0, id = SP_ID(self), decref_val = 0;
  int ndim = 0;
  char itype;
  number val, tempval;
  matrix *Il = NULL, *Jl = NULL;

  if (!value) PY_ERR_INT(PyExc_NotImplementedError,
      "cannot delete matrix entries");

  if (!(PY_NUMBER(value) || Matrix_Check(value) || SpMatrix_Check(value))){

    if (PyObject_CheckBuffer(value)) 
      value = (PyObject *)Matrix_NewFromPyBuffer(value, -1, &ndim);
    else
      value = (PyObject *)Matrix_NewFromSequence(value, SP_ID(self));

    if (!value)
      PY_ERR_INT(PyExc_NotImplementedError, "invalid type in assignment");

    decref_val = 1;
  }

  int val_id = get_id(value, (PY_NUMBER(value) ? 1 : 0));
  if (val_id > id)
    PY_ERR_INT(PyExc_TypeError, "invalid type in assignment");

  /* assignment value is matrix or number ? */
  if (PY_NUMBER(value)) {
    if (convert_num[id](&val, value, 1, 0))
      PY_ERR_INT(PyExc_TypeError, "invalid argument type");
    itype = 'n';
  }
  else if (Matrix_Check(value) && MAT_LGT(value)==1) {
    convert_num[id](&val, value, 0, 0);
    itype = 'n';
  }
  else if (Matrix_Check(value))
    itype = 'd';
  else
    itype = 's';

  /* single integer */
#if PY_MAJOR_VERSION >= 3
  if (PyLong_Check(args)) {
#else
  if (PyInt_Check(args)) {
#endif
    if (itype != 'n')
      PY_ERR_INT(PyExc_IndexError, "incompatible sizes in assignment");

#if PY_MAJOR_VERSION >= 3
    i = PyLong_AsLong(args);
#else
    i = PyInt_AsLong(args);
#endif
    if ( i<-SP_LGT(self) || i >= SP_LGT(self) )
      PY_ERR_INT(PyExc_IndexError, "index out of range");

    i = CWRAP(i,SP_LGT(self));

    if (spmatrix_getitem_i(self, i, &tempval))
      spmatrix_setitem_i(self, i, &val);
    else {
      if (!realloc_ccs(self->obj, SP_NNZ(self)+1))
        PY_ERR_INT(PyExc_MemoryError, "Cannot reallocate sparse matrix");
      spmatrix_setitem_i(self, i, &val);
    }
    return 0;
  }

  /* integer matrix list */
  if (PyList_Check(args) || Matrix_Check(args) || PySlice_Check(args)) {

    if (!(Il = create_indexlist(SP_LGT(self), args))) {
      if (decref_val) { Py_DECREF(value); }
      return -1;
    }

    int_t i, lgtI = MAT_LGT(Il);
    int_t nnz = SP_NNZ(self)+MAT_LGT(Il);

    if (((itype == 'd') &&
        ((lgtI != MAT_LGT(value) || MAT_NCOLS(value) != 1))) ||
        (((itype == 's') &&
            ((lgtI != SP_LGT(value)) || SP_NCOLS(value) != 1)))) {
      if (!Matrix_Check(args)) { Py_DECREF(Il); }
      if (decref_val) { Py_DECREF(value); }
      PY_ERR_INT(PyExc_TypeError, "incompatible sizes in assignment");
    }

    /* ass. argument is dense matrix or number */
    if  (itype == 'd' || itype == 'n') {

      int_t *col_merge = calloc(SP_NCOLS(self)+1,sizeof(int_t));
      int_t *row_merge = malloc(nnz*sizeof(int_t));
      void *val_merge = malloc(nnz*E_SIZE[id]);
      int_list *ilist = malloc(lgtI*sizeof(int_list));
      if (!col_merge || !row_merge || !val_merge || !ilist) {
        if (!Matrix_Check(args)) { Py_DECREF(Il); }
        free(col_merge); free(row_merge); free(val_merge); free(ilist);
        if (decref_val) { Py_DECREF(value); }
        PY_ERR_INT(PyExc_MemoryError, "insufficient memory");
      }

      for (i=0; i<lgtI; i++) {
        ilist[i].key = CWRAP(MAT_BUFI(Il)[i],SP_NROWS(self)*SP_NCOLS(self));
        ilist[i].value = i;
      }
      qsort(ilist, lgtI, sizeof(int_list), comp_int);

      /* merge lists */
      int_t rhs_cnt = 0, tot_cnt = 0;
      int_t rhs_j = ilist[rhs_cnt].key / SP_NROWS(self);
      int_t rhs_i = ilist[rhs_cnt].key % SP_NROWS(self);
      for (j=0; j<SP_NCOLS(self); j++) {
        for (i=SP_COL(self)[j]; i<SP_COL(self)[j+1]; i++) {
          while (rhs_cnt<lgtI && rhs_j == j && rhs_i < SP_ROW(self)[i]) {
            if (rhs_cnt == 0 || (rhs_cnt>0 &&
                ilist[rhs_cnt].key != ilist[rhs_cnt-1].key)) {
              row_merge[tot_cnt] = rhs_i;
              if (itype == 'n')
                write_num[id](val_merge, tot_cnt++, &val, 0);
              else
                convert_num[id]((unsigned char*)val_merge + E_SIZE[id]*tot_cnt++,
                    value, 0, ilist[rhs_cnt].value);

              col_merge[j+1]++;
            }
            if (rhs_cnt++ < lgtI-1) {
              rhs_j = ilist[rhs_cnt].key / SP_NROWS(self);
              rhs_i = ilist[rhs_cnt].key % SP_NROWS(self);
            }
          }
          if (rhs_cnt<lgtI && rhs_i == SP_ROW(self)[i] && rhs_j == j) {
            if (rhs_cnt == 0 ||
                (rhs_cnt>0 && ilist[rhs_cnt].key != ilist[rhs_cnt-1].key)) {
              row_merge[tot_cnt] = rhs_i;
              if (itype == 'n')
                write_num[id](val_merge, tot_cnt++, &val, 0);
              else
                convert_num[id]((unsigned char*)val_merge + E_SIZE[id]*tot_cnt++,
                    value, 0, ilist[rhs_cnt].value);
              col_merge[j+1]++;
            }
            if (rhs_cnt++ < lgtI-1) {
              rhs_j = ilist[rhs_cnt].key / SP_NROWS(self);
              rhs_i = ilist[rhs_cnt].key % SP_NROWS(self);
            }
          }
          else {
            row_merge[tot_cnt] = SP_ROW(self)[i];
            write_num[id](val_merge, tot_cnt++, SP_VAL(self), i);
            col_merge[j+1]++;
          }
        }
        while (rhs_cnt<lgtI && rhs_j == j) {
          if (rhs_cnt == 0 ||
              (rhs_cnt>0 && ilist[rhs_cnt].key != ilist[rhs_cnt-1].key)) {
            row_merge[tot_cnt] = rhs_i;
            if (itype == 'n')
              write_num[id](val_merge, tot_cnt++, &val, 0);
            else
              convert_num[id]((unsigned char*)val_merge + E_SIZE[id]*tot_cnt++,
                  value, 0, ilist[rhs_cnt].value);
            col_merge[j+1]++;
          }
          if (rhs_cnt++ < lgtI-1) {
            rhs_j = ilist[rhs_cnt].key / SP_NROWS(self);
            rhs_i = ilist[rhs_cnt].key % SP_NROWS(self);
          }
        }
      }

      for (i=0; i<SP_NCOLS(self); i++)
        col_merge[i+1] += col_merge[i];

      free(SP_COL(self)); SP_COL(self) = col_merge;
      free(SP_ROW(self)); SP_ROW(self) = row_merge;
      free(SP_VAL(self)); SP_VAL(self) = val_merge;
      free(ilist);

      //realloc_ccs(self->obj, SP_NNZ(self));
    }
    /* ass. argument is a sparse matrix */
    else
      {
      int_list *ilist = malloc(lgtI*sizeof(int_list));
      int_t *col_merge = calloc(SP_NCOLS(self)+1,sizeof(int_t));
      int_t *row_merge = malloc(nnz*sizeof(int_t));
      void *val_merge = malloc(nnz*E_SIZE[id]);
      if (!ilist || !col_merge || !row_merge || !val_merge) {
        free(ilist); free(col_merge); free(row_merge); free(val_merge);
        if (!Matrix_Check(args)) { Py_DECREF(Il); }
        if (decref_val) { Py_DECREF(value); }
        PY_ERR_INT(PyExc_MemoryError, "insufficient memory");
      }

      for (i=0; i<lgtI; i++) {
        ilist[i].key = CWRAP(MAT_BUFI(Il)[i],SP_NROWS(self)*SP_NCOLS(self));
        ilist[i].value = -1;
      }

      for (i=0; i<SP_NNZ(value); i++)
        ilist[SP_ROW(value)[i]].value = i;

      qsort(ilist, lgtI, sizeof(int_list), comp_int);

      /* merge lists */
      int_t rhs_cnt = 0, tot_cnt = 0;
      int_t rhs_j = ilist[rhs_cnt].key / SP_NROWS(self);
      int_t rhs_i = ilist[rhs_cnt].key % SP_NROWS(self);
      for (j=0; j<SP_NCOLS(self); j++) {
        for (i=SP_COL(self)[j]; i<SP_COL(self)[j+1]; i++) {

          while (rhs_cnt<lgtI && rhs_j == j && rhs_i < SP_ROW(self)[i]) {
            if (ilist[rhs_cnt].value >= 0 &&
                (rhs_cnt==0 || (rhs_cnt>0 && ilist[rhs_cnt].key !=
                    ilist[rhs_cnt-1].key))) {
              row_merge[tot_cnt] = rhs_i;
              convert_array((unsigned char*)val_merge + E_SIZE[id]*tot_cnt++,
			    (unsigned char*)SP_VAL(value) + E_SIZE[val_id]*ilist[rhs_cnt].value,
                  id, val_id, 1);
              col_merge[j+1]++;
            }
            if (rhs_cnt++ < lgtI-1) {
              rhs_j = ilist[rhs_cnt].key / SP_NROWS(self);
              rhs_i = ilist[rhs_cnt].key % SP_NROWS(self);
            }
          }

          if (rhs_cnt<lgtI && rhs_i == SP_ROW(self)[i] && rhs_j == j) {
            if (ilist[rhs_cnt].value >= 0 && (rhs_cnt==0 ||
                (rhs_cnt>0 && ilist[rhs_cnt].key !=
                    ilist[rhs_cnt-1].key))) {
              row_merge[tot_cnt] = rhs_i;
              convert_array((unsigned char*)val_merge + E_SIZE[id]*tot_cnt++,
			    (unsigned char*)SP_VAL(value) + E_SIZE[val_id]*ilist[rhs_cnt].value,
                  id, val_id, 1);
              col_merge[j+1]++;
            }
            if (rhs_cnt++ < lgtI-1) {
              rhs_j = ilist[rhs_cnt].key / SP_NROWS(self);
              rhs_i = ilist[rhs_cnt].key % SP_NROWS(self);
            }
          }
          else {
            row_merge[tot_cnt] = SP_ROW(self)[i];
            convert_array((unsigned char*)val_merge + E_SIZE[id]*tot_cnt++,
			  (unsigned char*)SP_VAL(self) + E_SIZE[id]*i, id, id, 1);
            col_merge[j+1]++;
          }
        }
        while (rhs_cnt<lgtI && rhs_j == j) {
          if (ilist[rhs_cnt].value >= 0 && (rhs_cnt==0 || (rhs_cnt>0 &&
              ilist[rhs_cnt].key != ilist[rhs_cnt-1].key))) {
            row_merge[tot_cnt] = rhs_i;
            convert_array((unsigned char*)val_merge + E_SIZE[id]*tot_cnt++,
			  (unsigned char*)SP_VAL(value) + E_SIZE[val_id]*ilist[rhs_cnt].value,
                id, val_id, 1);
            col_merge[j+1]++;
          }
          if (rhs_cnt++ < lgtI-1) {
            rhs_j = ilist[rhs_cnt].key / SP_NROWS(self);
            rhs_i = ilist[rhs_cnt].key % SP_NROWS(self);
          }
        }
      }

      for (i=0; i<SP_NCOLS(self); i++)
        col_merge[i+1] += col_merge[i];

      free(SP_COL(self)); SP_COL(self) = col_merge;
      free(SP_ROW(self)); SP_ROW(self) = row_merge;
      free(SP_VAL(self)); SP_VAL(self) = val_merge;
      free(ilist);

      //realloc_ccs(self->obj, SP_NNZ(self));
      }

    if (!Matrix_Check(args)) { Py_DECREF(Il); }
    if (decref_val) { Py_DECREF(value); }

    return 0;
  }

  /* remainding cases are different two argument indexing */

  PyObject *argI = NULL, *argJ = NULL;
  if (!PyArg_ParseTuple(args, "OO", &argI, &argJ))
    PY_ERR_INT(PyExc_TypeError, "invalid index arguments");

  /* two integers, subscript form, handle separately */
#if PY_MAJOR_VERSION >= 3
  if (PyLong_Check(argI) && PyLong_Check(argJ)) {
#else
  if (PyInt_Check(argI) && PyInt_Check(argJ)) {
#endif

    if (itype != 'n')
      PY_ERR_INT(PyExc_TypeError, "argument has wrong size");

#if PY_MAJOR_VERSION >= 3
    i = PyLong_AS_LONG(argI); j = PyLong_AS_LONG(argJ);
#else
    i = PyInt_AS_LONG(argI); j = PyInt_AS_LONG(argJ);
#endif
    if ( OUT_RNG(i, SP_NROWS(self)) || OUT_RNG(j, SP_NCOLS(self)) )
      PY_ERR_INT(PyExc_IndexError, "index out of range");

    i = CWRAP(i,SP_NROWS(self)); j = CWRAP(j,SP_NCOLS(self));
    if (spmatrix_getitem_ij(self, i, j, &tempval))
      spmatrix_setitem_ij(self, i, j, &val);
    else {
      if (!realloc_ccs(self->obj, SP_NNZ(self)+1))
        PY_ERR_INT(PyExc_MemoryError, "insufficient memory");

      spmatrix_setitem_ij(self, i, j, &val);
    }

    return 0;
  }

  if (!(Il = create_indexlist(SP_NROWS(self), argI)) ||
      !(Jl = create_indexlist(SP_NCOLS(self), argJ))) {
    PyErr_SetNone(PyExc_MemoryError);
    free_lists_exit(argI,argJ,Il,Jl,-1);
  }

  if (decref_val && ndim < 2 &&
      MAT_LGT(value) == MAT_LGT(Il)*MAT_LGT(Jl)) {
    MAT_NROWS(value) = MAT_LGT(Il); MAT_NCOLS(value) = MAT_LGT(Jl);
  }

  int_t lgtI = MAT_LGT(Il), lgtJ = MAT_LGT(Jl);

  if ((itype == 'd' && (lgtI != MAT_NROWS(value) ||
      lgtJ != MAT_NCOLS(value))) ||
      (itype == 's' && (lgtI != SP_NROWS(value) ||
          lgtJ != SP_NCOLS(value)))) {
    if (!Matrix_Check(argI)) { Py_DECREF(Il); }
    if (!Matrix_Check(argJ)) { Py_DECREF(Jl); }
    if (decref_val) { Py_DECREF(value); }
    PY_ERR_INT(PyExc_TypeError, "incompatible size of assignment");
  }

  /* ass. argument is dense matrix or number */
  if  ((itype == 'd' || itype == 'n') && lgtI*lgtJ> 0) {

    int_t nnz = SP_NNZ(self)+lgtI*lgtJ;

    int_t *col_merge = calloc(SP_NCOLS(self)+1,sizeof(int_t));
    int_t *row_merge = malloc(nnz*sizeof(int_t));
    void *val_merge  = malloc(nnz*E_SIZE[id]);
    int_list *Is = malloc(lgtI*sizeof(int_list));
    int_list *Js = malloc(lgtJ*sizeof(int_list));
    if (!Is || !Js || !col_merge || !row_merge || !val_merge) {
      if (!Matrix_Check(argI)) { Py_DECREF(Il); }
      if (!Matrix_Check(argJ)) { Py_DECREF(Jl); }
      free(Is); free(Js);
      free(col_merge); free(row_merge); free(val_merge);
      if (decref_val) { Py_DECREF(value); }
      PY_ERR_INT(PyExc_MemoryError, "insufficient memory");
    }

    for (i=0; i<lgtI; i++) {
      Is[i].key = CWRAP(MAT_BUFI(Il)[i],SP_NROWS(self));
      Is[i].value = i;
    }
    qsort(Is, lgtI, sizeof(int_list), comp_int);

    for (i=0; i<lgtJ; i++) {
      Js[i].key = CWRAP(MAT_BUFI(Jl)[i],SP_NCOLS(self));
      Js[i].value = i;
    }
    qsort(Js, lgtJ, sizeof(int_list), comp_int);

    int_t rhs_cnti, rhs_cntj = 0, tot_cnt = 0;
    int_t rhs_i, rhs_j = Js[0].key;
    for (j=0; j<SP_NCOLS(self); j++) {

      if (rhs_j < j && rhs_cntj++ < lgtJ-1) {
        rhs_j = Js[rhs_cntj].key;
      }

      rhs_cnti = 0; rhs_i = Is[0].key;
      for (i=SP_COL(self)[j]; i<SP_COL(self)[j+1]; i++) {

        while (rhs_cnti<lgtI && rhs_j == j && rhs_i < SP_ROW(self)[i]) {
          if (rhs_cnti == 0 || (rhs_cnti>0 &&
              Is[rhs_cnti].key != Is[rhs_cnti-1].key)) {
            row_merge[tot_cnt] = rhs_i;

            if (itype == 'n')
              write_num[id](val_merge, tot_cnt++, &val, 0);
            else
              convert_num[id]((unsigned char*)val_merge + E_SIZE[id]*tot_cnt++,
                  value, 0, Is[rhs_cnti].value + lgtI*Js[rhs_cntj].value);
            col_merge[j+1]++;
          }
          if (rhs_cnti++ < lgtI-1)
            rhs_i = Is[rhs_cnti].key;
        }

        if (rhs_cnti<lgtI && rhs_i == SP_ROW(self)[i] && rhs_j == j) {
          if (rhs_cnti == 0 || (rhs_cnti>0 &&
              Is[rhs_cnti].key != Is[rhs_cnti-1].key)) {
            row_merge[tot_cnt] = rhs_i;

            if (itype == 'n')
              write_num[id](val_merge, tot_cnt++, &val, 0);
            else
              convert_num[id]((unsigned char*)val_merge + E_SIZE[id]*tot_cnt++,
                  value, 0, Is[rhs_cnti].value + lgtI*Js[rhs_cntj].value);

            col_merge[j+1]++;
          }
          if (rhs_cnti++ < lgtI-1)
            rhs_i = Is[rhs_cnti].key;
        }
        else {
          row_merge[tot_cnt] = SP_ROW(self)[i];
          convert_array((unsigned char*)val_merge + E_SIZE[id]*tot_cnt++,
			(unsigned char*)SP_VAL(self) + E_SIZE[id]*i, id, id, 1);
          col_merge[j+1]++;
        }
      }
      while (rhs_cnti<lgtI && rhs_j == j) {
        if (rhs_cnti == 0 || (rhs_cnti>0 &&
            Is[rhs_cnti].key != Is[rhs_cnti-1].key)) {

          row_merge[tot_cnt] = rhs_i;

          if (itype == 'n')
            write_num[id](val_merge, tot_cnt++, &val, 0);
          else
            convert_num[id]((unsigned char*)val_merge + E_SIZE[id]*tot_cnt++,
                value, 0, Is[rhs_cnti].value + lgtI*Js[rhs_cntj].value);

          col_merge[j+1]++;
        }
        if (rhs_cnti++ < lgtI-1)
          rhs_i = Is[rhs_cnti].key;
      }
    }

    for (i=0; i<SP_NCOLS(self); i++)
      col_merge[i+1] += col_merge[i];

    free(SP_COL(self)); SP_COL(self) = col_merge;
    free(SP_ROW(self)); SP_ROW(self) = row_merge;
    free(SP_VAL(self)); SP_VAL(self) = val_merge;
    free(Is); free(Js);

    //realloc_ccs(self->obj, SP_NNZ(self));

  }
  /* ass. argument is a sparse matrix */
  else if (itype == 's' && lgtI*lgtJ > 0) {

    int_t nnz = SP_NNZ(self)+SP_NNZ(value);

    int_t *col_merge = calloc((SP_NCOLS(self)+1),sizeof(int_t));
    int_t *row_merge = malloc(nnz*sizeof(int_t));
    void *val_merge  = malloc(nnz*E_SIZE[id]);
    int_list *Is = malloc(lgtI*sizeof(int_list));
    int_list *Js = malloc(lgtJ*sizeof(int_list));
    if (!Is || !Js || !col_merge || !row_merge || !val_merge) {
      if (!Matrix_Check(argI)) { Py_DECREF(Il); }
      if (!Matrix_Check(argJ)) { Py_DECREF(Jl); }
      free(Is); free(Js);
      free(col_merge); free(row_merge); free(val_merge);
      if (decref_val) { Py_DECREF(value); }
      PY_ERR_INT(PyExc_MemoryError,"insufficient memory");
    }

    for (i=0; i<lgtJ; i++) {
      Js[i].key = CWRAP(MAT_BUFI(Jl)[i],SP_NCOLS(self));
      Js[i].value = i;
    }
    qsort(Js, lgtJ, sizeof(int_list), comp_int);

    int_t rhs_cnti, rhs_cntj = -1, tot_cnt = 0, rhs_offs_rptr = 0;
    int_t rhs_i, rhs_j = -1;
    for (j=0; j<SP_NCOLS(self); j++) {

      if (rhs_j < j && rhs_cntj++ < lgtJ-1) {
        rhs_j = Js[rhs_cntj].key;
        rhs_offs_rptr = SP_COL(value)[Js[rhs_cntj].value];

        for (i=0; i<lgtI; i++) {
          Is[i].key = CWRAP(MAT_BUFI(Il)[i],SP_NROWS(self));
          Is[i].value = -1;
        }

        for (i=rhs_offs_rptr; i<SP_COL(value)[Js[rhs_cntj].value+1]; i++)
          Is[SP_ROW(value)[i]].value = i-rhs_offs_rptr;

        qsort(Is, lgtI, sizeof(int_list), comp_int);
      }

      rhs_cnti = 0; rhs_i = Is[0].key;
      for (i=SP_COL(self)[j]; i<SP_COL(self)[j+1]; i++) {

        while (rhs_cnti<lgtI && rhs_j == j && rhs_i < SP_ROW(self)[i]) {
          if (Is[rhs_cnti].value >= 0 && (rhs_cnti==0 ||
              (rhs_cnti>0 && Is[rhs_cnti].key != Is[rhs_cnti-1].key))) {
            row_merge[tot_cnt] = rhs_i;

            convert_array((unsigned char*)val_merge + E_SIZE[id]*tot_cnt++,
			  (unsigned char*)SP_VAL(value) + E_SIZE[val_id]*
                (Is[rhs_cnti].value+rhs_offs_rptr), id, val_id, 1);

            col_merge[j+1]++;
          }
          if (rhs_cnti++ < lgtI-1)
            rhs_i = Is[rhs_cnti].key;
        }
        if (rhs_cnti<lgtI && rhs_i == SP_ROW(self)[i] && rhs_j == j) {
          if (Is[rhs_cnti].value >= 0 && (rhs_cnti==0 ||
              (rhs_cnti>0 && Is[rhs_cnti].key != Is[rhs_cnti-1].key))) {
            row_merge[tot_cnt] = rhs_i;

            convert_array((unsigned char*)val_merge + E_SIZE[id]*tot_cnt++,
			  (unsigned char*)SP_VAL(value) + E_SIZE[val_id]*
                (Is[rhs_cnti].value+rhs_offs_rptr), id, val_id, 1);

            col_merge[j+1]++;
          }
          if (rhs_cnti++ < lgtI-1)
            rhs_i = Is[rhs_cnti].key;
        }
        else {
          row_merge[tot_cnt] = SP_ROW(self)[i];
          convert_array((unsigned char*)val_merge + E_SIZE[id]*tot_cnt++,
			(unsigned char*)SP_VAL(self) + E_SIZE[id]*i, id, id, 1);
          col_merge[j+1]++;
        }
      }
      while (rhs_cnti<lgtI && rhs_j == j) {
        if (Is[rhs_cnti].value >= 0 && (rhs_cnti == 0 || (rhs_cnti>0 &&
            Is[rhs_cnti].key != Is[rhs_cnti-1].key))) {

          row_merge[tot_cnt] = rhs_i;

          convert_array((unsigned char*)val_merge + E_SIZE[id]*tot_cnt++,
			(unsigned char*)SP_VAL(value) + E_SIZE[val_id]*
              (Is[rhs_cnti].value+rhs_offs_rptr), id, val_id, 1);

          col_merge[j+1]++;
        }
        if (rhs_cnti++ < lgtI-1)
          rhs_i = Is[rhs_cnti].key;
      }
    }

    for (i=0; i<SP_NCOLS(self); i++)
      col_merge[i+1] += col_merge[i];

    free(SP_COL(self)); SP_COL(self) = col_merge;
    free(SP_ROW(self)); SP_ROW(self) = row_merge;
    free(SP_VAL(self)); SP_VAL(self) = val_merge;
    free(Is); free(Js);

    //realloc_ccs(self->obj, SP_NNZ(self));
  }

  if (!Matrix_Check(argI)) { Py_DECREF(Il); }
  if (!Matrix_Check(argJ)) { Py_DECREF(Jl); }
  if (decref_val) { Py_DECREF(value); }

  return 0;

}

static PyMappingMethods spmatrix_as_mapping = {
    (lenfunc)spmatrix_length,
    (binaryfunc)spmatrix_subscr,
    (objobjargproc)spmatrix_ass_subscr
};


static PyObject * spmatrix_neg(spmatrix *self)
{
  spmatrix *x = SpMatrix_NewFromSpMatrix(self,SP_ID(self));
  if (!x) return PyErr_NoMemory();

  int n=SP_NNZ(x);
  scal[SP_ID(self)](&n, &MinusOne[SP_ID(self)], SP_VAL(x), &intOne);

  return (PyObject *)x;
}

static PyObject * spmatrix_pos(spmatrix *self)
{
  spmatrix *x = SpMatrix_NewFromSpMatrix(self,SP_ID(self));
  if (!x) return PyErr_NoMemory();

  return (PyObject *)x;
}

static PyObject * spmatrix_abs(spmatrix *self)
{
  spmatrix *x = SpMatrix_New(SP_NROWS(self), SP_NCOLS(self),
      SP_NNZ(self), DOUBLE);
  if (!x) return PyErr_NoMemory();

  int_t i;

  if (SP_ID(self) == DOUBLE)
    for (i=0; i<SP_NNZ(self); i++) SP_VALD(x)[i] = fabs(SP_VALD(self)[i]);
  else
    for (i=0; i<SP_NNZ(self); i++) SP_VALD(x)[i] = cabs(SP_VALZ(self)[i]);

  memcpy(SP_ROW(x), SP_ROW(self), SP_NNZ(self)*sizeof(int_t));
  memcpy(SP_COL(x), SP_COL(self), (SP_NCOLS(self)+1)*sizeof(int_t));

  return (PyObject *)x;
}

static PyObject *
spmatrix_add_helper(PyObject *self, PyObject *other, int add)
{
  if (!SpMatrix_Check(self)  ||
      !(Matrix_Check(other) || SpMatrix_Check(other)))
    {
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
    }

  if ((X_NROWS(self) != X_NROWS(other)) || (X_NCOLS(self) != X_NCOLS(other)))
    PY_ERR_TYPE("incompatible dimensions");

  int id = MAX(SP_ID(self),X_ID(other));

  ccs *x, *z = NULL;
  void *y;

  if (!(x = convert_ccs(((spmatrix *)self)->obj, id)))
    return NULL;

  if (!(y = (Matrix_Check(other) ?
      (void *)Matrix_NewFromMatrix((matrix *)other, id) :
        (void *)convert_ccs(((spmatrix *)other)->obj, id)))) {
    if (x->id != id) free_ccs(x);
    return NULL;
  }

  if (sp_axpy[id]((add ? One[id] : MinusOne[id]), x,
      (Matrix_Check(other) ? MAT_BUF(y) : y),
      1, SpMatrix_Check(other), 0, (void *)&z))
    {
    if (x->id != id) free_ccs(x);
    if (Matrix_Check(other))
      Py_DECREF((PyObject *)y);
    else
      if (((ccs *)y)->id != id) free_ccs(y);

    return PyErr_NoMemory();
    }

  if (x->id != id) free_ccs(x);
  if (SpMatrix_Check(other)) {
    if (((ccs *)y)->id != id) free_ccs(y);
    spmatrix *ret = SpMatrix_New(SP_NROWS(other), SP_NCOLS(other), 0, id);
    if (!ret) return PyErr_NoMemory();
    free_ccs(ret->obj);
    ret->obj = z;
    return (PyObject *)ret;
  }
  else return (PyObject *)y;
}

static PyObject *
spmatrix_add(PyObject *self, PyObject *other)
{
  if (!SpMatrix_Check(self) && SpMatrix_Check(other)) {
    void *ptr = other; other = self; self = ptr;
  }

  PyObject *ret, *tmp;
  if (PY_NUMBER(other) || (Matrix_Check(other) && MAT_LGT(other)==1))
    if ((tmp = (PyObject *)dense((spmatrix *)self))) {
      ret = matrix_add(tmp, other);
      Py_DECREF(tmp);
      return ret;
    }
    else return NULL;

  else return spmatrix_add_helper(self, other, 1);
}


static PyObject *
spmatrix_iadd(PyObject *self, PyObject *other)
{
  if (!SpMatrix_Check(other))
    PY_ERR_TYPE("invalid inplace operation");

  int id = SP_ID(self);
  if (SP_ID(other) > id)
    PY_ERR_TYPE("incompatible types for inplace operation");

  if ((SP_NROWS(self) != SP_NROWS(other)) ||
      (SP_NCOLS(self) != SP_NCOLS(other)))
    PY_ERR_TYPE("incompatible dimensions");

  ccs *x = ((spmatrix *)self)->obj, *y;
  void *z;

  if (!(y = convert_ccs(((spmatrix *)other)->obj, id)))
    return NULL;

  if (sp_axpy[id](One[id], x, y, 1, 1, 0, &z))
    {
    if (y->id != id) free_ccs(y);
    return PyErr_NoMemory();
    }

  free_ccs(x); ((spmatrix *)self)->obj = z;
  if (y->id != id) free_ccs(y);

  Py_INCREF(self);
  return self;
}


static PyObject *
spmatrix_sub(PyObject *self, PyObject *other)
{
  PyObject *ret, *tmp;
  if (PY_NUMBER(self) || (Matrix_Check(self) && MAT_LGT(self)==1)) {
    if ((tmp = (PyObject *)dense((spmatrix *)other))) {
      ret = matrix_sub(self, tmp);
      Py_DECREF(tmp);
      return ret;
    }
    else return NULL;
  }
  else if (PY_NUMBER(other) || (Matrix_Check(other) && MAT_LGT(other)==1)) {
    if ((tmp = (PyObject *)dense((spmatrix *)self))) {
      ret = matrix_sub(tmp, other);
      Py_DECREF(tmp);
      return ret;
    }
    else return NULL;
  }
  else if (!SpMatrix_Check(self) && SpMatrix_Check(other))
    {
    return spmatrix_add_helper(other, self, 0);
    }
  else if (SpMatrix_Check(self) && !SpMatrix_Check(other)) {
    if ((ret = spmatrix_add_helper(self, other, 0))) {
      int n = MAT_LGT(other), id = MAT_ID(ret);
      scal[id](&n, &MinusOne[id], MAT_BUF(ret), &intOne);
      return ret;
    }
    else return NULL;
  }
  else return spmatrix_add_helper(other, self, 0);
}

static PyObject *
spmatrix_isub(PyObject *self, PyObject *other)
{
  if (!SpMatrix_Check(other))
    PY_ERR_TYPE("invalid inplace operation");

  int id = SP_ID(self);

  if (SP_ID(other) > id)
    PY_ERR_TYPE("incompatible types for inplace operation");

  if ((SP_NROWS(self) != SP_NROWS(other)) ||
      (SP_NCOLS(self) != SP_NCOLS(other)))
    PY_ERR_TYPE("incompatible dimensions");

  ccs *x = ((spmatrix *)self)->obj, *y;
  void *z;

  if (!(y = convert_ccs(((spmatrix *)other)->obj, id)))
    return NULL;

  if (sp_axpy[id](MinusOne[id], y, x, 1, 1, 0, &z))
    {
    if (y->id != id) free_ccs(y);
    return PyErr_NoMemory();
    }

  free_ccs(x); ((spmatrix *)self)->obj = z;
  if (y->id != id) free_ccs(y);

  Py_INCREF(self);
  return self;
}

static PyObject *
spmatrix_mul(PyObject *self, PyObject *other)
{
  if (!(SpMatrix_Check(self) || Matrix_Check(self) || PY_NUMBER(self)) ||
      !(SpMatrix_Check(other) || Matrix_Check(other) || PY_NUMBER(other)))
    {
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
    }

  int id = MAX(get_id(self, PY_NUMBER(self)),get_id(other, PY_NUMBER(other)));
  if (PY_NUMBER(self) || (Matrix_Check(self) && MAT_LGT(self) == 1 &&
      !(SpMatrix_Check(other) && SP_NROWS(other) == 1)) ||
      PY_NUMBER(other) || (Matrix_Check(other) && MAT_LGT(other) == 1 &&
          !(SpMatrix_Check(self) && SP_NCOLS(self) == 1)) )
    {

    spmatrix *ret = SpMatrix_NewFromSpMatrix((spmatrix *)
        (SpMatrix_Check(self) ? self : other), id);

    number val;
    convert_num[id](&val, !SpMatrix_Check(self) ? self : other,
        PY_NUMBER(other) || PY_NUMBER(self), 0);

    scal[id]((int *)&SP_NNZ(ret), &val, SP_VAL(ret), (void *)&One[INT]);
    return (PyObject *)ret;
    }

  else {
    if (X_NCOLS(self) != X_NROWS(other))
      PY_ERR_TYPE("incompatible dimensions");

    void *x, *y, *z = NULL;
    int sp_c = SpMatrix_Check(self) && SpMatrix_Check(other);
    PyObject *C = (sp_c ?
        (PyObject *)SpMatrix_New(SP_NROWS(self), SP_NCOLS(other), 0, id) :
          (PyObject *)Matrix_New(X_NROWS(self), X_NCOLS(other), id));

    if (SpMatrix_Check(self))
      x = convert_ccs(((spmatrix *)self)->obj, id);
    else
      x = convert_mtx_alloc((matrix *)self, id);

    if (SpMatrix_Check(other))
      y = convert_ccs(((spmatrix *)other)->obj, id);
    else
      y = convert_mtx_alloc((matrix *)other, id);

    if (!C || !x || !y) {
      PyErr_SetNone(PyExc_MemoryError);
      Py_XDECREF(C);
      C = NULL;
      goto cleanup;
    }
    if (sp_gemm[id]('N', 'N', One[id], x, y, Zero[id],
	    sp_c ? (unsigned char*)((spmatrix *)C)->obj : (unsigned char*)MAT_BUF(C),
            SpMatrix_Check(self), SpMatrix_Check(other), sp_c, 0, &z,
            X_NROWS(self), X_NCOLS(other), X_NROWS(other)))
      {
      PyErr_SetNone(PyExc_MemoryError);
      Py_DECREF(C); C = NULL;
      }

    if (z) {
      free_ccs( ((spmatrix *)C)->obj );
      ((spmatrix *)C)->obj = z;
    }

    cleanup:
    if (SpMatrix_Check(self)) {
      if (((ccs *)x)->id != id) free_ccs(x);
    }
    else if (MAT_ID(self) != id) free(x);

    if (SpMatrix_Check(other)) {
      if (((ccs *)y)->id != id) free_ccs(y);
    }
    else if (MAT_ID(other) != id) free(y);

    return (PyObject *)C;
  }
}

static PyObject *
spmatrix_imul(PyObject *self, PyObject *other)
{
  if (!(PY_NUMBER(other) || (Matrix_Check(other) && MAT_LGT(other) == 1)))
    PY_ERR_TYPE("invalid operands for sparse multiplication");

  if (SP_ID(self) < get_id(other, PY_NUMBER(other)))
    PY_ERR_TYPE("invalid operands for inplace sparse multiplication");

  number val;
  convert_num[SP_ID(self)](&val, other, !Matrix_Check(other), 0);
  scal[SP_ID(self)]((int *)&SP_NNZ(self), &val, SP_VAL(self),
      (void *)&One[INT]);

  Py_INCREF(self);
  return self;
}

static PyObject *
spmatrix_div_generic(spmatrix *A, PyObject *B, int inplace)
{
  if (!SpMatrix_Check(A) || !(PY_NUMBER(B) ||
      (Matrix_Check(B) && MAT_LGT(B)) == 1))
    PY_ERR_TYPE("invalid operands for sparse division");

  int idA = get_id(A, 0);
  int idB = get_id(B, (Matrix_Check(B) ? 0 : 1));
  int id  = MAX(idA,idB);

  number n;
  convert_num[id](&n, B, (Matrix_Check(B) ? 0 : 1), 0);

  if (!inplace) {
    PyObject *ret = (PyObject *)SpMatrix_NewFromSpMatrix((spmatrix *)A, id);
    if (!ret) return NULL;

    if (div_array[id](SP_VAL(ret), n, SP_NNZ(ret))) {
      Py_DECREF(ret); return NULL;
    }
    return ret;
  } else {
    if (id != idA) PY_ERR_TYPE("invalid inplace operation");

    if (div_array[id](SP_VAL(A), n, SP_NNZ(A)))
      return NULL;

    Py_INCREF(A);
    return (PyObject *)A;
  }
}

static PyObject * spmatrix_div(PyObject *self,PyObject *other) {
  return spmatrix_div_generic((spmatrix *)self, other, 0);
}

static PyObject * spmatrix_idiv(PyObject *self,PyObject *other) {
  return spmatrix_div_generic((spmatrix *)self, other, 1);
}

static int spmatrix_nonzero(matrix *self)
{
  int i, res = 0;
  for (i=0; i<SP_NNZ(self); i++) {
    if ((SP_ID(self) == DOUBLE) && (SP_VALD(self)[i] != 0.0)) res = 1;
    else if ((SP_ID(self) == COMPLEX) && (SP_VALZ(self)[i] != 0.0)) res = 1;
  }

  return res;
}


static PyNumberMethods spmatrix_as_number = {
    (binaryfunc)spmatrix_add,    /*nb_add*/
    (binaryfunc)spmatrix_sub,    /*nb_subtract*/
    (binaryfunc)spmatrix_mul,    /*nb_multiply*/
#if PY_MAJOR_VERSION < 3
    (binaryfunc)spmatrix_div,    /*nb_divide*/
#endif
    0,                           /*nb_remainder*/
    0,                           /*nb_divmod*/
    0,                           /*nb_power*/
    (unaryfunc)spmatrix_neg,     /*nb_negative*/
    (unaryfunc)spmatrix_pos,     /*nb_positive*/
    (unaryfunc)spmatrix_abs,     /*nb_absolute*/
    (inquiry)spmatrix_nonzero,   /*nb_nonzero*/
    0,                           /*nb_invert*/
    0,                           /*nb_lshift*/
    0,                           /*nb_rshift*/
    0,                           /*nb_and*/
    0,                           /*nb_xor*/
    0,                           /*nb_or*/
#if PY_MAJOR_VERSION < 3
    0,                           /*nb_coerce*/
#endif
    0,                           /*nb_int*/
#if PY_MAJOR_VERSION >= 3
    0,                           /*nb_reserved*/
#else
    0,                           /*nb_long*/
#endif
    0,                           /*nb_float*/
#if PY_MAJOR_VERSION < 3
    0,                           /*nb_oct*/
    0,                           /*nb_hex*/
#endif
    (binaryfunc)spmatrix_iadd,   /*nb_inplace_add*/
    (binaryfunc)spmatrix_isub,   /*nb_inplace_subtract*/
    (binaryfunc)spmatrix_imul,   /*nb_inplace_multiply*/
#if PY_MAJOR_VERSION < 3
    (binaryfunc)spmatrix_idiv,   /*nb_inplace_divide*/
#endif
    0,                           /*nb_inplace_remainder*/
    0,                           /*nb_inplace_power*/
    0,                           /*nb_inplace_lshift*/
    0,                           /*nb_inplace_rshift*/
    0,                           /*nb_inplace_and*/
    0,                           /*nb_inplace_xor*/
    0,                           /*nb_inplace_or*/
    0,                           /*nb_floor_divide */
#if PY_MAJOR_VERSION >= 3
    (binaryfunc)spmatrix_div,    /* nb_true_divide */
#else
    0,                           /* nb_true_divide */
#endif
    0,                           /* nb_inplace_floor_divide */
#if PY_MAJOR_VERSION >= 3
    (binaryfunc)spmatrix_idiv,   /* nb_inplace_true_divide */
    0,                           /* nb_index */
#else
    0,                           /* nb_inplace_true_divide */
#endif
};


/*********************** Iterator **************************/

typedef struct {
  PyObject_HEAD
  long index;
  spmatrix *mObj;   /* Set to NULL when iterator is exhausted */
} spmatrixiter;

static PyTypeObject spmatrixiter_tp;

#define SpMatrixIter_Check(O) PyObject_TypeCheck(O, &spmatrixiter_tp)

static PyObject *
spmatrix_iter(spmatrix *obj)
{
  spmatrixiter *it;

  if (!SpMatrix_Check(obj)) {
    PyErr_BadInternalCall();
    return NULL;
  }

  spmatrixiter_tp.tp_iter = PyObject_SelfIter;
  spmatrixiter_tp.tp_getattro = PyObject_GenericGetAttr;

  it = PyObject_GC_New(spmatrixiter, &spmatrixiter_tp);
  if (it == NULL)
    return NULL;

  Py_INCREF(obj);
  it->index = 0;
  it->mObj = obj;
  PyObject_GC_Track(it);

  return (PyObject *)it;
}

static void
spmatrixiter_dealloc(spmatrixiter *it)
{
  PyObject_GC_UnTrack(it);
  Py_XDECREF(it->mObj);
  PyObject_GC_Del(it);
}

static int
spmatrixiter_traverse(spmatrixiter *it, visitproc visit, void *arg)
{
  if (it->mObj == NULL)
    return 0;

  return visit((PyObject *)(it->mObj), arg);
}

static PyObject *
spmatrixiter_next(spmatrixiter *it)
{
  assert(SpMatrixIter_Check(it));
  if (it->index >= SP_NNZ(it->mObj))
    return NULL;

  return num2PyObject[SP_ID(it->mObj)](SP_VAL(it->mObj), it->index++);
}

static PyTypeObject spmatrixiter_tp = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                        /* ob_size */
#endif
    "spmatrixiter",                           /* tp_name */
    sizeof(spmatrixiter),                     /* tp_basicsize */
    0,                                        /* tp_itemsize */
    (destructor)spmatrixiter_dealloc,         /* tp_dealloc */
    0,                                        /* tp_print */
    0,                                        /* tp_getattr */
    0,                                        /* tp_setattr */
    0,                                        /* tp_compar */
    0,                                        /* tp_repr */
    0,                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,  /* tp_flags */
    0,                                        /* tp_doc */
    (traverseproc)spmatrixiter_traverse,      /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    (iternextfunc)spmatrixiter_next,          /* tp_iternext */
    0,                                        /* tp_methods */
};


PyTypeObject spmatrix_tp = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,
#endif
    "cvxopt.base.spmatrix",
    sizeof(spmatrix),
    0,
    (destructor)spmatrix_dealloc,              /* tp_dealloc */
    0,                                         /* tp_print */
    0,                                         /* tp_getattr */
    0,                                         /* tp_setattr */
#if PY_MAJOR_VERSION >= 3
    0,                                         /* tp_compare */
#else
    (cmpfunc)spmatrix_compare,                 /* tp_compare */
#endif
    (reprfunc)spmatrix_repr,                   /* tp_repr */
    &spmatrix_as_number,                       /* tp_as_number */
    0,                                         /* tp_as_sequence */
    &spmatrix_as_mapping,                      /* tp_as_mapping */
    0,                                         /* tp_hash */
    0,                                         /* tp_call */
    (reprfunc)spmatrix_str,                    /* tp_str */
    0,                                         /* tp_getattro */
    0,                                         /* tp_setattro */
    0,                                         /* tp_as_buffer */
#if PY_MAJOR_VERSION >= 3
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* tp_flags */
#else
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
    Py_TPFLAGS_CHECKTYPES,                     /* tp_flags */
#endif
    0,                                         /* tp_doc */
    0,                                         /* tp_traverse */
    0,                                         /* tp_clear */
    (richcmpfunc)spmatrix_richcompare,         /* tp_richcompare */
    0,                                         /* tp_weaklistoffset */
    (getiterfunc)spmatrix_iter,                /* tp_iter */
    0,                                         /* tp_iternext */
    spmatrix_methods,                          /* tp_methods */
    0,                                         /* tp_members */
    spmatrix_getsets,                          /* tp_getset */
    0,                                         /* tp_base */
    0,                                         /* tp_dict */
    0,                                         /* tp_descr_get */
    0,                                         /* tp_descr_set */
    0,                                         /* tp_dictoffset */
    0,                                         /* tp_init */
    0,                                         /* tp_alloc */
    spmatrix_new,                              /* tp_new */
};
