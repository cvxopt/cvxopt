/*
 * Copyright 2012-2013 M. Andersen and L. Vandenberghe.
 * Copyright 2010-2011 L. Vandenberghe.
 * Copyright 2004-2009 J. Dahl and L. Vandenberghe.
 *
 * This file is part of CVXOPT version 1.1.6.
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
    "Interface to the simplex and mixed integer LP algorithms in GLPK.\n\n"
    "The GLPK control parameters have the default values listed in \n"
    "the GLPK documentation, except for 'presolve', which is set\n"
    "to 1 and cannot be modified.  The other parameters can be\n"
    "modified by passing a smcp or iocp object to the appropriate function\n"
    "For example,  the commands param = glpk.smcp(msg_lev = 0), or \n"
    "param=glpk.smcp(); param.msg_lev=1 turn off the printed output during"
    " execution of glpk.simplex().\n"
    "See the documentation at www.gnu.org/software/glpk/glpk.html for\n"
    "the list of GLPK control parameters and their default values.");

static PyObject *glpk_module;

/* Wrappers around the option glpk structs */
typedef struct{
  PyObject_HEAD
  glp_smcp obj;
} pysmcp;

/* Deallocation of smcp object */
static void smcp_dealloc(pysmcp* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* New smcp method */
static PyObject *
smcp_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    pysmcp *self;
    self = (pysmcp *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/* Initialisation of smcp object */
static int
smcp_init(pysmcp *self, PyObject *args, PyObject *kwds)
{
    /*static char *kwlist[] = {"number", NULL};*/
    static char *kwlist[] = { "msg_lev", "meth", "pricing", "r_test", "tol_bnd", "tol_dj", "tol_piv", "obj_ll", "obj_ul", "it_lim", "tm_lim", "out_frq", "out_dly", "presolve" };
    glp_init_smcp(&self->obj);
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|iiiidddddiiiii", kwlist,
                &self->obj.msg_lev,
                &self->obj.meth,
                &self->obj.pricing,
                &self->obj.r_test,
                &self->obj.tol_bnd,
                &self->obj.tol_dj,
                &self->obj.tol_piv,
                &self->obj.obj_ll,
                &self->obj.obj_ul,
                &self->obj.it_lim,
                &self->obj.tm_lim,
                &self->obj.out_frq,
                &self->obj.out_dly,
                &self->obj.presolve))
        return -1;

    return 0;
}

/* smcp members declaration */
static PyMemberDef smcpMembers[] = {
  {"msg_lev", T_INT, offsetof(pysmcp,obj)+offsetof(glp_smcp,msg_lev), 0, "message level: "},
  {"meth", T_INT, offsetof(pysmcp,obj)+offsetof(glp_smcp,meth), 0, "simplex method option: "},
  {"pricing", T_INT, offsetof(pysmcp,obj)+offsetof(glp_smcp,pricing), 0, "pricing technique: "},
  {"r_test", T_INT, offsetof(pysmcp,obj)+offsetof(glp_smcp,r_test), 0, "ratio test technique: "},
  {"tol_bnd", T_DOUBLE, offsetof(pysmcp,obj)+offsetof(glp_smcp,tol_bnd), 0, "spx.tol_bnd "},
  {"tol_dj", T_DOUBLE, offsetof(pysmcp,obj)+offsetof(glp_smcp,tol_dj), 0, "spx.tol_dj "},
  {"tol_piv", T_DOUBLE, offsetof(pysmcp,obj)+offsetof(glp_smcp,tol_piv), 0, "spx.tol_piv "},
  {"obj_ll", T_DOUBLE, offsetof(pysmcp,obj)+offsetof(glp_smcp,obj_ll), 0, "spx.obj_ll "},
  {"obj_ul", T_DOUBLE, offsetof(pysmcp,obj)+offsetof(glp_smcp,obj_ul), 0, "spx.obj_ul "},
  {"it_lim", T_INT, offsetof(pysmcp,obj)+offsetof(glp_smcp,it_lim), 0, "spx.it_lim "},
  {"tm_lim", T_INT, offsetof(pysmcp,obj)+offsetof(glp_smcp,tm_lim), 0, "spx.tm_lim (milliseconds) "},
  {"out_frq", T_INT, offsetof(pysmcp,obj)+offsetof(glp_smcp,out_frq), 0, "spx.out_frq "},
  {"out_dly", T_INT, offsetof(pysmcp,obj)+offsetof(glp_smcp,out_dly), 0, "spx.out_dly (milliseconds) "},
  {"presolve", T_INT, offsetof(pysmcp,obj)+offsetof(glp_smcp,presolve), 0, "enable/disable using LP presolver "},
};

static PyTypeObject smcp_t = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "glpk.smcp",               /* tp_name */
    sizeof(pysmcp),            /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)smcp_dealloc,  /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "simplex method control parameters",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    smcpMembers,               /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)smcp_init,       /* tp_init */
    0,                         /* tp_alloc */
    smcp_new,                 /* tp_new */
};


/* Wrappers around the option glpk structs */
typedef struct{
  PyObject_HEAD
  glp_iocp obj;
} pyiocp;

/* Deallocation of iocp object */
static void iocp_dealloc(pysmcp* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* New iocp method */
static PyObject *
iocp_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    pyiocp *self;
    self = (pyiocp *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/* Initialisation of iocp object */
static int
iocp_init(pyiocp *self, PyObject *args, PyObject *kwds)
{
    /*static char *kwlist[] = {"number", NULL};*/
    static char *kwlist[] = { "msg_lev", "br_tech", "bt_tech", "tol_int", "tol_obj", "tm_lim", "out_frq", "out_dly", "cb_size", "pp_tech", "mip_gap", "mir_cuts", "gmi_cuts", "cov_cuts", "clq_cuts", "presolve", "binarize", "fp_heur", "ps_heur", "ps_tm_lim", "use_sol", "save_sol", "alien",NULL};
    glp_init_iocp(&self->obj);

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|iiiddiiiiidiiiiiiiiiisi", kwlist,
                &self->obj.msg_lev,
                &self->obj.br_tech,
                &self->obj.bt_tech,
                &self->obj.tol_int,
                &self->obj.tol_obj,
                &self->obj.tm_lim,
                &self->obj.out_frq,
                &self->obj.out_dly,
                &self->obj.cb_size,
                &self->obj.pp_tech,
                &self->obj.mip_gap,
                &self->obj.mir_cuts,
                &self->obj.gmi_cuts,
                &self->obj.cov_cuts,
                &self->obj.clq_cuts,
                &self->obj.presolve,
                &self->obj.binarize,
                &self->obj.fp_heur,
                &self->obj.ps_heur,
                &self->obj.ps_tm_lim,
                &self->obj.use_sol,
                &self->obj.save_sol,
                &self->obj.alien))
                    return -1;

    return 0;
}

/* iocp members declaration */
static PyMemberDef iocpMembers[] = {
  {"msg_lev", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,msg_lev), 0, "message level (see glp_smcp) "},
  {"br_tech", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,br_tech), 0, "branching technique: "},
  {"bt_tech", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,bt_tech), 0, "backtracking technique: "},
  {"tol_int", T_DOUBLE, offsetof(pysmcp,obj)+offsetof(glp_iocp,tol_int), 0, "mip.tol_int "},
  {"tol_obj", T_DOUBLE, offsetof(pysmcp,obj)+offsetof(glp_iocp,tol_obj), 0, "mip.tol_obj "},
  {"tm_lim", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,tm_lim), 0, "mip.tm_lim (milliseconds) "},
  {"out_frq", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,out_frq), 0, "mip.out_frq (milliseconds) "},
  {"out_dly", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,out_dly), 0, "mip.out_dly (milliseconds) "},
  /*void (*cb_func)(glp_tree *T, void *info); [> mip.cb_func <]*/
  /*void *cb_info;          [> mip.cb_info <]*/
  {"cb_size", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,cb_size), 0, "mip.cb_size "},
  {"pp_tech", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,pp_tech), 0, "preprocessing technique: "},
  {"mip_gap", T_DOUBLE, offsetof(pysmcp,obj)+offsetof(glp_iocp,mip_gap), 0, "relative MIP gap tolerance "},
  {"mir_cuts", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,mir_cuts), 0, "MIR cuts       (GLP_ON/GLP_OFF) "},
  {"gmi_cuts", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,gmi_cuts), 0, "Gomory's cuts  (GLP_ON/GLP_OFF) "},
  {"cov_cuts", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,cov_cuts), 0, "cover cuts     (GLP_ON/GLP_OFF) "},
  {"clq_cuts", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,clq_cuts), 0, "clique cuts    (GLP_ON/GLP_OFF) "},
  {"presolve", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,presolve), 0, "enable/disable using MIP presolver "},
  {"binarize", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,binarize), 0, "try to binarize integer variables "},
  {"fp_heur", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,fp_heur), 0, "feasibility pump heuristic "},
  {"ps_heur", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,ps_heur), 0, "proximity search heuristic "},
  {"ps_tm_lim", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,ps_tm_lim), 0, "proxy time limit, milliseconds "},
  {"use_sol", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,use_sol), 0, "use existing solution "},
  {"save_sol",T_STRING,offsetof(pysmcp,obj)+offsetof(glp_iocp,save_sol),0, "filename to save every new solution"},
  {"alien", T_INT, offsetof(pysmcp,obj)+offsetof(glp_iocp,alien), 0, "use alien solver "},
};

static PyTypeObject iocp_t = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "glpk.iocp",             /* tp_name */
    sizeof(pyiocp),            /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)iocp_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "integer optimizer control parameters",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    iocpMembers,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)iocp_init,       /* tp_init */
    0,                         /* tp_alloc */
    iocp_new,                 /* tp_new */
};



/* Small helper function to generate the output string of the simplex function */
inline static void set_output_string(PyObject* t,const char s[]) {
            PyTuple_SET_ITEM(t, 0, (PyObject *)
#if PY_MAJOR_VERSION >= 3
                PyUnicode_FromString(s));
#else
                PyString_FromString(s));
#endif
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
    PyObject *G, *A=NULL, *t=NULL;
    glp_prob *lp;
    glp_smcp *options = NULL;
    pysmcp *smcpParm = NULL;
    int m, n, p, i, j, k, nnz, nnzmax, *rn=NULL, *cn=NULL;
    double *a=NULL, val;
    char *kwlist[] = {"c", "G", "h", "A", "b","options", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|OOO!", kwlist, &c,
        &G, &h, &A, &b,&smcp_t,&smcpParm)) return NULL;

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
    if(!smcpParm) 
    {
      smcpParm = (pysmcp*)malloc(sizeof(*smcpParm));
      glp_init_smcp(&(smcpParm->obj));
    }
    if(smcpParm) 
    {
      Py_INCREF(smcpParm);
      options = &smcpParm->obj;
      options->presolve = 1;
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

    if (!(t = PyTuple_New(A ? 4 : 3))){
        glp_delete_prob(lp);
        return PyErr_NoMemory();
    }


    switch (glp_simplex(lp,options)){

        case 0:

            x = (matrix *) Matrix_New(n,1,DOUBLE);
            z = (matrix *) Matrix_New(m,1,DOUBLE);
            if (A) y = (matrix *) Matrix_New(p,1,DOUBLE);
            if (!x || !z || (A && !y)){
                Py_XDECREF(x);
                Py_XDECREF(z);
                Py_XDECREF(y);
                Py_XDECREF(t);
                Py_XDECREF(smcpParm);
                glp_delete_prob(lp);
                return PyErr_NoMemory();
            }

            set_output_string(t,"optimal");

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

            Py_XDECREF(smcpParm);
            glp_delete_prob(lp);
            return (PyObject *) t;
        case GLP_EBADB:
            set_output_string(t,"incorrect initial basis");
            break;
        case GLP_ESING:
            set_output_string(t,"singular initial basis matrix");
            break;
        case GLP_ECOND:
            set_output_string(t,"ill-conditioned initial basis matrix");
            break;
        case GLP_EBOUND:
            set_output_string(t,"incorrect bounds");
            break;
        case GLP_EFAIL:
            set_output_string(t,"solver failure");
            break;
        case GLP_EOBJLL:
            set_output_string(t,"objective function reached lower limit");
            break;
        case GLP_EOBJUL:
            set_output_string(t,"objective function reached upper limit");
            break;
        case GLP_EITLIM:
            set_output_string(t,"iteration limit exceeded");
            break;
        case GLP_ETMLIM:
            set_output_string(t,"time limit exceeded");
            break;
        case GLP_ENOPFS:
            set_output_string(t,"primal infeasible");
            break;
        case GLP_ENODFS:
            set_output_string(t,"dual infeasible");
            break;
        default:
            set_output_string(t,"unknown");
            break;
    }

    Py_XDECREF(smcpParm);
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
    "x            a (sub-)optimal solution if status is 'optimal' or \n"
    "             'time limit exceeded'; None otherwise";

static PyObject *integer(PyObject *self, PyObject *args,
    PyObject *kwrds)
{
    matrix *c, *h, *b=NULL, *x=NULL;
    PyObject *G, *A=NULL, *IntSet=NULL, *BinSet = NULL;
    PyObject *t=NULL;
    pyiocp *iocpParm = NULL;;
    glp_iocp *options = NULL;
    glp_prob *lp;
    int m, n, p, i, j, k, nnz, nnzmax, *rn=NULL, *cn=NULL;
    double *a=NULL, val;
    char *kwlist[] = {"c", "G", "h", "A", "b", "I", "B","iocp", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwrds, "OOO|OOOOO!", kwlist, &c,
	    &G, &h, &A, &b, &IntSet, &BinSet,iocp_t,&iocpParm)) return NULL;

    if(!iocpParm) 
    {
      iocpParm = (pyiocp*)malloc(sizeof(*iocpParm));
      glp_init_iocp(&(iocpParm->obj));
    }
    if(iocpParm) 
    {
      Py_INCREF(iocpParm);
      options = &iocpParm->obj;
      options->presolve = 1;
    }

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

    if (IntSet) {
      PyObject *iter = PySequence_Fast(IntSet, "Critical error: not sequence");

      for (i=0; i<PySet_GET_SIZE(IntSet); i++) {

	PyObject *tmp = PySequence_Fast_GET_ITEM(iter, i);
#if PY_MAJOR_VERSION >= 3
	if (!PyLong_Check(tmp)) {
#else
	if (!PyInt_Check(tmp)) {
#endif
	  glp_delete_prob(lp);
	  Py_DECREF(iter);
	  PY_ERR_TYPE("non-integer element in I");
	}
#if PY_MAJOR_VERSION >= 3
	int k = PyLong_AS_LONG(tmp);
#else
	int k = PyInt_AS_LONG(tmp);
#endif
	if ((k < 0) || (k >= n)) {
	  glp_delete_prob(lp);
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
#if PY_MAJOR_VERSION >= 3
	if (!PyLong_Check(tmp)) {
#else
	if (!PyInt_Check(tmp)) {
#endif
	  glp_delete_prob(lp);
	  Py_DECREF(iter);
	  PY_ERR_TYPE("non-binary element in I");
	}
#if PY_MAJOR_VERSION >= 3
	int k = PyLong_AS_LONG(tmp);
#else
	int k = PyInt_AS_LONG(tmp);
#endif
	if ((k < 0) || (k >= n)) {
	  glp_delete_prob(lp);
	  Py_DECREF(iter);
	  PY_ERR(PyExc_IndexError, "index element out of range in B");
	}
	glp_set_col_kind(lp, k+1, GLP_BV);
      }

      Py_DECREF(iter);

    }


      switch (glp_intopt(lp,options)){

          case 0:

              x = (matrix *) Matrix_New(n,1,DOUBLE);
              if (!x) {
                  Py_XDECREF(iocpParm);
                  Py_XDECREF(t);
                  glp_delete_prob(lp);
                  return PyErr_NoMemory();
              }
              set_output_string(t,"optimal");
              set_output_string(t,"optimal");

              for (i=0; i<n; i++)
                  MAT_BUFD(x)[i] = glp_mip_col_val(lp, i+1);
              PyTuple_SET_ITEM(t, 1, (PyObject *) x);

              Py_XDECREF(iocpParm);
              glp_delete_prob(lp);
              return (PyObject *) t;

          case GLP_ETMLIM:

              x = (matrix *) Matrix_New(n,1,DOUBLE);
              if (!x) {
                  Py_XDECREF(t);
                  Py_XDECREF(iocpParm);
                  glp_delete_prob(lp);
                  return PyErr_NoMemory();
              }
              set_output_string(t,"time limit exceeded");

              for (i=0; i<n; i++)
                  MAT_BUFD(x)[i] = glp_mip_col_val(lp, i+1);
              PyTuple_SET_ITEM(t, 1, (PyObject *) x);

              Py_XDECREF(iocpParm);
              glp_delete_prob(lp);
              return (PyObject *) t;


          case GLP_EBOUND:
              set_output_string(t,"incorrect bounds");
              break;
          case GLP_EFAIL:
              set_output_string(t,"invalid MIP formulation");
              break;

          case GLP_ENOPFS:
              set_output_string(t,"primal infeasible");
              break;

          case GLP_ENODFS:
              set_output_string(t,"dual infeasible");
              break;

          case GLP_EMIPGAP:
              set_output_string(t,"Relative mip gap tolerance reached");
              break;

              /*case LPX_E_ITLIM:

                set_output_string(t,"maxiters exceeded");
                break;*/

              /*case LPX_E_SING:

                set_output_string(t,"singular or ill-conditioned basis");
                break;*/


          default:

              set_output_string(t,"unknown");
      }

      Py_XDECREF(iocpParm);
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

void addglpkConstants (void)
{
  PyModule_AddIntMacro(glpk_module, GLP_ON);
  PyModule_AddIntMacro(glpk_module,GLP_OFF);

  /* reason codes: */
  PyModule_AddIntMacro(glpk_module,GLP_IROWGEN);
  PyModule_AddIntMacro(glpk_module,GLP_IBINGO);
  PyModule_AddIntMacro(glpk_module,GLP_IHEUR);
  PyModule_AddIntMacro(glpk_module,GLP_ICUTGEN);
  PyModule_AddIntMacro(glpk_module,GLP_IBRANCH);
  PyModule_AddIntMacro(glpk_module,GLP_ISELECT);
  PyModule_AddIntMacro(glpk_module,GLP_IPREPRO);

  /* branch selection indicator: */
  PyModule_AddIntMacro(glpk_module,GLP_NO_BRNCH);
  PyModule_AddIntMacro(glpk_module,GLP_DN_BRNCH);
  PyModule_AddIntMacro(glpk_module,GLP_UP_BRNCH);

  /* return codes: */
  PyModule_AddIntMacro(glpk_module,GLP_EBADB);
  PyModule_AddIntMacro(glpk_module,GLP_ESING);
  PyModule_AddIntMacro(glpk_module,GLP_ECOND);
  PyModule_AddIntMacro(glpk_module,GLP_EBOUND);
  PyModule_AddIntMacro(glpk_module,GLP_EFAIL);
  PyModule_AddIntMacro(glpk_module,GLP_EOBJLL);
  PyModule_AddIntMacro(glpk_module,GLP_EOBJUL);
  PyModule_AddIntMacro(glpk_module,GLP_EITLIM);
  PyModule_AddIntMacro(glpk_module,GLP_ETMLIM);
  PyModule_AddIntMacro(glpk_module,GLP_ENOPFS);
  PyModule_AddIntMacro(glpk_module,GLP_ENODFS);
  PyModule_AddIntMacro(glpk_module,GLP_EROOT);
  PyModule_AddIntMacro(glpk_module,GLP_ESTOP);
  PyModule_AddIntMacro(glpk_module,GLP_EMIPGAP);
  PyModule_AddIntMacro(glpk_module,GLP_ENOFEAS);
  PyModule_AddIntMacro(glpk_module,GLP_ENOCVG);
  PyModule_AddIntMacro(glpk_module,GLP_EINSTAB);
  PyModule_AddIntMacro(glpk_module,GLP_EDATA);
  PyModule_AddIntMacro(glpk_module,GLP_ERANGE);

  /* condition indicator: */
  PyModule_AddIntMacro(glpk_module,GLP_KKT_PE);
  PyModule_AddIntMacro(glpk_module,GLP_KKT_PB);
  PyModule_AddIntMacro(glpk_module,GLP_KKT_DE);
  PyModule_AddIntMacro(glpk_module,GLP_KKT_DB);
  PyModule_AddIntMacro(glpk_module,GLP_KKT_CS);

  /* MPS file format: */
  PyModule_AddIntMacro(glpk_module,GLP_MPS_DECK);
  PyModule_AddIntMacro(glpk_module,GLP_MPS_FILE);

  /* simplex method control parameters */
  /* message level: */
  PyModule_AddIntMacro(glpk_module,GLP_MSG_OFF);
  PyModule_AddIntMacro(glpk_module,GLP_MSG_ERR);
  PyModule_AddIntMacro(glpk_module,GLP_MSG_ON);
  PyModule_AddIntMacro(glpk_module,GLP_MSG_ALL);
  PyModule_AddIntMacro(glpk_module,GLP_MSG_DBG);
  /* simplex method option: */
  PyModule_AddIntMacro(glpk_module,GLP_PRIMAL);
  PyModule_AddIntMacro(glpk_module,GLP_DUALP);
  PyModule_AddIntMacro(glpk_module,GLP_DUAL);
  /* pricing technique: */
  PyModule_AddIntMacro(glpk_module,GLP_PT_STD);
  PyModule_AddIntMacro(glpk_module,GLP_PT_PSE);
  /* ratio test technique: */
  PyModule_AddIntMacro(glpk_module,GLP_RT_STD);
  PyModule_AddIntMacro(glpk_module,GLP_RT_HAR);

  /* interior-point solver control parameters */
  /* ordering algorithm: */
  PyModule_AddIntMacro(glpk_module,GLP_ORD_NONE);
  PyModule_AddIntMacro(glpk_module,GLP_ORD_QMD);
  PyModule_AddIntMacro(glpk_module,GLP_ORD_AMD);
  PyModule_AddIntMacro(glpk_module,GLP_ORD_SYMAMD);
}

PyMODINIT_FUNC PyInit_glpk(void)
{
  if (!(glpk_module = PyModule_Create(&glpk_module_def))) return NULL;
  if (PyType_Ready(&iocp_t) < 0 || (PyType_Ready(&smcp_t) < 0)) return NULL;
  /*  Adding macros */
  addglpkConstants();
  /* Adding  option lists as objects */
  Py_INCREF(&smcp_t);
  PyModule_AddObject(glpk_module,"smcp",(PyObject*)&smcp_t);
  Py_INCREF(&iocp_t);
  PyModule_AddObject(glpk_module,"iocp",(PyObject*)&iocp_t);
  if (import_cvxopt() < 0) return NULL;
  return glpk_module;
}

#else

PyMODINIT_FUNC initglpk(void)
{
    glpk_module = Py_InitModule3("cvxopt.glpk", glpk_functions, 
            glpk__doc__);
    if (PyType_Ready(&iocp_t) < 0 || (PyType_Ready(&smcp_t) < 0)) return NULL;
    addglpkConstants();
    Py_INCREF(&smcp_t);
    PyModule_AddObject(glpk_module,"smcp",(PyObject*)&smcp_t);
    Py_INCREF(&iocp_t);
    PyModule_AddObject(glpk_module,"iocp",(PyObject*)&iocp_t);
    if (import_cvxopt() < 0) return;
}

#endif
