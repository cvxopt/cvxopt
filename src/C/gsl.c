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

#include <complex.h>

#include "misc.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <time.h>

PyDoc_STRVAR(gsl__doc__,"Random Module.");

static unsigned long seed = 0;
static const gsl_rng_type *rng_type;
static gsl_rng *rng;

static char doc_getseed[] =
  "Returns the seed value for the random number generator.\n\n"
  "getseed()";

static PyObject * getseed(PyObject *self)
{
  return Py_BuildValue("l",seed);
}

static char doc_setseed[] =
  "Sets the seed value for the random number generator.\n\n"
  "setseed(value = 0)\n\n"
  "ARGUMENTS\n"
  "value     integer seed. If the value is 0, then the system clock\n"
  "          measured in seconds is used instead";

static PyObject * setseed(PyObject *self, PyObject *args)
{
  unsigned long seed_ = 0;
  time_t seconds;

  if (!PyArg_ParseTuple(args, "|l", &seed_))
    return NULL;

  if (!seed_) {
    time(&seconds);
    seed = (unsigned long)seconds;
  }
  else seed = seed_;

  return Py_BuildValue("");
}


static char doc_normal[] =
  "Randomly generates a matrix with normally distributed entries.\n\n"
  "normal(nrows, ncols=1, mean=0, std=1)\n\n"
  "PURPOSE\n"
  "Returns a matrix with typecode 'd' and size nrows by ncols, with\n"
  "its entries randomly generated from a normal distribution with mean\n"
  "m and standard deviation std.\n\n"
  "ARGUMENTS\n"
  "nrows     number of rows\n\n"
  "ncols     number of columns\n\n"
  "mean      approximate mean of the distribution\n\n"
  "std       standard deviation of the distribution";
static PyObject *
normal(PyObject *self, PyObject *args, PyObject *kwrds)
{
  matrix *obj;
  int i, nrows, ncols = 1;
  double m = 0, s = 1;
  char *kwlist[] = {"nrows", "ncols", "mean", "std",  NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "i|idd", kwlist,
	  &nrows, &ncols, &m, &s)) return NULL;

  if (s < 0.0) PY_ERR(PyExc_ValueError, "std must be non-negative");

  if ((nrows<0) || (ncols<0)) {
    PyErr_SetString(PyExc_TypeError, "dimensions must be non-negative");
    return NULL;
  }

  if (!(obj = Matrix_New(nrows, ncols, DOUBLE)))
    return PyErr_NoMemory();

  gsl_rng_env_setup();
  rng_type = gsl_rng_default;
  rng = gsl_rng_alloc (rng_type);
  gsl_rng_set(rng, seed);

  for (i = 0; i < nrows*ncols; i++)
    MAT_BUFD(obj)[i] = gsl_ran_gaussian (rng, s) + m;

  seed = gsl_rng_get (rng);
  gsl_rng_free(rng);

  return (PyObject *)obj;
}

static char doc_uniform[] =
  "Randomly generates a matrix with uniformly distributed entries.\n\n"
  "uniform(nrows, ncols=1, a=0, b=1)\n\n"
  "PURPOSE\n"
  "Returns a matrix with typecode 'd' and size nrows by ncols, with\n"
  "its entries randomly generated from a uniform distribution on the\n"
  "interval (a,b).\n\n"
  "ARGUMENTS\n"
  "nrows     number of rows\n\n"
  "ncols     number of columns\n\n"
  "a         lower bound\n\n"
  "b         upper bound";

static PyObject *
uniform(PyObject *self, PyObject *args, PyObject *kwrds)
{
  matrix *obj;
  int i, nrows, ncols = 1;
  double a = 0, b = 1;

  char *kwlist[] = {"nrows", "ncols", "a", "b", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwrds, "i|idd", kwlist,
	  &nrows, &ncols, &a, &b)) return NULL;

  if (a>b) PY_ERR(PyExc_ValueError, "a must be less than b");

  if ((nrows<0) || (ncols<0))
    PY_ERR_TYPE("dimensions must be non-negative");

  if (!(obj = (matrix *)Matrix_New(nrows, ncols, DOUBLE)))
    return PyErr_NoMemory();

  gsl_rng_env_setup();
  rng_type = gsl_rng_default;
  rng = gsl_rng_alloc (rng_type);
  gsl_rng_set(rng, seed);

  for (i= 0; i < nrows*ncols; i++)
    MAT_BUFD(obj)[i] = gsl_ran_flat (rng, a, b);

  seed = gsl_rng_get (rng);
  gsl_rng_free(rng);

  return (PyObject *)obj;
}

static PyMethodDef gsl_functions[] = {
{"getseed", (PyCFunction)getseed, METH_VARARGS|METH_KEYWORDS, doc_getseed},
{"setseed", (PyCFunction)setseed, METH_VARARGS|METH_KEYWORDS, doc_setseed},
{"normal", (PyCFunction)normal, METH_VARARGS|METH_KEYWORDS, doc_normal},
{"uniform", (PyCFunction)uniform, METH_VARARGS|METH_KEYWORDS, doc_uniform},
{NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static PyModuleDef gsl_module = {
    PyModuleDef_HEAD_INIT,
    "gsl",
    gsl__doc__,
    -1,
    gsl_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_gsl(void)
{
  PyObject *m;
  if (!(m = PyModule_Create(&gsl_module))) return NULL;
  if (import_cvxopt() < 0) return NULL;
  return m;
}

#else

PyMODINIT_FUNC initgsl(void)
{
  PyObject *m;
  m = Py_InitModule3("cvxopt.gsl", gsl_functions, gsl__doc__);
  if (import_cvxopt() < 0) return;
}

#endif
