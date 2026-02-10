
/*
    A modified version of the example in the Python documentation:
    http://docs.python.org/ext/pure-embedding.html

    Solves the LP:

    minimize -4*x1 -5*x2
    subject  2*x1 + x2 <= 3
             x1 + 2*x2 <= 3
             x1 >= 0,  x2 >= 0

    and prints the solution from C.

    On Linux, set CVXOPT_SRC to the root of the src directory
    in the cvxopt source tree and compile with:

    gcc -o embed_cvxopt embed_cvxopt.c \
        -I${CVXOPT_SRC}/C \
        $(python3.13-config --cflags) \
        $(python3.13-config --embed --ldflags)

    Change the Python version as needed.

    On macOS/conda, additional flags may be needed to link against the correct Python framework:

    gcc -o embed_cvxopt embed_cvxopt.c \
        -I${CVXOPT_SRC}/C \
        $(python3.13-config --cflags) \
        -L"$CONDA_PREFIX/lib" \
        -Wl,-rpath,"$CONDA_PREFIX/lib" \
        $(python3.13-config --embed --ldflags)
*/

#include "cvxopt.h"

int main(int argc, char *argv[])
{
    Py_Initialize();

    if (import_cvxopt() < 0)
    {
        fprintf(stderr, "error importing cvxopt");
        return 1;
    }

    /* import cvxopt.solvers */
    PyObject *solvers = PyImport_ImportModule("cvxopt.solvers");
    if (!solvers)
    {
        fprintf(stderr, "error importing cvxopt.solvers");
        return 1;
    }

    /* get reference to solvers.solvelp */
    PyObject *lp = PyObject_GetAttrString(solvers, "lp");
    if (!lp)
    {
        fprintf(stderr, "error referencing cvxopt.solvers.lp");
        Py_DECREF(solvers);
        return 1;
    }

    /* create matrices */
    PyObject *c = (PyObject *)Matrix_New(2, 1, DOUBLE);
    PyObject *G = (PyObject *)Matrix_New(4, 2, DOUBLE);
    PyObject *h = (PyObject *)Matrix_New(4, 1, DOUBLE);
    PyObject *pArgs = PyTuple_New(3);
    if (!c || !G || !h || !pArgs)
    {
        fprintf(stderr, "error creating matrices");
        Py_DECREF(solvers);
        Py_DECREF(lp);
        Py_XDECREF(c);
        Py_XDECREF(G);
        Py_XDECREF(h);
        Py_XDECREF(pArgs);
        return 1;
    }

    MAT_BUFD(c)
    [0] = -4;
    MAT_BUFD(c)
    [1] = -5;

    MAT_BUFD(G)
    [0] = 2;
    MAT_BUFD(G)
    [1] = 1;
    MAT_BUFD(G)
    [2] = -1;
    MAT_BUFD(G)
    [4] = 1;
    MAT_BUFD(G)
    [5] = 2;
    MAT_BUFD(G)
    [7] = -1;

    MAT_BUFD(h)
    [0] = 3;
    MAT_BUFD(h)
    [1] = 3;

    /* pack matrices into an argument tuple - references are stolen*/
    PyTuple_SetItem(pArgs, 0, c);
    PyTuple_SetItem(pArgs, 1, G);
    PyTuple_SetItem(pArgs, 2, h);

    PyObject *sol = PyObject_CallObject(lp, pArgs);
    if (!sol)
    {
        PyErr_Print();
        Py_DECREF(solvers);
        Py_DECREF(lp);
        Py_DECREF(pArgs);
        return 1;
    }

    PyObject *x = PyDict_GetItemString(sol, "x");
    fprintf(stdout, "\n\nx = (%5.4e, %5.4e)\n", MAT_BUFD(x)[0], MAT_BUFD(x)[1]);

    Py_DECREF(solvers);
    Py_DECREF(lp);
    Py_DECREF(pArgs);
    Py_DECREF(sol);

    Py_Finalize();
    return 0;
}
