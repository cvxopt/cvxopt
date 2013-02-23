.. _intro:

************
Introduction
************

CVXOPT is a free software package for convex optimization based on the 
Python programming language.  It can be used with the interactive Python 
interpreter, on the command line by executing Python scripts, or integrated
in other software via Python extension modules.  Its main purpose is to 
make the development of software for convex optimization applications 
straightforward by building on Python's extensive standard library and on 
the strengths of Python as a high-level programming language.  

CVXOPT extends the built-in Python objects with two matrix objects: a 
:class:`matrix <cvxopt.matrix>`  object for dense matrices and an 
:class:`spmatrix <cvxopt.spmatrix>` object for sparse matrices.  These two 
matrix types are introduced in the chapter :ref:`c-matrices`, together 
with the arithmetic operations and functions defined for them.  The 
following chapters (:ref:`c-blas` and :ref:`c-spsolvers`) describe 
interfaces to several libraries for dense and sparse matrix computations.  
The CVXOPT optimization routines are described in the chapters 
:ref:`c-coneprog` and :ref:`c-modeling`.
These include convex optimization solvers written in Python, 
interfaces to a few other optimization libraries, and a modeling tool 
for piecewise-linear convex optimization problems.

CVXOPT is organized in different modules.  

:mod:`cvxopt.blas <cvxopt.blas>` 
  Interface to most of the double-precision real and complex BLAS 
  (:ref:`c-blas`).

:mod:`cvxopt.lapack <cvxopt.lapack>`
  Interface to dense double-precision real and complex linear equation 
  solvers and eigenvalue routines from LAPACK (:ref:`c-lapack`).

:mod:`cvxopt.fftw <cvxopt.fftw>` 
  An optional interface to the discrete transform routines from FFTW 
  (:ref:`c-fftw`). 

:mod:`cvxopt.amd <cvxopt.amd>`  
  Interface to the approximate minimum degree ordering routine from AMD 
  (:ref:`s-orderings`).

:mod:`cvxopt.umfpack <cvxopt.umfpack>` 
  Interface to the sparse LU solver from UMFPACK (:ref:`s-umfpack`).

:mod:`cvxopt.cholmod <cvxopt.cholmod>`  
  Interface to the sparse Cholesky solver from CHOLMOD (:ref:`s-cholmod`).

:mod:`cvxopt.solvers <cvxopt.solvers>` 
  Convex optimization routines and optional interfaces to solvers from 
  GLPK, MOSEK, and DSDP5 (:ref:`c-coneprog` and :ref:`c-solvers`).

:mod:`cvxopt.modeling <cvxopt.modeling>`   
  Routines for specifying and solving linear programs and convex 
  optimization problems with piecewise-linear cost and constraint functions
  (:ref:`c-modeling`).

:mod:`cvxopt.info <cvxopt.info>`  
  Defines a string :const:`version` with the version number of the CVXOPT 
  installation and a function :func:`license` that prints the CVXOPT 
  license.  

:mod:`cvxopt.printing <cvxopt.printing>` 
  Contains functions and parameters that control how matrices are formatted.

The modules are described in detail in this manual and in the on-line Python
help facility :program:`pydoc`.  Several example scripts are included in 
the distribution. 
