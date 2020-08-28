.. _c-printing:

*****************
Matrix Formatting
*****************

This appendix describes ways to customize the formatting of kvxopt matrices.

As with other Python objects, the functions :func:`repr` and :func:`str` 
return strings with printable representations of matrices.  The command 
'``print A``' executes '``str(A)``', whereas the command '``A``'
calls '``repr(A)``'.  The following example illustrates the default 
formatting of dense matrices.

>>> from kvxopt import matrix 
>>> A = matrix(range(50), (5,10), 'd')
>>> A  
<5x10 matrix, tc='d'>
>>> print(A)
[ 0.00e+00  5.00e+00  1.00e+01  1.50e+01  2.00e+01  2.50e+01  3.00e+01 ... ]
[ 1.00e+00  6.00e+00  1.10e+01  1.60e+01  2.10e+01  2.60e+01  3.10e+01 ... ]
[ 2.00e+00  7.00e+00  1.20e+01  1.70e+01  2.20e+01  2.70e+01  3.20e+01 ... ]
[ 3.00e+00  8.00e+00  1.30e+01  1.80e+01  2.30e+01  2.80e+01  3.30e+01 ... ]
[ 4.00e+00  9.00e+00  1.40e+01  1.90e+01  2.40e+01  2.90e+01  3.40e+01 ... ]

The format is parameterized by the dictionary :data:`options` in the 
module :mod:`kvxopt.printing`.  The parameters :attr:`options['iformat']` 
and :attr:`options['dformat']` determine, respectively, how integer and 
double/complex numbers are printed.  The entries are Python format strings 
with default values :const:`'\% .2e'` for :const:`'d'` and :const:`'z'` 
matrices and :const:`\% i'` for :const:`'i'` matrices.  The parameters 
:attr:`options['width']` and :attr:`options['height']` specify the maximum 
number of columns and rows that are shown.  If :attr:`options['width']` is 
set to a negative value, all columns are displayed.  If 
:attr:`options['height']` is set to a negative value, all rows are 
displayed.  The default values of :attr:`options['width']` and 
:attr:`options['height']` are 7 and -1, respectively.

>>> from kvxopt import printing
>>> printing.options
{'width': 7, 'dformat': '% .2e', 'iformat': '% i', 'height': -1}
>>> printing.options['dformat'] = '%.1f'
>>> printing.options['width'] = -1
>>> print(A)
[ 0.0  5.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0 45.0]
[ 1.0  6.0 11.0 16.0 21.0 26.0 31.0 36.0 41.0 46.0]
[ 2.0  7.0 12.0 17.0 22.0 27.0 32.0 37.0 42.0 47.0]
[ 3.0  8.0 13.0 18.0 23.0 28.0 33.0 38.0 43.0 48.0]
[ 4.0  9.0 14.0 19.0 24.0 29.0 34.0 39.0 44.0 49.0]


In order to make the built-in Python functions :func:`repr` and :func:`str`
accessible for further customization, two functions are provided in 
kvxopt.  The function :func:`kvxopt.matrix_repr` is used when 
:func:`repr` is called with a matrix argument; and 
:func:`kvxopt.matrix_str` is used when :func:`str` is called with a matrix 
argument.  By default, the functions are set to 
:func:`printing.matrix_repr_default` and
:func:`printing.matrix_str_default`, respectively, but they can be 
redefined to any other Python functions.  For example, if we prefer 
``A`` to return the same output as ``print A``, we can simply 
redefine :func:`kvxopt.matrix_repr` as shown below.

>>> import kvxopt
>>> from kvxopt import matrix, printing
>>> A = matrix(range(4), (2,2), 'd')
>>> A
<2x2 matrix, tc='d'>
>>> kvxopt.matrix_repr = printing.matrix_str_default
>>> A
[ 0.00e+00  2.00e+00]
[ 1.00e+00  3.00e+00]


The formatting for sparse matrices is similar.  The functions :func:`repr` 
and :func:`str` for sparse matrices are :func:`kvxopt.spmatrix_repr` 
and :func:`kvxopt.spmatrix_str`, respectively.  By default, they are set to
:func:`printing.spmatrix_repr_default` and 
:func:`printing.spmatrix_repr_str`.


>>> import kvxopt
>>> from kvxopt import printing, spmatrix 
>>> A = spmatrix(range(5), range(5), range(5), (5,10))
>>> A
<5x10 sparse matrix, tc='d', nnz=5>
>>> print(A)
[ 0.00e+00     0         0         0         0         0         0     ... ]
[    0      1.00e+00     0         0         0         0         0     ... ]
[    0         0      2.00e+00     0         0         0         0     ... ]
[    0         0         0      3.00e+00     0         0         0     ... ]
[    0         0         0         0      4.00e+00     0         0     ... ]

>>> kvxopt.spmatrix_repr = printing.spmatrix_str_default
>>> A
[ 0.00e+00     0         0         0         0         0         0     ... ]
[    0      1.00e+00     0         0         0         0         0     ... ]
[    0         0      2.00e+00     0         0         0         0     ... ]
[    0         0         0      3.00e+00     0         0         0     ... ]
[    0         0         0         0      4.00e+00     0         0     ... ]


As can be seen from the example, the default behaviour is to print the 
entire matrix including structural zeros. An alternative triplet printing 
style is defined in :func:`printing.spmatrix_str_triplet`. 

>>> kvxopt.spmatrix_str = printing.spmatrix_str_triplet
>>> print(A)
(0,0)  0.00e+00
(1,1)  1.00e+00
(2,2)  2.00e+00
(3,3)  3.00e+00
(4,4)  4.00e+00
