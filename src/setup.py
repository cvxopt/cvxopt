from distutils.core import setup, Extension
from glob import glob

# Modifiy this if BLAS and LAPACK libraries are not in /usr/lib.
BLAS_LIB_DIR = '/usr/lib'

# Default names of BLAS and LAPACK libraries
BLAS_LIB = ['blas']
LAPACK_LIB = ['lapack']
BLAS_EXTRA_LINK_ARGS = []

# Set environment variable BLAS_NOUNDERSCORES=1 if your BLAS/LAPACK do
# not use trailing underscores
BLAS_NOUNDERSCORES = False

# Set to 1 if you are using the random number generators in the GNU
# Scientific Library.
BUILD_GSL = 0

# Directory containing libgsl (used only when BUILD_GSL = 1).
GSL_LIB_DIR = '/usr/lib'

# Directory containing the GSL header files (used only when BUILD_GSL = 1).
GSL_INC_DIR = '/usr/include/gsl'

# Set to 1 if you are installing the fftw module.
BUILD_FFTW = 0 

# Directory containing libfftw3 (used only when BUILD_FFTW = 1).
FFTW_LIB_DIR = '/usr/lib'

# Directory containing fftw.h (used only when BUILD_FFTW = 1).
FFTW_INC_DIR = '/usr/include'

# Set to 1 if you are installing the glpk module.
BUILD_GLPK = 0 

# Directory containing libglpk (used only when BUILD_GLPK = 1).
GLPK_LIB_DIR = '/usr/lib'

# Directory containing glpk.h (used only when BUILD_GLPK = 1).
GLPK_INC_DIR = '/usr/include'

# Set to 1 if you are installing the DSDP module.
BUILD_DSDP = 0

# Directory containing libdsdp (used only when BUILD_DSDP = 1).
DSDP_LIB_DIR = '/usr/lib'
 
# Directory containing dsdp5.h (used only when BUILD_DSDP = 1).
DSDP_INC_DIR = '/usr/include/dsdp'

# No modifications should be needed below this line.

extmods = []

# Macros
MACROS = []
if BLAS_NOUNDERSCORES: MACROS.append(('BLAS_NO_UNDERSCORE',''))

# optional modules

if BUILD_GSL:
    gsl = Extension('gsl', libraries = ['m', 'gsl'] + BLAS_LIB,
        include_dirs = [ GSL_INC_DIR ],
        library_dirs = [ GSL_LIB_DIR, BLAS_LIB_DIR ],
        extra_link_args = BLAS_EXTRA_LINK_ARGS,
        sources = ['C/gsl.c'] )
    extmods += [gsl];

if BUILD_FFTW:
    fftw = Extension('fftw', libraries = ['fftw3'] + BLAS_LIB,
        include_dirs = [ FFTW_INC_DIR ],
        library_dirs = [ FFTW_LIB_DIR, BLAS_LIB_DIR ],
        extra_link_args = BLAS_EXTRA_LINK_ARGS,
        sources = ['C/fftw.c'] )
    extmods += [fftw];

if BUILD_GLPK:
    glpk = Extension('glpk', libraries = ['glpk'],
        include_dirs = [ GLPK_INC_DIR ],
        library_dirs = [ GLPK_LIB_DIR ],
        sources = ['C/glpk.c'] )
    extmods += [glpk];

if BUILD_DSDP:
    dsdp = Extension('dsdp', libraries = ['dsdp'] + LAPACK_LIB + BLAS_LIB,
        include_dirs = [ DSDP_INC_DIR ],
        library_dirs = [ DSDP_LIB_DIR, BLAS_LIB_DIR ],
        extra_link_args = BLAS_EXTRA_LINK_ARGS,
        sources = ['C/dsdp.c'] )
    extmods += [dsdp];

# Required modules

base = Extension('base', libraries = ['m'] + LAPACK_LIB + BLAS_LIB,
    library_dirs = [ BLAS_LIB_DIR ],
    define_macros = MACROS,
    extra_link_args = BLAS_EXTRA_LINK_ARGS,
    sources = ['C/base.c','C/dense.c','C/sparse.c']) 

blas = Extension('blas', libraries = BLAS_LIB,
    library_dirs = [ BLAS_LIB_DIR ],
    define_macros = MACROS,
    extra_link_args = BLAS_EXTRA_LINK_ARGS,
    sources = ['C/blas.c'] )

lapack = Extension('lapack', libraries = LAPACK_LIB + BLAS_LIB,
    library_dirs = [ BLAS_LIB_DIR ],
    define_macros = MACROS,
    extra_link_args = BLAS_EXTRA_LINK_ARGS,
    sources = ['C/lapack.c'] )

umfpack = Extension('umfpack', 
    include_dirs = [ 'C/SuiteSparse/UMFPACK/Include',
        'C/SuiteSparse/AMD/Include', 'C/SuiteSparse/AMD/Source', 
        'C/SuiteSparse/UFconfig' ],
    library_dirs = [ BLAS_LIB_DIR ],
    define_macros = MACROS,
    libraries = LAPACK_LIB + BLAS_LIB,
    extra_link_args = BLAS_EXTRA_LINK_ARGS,
    sources = [ 'C/umfpack.c',
        'C/SuiteSparse/UMFPACK/Source/umfpack_global.c',
        'C/SuiteSparse/UMFPACK/Source/umfpack_tictoc.c' ] +
        glob('C/SuiteSparse_cvxopt_extra/umfpack/*'))

# Build for int or long? 
import sys
if sys.maxsize > 2**31: MACROS += [('DLONG','')]

cholmod = Extension('cholmod',
    library_dirs = [ BLAS_LIB_DIR ],
    libraries = LAPACK_LIB + BLAS_LIB,
    include_dirs = [ 'C/SuiteSparse/CHOLMOD/Include', 
        'C/SuiteSparse/COLAMD', 'C/SuiteSparse/AMD/Include', 
        'C/SuiteSparse/UFconfig', 'C/SuiteSparse/COLAMD/Include' ],
    define_macros = MACROS + [('NPARTITION', '1')],
    extra_link_args = BLAS_EXTRA_LINK_ARGS,
    sources = [ 'C/cholmod.c' ] + 
        ['C/SuiteSparse/AMD/Source/' + s for s in ['amd_global.c',
            'amd_postorder.c', 'amd_post_tree.c', 'amd_2.c']] +
        ['C/SuiteSparse/COLAMD/Source/' + s for s in ['colamd.c',
            'colamd_global.c']] +
        glob('C/SuiteSparse/CHOLMOD/Core/c*.c') +
        glob('C/SuiteSparse/CHOLMOD/Cholesky/c*.c') +
        ['C/SuiteSparse/CHOLMOD/Check/cholmod_check.c'] +
        glob('C/SuiteSparse/CHOLMOD/Supernodal/c*.c') )

amd = Extension('amd', 
    include_dirs = [ 'C/SuiteSparse/AMD/Include', 
        'C/SuiteSparse/UFconfig' ],
    define_macros = MACROS,
    sources = [ 'C/amd.c' ] + glob('C/SuiteSparse/AMD/Source/*.c') )

misc_solvers = Extension('misc_solvers',
    libraries = LAPACK_LIB + BLAS_LIB,
    library_dirs = [ BLAS_LIB_DIR ],
    define_macros = MACROS,
    extra_link_args = BLAS_EXTRA_LINK_ARGS,
    sources = ['C/misc_solvers.c'] )

extmods += [base, blas, lapack, umfpack, cholmod, amd, misc_solvers] 

setup (name = 'cvxopt', 
    description = 'Convex optimization package',
    version = '1.1.5', 
    long_description = '''
CVXOPT is a free software package for convex optimization based on the 
Python programming language. It can be used with the interactive Python 
interpreter, on the command line by executing Python scripts, or 
integrated in other software via Python extension modules. Its main 
purpose is to make the development of software for convex optimization 
applications straightforward by building on Python's extensive standard 
library and on the strengths of Python as a high-level programming 
language.''', 
    author = 'M. Andersen, J. Dahl, and L. Vandenberghe',
    author_email = 'martin.skovgaard.andersen@gmail.com, dahl.joachim@gmail.com, vandenbe@ee.ucla.edu',
    url = 'http://abel.ee.ucla.edu/cvxopt',
    license = 'GNU GPL version 3',
    ext_package = "cvxopt",
    ext_modules = extmods,
    package_dir = {"cvxopt": "python"},
    packages = ["cvxopt"])
