try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
from glob import glob
import os

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

# Set to 1 to use external SuiteSparse library
SUITESPARSE_EXT_LIB = 0

# Directory containing external SuiteSparse library
SUITESPARSE_LIB_DIR = '/usr/local/lib'

# Directory containing SuiteSparse header files
SUITESPARSE_INC_DIR = '/usr/local/include'

# No modifications should be needed below this line.

BLAS_NOUNDERSCORES = int(os.environ.get("CVXOPT_BLAS_NOUNDERSCORES",BLAS_NOUNDERSCORES)) == True
BLAS_LIB = os.environ.get("CVXOPT_BLAS_LIB",BLAS_LIB)
LAPACK_LIB = os.environ.get("CVXOPT_LAPACK_LIB",LAPACK_LIB)
BLAS_LIB_DIR = os.environ.get("CVXOPT_BLAS_LIB_DIR",BLAS_LIB_DIR)
BLAS_EXTRA_LINK_ARGS = os.environ.get("CVXOPT_BLAS_EXTRA_LINK_ARGS",BLAS_EXTRA_LINK_ARGS)
if type(BLAS_LIB) is str: BLAS_LIB = BLAS_LIB.strip().split(',')
if type(LAPACK_LIB) is str: LAPACK_LIB = LAPACK_LIB.strip().split(',')
if type(BLAS_EXTRA_LINK_ARGS) is str: BLAS_EXTRA_LINK_ARGS = BLAS_EXTRA_LINK_ARGS.strip().split(',')
BUILD_GSL = int(os.environ.get("CVXOPT_BUILD_GSL",BUILD_GSL))
GSL_LIB_DIR = os.environ.get("CVXOPT_GSL_LIB_DIR",GSL_LIB_DIR)
GSL_INC_DIR = os.environ.get("CVXOPT_GSL_INC_DIR",GSL_INC_DIR)
BUILD_FFTW = int(os.environ.get("CVXOPT_BUILD_FFTW",BUILD_FFTW))
FFTW_LIB_DIR = os.environ.get("CVXOPT_FFTW_LIB_DIR",FFTW_LIB_DIR)
FFTW_INC_DIR = os.environ.get("CVXOPT_FFTW_INC_DIR",FFTW_INC_DIR)
BUILD_GLPK = int(os.environ.get("CVXOPT_BUILD_GLPK",BUILD_GLPK))
GLPK_LIB_DIR = os.environ.get("CVXOPT_GLPK_LIB_DIR",GLPK_LIB_DIR)
GLPK_INC_DIR = os.environ.get("CVXOPT_GLPK_INC_DIR",GLPK_INC_DIR)
BUILD_DSDP = int(os.environ.get("CVXOPT_BUILD_DSDP",BUILD_DSDP))
DSDP_LIB_DIR = os.environ.get("CVXOPT_DSDP_LIB_DIR",DSDP_LIB_DIR)
DSDP_INC_DIR = os.environ.get("CVXOPT_DSDP_INC_DIR",DSDP_INC_DIR)
SUITESPARSE_EXT_LIB = os.environ.get("CVXOPT_SUITESPARSE_EXT_LIB",SUITESPARSE_EXT_LIB)
SUITESPARSE_LIB_DIR = os.environ.get("CVXOPT_AMD_EXT_LIB",SUITESPARSE_LIB_DIR)
SUITESPARSE_INC_DIR = os.environ.get("CVXOPT_AMD_EXT_LIB",SUITESPARSE_INC_DIR)

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
        sources = ['src/C/gsl.c'] )
    extmods += [gsl];

if BUILD_FFTW:
    fftw = Extension('fftw', libraries = ['fftw3'] + BLAS_LIB,
        include_dirs = [ FFTW_INC_DIR ],
        library_dirs = [ FFTW_LIB_DIR, BLAS_LIB_DIR ],
        extra_link_args = BLAS_EXTRA_LINK_ARGS,
        sources = ['src/C/fftw.c'] )
    extmods += [fftw];

if BUILD_GLPK:
    glpk = Extension('glpk', libraries = ['glpk'],
        include_dirs = [ GLPK_INC_DIR ],
        library_dirs = [ GLPK_LIB_DIR ],
        sources = ['src/C/glpk.c'] )
    extmods += [glpk];

if BUILD_DSDP:
    dsdp = Extension('dsdp', libraries = ['dsdp'] + LAPACK_LIB + BLAS_LIB,
        include_dirs = [ DSDP_INC_DIR ],
        library_dirs = [ DSDP_LIB_DIR, BLAS_LIB_DIR ],
        extra_link_args = BLAS_EXTRA_LINK_ARGS,
        sources = ['src/C/dsdp.c'] )
    extmods += [dsdp];

# Required modules

base = Extension('base', libraries = ['m'] + LAPACK_LIB + BLAS_LIB,
    library_dirs = [ BLAS_LIB_DIR ],
    define_macros = MACROS,
    extra_link_args = BLAS_EXTRA_LINK_ARGS,
    sources = ['src/C/base.c','src/C/dense.c','src/C/sparse.c']) 

blas = Extension('blas', libraries = BLAS_LIB,
    library_dirs = [ BLAS_LIB_DIR ],
    define_macros = MACROS,
    extra_link_args = BLAS_EXTRA_LINK_ARGS,
    sources = ['src/C/blas.c'] )

lapack = Extension('lapack', libraries = LAPACK_LIB + BLAS_LIB,
    library_dirs = [ BLAS_LIB_DIR ],
    define_macros = MACROS,
    extra_link_args = BLAS_EXTRA_LINK_ARGS,
    sources = ['src/C/lapack.c'] )

if SUITESPARSE_EXT_LIB:
    umfpack = Extension('umfpack',
        libraries = ['amd','colamd','suitesparseconfig','cholmod','umfpack'],
        include_dirs = [SUITESPARSE_INC_DIR],
        library_dirs = [SUITESPARSE_LIB_DIR],
        sources = ['src/C/umfpack.c'])
else:
    umfpack = Extension('umfpack', 
        include_dirs = [ 'src/C/SuiteSparse/UMFPACK/Include',
            'src/C/SuiteSparse/AMD/Include', 
            'src/C/SuiteSparse/AMD/Source', 
            'src/C/SuiteSparse/SuiteSparse_config' ],
        library_dirs = [ BLAS_LIB_DIR ],
        define_macros = MACROS + [('NTIMER', '1'), ('NCHOLMOD', '1')],
        libraries = LAPACK_LIB + BLAS_LIB,
        extra_compile_args = ['-Wno-unknown-pragmas'],
        extra_link_args = BLAS_EXTRA_LINK_ARGS,
        sources = [ 'src/C/umfpack.c',
            'src/C/SuiteSparse/UMFPACK/Source/umfpack_global.c',
            'src/C/SuiteSparse/UMFPACK/Source/umfpack_tictoc.c' ] +
            ['src/C/SuiteSparse/SuiteSparse_config/SuiteSparse_config.c'] +
            glob('src/C/SuiteSparse_cvxopt_extra/umfpack/*'))

# Build for int or long? 
import sys
if sys.maxsize > 2**31: MACROS += [('DLONG',None)]

if SUITESPARSE_EXT_LIB:
    cholmod = Extension('cholmod',
        libraries = ['amd','colamd','suitesparseconfig','cholmod'],
        include_dirs = [SUITESPARSE_INC_DIR],
        library_dirs = [SUITESPARSE_LIB_DIR],
        sources = [ 'src/C/cholmod.c' ])
else:
    cholmod = Extension('cholmod',
        library_dirs = [ BLAS_LIB_DIR ],
        libraries = LAPACK_LIB + BLAS_LIB,
        include_dirs = [ 'src/C/SuiteSparse/CHOLMOD/Include', 
            'src/C/SuiteSparse/COLAMD', 
            'src/C/SuiteSparse/AMD/Include', 
            'src/C/SuiteSparse/COLAMD/Include',
            'src/C/SuiteSparse/SuiteSparse_config' ],
        define_macros = MACROS + [('NPARTITION', '1'), ('NTIMER', '1')],
        extra_link_args = BLAS_EXTRA_LINK_ARGS,
        sources = [ 'src/C/cholmod.c' ] + 
            ['src/C/SuiteSparse/AMD/Source/' + s for s in ['amd_global.c',
                'amd_postorder.c', 'amd_post_tree.c', 'amd_2.c']] +
            ['src/C/SuiteSparse/COLAMD/Source/' + s for s in ['colamd.c',
                'colamd_global.c']] +
            ['src/C/SuiteSparse/SuiteSparse_config/SuiteSparse_config.c'] +
            glob('src/C/SuiteSparse/CHOLMOD/Core/c*.c') +
            glob('src/C/SuiteSparse/CHOLMOD/Cholesky/c*.c') +
            ['src/C/SuiteSparse/CHOLMOD/Check/cholmod_check.c'] +
            glob('src/C/SuiteSparse/CHOLMOD/Supernodal/c*.c') )

if SUITESPARSE_EXT_LIB:
    amd = Extension('amd',
        libraries = ['amd','suitesparseconfig'],
        include_dirs = [SUITESPARSE_INC_DIR],
        library_dirs = [SUITESPARSE_LIB_DIR],
        sources = ['src/C/amd.c'])
else:
    amd = Extension('amd', 
        include_dirs = [ 'src/C/SuiteSparse/AMD/Include', 
            'src/C/SuiteSparse/SuiteSparse_config' ],
        define_macros = MACROS,
        sources = [ 'src/C/amd.c', 'src/C/SuiteSparse/SuiteSparse_config/SuiteSparse_config.c'] +
        glob('src/C/SuiteSparse/AMD/Source/*.c') )

misc_solvers = Extension('misc_solvers',
    libraries = LAPACK_LIB + BLAS_LIB,
    library_dirs = [ BLAS_LIB_DIR ],
    define_macros = MACROS,
    extra_link_args = BLAS_EXTRA_LINK_ARGS,
    sources = ['src/C/misc_solvers.c'] )

extmods += [base, blas, lapack, umfpack, cholmod, amd, misc_solvers] 

setup (name = 'cvxopt', 
    description = 'Convex optimization package',
    version = '1.1.8', 
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
    url = 'http://cvxopt.org',
    license = 'GNU GPL version 3',
    ext_package = "cvxopt",
    ext_modules = extmods,
    package_dir = {"cvxopt": "src/python"},
    packages = ["cvxopt"],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        ],
    )
