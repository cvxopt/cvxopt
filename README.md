CVXOPT
======

Python Software for Convex Optimization

* [Website](http://cvxopt.org)
* [Documentation](http://cvxopt.org/userguide)
* [Google Groups forum](https://groups.google.com/forum/#!forum/cvxopt)


Release info
------------

[![License](https://img.shields.io/badge/license-GPL3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![GitHub release](https://img.shields.io/github/release/cvxopt/cvxopt.svg)](https://github.com/cvxopt/cvxopt/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/cvxopt.svg)](https://pypi.python.org/pypi/cvxopt)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/cvxopt/badges/version.svg)](https://anaconda.org/conda-forge/cvxopt)


Installation
------------

Pre-built wheels for Linux, macOS and Windows are on PyPI:

```bash
pip install cvxopt
```

Packages are also available on conda-forge:

```bash
conda install -c conda-forge cvxopt
```

The wheels include the optional GLPK, DSDP, FFTW and GSL extensions (GLPK
only on Windows). The MOSEK interface additionally needs the `mosek`
Python package:

```bash
pip install cvxopt mosek
```

See [INSTALL](INSTALL) for details on the pre-built packages.


Quickstart
----------

Solve the small linear program

    minimize    -4x - 5y
    subject to  2x + y <= 3
                x + 2y <= 3
                x >= 0, y >= 0

```python
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

c = matrix([-4.0, -5.0])
G = matrix([[2.0, 1.0, -1.0, 0.0], [1.0, 2.0, 0.0, -1.0]])
h = matrix([3.0, 3.0, 0.0, 0.0])

sol = solvers.lp(c, G, h)
print(sol['status'])
print(sol['x'])
```

```result
optimal
[ 1.00e+00]
[ 1.00e+00]
```

Note that `matrix` is column-major: each inner list of `G` is one column.
More examples are in the [examples](examples) directory and the
[user guide](http://cvxopt.org/userguide).


Building from source
--------------------

CVXOPT is a C extension package. It needs a C compiler, BLAS and LAPACK,
and [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse)
(AMD, CHOLMOD and UMFPACK). The optional GLPK, DSDP, FFTW and GSL
extensions need the corresponding libraries.

Install the system libraries:

```bash
# Debian / Ubuntu
sudo apt-get install libopenblas-dev libsuitesparse-dev libdsdp-dev libfftw3-dev libglpk-dev libgsl0-dev

# macOS (Homebrew)
brew install suite-sparse glpk gsl fftw
```

Then build and install from the repository root:

```bash
pip install build
python -m build --wheel
pip install --no-index --find-links=dist cvxopt
```

The build is configured through environment variables read by
`setup.py`. The most common ones are

| Variable | Purpose | Default |
| --- | --- | --- |
| `CVXOPT_BLAS_LIB`, `CVXOPT_LAPACK_LIB` | BLAS/LAPACK library names, `;`-separated | `blas`, `lapack` |
| `CVXOPT_BLAS_LIB_DIR` | directory containing BLAS/LAPACK | platform guess |
| `CVXOPT_SUITESPARSE_LIB_DIR`, `CVXOPT_SUITESPARSE_INC_DIR` | installed SuiteSparse libraries and headers | platform guess |
| `CVXOPT_SUITESPARSE_SRC_DIR` | build the SuiteSparse parts from source instead | unset |
| `CVXOPT_BUILD_GLPK`, `CVXOPT_BUILD_DSDP`, `CVXOPT_BUILD_FFTW`, `CVXOPT_BUILD_GSL` | set to `1` to build an optional extension | `0` |
| `CVXOPT_MSVC` | set to `1` when compiling with MSVC | `0` |

For example, on macOS with Homebrew SuiteSparse and the Accelerate
framework the two SuiteSparse variables are enough:

```bash
export CVXOPT_SUITESPARSE_LIB_DIR=/opt/homebrew/lib
export CVXOPT_SUITESPARSE_INC_DIR=/opt/homebrew/include/suitesparse
python -m build --wheel
```

The complete, tested recipes for all three platforms are the workflows
in [.github/workflows](.github/workflows), and the
[installation guide](https://cvxopt.org/install/index.html) on the
website covers further options.


Running the tests
-----------------

With CVXOPT installed (a source build or a wheel):

```bash
pip install pytest pytest-cov
pytest --cov=cvxopt tests
```

Tests for optional extensions (GLPK, DSDP, GSL, MOSEK) are skipped when
the corresponding module is not built.


Contributing
------------

Bug reports and pull requests are welcome on the
[issue tracker](https://github.com/cvxopt/cvxopt/issues). Continuous
integration builds and tests every push on Linux, macOS and Windows.
Please add a test under `tests/` with any code change.


Related projects
----------------

* [CVXOPT wheels](https://github.com/cvxopt/cvxopt-wheels)
* [CVXOPT conda-forge package](https://github.com/conda-forge/cvxopt-feedstock)
* [Julia interface to CVXOPT](https://github.com/cvxopt/CVXOPT.jl)
* [MATLAB interface to CVXOPT](https://github.com/cvxopt/cvxopt-matlab)


Build status
------------

[![Linux build Actions Status](https://github.com/cvxopt/cvxopt/workflows/Linux%20build/badge.svg)](https://github.com/cvxopt/cvxopt/actions)
 [![macOS build Actions Status](https://github.com/cvxopt/cvxopt/workflows/macOS%20build/badge.svg)](https://github.com/cvxopt/cvxopt/actions) [![Windows build Actions Status](https://github.com/cvxopt/cvxopt/workflows/Windows%20build/badge.svg)](https://github.com/cvxopt/cvxopt/actions)
