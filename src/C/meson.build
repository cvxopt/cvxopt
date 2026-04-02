# Required C extension modules

py.extension_module('base',
  ['base.c', 'dense.c', 'sparse.c'],
  dependencies: [py_dep, m_dep, lapack_dep, blas_dep],
  c_args: c_args,
  subdir: 'cvxopt',
  install: true,
)

py.extension_module('blas',
  'blas.c',
  dependencies: [py_dep, blas_dep],
  c_args: c_args,
  subdir: 'cvxopt',
  install: true,
)

py.extension_module('lapack',
  'lapack.c',
  dependencies: [py_dep, lapack_dep, blas_dep],
  c_args: c_args,
  subdir: 'cvxopt',
  install: true,
)

py.extension_module('umfpack',
  'umfpack.c',
  dependencies: [py_dep, umfpack_dep, cholmod_dep, amd_lib_dep, colamd_dep,
                  suitesparseconfig_dep, lapack_dep, blas_dep, rt_dep],
  subdir: 'cvxopt',
  install: true,
)

py.extension_module('cholmod',
  'cholmod.c',
  dependencies: [py_dep, cholmod_dep, colamd_dep, amd_lib_dep,
                  suitesparseconfig_dep, lapack_dep, blas_dep, rt_dep],
  subdir: 'cvxopt',
  install: true,
)

py.extension_module('amd',
  'amd.c',
  dependencies: [py_dep, amd_lib_dep, suitesparseconfig_dep, rt_dep],
  subdir: 'cvxopt',
  install: true,
)

py.extension_module('misc_solvers',
  'misc_solvers.c',
  dependencies: [py_dep, lapack_dep, blas_dep],
  c_args: c_args,
  subdir: 'cvxopt',
  install: true,
)

# Optional C extension modules

if get_option('build_gsl')
  _gsl_c_args = c_args
  if host_machine.system() == 'windows'
    _gsl_c_args += ['-DGSL_DLL', '-DWIN32']
  endif
  py.extension_module('gsl',
    'gsl.c',
    dependencies: [py_dep, m_dep, gsl_dep, blas_dep],
    c_args: _gsl_c_args,
    subdir: 'cvxopt',
    install: true,
  )
endif

if get_option('build_fftw')
  _fftw_c_args = c_args
  if host_machine.system() == 'windows'
    _fftw_c_args += ['-DFFTW_DLL', '-DFFTW_NO_Complex']
  endif
  py.extension_module('fftw',
    'fftw.c',
    dependencies: [py_dep, fftw_dep, blas_dep],
    c_args: _fftw_c_args,
    subdir: 'cvxopt',
    install: true,
  )
endif

if get_option('build_glpk')
  py.extension_module('glpk',
    'glpk.c',
    dependencies: [py_dep, glpk_dep],
    subdir: 'cvxopt',
    install: true,
  )
endif

if get_option('build_dsdp')
  py.extension_module('dsdp',
    'dsdp.c',
    dependencies: [py_dep, dsdp_dep, lapack_dep, blas_dep],
    c_args: c_args,
    subdir: 'cvxopt',
    install: true,
  )
endif
