project('cvxopt', 'c',
  version: run_command(
    find_program('python3'), [
      '-c',
      'from setuptools_scm import get_version; print(get_version())',
    ],
    check: true,
  ).stdout().strip(),
  license: 'GPL-3.0-or-later',
  meson_version: '>=1.1.0',
  default_options: [
    'c_std=c17',
  ],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()
cc = meson.get_compiler('c')

# --- Common dependencies ---
blas_dep = dependency('blas')
lapack_dep = dependency('lapack')
m_dep = cc.find_library('m', required: false)
rt_dep = cc.find_library('rt', required: false)

# Global compile args
c_args = []
if get_option('blas_no_underscore')
  c_args += '-DBLAS_NO_UNDERSCORE'
endif

# --- SuiteSparse ---
suitesparseconfig_dep = dependency('SuiteSparse_config')
amd_lib_dep = dependency('AMD')
colamd_dep = dependency('COLAMD')
cholmod_dep = dependency('CHOLMOD')
umfpack_dep = dependency('UMFPACK')

# --- Optional dependencies ---

if get_option('build_gsl')
  gsl_dep = dependency('gsl')
endif

if get_option('build_fftw')
  fftw_dep = dependency('fftw3')
endif

if get_option('build_glpk')
  glpk_dep = dependency('glpk', required: false)
  if not glpk_dep.found()
    glpk_dep = cc.find_library('glpk')
  endif
endif

if get_option('build_dsdp')
  dsdp_lib = cc.find_library('dsdp')
  _dsdp_inc_dir = get_option('dsdp_inc_dir')
  if _dsdp_inc_dir != ''
    dsdp_dep = declare_dependency(
      dependencies: dsdp_lib,
      compile_args: ['-I' + _dsdp_inc_dir],
    )
  else
    dsdp_dep = dsdp_lib
  endif
endif

subdir('src')
