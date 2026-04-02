project('cvxopt', 'c',
  version: run_command(
    find_program('python3'), files('tools/gitversion.py'),
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
blas_dep = dependency('blas', required: false)
if not blas_dep.found()
  blas_dep = dependency('openblas')
endif

lapack_dep = dependency('lapack', required: false)
if not lapack_dep.found()
  lapack_dep = dependency('openblas')
endif
m_dep = cc.find_library('m', required: false)
rt_dep = cc.find_library('rt', required: false)

# Global compile args
c_args = []
if get_option('blas_no_underscore')
  c_args += '-DBLAS_NO_UNDERSCORE'
endif

# --- SuiteSparse ---
suitesparseconfig_dep = dependency('SuiteSparse_config', required: false)
if not suitesparseconfig_dep.found()
  suitesparseconfig_dep = cc.find_library('suitesparseconfig')
endif

amd_lib_dep = dependency('AMD', required: false)
if not amd_lib_dep.found()
  amd_lib_dep = cc.find_library('amd')
endif

colamd_dep = dependency('COLAMD', required: false)
if not colamd_dep.found()
  colamd_dep = cc.find_library('colamd')
endif

cholmod_dep = dependency('CHOLMOD', required: false)
if not cholmod_dep.found()
  cholmod_dep = cc.find_library('cholmod')
endif

umfpack_dep = dependency('UMFPACK', required: false)
if not umfpack_dep.found()
  umfpack_dep = cc.find_library('umfpack')
endif

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
  _dsdp_lib_dirs = []
  _dsdp_lib_dir = get_option('dsdp_lib_dir')
  if _dsdp_lib_dir != ''
    _dsdp_lib_dirs = [_dsdp_lib_dir]
  endif
  dsdp_lib = cc.find_library('dsdp', dirs: _dsdp_lib_dirs)
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
