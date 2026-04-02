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
# Detect SuiteSparse include directory for find_library fallback.
# When pkg-config files are available (recommended), this is not needed.
_ss_inc_args = []
foreach d : ['/usr/include/suitesparse',
             '/usr/local/include/suitesparse',
             '/opt/homebrew/include/suitesparse']
  if _ss_inc_args.length() == 0 and cc.has_header('umfpack.h', args: '-I' + d)
    _ss_inc_args = ['-I' + d]
  endif
endforeach

# SuiteSparse_config
suitesparseconfig_dep = dependency('SuiteSparse_config', required: false)
if not suitesparseconfig_dep.found()
  suitesparseconfig_dep = dependency('suitesparseconfig', required: false)
endif
if not suitesparseconfig_dep.found()
  suitesparseconfig_dep = declare_dependency(
    dependencies: cc.find_library('suitesparseconfig'),
    compile_args: _ss_inc_args,
  )
endif

# AMD
amd_lib_dep = dependency('AMD', required: false)
if not amd_lib_dep.found()
  amd_lib_dep = dependency('amd', required: false)
endif
if not amd_lib_dep.found()
  amd_lib_dep = declare_dependency(
    dependencies: cc.find_library('amd'),
    compile_args: _ss_inc_args,
  )
endif

# COLAMD
colamd_dep = dependency('COLAMD', required: false)
if not colamd_dep.found()
  colamd_dep = dependency('colamd', required: false)
endif
if not colamd_dep.found()
  colamd_dep = declare_dependency(
    dependencies: cc.find_library('colamd'),
    compile_args: _ss_inc_args,
  )
endif

# CHOLMOD
cholmod_dep = dependency('CHOLMOD', required: false)
if not cholmod_dep.found()
  cholmod_dep = dependency('cholmod', required: false)
endif
if not cholmod_dep.found()
  cholmod_dep = declare_dependency(
    dependencies: cc.find_library('cholmod'),
    compile_args: _ss_inc_args,
  )
endif

# UMFPACK
umfpack_dep = dependency('UMFPACK', required: false)
if not umfpack_dep.found()
  umfpack_dep = dependency('umfpack', required: false)
endif
if not umfpack_dep.found()
  umfpack_dep = declare_dependency(
    dependencies: cc.find_library('umfpack'),
    compile_args: _ss_inc_args,
  )
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
