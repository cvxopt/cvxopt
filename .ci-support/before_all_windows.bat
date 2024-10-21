call .ci-support\config_openblas.bat 
call .ci-support\config_suitesparse_src.bat
if [%CVXOPT_BUILD_GLPK%]==[1] ( call .ci-support\config_glpk.bat )