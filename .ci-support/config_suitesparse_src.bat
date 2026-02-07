set SUITESPARSE_VERSION=7.12.1
set SUITESPARSE_SHA256=794ae22f7e38e2ac9f5cbb673be9dd80cdaff2cdf858f5104e082694f743b0ba

wget -nv https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v%SUITESPARSE_VERSION%.tar.gz -O SuiteSparse-%SUITESPARSE_VERSION%.tar.gz
checksum -t sha256 -c %SUITESPARSE_SHA256% SuiteSparse-%SUITESPARSE_VERSION%.tar.gz
mkdir SuiteSparse
7z x -bso0 -bsp0 SuiteSparse-%SUITESPARSE_VERSION%.tar.gz 
7z x -bso0 -bsp0 SuiteSparse-%SUITESPARSE_VERSION%.tar
SET CVXOPT_SUITESPARSE_SRC_DIR=%cd%\SuiteSparse-%SUITESPARSE_VERSION%
