set SUITESPARSE_VERSION=7.8.2
set SUITESPARSE_SHA256=996c48c87baaeb5fc04bd85c7e66d3651a56fe749c531c60926d75b4db5d2181

wget -nv https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v%SUITESPARSE_VERSION%.tar.gz -O SuiteSparse-%SUITESPARSE_VERSION%.tar.gz
checksum -t sha256 -c %SUITESPARSE_SHA256% SuiteSparse-%SUITESPARSE_VERSION%.tar.gz
mkdir SuiteSparse
7z x -bso0 -bsp0 SuiteSparse-%SUITESPARSE_VERSION%.tar.gz 
7z x -bso0 -bsp0 SuiteSparse-%SUITESPARSE_VERSION%.tar
SET CVXOPT_SUITESPARSE_SRC_DIR=%cd%\SuiteSparse-%SUITESPARSE_VERSION%
