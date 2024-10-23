set GLPK_VERSION=5.0
set GLPK_SHA256=26aa624525a636de272c0b329e2dfd01a0d5b7827f1c1c76f393d71e37dead70

wget -nv http://ftp.gnu.org/gnu/glpk/glpk-%GLPK_VERSION%.tar.gz 
checksum -t sha256 -c %GLPK_SHA256% glpk-%GLPK_VERSION%.tar.gz 
7z x -bso0 -bsp0 glpk-%GLPK_VERSION%.tar.gz && 7z x -bso0 -bsp0 glpk-%GLPK_VERSION%.tar 
cd glpk-%GLPK_VERSION%\w64 
copy config_VC config.h 
nmake /f Makefile_VC glpk.lib 
cd ..\.. 
SET "CVXOPT_GLPK_LIB_DIR=%cd%\glpk-%GLPK_VERSION%\w64" 
SET "CVXOPT_GLPK_INC_DIR=%cd%\glpk-%GLPK_VERSION%\src"