set OPENBLAS_VERSION=0.3.31
set OPENBLAS_SHA256_x64=e7595359700e8bb5a15c41af1920850b1be37078eb22813201b3d4bc5bd9227e
set OPENBLAS_SHA256_x86=1ad9181595e1d3a6de52d50309721324f3a4e78f3000e413d40d9872bc0ab8f5
if [%PLATFORM%]==[x64] ( 
    set OPENBLAS_SHA256=%OPENBLAS_SHA256_x64%
) else (
    set OPENBLAS_SHA256=%OPENBLAS_SHA256_x86%
)

wget -nv https://github.com/OpenMathLib/OpenBLAS/releases/download/v%OPENBLAS_VERSION%/OpenBLAS-%OPENBLAS_VERSION%-%PLATFORM%.zip 
checksum -t sha256 -c %OPENBLAS_SHA256% OpenBLAS-%OPENBLAS_VERSION%-%PLATFORM%.zip 
mkdir OpenBLAS 
7z x -oOpenBLAS -bso0 -bsp0 OpenBLAS-%OPENBLAS_VERSION%-%PLATFORM%.zip 
set "CVXOPT_BLAS_LIB=libopenblas" 
set "CVXOPT_LAPACK_LIB=libopenblas" 
set "OPENBLAS_DLL=%cd%\OpenBLAS\bin\libopenblas.dll"
set "OPENBLAS_LIB=%cd%\OpenBLAS\lib\libopenblas.lib"

wget https://raw.githubusercontent.com/OpenMathLib/OpenBLAS/v%OPENBLAS_VERSION%/LICENSE -O LICENSE_OpenBLAS-%OPENBLAS_VERSION% &
set OPENBLAS_LICENSE=%cd%\LICENSE_OpenBLAS-%OPENBLAS_VERSION%        
