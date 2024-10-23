set OPENBLAS_VERSION=0.3.28
set OPENBLAS_SHA256_x64=4cbd0e5daa3fb083b18f5e5fa6eefe79e2f2c51a6d539f98a3c6309a21160042
set OPENBLAS_SHA256_x86=4a14ba2b43937278616cd0883e033cc07ee1331afdd2d264ad81432bd7b16c7b
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
