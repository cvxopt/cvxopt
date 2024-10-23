#!/bin/bash

if ldd --version 2>&1 | grep -q musl; then
    # Musl libc
    apk add --no-cache openblas-dev suitesparse-dev fftw-dev glpk-dev gsl-dev
    LIB_PATH="/usr/lib"
else
    # Glibc
    ARCH=$(uname -m)
    yum install -y openblas-devel suitesparse-devel fftw-devel gsl-devel
    dnf install -y epel-release && yum install -y glpk-devel
    if [ "$ARCH" == "x86_64" ] || [ "$ARCH" == "aarch64" ]; then
        LIB_PATH="/usr/lib64"
    elif [ "$ARCH" == "i686" ]; then
        LIB_PATH="/usr/lib"
    else
        echo "Unknown architecture: $ARCH"
        exit 1
    fi
fi

# Build DSDP
curl -o DSDP5.8.tar.gz https://www.mcs.anl.gov/hs/software/DSDP/DSDP5.8.tar.gz
tar xzf DSDP5.8.tar.gz
export CC=gcc
(cd DSDP5.8 && patch -p1 < ../.github/workflows/dsdp.patch && make LAPACKBLAS="-L$LIB_PATH -lopenblas" PREFIX=/usr/local DSDPROOT=`pwd` install)