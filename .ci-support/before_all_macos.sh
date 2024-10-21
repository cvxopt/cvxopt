#!/bin/bash

brew install suitesparse fftw glpk gsl

# Download, build and install DSDP
DSDP_VERSION="5.8"
DSDP_SHA256="26aa624525a636de272c0b329e2dfd01a0d5b7827f1c1c76f393d71e37dead70"
if [ ! -f DSDP${DSDP_VERSION}.tar.gz ]; then
    curl -o DSDP${DSDP_VERSION}.tar.gz https://www.mcs.anl.gov/hs/software/DSDP/DSDP${DSDP_VERSION}.tar.gz
fi
if [ -d DSDP${DSDP_VERSION} ]; then
    rm -rf DSDP${DSDP_VERSION}
fi
echo "${DSDP_SHA256}  DSDP${DSDP_VERSION}.tar.gz" | shasum -a 256 -c || exit 1
tar xzf DSDP${DSDP_VERSION}.tar.gz
ARCH=$(uname -m)
if [ "$ARCH" == "x86_64" ]; then
    brew install openblas
    export ARCH_FLAGS="-target x86_64-apple-macos11"
    export LAPACKBLAS="-lopenblas -L/usr/local/opt/openblas/lib"
elif [ "$ARCH" == "arm64" ]; then
    export ARCH_FLAGS="-target arm64-apple-macos11"
    export LAPACKBLAS="-framework Accelerate"
else
    echo "Unknown architecture: $ARCH"
    exit 1
fi
(cd DSDP${DSDP_VERSION} \
    && patch -p1 < ../.github/workflows/dsdp.patch \
    && make CC=gcc LAPACKBLAS="${LAPACKBLAS}" PREFIX="/usr/local" IS_OSX=1 DSDPROOT=`pwd` install)
