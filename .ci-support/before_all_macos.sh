#!/bin/bash

brew install suitesparse fftw glpk gsl

if [ -f .local/lib/libdsdp.dylib ]; then
    echo "DSDP found in .local/lib, skipping build."
else
    echo "DSDP not found in .local/lib, building DSDP..."

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
    export ARCH_FLAGS="-target arm64-apple-macos11"
    export LAPACKBLAS="-framework Accelerate"
    mkdir -p .local/lib .local/include
    PREFIX="$(pwd)/.local"
    (cd DSDP${DSDP_VERSION} \
        && patch -p1 < ../.github/workflows/dsdp.patch \
        && make CC=gcc LAPACKBLAS="${LAPACKBLAS}" PREFIX=$PREFIX IS_OSX=1 DSDPROOT=`pwd` install)
fi
