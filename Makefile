# Root Makefile for CVXOPT.
#
# The C extensions are configured through CVXOPT_* environment variables
# read by setup.py (see README.md); they pass straight through, e.g.
#
#     CVXOPT_BUILD_GLPK=1 make build
#
# PYTHON selects the interpreter (and therefore the virtual environment).

PYTHON ?= python3
PIP    := $(PYTHON) -m pip

.DEFAULT_GOAL := help

.PHONY: help build install test docs clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

build:  ## Build a wheel into dist/ (needs BLAS/LAPACK and SuiteSparse)
	$(PIP) install --quiet --upgrade build
	$(PYTHON) -m build --wheel

install: build  ## Build and install the wheel into the current environment
	$(PIP) install --no-index --find-links=dist cvxopt

test:  ## Run the test suite with coverage (cvxopt must be installed)
	$(PIP) install --quiet pytest pytest-cov
	$(PYTHON) -m pytest --cov=cvxopt tests

docs:  ## Build the Sphinx HTML documentation into doc/build/html
	$(PIP) install --quiet -r doc/requirements.txt
	$(MAKE) -C doc html SPHINXBUILD="$(PYTHON) -m sphinx"

clean:  ## Remove build, test and documentation artefacts
	rm -rf build dist src/python/*.egg-info *.egg-info .coverage .pytest_cache
	rm -rf src/python/__pycache__ tests/__pycache__ src/python/_version.py
	$(MAKE) -C doc clean
