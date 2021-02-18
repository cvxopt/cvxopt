#!/usr/bin/env python

import os
import sys
import textwrap


def add_bootstrap(bld_dir):

    with open(os.path.join(bld_dir,'__init__.py'),'r') as fd:
        init_str = fd.read()

    if init_str.find('_openblas_bootstrap') == -1:
        with open(os.path.join(bld_dir,'__init__.py'),'w') as fd:
            fd.write(init_str.replace('import cvxopt.base','from . import _openblas_bootstrap\nimport cvxopt.base'))

    with open(os.path.join(bld_dir,'_openblas_bootstrap.py'),'w') as fd:
        fd.write(textwrap.dedent("""
            '''
            OpenBLAS bootstrap to preload Windows DLL in order to avoid ImportError.
            '''
            import os
            import threading

            _openblas_loaded = threading.Event()

            def _load_openblas_win():
                
                if _openblas_loaded.is_set(): 
                    return

                old_path = os.environ['PATH']
                lib_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '.lib')
                try:
                    os.add_dll_directory(lib_path)
                except:
                    os.environ['PATH'] = os.pathsep.join([lib_path, old_path])

                from ctypes import cdll
                try:
                    cdll.LoadLibrary(os.path.join(lib_path, 'libopenblas.dll'))
                except OSError as e:
                    raise ImportError("Cannot find OpenBLAS") from e
                else:
                    _openblas_loaded.set()
                finally:
                    os.environ['PATH'] = old_path

            if os.name == 'nt':
                _load_openblas_win()
            """))

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: %s build_dir" % (sys.argv[0])) 
        sys.exit(0)

    if not os.path.isdir(sys.argv[1]):
        print("Directory %s does not exist." % (sys.argv[1]))
        sys.exit(-1)

    add_bootstrap(sys.argv[1])
