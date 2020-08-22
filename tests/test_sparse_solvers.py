# -*- coding: utf-8 -*-
# @Author: Uriel Sandoval
# @Date:   2020-08-21 11:06:17
# @Last Modified by:   Uriel Sandoval
# @Last Modified time: 2020-08-24 14:44:41

from os import path
import unittest
from itertools import product


class SparseSolver(unittest.TestCase):
    # Base matrix
    _A = None

    # Symbolic factorization for A
    _Fs = None

    # Numeric factorization for A
    _Fn = None

    # Convert to complex matrix as Ac = A + 1j*A 
    _complex = False

    _tol = 1e-12

    __test__ = True

    _cases = ['ACTIVSg2000.mtx', 'bcsstk13.mtx',
              'bcsstk24.mtx', 'bp_800.mtx']

    # Number of columns to consider b: Ax = b
    _columns_B = 3


    def read_mtx(self):
        from kvxopt import spmatrix

        fn = path.join(path.dirname(path.realpath(__file__)), self._matx_fn)
        with open(fn, 'r') as fd:
            size = None
            for row in fd.readlines():
                if row.startswith('%'):
                    continue
                elif not size:
                    size = list(map(int, row.split(' ')))
                    I = [None] * size[2]
                    J = [None] * size[2]
                    V = [None] * size[2]
                    i = 0
                    continue

                val = row.split(' ')
                I[i] = int(val[0]) - 1
                J[i] = int(val[1]) - 1
                V[i] = float(val[2])
                i += 1


            return spmatrix(V, I, J, (size[0], size[1]))

    def assertAlmostEqualLists(self,L1,L2,places=7, msg=None):
        self.assertEqual(len(L1),len(L2))
        for u,v in zip(L1,L2): self.assertAlmostEqual(u,v,places, msg=msg)


    def set_up_matrix(self, build_B = False):

        from kvxopt import normal

        # Types of tranpose systems:
        # 'N', solves A*X = B.
        # 'T', solves A^T*X = B.
        # 'C', solves A^H*X = B.
        if self._complex:
            self._trans = ['N', 'T', 'C']
        else:
            self._trans = ['N', 'T']


        self._A = self.read_mtx()
        if self._complex:
            self._A = +self._A + self._A*1j

        if build_B:
            self.b = normal(self._A.size[0], self._columns_B)
            if self._complex:
                self.b = +self.b*1j


class TestUMFPACK(SparseSolver):

    def test_lu(self):
        """
        Test UMFPACK factorization:

        P * R * A * Q = L * U

        """

        from kvxopt.umfpack import numeric, symbolic, get_numeric
        from kvxopt import norm

        for self._matx_fn, self._complex in product(self._cases, [True, False]):
            self.set_up_matrix(False)

            Fs = symbolic(self._A)
            Fn = numeric(self._A, Fs)

            L, U, P, Q, R = get_numeric(self._A, Fn)

            rho = norm(P*R*self._A*Q - L*U, '1')
            self.assertAlmostEqual(rho, 0.0, msg='Test %s failed'%self._matx_fn)

    def test_linsolve(self):

        from kvxopt.umfpack import linsolve

        for self._matx_fn, self._complex in product(self._cases, [True, False]):
            self.set_up_matrix(True)

            for tran in self._trans:

                x = +self.b
                linsolve(self._A, x, trans = tran)

                if tran == 'T':
                    b1 = self._A.trans() * x
                elif tran == 'C':
                    b1 = self._A.ctrans() * x
                else:
                    b1 = self._A * x 
                
                self.assertAlmostEqualLists(list(b1), list(self.b), 
                                            msg='Test %s failed'%self._matx_fn)

    
    def test_solve(self):

        from kvxopt.umfpack import numeric, symbolic, solve

        for self._matx_fn, self._complex in product(self._cases, [True, False]):
            self.set_up_matrix(True)


            Fs = symbolic(self._A)
            Fn = numeric(self._A, Fs)

            for tran in self._trans:

                x = +self.b
                solve(self._A, Fn, x, trans = tran)

                if tran == 'T':
                    b1 = self._A.trans() * x
                elif tran == 'C':
                    b1 = self._A.ctrans() * x
                else:
                    b1 = self._A * x 

                self.assertAlmostEqualLists(list(b1), list(self.b), 
                                            msg='Test %s failed'%self._matx_fn)

 
    def test_get_det(self):

        try:
            import numpy as np
        except ImportError:
            self.skipTest("Numpy not available for determinant testing")

        from kvxopt.umfpack import numeric, symbolic, get_det
        from kvxopt import matrix, spmatrix

        V = [2,3, 3,-1,4, 4,-3,1,2, 2, 6,1]
        Vc = list(map(lambda x: x+x*1j, V))
        I = [0,1, 0, 2,4, 1, 2,3,4, 2, 1, 4]
        J = [0,0, 1, 1,1, 2, 2,2,2, 3, 4,4]
        B =  matrix([1.0j]*5)
        A = spmatrix(V,I,J)
        Ac = spmatrix(Vc,I,J)

        # Double matrix case
        Fs =  symbolic(A)
        Fn = numeric(A, Fs)
            
        det1 = get_det(A, Fs, Fn)
        det2 = np.linalg.det(np.array(matrix(A)))

        self.assertAlmostEqual(det1, det2)


        # Complex matrix case
        Fs =  symbolic(Ac)
        Fn = numeric(Ac, Fs)
            
        det1 = get_det(Ac, Fs, Fn)
        det2 = np.linalg.det(np.array(matrix(Ac)))

        self.assertAlmostEqual(det1, det2)



class TestKLU(SparseSolver):

    def test_lu(self):
        """
        Test UMFPACK factorization:

        P * R * A * Q = L * U

        """

        from kvxopt.klu import numeric, symbolic, get_numeric
        from kvxopt import norm

        for self._matx_fn, self._complex in product(self._cases, [True, False]):
            self.set_up_matrix(False)

            Fs = symbolic(self._A)
            Fn = numeric(self._A, Fs)

            L, U, P, Q, R, F, r = get_numeric(self._A, Fs, Fn)

            rho = norm(R*P*self._A*Q - (L*U + F), '1')

            self.assertAlmostEqual(rho, 0.0, msg='Test %s failed'%self._matx_fn)

    def test_linsolve(self):

        from kvxopt.klu import linsolve

        for self._matx_fn, self._complex in product(self._cases, [True, False]):
            self.set_up_matrix(True)

            for tran in self._trans:

                x = +self.b
                linsolve(self._A, x, trans = tran)

                if tran == 'T':
                    b1 = self._A.trans() * x
                elif tran == 'C':
                    b1 = self._A.ctrans() * x
                else:
                    b1 = self._A * x 
                
                self.assertAlmostEqualLists(list(b1), list(self.b), 
                                            msg='Test %s failed'%self._matx_fn)

    
    def test_solve(self):

        from kvxopt.klu import numeric, symbolic, solve

        for self._matx_fn, self._complex in product(self._cases, [True, False]):
            self.set_up_matrix(True)


            Fs = symbolic(self._A)
            Fn = numeric(self._A, Fs)

            for tran in self._trans:

                x = +self.b
                solve(self._A, Fs, Fn, x, trans = tran)

                if tran == 'T':
                    b1 = self._A.trans() * x
                elif tran == 'C':
                    b1 = self._A.ctrans() * x
                else:
                    b1 = self._A * x 

                self.assertAlmostEqualLists(list(b1), list(self.b), 
                                            msg='Test %s failed'%self._matx_fn)

    def test_get_det(self):

        try:
            import numpy as np
        except ImportError:
            self.skipTest("Numpy not available for determinant testing")

        from kvxopt.klu import numeric, symbolic, get_det
        from kvxopt import matrix, spmatrix

        V = [2,3, 3,-1,4, 4,-3,1,2, 2, 6,1]
        Vc = list(map(lambda x: x+x*1j, V))
        I = [0,1, 0, 2,4, 1, 2,3,4, 2, 1, 4]
        J = [0,0, 1, 1,1, 2, 2,2,2, 3, 4,4]
        B =  matrix([1.0j]*5)
        A = spmatrix(V,I,J)
        Ac = spmatrix(Vc,I,J)

        # Double matrix case
        Fs =  symbolic(A)
        Fn = numeric(A, Fs)
            
        det1 = get_det(A, Fs, Fn)
        det2 = np.linalg.det(np.array(matrix(A)))

        self.assertAlmostEqual(det1, det2)


        # Complex matrix case
        Fs =  symbolic(Ac)
        Fn = numeric(Ac, Fs)
            
        det1 = get_det(Ac, Fs, Fn)
        det2 = np.linalg.det(np.array(matrix(Ac)))

        self.assertAlmostEqual(det1, det2)



if __name__ == '__main__':
    unittest.main()













