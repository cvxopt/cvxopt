import unittest
from cvxopt import matrix, spmatrix, umfpack


class TestUMFPACK(unittest.TestCase):

    def setUp(self):
        # unsymmetric nonsingular matrix
        #   [ 2  0  1 ]
        #   [ 1  3  0 ]
        #   [ 0  1  4 ]
        self.A = spmatrix([2.0, 1.0, 3.0, 1.0, 1.0, 4.0],
                          [0, 1, 1, 0, 2, 2],
                          [0, 0, 1, 2, 1, 2])
        self.x = matrix([1.0, 2.0, 3.0])

    def assertAlmostEqualLists(self, L1, L2, places=7):
        self.assertEqual(len(L1), len(L2))
        for u, v in zip(L1, L2):
            self.assertAlmostEqual(u, v, places)

    def test_linsolve(self):
        B = self.A * self.x
        umfpack.linsolve(self.A, B)
        self.assertAlmostEqualLists(list(B), list(self.x))

    def test_linsolve_trans(self):
        B = self.A.T * self.x
        umfpack.linsolve(self.A, B, trans='T')
        self.assertAlmostEqualLists(list(B), list(self.x))

    def test_symbolic_numeric_solve(self):
        Fs = umfpack.symbolic(self.A)
        Fn = umfpack.numeric(self.A, Fs)
        B = self.A * self.x
        umfpack.solve(self.A, Fn, B)
        self.assertAlmostEqualLists(list(B), list(self.x))

    def test_singular(self):
        A = spmatrix([1.0, 1.0], [0, 0], [0, 1], (2, 2))
        with self.assertRaises(ArithmeticError):
            umfpack.linsolve(A, matrix([1.0, 1.0]))


if __name__ == '__main__':
    unittest.main()
