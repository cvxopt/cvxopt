import unittest
from cvxopt import matrix, spmatrix, cholmod


class TestCHOLMOD(unittest.TestCase):

    def setUp(self):
        # symmetric positive definite tridiagonal matrix
        #   [ 4 -1  0 ]
        #   [-1  4 -1 ]      det = 56
        #   [ 0 -1  4 ]
        self.A = spmatrix([4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
                          [0, 1, 0, 1, 2, 1, 2],
                          [0, 0, 1, 1, 1, 2, 2])
        self.x = matrix([1.0, 2.0, 3.0])
        self.b = self.A * self.x

    def assertAlmostEqualLists(self, L1, L2, places=7):
        self.assertEqual(len(L1), len(L2))
        for u, v in zip(L1, L2):
            self.assertAlmostEqual(u, v, places)

    def test_linsolve(self):
        B = +self.b
        cholmod.linsolve(self.A, B)
        self.assertAlmostEqualLists(list(B), list(self.x))

    def test_splinsolve_inverse(self):
        I = spmatrix(1.0, range(3), range(3))
        X = cholmod.splinsolve(self.A, I)
        self.assertIsInstance(X, spmatrix)
        AX = matrix(self.A) * matrix(X)
        self.assertAlmostEqualLists(list(AX), list(matrix(I)))

    def test_symbolic_numeric_solve(self):
        F = cholmod.symbolic(self.A)
        cholmod.numeric(self.A, F)
        B = +self.b
        cholmod.solve(F, B)
        self.assertAlmostEqualLists(list(B), list(self.x))

    def test_diag(self):
        F = cholmod.symbolic(self.A)
        cholmod.numeric(self.A, F)
        d = cholmod.diag(F)
        # prod(diag(L))**2 == det(A) == 56
        prod = 1.0
        for v in d:
            prod *= v
        self.assertAlmostEqual(prod ** 2, 56.0)


if __name__ == '__main__':
    unittest.main()
