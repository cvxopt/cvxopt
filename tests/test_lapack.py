import unittest
from cvxopt import matrix, lapack


class TestLAPACK(unittest.TestCase):

    def assertAlmostEqualLists(self, L1, L2, places=7):
        self.assertEqual(len(L1), len(L2))
        for u, v in zip(L1, L2):
            self.assertAlmostEqual(u, v, places)

    def test_gesv(self):
        # A = [2 1; 1 3], x = [1, 2]  =>  b = A*x = [4, 7]
        A = matrix([[2.0, 1.0], [1.0, 3.0]])
        b = matrix([4.0, 7.0])
        lapack.gesv(A, b)
        self.assertAlmostEqualLists(list(b), [1.0, 2.0])

    def test_gesv_singular(self):
        A = matrix([[1.0, 1.0], [1.0, 1.0]])
        b = matrix([1.0, 1.0])
        with self.assertRaises(ArithmeticError):
            lapack.gesv(A, b)

    def test_potrf_potrs(self):
        # A = [4 2; 2 3] = L*L' with L = [2 0; 1 sqrt(2)]
        A = matrix([[4.0, 2.0], [2.0, 3.0]])
        L = +A
        lapack.potrf(L)
        self.assertAlmostEqual(L[0, 0], 2.0)
        self.assertAlmostEqual(L[1, 0], 1.0)
        self.assertAlmostEqual(L[1, 1], 2.0 ** 0.5)
        # b = A*[1, 1] = [6, 5]
        b = matrix([6.0, 5.0])
        lapack.potrs(L, b)
        self.assertAlmostEqualLists(list(b), [1.0, 1.0])

    def test_syev(self):
        # eigenvalues of [2 1; 1 2] are 1 and 3
        A = matrix([[2.0, 1.0], [1.0, 2.0]])
        V = +A
        W = matrix(0.0, (2, 1))
        lapack.syev(V, W, jobz='V')
        self.assertAlmostEqualLists(list(W), [1.0, 3.0])
        # A*V == V*diag(W)
        D = matrix([[W[0], 0.0], [0.0, W[1]]])
        self.assertAlmostEqualLists(list(A * V - V * D), 4 * [0.0])

    def test_gels(self):
        # columns [1,1,1] and [0,1,2]; b = 1*col0 + 1*col1 exactly
        A = matrix([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
        B = matrix([1.0, 2.0, 3.0])
        lapack.gels(A, B)
        self.assertAlmostEqualLists(list(B[:2]), [1.0, 1.0])

    def test_gesvd(self):
        A = matrix([[3.0, 0.0], [0.0, 1.0]])
        S = matrix(0.0, (2, 1))
        lapack.gesvd(A, S)
        self.assertAlmostEqualLists(list(S), [3.0, 1.0])


if __name__ == '__main__':
    unittest.main()
