import unittest
from cvxopt import matrix, solvers


class CvxoptTestCase(unittest.TestCase):

    def assertMatrixAlmostEqual(self, A, B, places=3):
        """Assert that two matrices are similar."""
        self.assertEqual(A.size, B.size)
        N, M = A.size
        for i in range(N):
            for j in range(M):
                self.assertAlmostEqual(A[i, j], B[i, j], places=places)


class TestSDP(CvxoptTestCase):

    def setUp(self):
        solvers.options['show_progress'] = False

    def test_dsdp(self):
        # Setup problem
        c = matrix([1., -1., 1.])
        G = [matrix([[-7., -11., -11., 3.],
                     [7., -18., -18., 8.],
                     [-2.,  -8.,  -8., 1.]])]
        G += [matrix([[-21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.],
                      [0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.],
                      [-5.,   2., -17.,   2.,  -6.,   8., -17.,   8., 6.]])]
        h = [matrix([[33., -9.], [-9., 26.]])]
        h += [matrix([[14., 9., 40.], [9., 91., 10.], [40., 10., 15.]])]

        # Solve with default solver
        sol = solvers.sdp(c, Gs=G, hs=h)

        # Solve with DSDP
        sol_dsdp = solvers.sdp(c, Gs=G, hs=h, solver='dsdp')

        # Compare solutions
        self.assertMatrixAlmostEqual(sol['x'], sol_dsdp['x'])
        self.assertMatrixAlmostEqual(sol['zs'][0], sol_dsdp['zs'][0])
        self.assertMatrixAlmostEqual(sol['zs'][1], sol_dsdp['zs'][1])


if __name__ == '__main__':
    unittest.main()
