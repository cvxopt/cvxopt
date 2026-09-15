import unittest
from cvxopt import blas, matrix

class TestBLAS(unittest.TestCase):

    def setUp(self):
        from cvxopt import blas, matrix
        self.a = matrix([1.0,2.0,3.0,4.0])
        self.b = matrix([2.5,-2.0,-4.0,1.0,3.0]) 

    def assertEqualLists(self,L1,L2):
        self.assertEqual(len(L1),len(L2))
        for u,v in zip(L1,L2): self.assertEqual(u,v)

    def assertAlmostEqualLists(self,L1,L2,places=7):
        self.assertEqual(len(L1),len(L2))
        for u,v in zip(L1,L2): self.assertAlmostEqual(u,v,places)

    def test_iamax(self):
        self.assertTrue(blas.iamax(self.a) == 3)
        self.assertTrue(blas.iamax(self.b) == 2)
        self.assertTrue(blas.iamax(self.b,inc=2) == 1)
        self.assertTrue(blas.iamax(self.b,offset=1) == 1)

    def test_level1(self):
        x = matrix([1.0, 2.0, 3.0])
        y = matrix([4.0, 5.0, 6.0])
        self.assertAlmostEqual(blas.dot(x, y), 32.0)
        self.assertAlmostEqual(blas.nrm2(x), 14.0 ** 0.5)
        self.assertAlmostEqual(blas.asum(matrix([-1.0, 2.0])), 3.0)
        z = +y
        blas.axpy(x, z, alpha=2.0)          # z := 2*x + y
        self.assertAlmostEqualLists(list(z), [6.0, 9.0, 12.0])
        blas.scal(0.5, z)                   # z := 0.5*z
        self.assertAlmostEqualLists(list(z), [3.0, 4.5, 6.0])
        w = matrix(0.0, (3, 1))
        blas.copy(x, w)
        self.assertEqualLists(list(w), list(x))

    def test_gemv(self):
        A = matrix([[1.0, 3.0], [2.0, 4.0]])   # [1 2; 3 4]
        x = matrix([1.0, 1.0])
        y = matrix(0.0, (2, 1))
        blas.gemv(A, x, y)                  # y := A*x
        self.assertAlmostEqualLists(list(y), [3.0, 7.0])
        blas.gemv(A, x, y, trans='T', alpha=2.0, beta=1.0)  # y := 2*A'*x + y
        self.assertAlmostEqualLists(list(y), [11.0, 19.0])

    def test_gemm(self):
        A = matrix([[1.0, 3.0], [2.0, 4.0]])   # [1 2; 3 4]
        C = matrix(0.0, (2, 2))
        blas.gemm(A, A, C)                  # C := A*A = [7 10; 15 22]
        self.assertAlmostEqualLists(list(C), [7.0, 15.0, 10.0, 22.0])


if __name__ == '__main__':
    unittest.main()
