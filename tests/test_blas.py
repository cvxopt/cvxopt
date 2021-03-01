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