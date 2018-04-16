import unittest

class TestMOSEK(unittest.TestCase):

    def setUp(self):
        try:
            from cvxopt import msk
        except:
            self.skipTest("MOSEK not available")

    def assertAlmostEqualLists(self,L1,L2,places=3):
        self.assertEqual(len(L1),len(L2))
        for u,v in zip(L1,L2): self.assertAlmostEqual(u,v,places)

    def test_conelp(self):
        from cvxopt import matrix, msk, solvers
        c = matrix([-6., -4., -5.])
        G = matrix([[ 16., 7.,  24.,  -8.,   8.,  -1.,  0., -1.,  0.,  0.,   7.,  -5.,   1.,  -5.,   1.,  -7.,   1.,   -7.,  -4.],
                    [-14., 2.,   7., -13., -18.,   3.,  0.,  0., -1.,  0.,   3.,   13.,  -6.,  13.,  12., -10.,  -6.,  -10., -28.],
                    [  5., 0., -15.,  12.,  -6.,  17.,  0.,  0.,  0., -1.,   9.,    6.,  -6.,   6.,  -7.,  -7.,  -6.,   -7., -11.]])
        h = matrix( [ -3., 5.,  12.,  -2., -14., -13., 10.,  0.,  0.,  0.,  68.,  -30., -19., -30.,  99.,  23., -19.,   23.,  10.] )
        dims = {'l': 2, 'q': [4, 4], 's': [3]}
        self.assertAlmostEqualLists(list(solvers.conelp(c, G, h, dims)['x']),list(msk.conelp(c, G, h, dims)[1]))

    def test_lp(self):
        from cvxopt import matrix, msk, solvers
        c = matrix([-4., -5.])
        G = matrix([[2., 1., -1., 0.], [1., 2., 0., -1.]])
        h = matrix([3., 3., 0., 0.])
        self.assertAlmostEqualLists(list(solvers.lp(c, G, h)['x']),list(msk.lp(c, G, h)[1]))

    # TODO: test msk.qp() and msk.ilp()
    #def test_ilp(self):
    #    msk.ilp()
    #
    #def test_qp(self):
    #    msk.qp()

    def test_socp(self):
        from cvxopt import matrix, msk, solvers
        c = matrix([-2., 1., 5.])
        G = [ matrix( [[12., 13., 12.], [6., -3., -12.], [-5., -5., 6.]] ) ]
        G += [ matrix( [[3., 3., -1., 1.], [-6., -6., -9., 19.], [10., -2., -2., -3.]] ) ]
        h = [ matrix( [-12., -3., -2.] ),  matrix( [27., 0., 3., -42.] ) ]
        self.assertAlmostEqualLists(list(solvers.socp(c, Gq = G, hq = h)['x']),list(msk.socp(c, Gq = G, hq = h)[1]))

    def test_options(self):
        from cvxopt import matrix, msk, solvers
        msk.options = {msk.mosek.iparam.log: 0}
        c = matrix([-4., -5.])
        G = matrix([[2., 1., -1., 0.], [1., 2., 0., -1.]])
        h = matrix([3., 3., 0., 0.])
        msk.lp(c, G, h)
        msk.lp(c, G, h, options={msk.mosek.iparam.log: 1})
        solvers.lp(c, G, h, solver='mosek', options={'mosek':{msk.mosek.iparam.log: 1}})

if __name__ == '__main__':
    unittest.main()
