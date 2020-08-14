import unittest

class TestDSDP(unittest.TestCase):

    def setUp(self):
        try:
            from kvxopt import dsdp, matrix
            c = matrix([1.,-1.,1.])
            G = [ matrix([[-7., -11., -11., 3.],
                        [ 7., -18., -18., 8.],
                        [-2.,  -8.,  -8., 1.]]) ]
            G += [ matrix([[-21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.],
                        [  0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.],
                        [ -5.,   2., -17.,   2.,  -6.,   8., -17.,  8., 6.]]) ]
            h = [ matrix([[33., -9.], [-9., 26.]]) ]
            h += [ matrix([[14., 9., 40.], [9., 91., 10.], [40., 10., 15.]]) ]
            self._prob_data = (c,G,h)
        except:
            self.skipTest("DSDP not available")

    def test_sdp(self):
        from kvxopt import solvers, dsdp
        c,Gs,hs = self._prob_data
        sol_ref1 = solvers.sdp(c,None,None,Gs,hs)
        self.assertTrue(sol_ref1['status']=='optimal')
        sol_ref2 = solvers.sdp(c,Gs=Gs,hs=hs)
        self.assertTrue(sol_ref2['status']=='optimal')
        sol1 = solvers.sdp(c,None,None,Gs,hs,solver='dsdp')
        self.assertTrue(sol1['status']=='optimal')
        sol2 = solvers.sdp(c,Gs=Gs,hs=hs,solver='dsdp')
        self.assertTrue(sol2['status']=='optimal')
        sol3 = dsdp.sdp(c,None,None,Gs,hs)
        self.assertTrue(sol3[0] == 'DSDP_PDFEASIBLE')
        sol4 = dsdp.sdp(c,Gs=Gs,hs=hs)
        self.assertTrue(sol4[0] == 'DSDP_PDFEASIBLE')

    def test_options(self):
        from kvxopt import dsdp
        c,Gs,hs = self._prob_data
        dsdp.options = {'DSDP_Monitor':1}
        sol1 = dsdp.sdp(c,Gs=Gs,hs=hs)
        self.assertTrue(sol1[0] == 'DSDP_PDFEASIBLE')
        sol2 = dsdp.sdp(c,Gs=Gs,hs=hs,options={})
        self.assertTrue(sol2[0] == 'DSDP_PDFEASIBLE')
        sol3 = dsdp.sdp(c,Gs=Gs,hs=hs,options={'DSDP_Monitor':1})
        self.assertTrue(sol3[0] == 'DSDP_PDFEASIBLE')
        sol4 = dsdp.sdp(c,Gs=Gs,hs=hs,options={'DSDP_MaxIts':2})
        self.assertTrue(sol4[0] == 'DSDP_UNKNOWN')
        sol5 = dsdp.sdp(c,Gs=Gs,hs=hs,options={'DSDP_GapTolerance':1e-8})
        self.assertTrue(sol5[0] == 'DSDP_PDFEASIBLE')

if __name__ == '__main__':
    unittest.main()
