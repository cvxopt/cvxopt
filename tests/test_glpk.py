import unittest

class TestGLPK(unittest.TestCase):

    def setUp(self):
        try:
            from kvxopt import glpk, matrix
            c = matrix([-4., -5.])
            G = matrix([[2., 1., -1., 0.], [1., 2., 0., -1.]])
            h = matrix([3., 3., 0., 0.])
            A = matrix([1.0,1.0],(1,2))
            b = matrix(1.0)
            self._prob_data = (c,G,h,A,b)
        except:
            self.skipTest("GLPK not available")

    def test_lp(self):
        from kvxopt import solvers, glpk
        c,G,h,A,b = self._prob_data
        sol1 = solvers.lp(c,G,h)
        self.assertTrue(sol1['status']=='optimal')
        sol2 = solvers.lp(c,G,h,A,b)
        self.assertTrue(sol2['status']=='optimal')
        sol3 = solvers.lp(c,G,h,solver='glpk')
        self.assertTrue(sol3['status']=='optimal')
        sol4 = solvers.lp(c,G,h,A,b,solver='glpk')
        self.assertTrue(sol4['status']=='optimal')
        sol5 = glpk.lp(c,G,h)
        self.assertTrue(sol5[0]=='optimal')
        sol6 = glpk.lp(c,G,h,A,b)
        self.assertTrue(sol6[0]=='optimal')
        sol7 = glpk.lp(c,G,h,None,None)
        self.assertTrue(sol7[0]=='optimal')

    def test_ilp(self):
        from kvxopt import glpk, matrix
        c,G,h,A,b = self._prob_data
        sol1 = glpk.ilp(c, G, h, A, b, set([0]), set())
        self.assertTrue(sol1[0]=='optimal')
        sol2 = glpk.ilp(c, G, h, A, b, set([0]), set())
        self.assertTrue(sol2[0]=='optimal')
        sol3 = glpk.ilp(c, G, h, None, None, set([0, 1]), set())
        self.assertTrue(sol3[0]=='optimal')
        sol4 = glpk.ilp(c, G, h, None, None, set(), set([1]))
        self.assertTrue(sol4[0]=='optimal')
        sol5 = glpk.ilp(c, G, h, A, matrix(-1.0), set(), set([0,1]))
        self.assertTrue(sol5[0]=='LP relaxation is primal infeasible')

    def test_options(self):
        from kvxopt import glpk, solvers
        c,G,h,A,b = self._prob_data
        glpk.options = {'msg_lev' : 'GLP_MSG_OFF'}

        sol1 = glpk.lp(c,G,h)
        self.assertTrue(sol1[0]=='optimal')
        sol2 = glpk.lp(c,G,h,A,b)
        self.assertTrue(sol2[0]=='optimal')
        sol3 = glpk.lp(c,G,h,options={'msg_lev' : 'GLP_MSG_ON'})
        self.assertTrue(sol3[0]=='optimal')
        sol4 = glpk.lp(c,G,h,A,b,options={'msg_lev' : 'GLP_MSG_ERR'})
        self.assertTrue(sol4[0]=='optimal')

        sol5 = solvers.lp(c,G,h,solver='glpk',options={'glpk':{'msg_lev' : 'GLP_MSG_ON'}})
        self.assertTrue(sol5['status']=='optimal')

        sol1 = glpk.ilp(c,G,h,None,None,set(),set([0,1]))
        self.assertTrue(sol1[0]=='optimal')
        sol2 = glpk.ilp(c,G,h,A,b,set([0,1]),set())
        self.assertTrue(sol2[0]=='optimal')
        sol3 = glpk.ilp(c,G,h,None,None,set(),set([0,1]),options={'msg_lev' : 'GLP_MSG_ALL'})
        self.assertTrue(sol3[0]=='optimal')
        sol4 = glpk.ilp(c,G,h,A,b,set(),set([0]),options={'msg_lev' : 'GLP_MSG_ALL'})
        self.assertTrue(sol4[0]=='optimal')

        solvers.options['glpk'] = {'msg_lev' : 'GLP_MSG_ON'}
        sol5 = solvers.lp(c,G,h,solver='glpk')
        self.assertTrue(sol5['status']=='optimal')

if __name__ == '__main__':
    unittest.main()
