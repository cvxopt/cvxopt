import unittest

class TestGSL(unittest.TestCase):

    def setUp(self):
        try:
            from cvxopt import gsl
        except:
            self.skipTest("GSL not available")

    def test1(self):
        from cvxopt import gsl
        gsl.setseed(123)
        self.assertTrue(gsl.getseed()==123)
        gsl.setseed()
        self.assertTrue(gsl.getseed()>0)

    def test2(self):
        from cvxopt import gsl
        x = gsl.normal(3,2)
        self.assertTrue(x.size[0] == 3 and x.size[1] == 2)

    def test3(self):
        from cvxopt import gsl
        x = gsl.uniform(4,3)
        self.assertTrue(x.size[0] == 4 and x.size[1] == 3)

if __name__ == '__main__':
    unittest.main()
