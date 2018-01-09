import unittest, os
from cvxopt import matrix, normal, setseed
from cvxopt.modeling import op, variable, dot, max, sum

class TestModeling(unittest.TestCase):

    def test_exceptions(self):
        with self.assertRaises(TypeError):
            x = variable(0)

    def test_case1(self):
        x = variable()
        y = variable()
        c1 = ( 2*x+y <= 3 )
        c2 = ( x+2*y <= 3 )
        c3 = ( x >= 0 )
        c4 = ( y >= 0 )
        lp1 = op(-4*x-5*y, [c1,c2,c3,c4])
        print(repr(x))
        print(str(x))
        print(repr(lp1))
        print(str(lp1))
        lp1.solve()
        print(repr(x))
        print(str(x))
        self.assertTrue(lp1.status == 'optimal')

    def test_case2(self):
        x = variable(2)
        A = matrix([[2.,1.,-1.,0.], [1.,2.,0.,-1.]])
        b = matrix([3.,3.,0.,0.])
        c = matrix([-4.,-5.])
        ineq = ( A*x <= b )
        lp2 = op(dot(c,x), ineq)
        lp2.solve()
        self.assertAlmostEqual(lp2.objective.value()[0], -9.0, places=4)

    def test_case3(self):
        m, n = 500, 100
        setseed(100)
        A = normal(m,n)
        b = normal(m)

        x1 = variable(n)
        lp1 = op(max(abs(A*x1-b)))
        lp1.solve()
        self.assertTrue(lp1.status == 'optimal')

        x2 = variable(n)
        lp2 = op(sum(abs(A*x2-b)))
        lp2.solve()
        self.assertTrue(lp2.status == 'optimal')

        x3 = variable(n)
        lp3 = op(sum(max(0, abs(A*x3-b)-0.75, 2*abs(A*x3-b)-2.25)))
        lp3.solve()
        self.assertTrue(lp3.status == 'optimal')

    def test_loadfile(self):
        lp = op()
        lp.fromfile(os.path.join(os.path.dirname(__file__),"boeing2.mps"))
        lp.solve()
        self.assertTrue(lp.status == 'optimal')


if __name__ == '__main__':
    unittest.main()
