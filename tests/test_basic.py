import unittest

class TestBasic(unittest.TestCase):

    def assertEqualLists(self,L1,L2):
        self.assertEqual(len(L1),len(L2))
        for u,v in zip(L1,L2): self.assertEqual(u,v)

    def assertAlmostEqualLists(self,L1,L2,places=7):
        self.assertEqual(len(L1),len(L2))
        for u,v in zip(L1,L2): self.assertAlmostEqual(u,v,places)

    def test_cvxopt_init(self):
        import cvxopt
        cvxopt.copyright()
        cvxopt.license()

    def test_basic(self):
        import cvxopt
        a = cvxopt.matrix([1.0,2.0,3.0])
        assert list(a) == [1.0, 2.0, 3.0]
        b = cvxopt.matrix([3.0,-2.0,-1.0])
        c = cvxopt.spmatrix([1.0,-2.0,3.0],[0,2,4],[1,2,4],(6,5))
        d = cvxopt.spmatrix([1.0,2.0,5.0],[0,1,2],[0,0,0],(3,1))
        e = cvxopt.mul(a, b)
        self.assertEqualLists(e, [3.0,-4.0,-3.0])
        self.assertAlmostEqualLists(list(cvxopt.div(a,b)),[1.0/3.0,-1.0,-3.0])
        self.assertAlmostEqual(cvxopt.div([1.0,2.0,0.25]),2.0)
        self.assertEqualLists(list(cvxopt.min(a,b)),[1.0,-2.0,-1.0])
        self.assertEqualLists(list(cvxopt.max(a,b)),[3.0,2.0,3.0])
        self.assertEqual(cvxopt.max([1.0,2.0]),2.0)
        self.assertEqual(cvxopt.max(a),3.0)
        self.assertEqual(cvxopt.max(c),3.0)
        self.assertEqual(cvxopt.max(d),5.0)
        self.assertEqual(cvxopt.min([1.0,2.0]),1.0)
        self.assertEqual(cvxopt.min(a),1.0)
        self.assertEqual(cvxopt.min(c),-2.0)
        self.assertEqual(cvxopt.min(d),1.0)
        self.assertEqual(len(c.imag()),0)
        with self.assertRaises(OverflowError):
            cvxopt.matrix(1.0,(32780*4,32780))
        with self.assertRaises(OverflowError):
            cvxopt.spmatrix(1.0,(0,32780*4),(0,32780))+1

    def test_basic_complex(self):
        import cvxopt
        a = cvxopt.matrix([1,-2,3])
        b = cvxopt.matrix([1.0,-2.0,3.0])
        c = cvxopt.matrix([1.0+2j,1-2j,0+1j])
        d = cvxopt.spmatrix([complex(1.0,0.0), complex(0.0,1.0), complex(2.0,-1.0)],[0,1,3],[0,2,3],(4,4))
        e = cvxopt.spmatrix([complex(1.0,0.0), complex(0.0,1.0), complex(2.0,-1.0)],[2,3,3],[1,2,3],(4,4))
        self.assertAlmostEqualLists(list(cvxopt.div(b,c)),[0.2-0.4j,-0.4-0.8j,-3j])
        self.assertAlmostEqualLists(list(cvxopt.div(b,2.0j)),[-0.5j,1j,-1.5j])
        self.assertAlmostEqualLists(list(cvxopt.div(a,c)),[0.2-0.4j,-0.4-0.8j,-3j])
        self.assertAlmostEqualLists(list(cvxopt.div(c,a)),[(1+2j),(-0.5+1j),0.3333333333333333j])
        self.assertAlmostEqualLists(list(cvxopt.div(c,c)),[1.0,1.0,1.0])
        self.assertAlmostEqualLists(list(cvxopt.div(a,2.0j)),[-0.5j,1j,-1.5j])
        self.assertAlmostEqualLists(list(cvxopt.div(c,1.0j)),[2-1j,-2-1j,1+0j])
        self.assertAlmostEqualLists(list(cvxopt.div(1j,c)),[0.4+0.2j,-0.4+0.2j,1+0j])
        self.assertTrue(len(d)+len(e)==len(cvxopt.sparse([d,e])))
        self.assertTrue(len(d)+len(e)==len(cvxopt.sparse([[d],[e]])))

    def test_basic_no_gsl(self):
        import sys
        sys.modules['gsl'] = None
        import cvxopt
        cvxopt.normal(4,8)
        cvxopt.uniform(4,8)


    def test_print(self):
        from cvxopt import printing, matrix, spmatrix
        printing.options['height']=2
        A = spmatrix(1.0,range(3),range(3), tc='d')
        print(printing.matrix_repr_default(matrix(A)))
        print(printing.matrix_str_default(matrix(A)))
        print(printing.spmatrix_repr_default(A))
        print(printing.spmatrix_str_default(A))
        print(printing.spmatrix_str_triplet(A))

        A = spmatrix(1.0,range(3),range(3), tc='z')
        print(printing.matrix_repr_default(matrix(A)))
        print(printing.matrix_str_default(matrix(A)))
        print(printing.spmatrix_repr_default(A))
        print(printing.spmatrix_str_default(A))
        print(printing.spmatrix_str_triplet(A))

        A = spmatrix([],[],[],(3,3))
        print(printing.spmatrix_repr_default(A))
        print(printing.spmatrix_str_default(A))
        print(printing.spmatrix_str_triplet(A))

if __name__ == '__main__':
    unittest.main()
