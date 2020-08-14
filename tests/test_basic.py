import unittest

class TestBasic(unittest.TestCase):

    def assertEqualLists(self,L1,L2):
        self.assertEqual(len(L1),len(L2))
        for u,v in zip(L1,L2): self.assertEqual(u,v)

    def assertAlmostEqualLists(self,L1,L2,places=7):
        self.assertEqual(len(L1),len(L2))
        for u,v in zip(L1,L2): self.assertAlmostEqual(u,v,places)

    def test_kvxopt_init(self):
        import kvxopt
        kvxopt.copyright()
        kvxopt.license()

    def test_basic(self):
        import kvxopt
        a = kvxopt.matrix([1.0,2.0,3.0])
        assert list(a) == [1.0, 2.0, 3.0]
        b = kvxopt.matrix([3.0,-2.0,-1.0])
        c = kvxopt.spmatrix([1.0,-2.0,3.0],[0,2,4],[1,2,4],(6,5))
        d = kvxopt.spmatrix([1.0,2.0,5.0],[0,1,2],[0,0,0],(3,1))
        e = kvxopt.mul(a, b)
        self.assertEqualLists(e, [3.0,-4.0,-3.0])
        self.assertAlmostEqualLists(list(kvxopt.div(a,b)),[1.0/3.0,-1.0,-3.0])
        self.assertAlmostEqual(kvxopt.div([1.0,2.0,0.25]),2.0)
        self.assertEqualLists(list(kvxopt.min(a,b)),[1.0,-2.0,-1.0])
        self.assertEqualLists(list(kvxopt.max(a,b)),[3.0,2.0,3.0])
        self.assertEqual(kvxopt.max([1.0,2.0]),2.0)
        self.assertEqual(kvxopt.max(a),3.0)
        self.assertEqual(kvxopt.max(c),3.0)
        self.assertEqual(kvxopt.max(d),5.0)
        self.assertEqual(kvxopt.min([1.0,2.0]),1.0)
        self.assertEqual(kvxopt.min(a),1.0)
        self.assertEqual(kvxopt.min(c),-2.0)
        self.assertEqual(kvxopt.min(d),1.0)
        self.assertEqual(len(c.imag()),0)
        with self.assertRaises(OverflowError):
            kvxopt.matrix(1.0,(32780*4,32780))
        with self.assertRaises(OverflowError):
            kvxopt.spmatrix(1.0,(0,32780*4),(0,32780))+1

    def test_basic_complex(self):
        import kvxopt
        a = kvxopt.matrix([1,-2,3])
        b = kvxopt.matrix([1.0,-2.0,3.0])
        c = kvxopt.matrix([1.0+2j,1-2j,0+1j])
        d = kvxopt.spmatrix([complex(1.0,0.0), complex(0.0,1.0), complex(2.0,-1.0)],[0,1,3],[0,2,3],(4,4))
        e = kvxopt.spmatrix([complex(1.0,0.0), complex(0.0,1.0), complex(2.0,-1.0)],[2,3,3],[1,2,3],(4,4))
        self.assertAlmostEqualLists(list(kvxopt.div(b,c)),[0.2-0.4j,-0.4-0.8j,-3j])
        self.assertAlmostEqualLists(list(kvxopt.div(b,2.0j)),[-0.5j,1j,-1.5j])
        self.assertAlmostEqualLists(list(kvxopt.div(a,c)),[0.2-0.4j,-0.4-0.8j,-3j])
        self.assertAlmostEqualLists(list(kvxopt.div(c,a)),[(1+2j),(-0.5+1j),0.3333333333333333j])
        self.assertAlmostEqualLists(list(kvxopt.div(c,c)),[1.0,1.0,1.0])
        self.assertAlmostEqualLists(list(kvxopt.div(a,2.0j)),[-0.5j,1j,-1.5j])
        self.assertAlmostEqualLists(list(kvxopt.div(c,1.0j)),[2-1j,-2-1j,1+0j])
        self.assertAlmostEqualLists(list(kvxopt.div(1j,c)),[0.4+0.2j,-0.4+0.2j,1+0j])
        self.assertTrue(len(d)+len(e)==len(kvxopt.sparse([d,e])))
        self.assertTrue(len(d)+len(e)==len(kvxopt.sparse([[d],[e]])))

    def test_basic_no_gsl(self):
        import sys
        sys.modules['gsl'] = None
        import kvxopt
        kvxopt.normal(4,8)
        kvxopt.uniform(4,8)


    def test_print(self):
        from kvxopt import printing, matrix, spmatrix
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
