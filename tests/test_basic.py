import unittest

class TestBasic(unittest.TestCase):

    def test_cvxopt_init(self):
        import cvxopt
        cvxopt.copyright()
        cvxopt.license()
        
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
