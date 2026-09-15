import unittest
from cvxopt import matrix, spmatrix, amd


class TestAMD(unittest.TestCase):

    def test_order_is_permutation(self):
        # arrow-shaped symmetric pattern: dense first row/column
        n = 5
        I = list(range(n)) + list(range(1, n))
        J = [0] * n + list(range(1, n))
        A = spmatrix(1.0, I, J, (n, n))
        p = amd.order(A)
        self.assertIsInstance(p, matrix)
        self.assertEqual(p.typecode, 'i')
        self.assertEqual(len(p), n)
        self.assertEqual(sorted(list(p)), list(range(n)))

    def test_order_arrow_puts_dense_node_last(self):
        # ordering the dense node last eliminates fill-in entirely
        n = 5
        I = list(range(n)) + list(range(1, n))
        J = [0] * n + list(range(1, n))
        A = spmatrix(1.0, I, J, (n, n))
        p = amd.order(A)
        self.assertEqual(p[n - 1], 0)


if __name__ == '__main__':
    unittest.main()
