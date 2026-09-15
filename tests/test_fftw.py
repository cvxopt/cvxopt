import unittest
from cvxopt import matrix


class TestFFTW(unittest.TestCase):

    def setUp(self):
        try:
            from cvxopt import fftw
        except ImportError:
            self.skipTest("FFTW not available")
        self.fftw = fftw

    def assertAlmostEqualLists(self, L1, L2, places=7):
        self.assertEqual(len(L1), len(L2))
        for u, v in zip(L1, L2):
            self.assertAlmostEqual(u, v, places)

    def test_dft_constant(self):
        # DFT of a constant vector is n at frequency 0 and zero elsewhere
        n = 4
        X = matrix(1.0 + 0j, (n, 1))
        self.fftw.dft(X)
        self.assertAlmostEqualLists(list(X), [n] + (n - 1) * [0.0])

    def test_dft_impulse(self):
        # DFT of a unit impulse is all ones
        n = 4
        X = matrix(0.0 + 0j, (n, 1))
        X[0] = 1.0
        self.fftw.dft(X)
        self.assertAlmostEqualLists(list(X), n * [1.0])

    def test_dft_idft_roundtrip(self):
        x = matrix([1.0 + 2j, -3.0 + 0j, 0.5 - 1j, 4.0 + 0j])
        X = +x
        self.fftw.dft(X)
        self.fftw.idft(X)
        self.assertAlmostEqualLists(list(X), list(x))


if __name__ == '__main__':
    unittest.main()
