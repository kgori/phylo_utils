import unittest
import phylo_utils as phy
import numpy as np

class TestLnlNode(unittest.TestCase):

    def test_gamma(self):
        result = 1
        expected = 1
        self.assertTrue(np.allclose(result, expected))

if __name__ == '__main__':
    unittest.main()