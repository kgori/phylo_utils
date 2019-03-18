import unittest
from abc import abstractmethod
from unittest import TestCase
import numpy as np
import phylo_utils as phy
import phylo_utils.substitution_models.abstract
from phylo_utils.substitution_models import TN93
from phylo_utils.substitution_models.unrest import Unrest
from phylo_utils.substitution_models.gtr import GTR
from phylo_utils.substitution_models.hky85 import HKY85
from phylo_utils.substitution_models.f84 import F84
from phylo_utils.substitution_models.f81 import F81
from phylo_utils.substitution_models.k80 import K80
from phylo_utils.substitution_models.jc69 import JC69
from phylo_utils.substitution_models.wag import WAG
from phylo_utils.substitution_models.lg import LG


class TestInputs(TestCase):
    def setUp(self):
        self.good_array = np.array([0.25, 0.25, 0.25, 0.25])
        self.bad_array = np.array([0.250001, 0.25, 0.25, 0.25])

    def test_check_frequencies_mismatched_length(self):
        with self.assertRaises(ValueError):
            phylo_utils.substitution_models.abstract.check_frequencies(self.good_array, 5)

    def test_check_frequencies_bad_input(self):
        with self.assertRaises(ValueError):
            phylo_utils.substitution_models.abstract.check_frequencies(self.bad_array, 4)

    def test_check_frequencies_good_input(self):
        self.assertTrue(np.allclose(self.good_array, phylo_utils.substitution_models.abstract.check_frequencies(self.good_array, 4)))


class TestModelGeneric(TestCase):
    def test_detailed_balance(self):
        self.assertTrue(self.model.detailed_balance())

    def test_q_scale(self):
        self.assertAlmostEqual(self.model.freqs.T.dot(-np.diag(self.model.q())), 1.0)


class TestJC(TestModelGeneric):
    model = JC69()


class TestK80(TestModelGeneric):
    model = K80(1.5)


class TestF81(TestModelGeneric):
    model = F81([0.1, 0.2, 0.3, 0.4])


class TestF84(TestModelGeneric):
    model = F84(1.5, [0.1, 0.2, 0.3, 0.4])


class TestHKY85(TestModelGeneric):
    model = HKY85(1.5, [0.1, 0.2, 0.3, 0.4])


class TestTN93(TestModelGeneric):
    model = TN93(2.5, 2.4, freqs = [0.1, 0.2, 0.3, 0.4])


class TestGTR(TestModelGeneric):
    model = GTR([6., 5., 4., 3., 2., 1.], [0.1, 0.2, 0.3, 0.4])


class TestUnrest(TestModelGeneric):
    model = Unrest(rates = [[0.,  1.,  2.,  3.],
                            [4.,  0.,  5.,  6.],
                            [7.,  8.,  0.,  9.],
                            [10., 11., 12., 0.]])

    def test_detailed_balance(self):
        self.assertFalse(self.model.detailed_balance())


class TestWAG(TestModelGeneric):
    model = WAG()


class TestLG(TestModelGeneric):
    model = LG()


del TestModelGeneric  # don't test the abstract case

if __name__ == '__main__':
    unittest.main()
