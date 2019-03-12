from unittest import TestCase
import numpy as np
import phylo_utils as phy
from phylo_utils.substitution_models import JC69, K80, F81, F84, HKY85, TN93, GTR, Unrest
from phylo_utils.substitution_models import LG, WAG

class TestInputs(TestCase):
    def setUp(self):
        self.good_array = np.array([0.25, 0.25, 0.25, 0.25])
        self.bad_array = np.array([0.250001, 0.25, 0.25, 0.25])

    def test_check_frequencies_mismatched_length(self):
        with self.assertRaises(ValueError):
            phy.substitution_models.check_frequencies(self.good_array, 5)

    def test_check_frequencies_bad_input(self):
        with self.assertRaises(ValueError):
            phy.substitution_models.check_frequencies(self.bad_array, 4)

    def test_check_frequencies_good_input(self):
        self.assertTrue(np.allclose(self.good_array, phy.substitution_models.check_frequencies(self.good_array, 4)))


class TestJC(object):
    model = JC69()


class TestK80(object):
    model = K80(1.5)


class TestF81(object):
    model = F81([0.1, 0.2, 0.3, 0.4])


class TestF84(object):
    model = F84(1.5, [0.1, 0.2, 0.3, 0.4])


class TestHKY85(object):
    model = HKY85(1.5, [0.1, 0.2, 0.3, 0.4])


class TestGTR(object):
    model = GTR([6., 5., 4., 3., 2., 1.], [0.1, 0.2, 0.3, 0.4])


class TestWAG(object):
    model = WAG()


class TestLG(object):
    model = LG()