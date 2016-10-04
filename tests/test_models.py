from unittest import TestCase
import numpy as np
import phylo_utils as phy

class TestModels(TestCase):
    def setUp(self):
        self.good_array = np.array([0.25, 0.25, 0.25, 0.25])
        self.bad_array = np.array([0.250001, 0.25, 0.25, 0.25])

    def test_check_frequencies_mismatched_length(self):
        with self.assertRaises(ValueError):
            phy.models.check_frequencies(self.good_array, 5)

    def test_check_frequencies_bad_input(self):
        with self.assertRaises(ValueError):
            phy.models.check_frequencies(self.bad_array, 4)

    def test_check_frequencies_good_input(self):
        self.assertTrue(np.allclose(self.good_array, phy.models.check_frequencies(self.good_array, 4)))

    def test_JC(self):
        model = phy.models.JC69()