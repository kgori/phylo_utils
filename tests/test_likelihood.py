import unittest
import phylo_utils as phy
import numpy as np


class TestLnlNode(unittest.TestCase):
    def setUp(self):
        model = phy.substitution_models.K80(2.)
        self.root = phy.likelihood.LnlNode(model)
        self.left = phy.likelihood.LnlNode(model)
        self.right = phy.likelihood.LnlNode(model)
        partials_1 = np.ascontiguousarray(np.array([[1, 0, 0, 0]], dtype=np.double))
        partials_2 = np.ascontiguousarray(np.array([[0, 1, 0, 0]], dtype=np.double))
        self.left.set_partials(partials_1)
        self.right.set_partials(partials_2)

    def test_update_transition_probabilities(self):
        self.root.update_transition_probabilities(0.1, 0.2)
        self.assertTrue(np.allclose(self.root.probs1, self.root.subst_model.p(0.1)))
        self.assertTrue(np.allclose(self.root.probs2, self.root.subst_model.p(0.2)))

    def test_set_partials(self):
        self.root.set_partials([1,0,0,0])
        self.assertEqual(self.root.partials.dtype, np.double)
        self.assertEqual(self.root.partials.shape, (1, 4))

    def test_compute_partials(self):
        self.root.update_transition_probabilities(0.1, 0.2)
        self.root.compute_partials(self.left, self.right)
        self.assertTrue(np.allclose([[ 0.0764, 0.0378, 0.0011, 0.0011]], self.root.partials.round(4)))

    def test_compute_edge_sitewise_likelihood(self):
        results = []
        for n in np.linspace(0.1,1.0,10):
            self.left.compute_edge_sitewise_likelihood(self.right, n)
            self.right.compute_edge_sitewise_likelihood(self.left, n)
            results.append((self.left.sitewise == self.right.sitewise).all())
        self.assertTrue(all(results))

    def test_compute_likelihood_at_root(self):
        self.root.update_transition_probabilities(0.1, 0.2)
        self.root.compute_partials(self.left, self.right)
        self.assertAlmostEqual(np.log((self.root.subst_model.freqs * self.root.partials).sum()), -3.5371, places=4)

    def test_compute_likelihood(self):
        self.assertAlmostEqual(self.left.compute_likelihood(self.right, 0.3), -3.5371, places=4)


class TestPairDistOpt(unittest.TestCase):

    def testfnc(self):
        kappa = 0.7345
        k80 = phy.substitution_models.K80(kappa)
        # Simulated data from K80, kappa=1, distance = 0.8
        sites_a = phy.seq_to_partials('ACCCTCCGCGTTGGGTAGTCCTAGGCCCAATGGCGTTTATGCCTCGATTTTTAGTTCTACCGTCCCTACAGATGGATGCCGTCGCATAGACACTGTCAATTCCATTCGGCAGGCTTCACACTGTTGCATTTTCATTTTGTACACGGTACCAACATAGGAGTGCTGTATTGCTATATTTCCAGTACACGGCGTTGAGTCGGATGGAAACGCCGGCGGAAGACAGCTTGGCGGGTCTTCACGCATCACCGCGGGGTCTGAAAGGTATTATCGCTGCTTAAATCAGACCGGTCAAGCTTCCTGGCGGAAGGCGGCAAGGTCCAGCCACAGCATGCTTATTCCTTGTCACGCCGGGTGGAAATCTAGAGCGTCCGGTGGACACAGAGTGATTTTGTACGGGGGGTTCCATACCAGGACATTAGGGTCGGTTTACGGTCTGAGATGTATGTTGCCTTGCGGTCGACGAGCACTGATTCCCCTGAACTTCGTAAGACACATATAGTTTTAATGAAATCCCCAAAACGAGCATGGTTTCAGTATACGCGACAACTTAGGATACAACATACTGAACCAGTCCGCATTGAGGTGCCAATCAAACGGGACCGGGACTGATAAGTATAAAATAGGTTTCCCTGTCCTCTACCTACGTTATCCTCGCGTCGATTTTGATTCTTACCAAGACTGCTAATCAGGCCCTGTGGCCTGCATGTCACCATGTCAGCGTGTTTGGCTAAATTCACGGGATTGGCCTTACCGACTTACATCAGTATTTCATACATAGTTACTCGAGTTTAACGTTGACAGTTAGTCCCATGATACGGCAAAGCCTGGTTCGGCGGATTTCCGAGTACAGCATCTTCGCCCCCGAGATTGCCGCCAATGGACACCCTCCTGAGATGCAGATATGAGTGTTTTTGACACTCTGAGGCTGAGATCCTCACACTTCCGGAGCTTCCGCGATAGTCACGTGGTTATTAGACTTACGGCAGGAAAAATCATGTTA', alphabet='dna')
        sites_b = phy.seq_to_partials('AAGCTCCGCGTAAGCTAACGACCAGTCAGCTAGGTTTAGTGCCACCAGTATGGCTAGTTCCGGAGGGCAAACCGGATGCTACCGATTGGTCACCCTCAGGGTGATTTCGCAGGGCGCTCACTTATTCCTTTTAAATCCTGCCAACAGACTAAGAAAGTTGTACGGTATTCCTATATCTTCAGTACTGCTCTTGGCCGTGCATGTAGCCGAACGACGAGGACGGTACATGAGTTTCTCACCAATTACAGGCGGTTCCATTAGGCAGTAGCTGCGGTTAGTTCATACTGCTAAAGAATCTTCTTGGAACGTGCCAAGGACCAGTCACACACATGTTGTAGTCCCTCATCGTGGTAGGCGTTCCAGACCGTCCGTGGTACACATACCAAATTTCGTACCGGCTGACTCAAAGCGGGAGTTCGCATGATACCAGGGAACGAGATGTTCAAAACGATCAGGTAGTGCCGCCATCTTTCAGGTTCTTTCGTTTCGTCCTATGATACTTGAGTAGCGGTCAAACGAAGCTCGTAGGTGACAGTTACGAGACATGCTGGGATGCAACATACTTTCGCAGTTAGCTAGTAGGTACCTATCTAGCGAATCGAGCTAGGATACCCTGATTATGCTTGTCTCCGTCCTCTTACTATGATCTCCTCGCGTGGTTTTTGCTGCTTAACCGTTGTGCCGTATAAAACAAGAGGCGGGAGTTTAGCTGTGGGAACTTCGTAGACCTTGTAAGCTGGATAGGCCCGTCCGTCGTAATTAATTACCTAAAAGAGAGTCAAACAAGCTTAAGTCGCCGAGTTAGTCGGATAAGAAGCCATTCTCTGGTCCGCCAACCTTCCCATGCCAGTACGGTTGCCGAGGTCCATTCGGTGACTGTGGGATAACCGTTGCCGGAGCTATGAGATCCATTACAACTCTGCGCCTAGGATGTTAACTCTACCGAAGTTTGCGACCCCGGAACCTGTAAATTGTCCTTAGGGTCGTAACATTTTCAAGC', alphabet='dna')
        phy.likelihood.optimise(k80, sites_a, sites_b)
        root = phy.likelihood.LnlNode(k80)
        root.set_partials(sites_a)
        leaf = phy.likelihood.Leaf(sites_b)
        phy.likelihood.brent_optimise(root, leaf)

if __name__ == '__main__':
    unittest.main()