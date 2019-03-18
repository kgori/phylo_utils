import unittest
from unittest import TestCase
import numpy as np
from phylo_utils.utils import seq_to_partials

A_POS=0
C_POS=1
G_POS=2
T_POS=3

def create_dna_partials(bases):
    arr = np.zeros(4, dtype=float)
    for base in bases:
        arr[base] = 1
    return arr

class TestSeqToPartialsDNA(TestCase):
    def test_A(self):
        self.assertTrue(np.allclose(
            seq_to_partials('A', alphabet='dna'),
            create_dna_partials([A_POS]))
        )

    def test_a(self):
        self.assertTrue(np.allclose(
            seq_to_partials('a', alphabet='dna'),
            create_dna_partials([A_POS]))
        )

    def test_C(self):
        self.assertTrue(np.allclose(
            seq_to_partials('C', alphabet='dna'),
            create_dna_partials([C_POS]))
        )

    def test_c(self):
        self.assertTrue(np.allclose(
            seq_to_partials('c', alphabet='dna'),
            create_dna_partials([C_POS]))
        )

    def test_G(self):
        self.assertTrue(np.allclose(
            seq_to_partials('G', alphabet='dna'),
            create_dna_partials([G_POS]))
        )

    def test_g(self):
        self.assertTrue(np.allclose(
            seq_to_partials('g', alphabet='dna'),
            create_dna_partials([G_POS]))
        )

    def test_T(self):
        self.assertTrue(np.allclose(
            seq_to_partials('T', alphabet='dna'),
            create_dna_partials([T_POS]))
        )

    def test_t(self):
        self.assertTrue(np.allclose(
            seq_to_partials('t', alphabet='dna'),
            create_dna_partials([T_POS]))
        )

    def test_N(self):
        self.assertTrue(np.allclose(
            seq_to_partials('N', alphabet='dna'),
            create_dna_partials([A_POS, C_POS, G_POS, T_POS]))
        )

    def test_n(self):
        self.assertTrue(np.allclose(
            seq_to_partials('n', alphabet='dna'),
            create_dna_partials([A_POS, C_POS, G_POS, T_POS]))
        )

    def test_R(self):
        self.assertTrue(np.allclose(
            seq_to_partials('R', alphabet='dna'),
            create_dna_partials([A_POS, G_POS]))
        )

    def test_r(self):
        self.assertTrue(np.allclose(
            seq_to_partials('r', alphabet='dna'),
            create_dna_partials([A_POS, G_POS]))
        )

    def test_Y(self):
        self.assertTrue(np.allclose(
            seq_to_partials('Y', alphabet='dna'),
            create_dna_partials([C_POS, T_POS]))
        )

    def test_y(self):
        self.assertTrue(np.allclose(
            seq_to_partials('y', alphabet='dna'),
            create_dna_partials([C_POS, T_POS]))
        )

    def test_S(self):
        self.assertTrue(np.allclose(
            seq_to_partials('S', alphabet='dna'),
            create_dna_partials([C_POS, G_POS]))
        )

    def test_s(self):
        self.assertTrue(np.allclose(
            seq_to_partials('s', alphabet='dna'),
            create_dna_partials([C_POS, G_POS]))
        )

    def test_W(self):
        self.assertTrue(np.allclose(
            seq_to_partials('W', alphabet='dna'),
            create_dna_partials([A_POS, T_POS]))
        )

    def test_w(self):
        self.assertTrue(np.allclose(
            seq_to_partials('w', alphabet='dna'),
            create_dna_partials([A_POS, T_POS]))
        )

    def test_K(self):
        self.assertTrue(np.allclose(
            seq_to_partials('K', alphabet='dna'),
            create_dna_partials([T_POS, G_POS]))
        )

    def test_k(self):
        self.assertTrue(np.allclose(
            seq_to_partials('k', alphabet='dna'),
            create_dna_partials([T_POS, G_POS]))
        )

    def test_M(self):
        self.assertTrue(np.allclose(
            seq_to_partials('M', alphabet='dna'),
            create_dna_partials([C_POS, A_POS]))
        )

    def test_m(self):
        self.assertTrue(np.allclose(
            seq_to_partials('m', alphabet='dna'),
            create_dna_partials([C_POS, A_POS]))
        )

    def test_B(self):
        self.assertTrue(np.allclose(
            seq_to_partials('B', alphabet='dna'),
            create_dna_partials([C_POS, G_POS, T_POS]))
        )

    def test_b(self):
        self.assertTrue(np.allclose(
            seq_to_partials('b', alphabet='dna'),
            create_dna_partials([C_POS, G_POS, T_POS]))
        )

    def test_D(self):
        self.assertTrue(np.allclose(
            seq_to_partials('D', alphabet='dna'),
            create_dna_partials([A_POS, G_POS, T_POS]))
        )

    def test_d(self):
        self.assertTrue(np.allclose(
            seq_to_partials('d', alphabet='dna'),
            create_dna_partials([A_POS, G_POS, T_POS]))
        )

    def test_H(self):
        self.assertTrue(np.allclose(
            seq_to_partials('H', alphabet='dna'),
            create_dna_partials([A_POS, C_POS, T_POS]))
        )

    def test_h(self):
        self.assertTrue(np.allclose(
            seq_to_partials('h', alphabet='dna'),
            create_dna_partials([A_POS, C_POS, T_POS]))
        )

    def test_V(self):
        self.assertTrue(np.allclose(
            seq_to_partials('V', alphabet='dna'),
            create_dna_partials([A_POS, C_POS, G_POS]))
        )

    def test_v(self):
        self.assertTrue(np.allclose(
            seq_to_partials('v', alphabet='dna'),
            create_dna_partials([A_POS, C_POS, G_POS]))
        )


    def test_gap(self):
        self.assertTrue(np.allclose(
            seq_to_partials('-', alphabet='dna'),
            create_dna_partials([A_POS, C_POS, G_POS, T_POS]))
        )
