import numpy as np

from phylo_utils.data import fixed_equal_nucleotide_frequencies
from phylo_utils.substitution_models.abstract import check_rates, Eigen, DNAReversibleModel

# Precomputed values for Jukes Cantor model
jc_q_mtx = np.array([[-3,1,1,1],
                     [1,-3,1,1],
                     [1,1,-3,1],
                     [1,1,1,-3]],
                    dtype=np.double) / 3
jc_evecs = np.ascontiguousarray([[1,2,0,0.5],
                                 [1,2,0,-0.5],
                                 [1,-2,0.5,0],
                                 [1,-2,-0.5,0]],
                                dtype=np.double)
jc_ivecs = np.asfortranarray([[0.25,0.25,0.25,0.25],
                              [0.125,0.125,-0.125,-0.125],
                              [0,0,1,-1],
                              [1,-1,0,0]],
                             dtype=np.double)
jc_evals = np.ascontiguousarray([0, -4/3, -4/3, -4/3], dtype=np.double)


class JC69(DNAReversibleModel):
    _name = 'JC69'
    _freqs = fixed_equal_nucleotide_frequencies.copy()
    def __init__(self):
        mtx = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0]])
        self._rates = check_rates(mtx, 4)
        self._q_mtx = jc_q_mtx
        self.eigen = Eigen(jc_evecs, jc_evals, jc_ivecs)

    def p(self, t):
        e1 = 0.25 + 0.75*np.exp(-4*t/3.)
        e2 = 0.25 - 0.25*np.exp(-4*t/3.)
        return np.array([[e1, e2, e2, e2],
                         [e2, e1, e2, e2],
                         [e2, e2, e1, e2],
                         [e2, e2, e2, e1]])
