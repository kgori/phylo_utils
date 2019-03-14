import numpy as np

from phylo_utils.data import fixed_equal_nucleotide_rates
from phylo_utils.substitution_models.abstract import compute_q_matrix, q_to_freqs, get_eigen, Eigen, \
    DNANonReversibleModel


class Unrest(DNANonReversibleModel):
    _name = 'UNREST'
    def __init__(self, rates=None):
        if rates is None:
            rates = fixed_equal_nucleotide_rates.copy()
        else:
            rates = np.ascontiguousarray(rates)
        super(Unrest, self).__init__(rates)
        self._q_mtx = compute_q_matrix(self._rates, None)
        self.eigen = Eigen(*get_eigen(self._q_mtx))

    @property
    def freqs(self):
        return q_to_freqs(self._q_mtx)
