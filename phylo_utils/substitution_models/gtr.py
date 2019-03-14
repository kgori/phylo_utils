import numpy as np

from phylo_utils.data import fixed_equal_nucleotide_rates, fixed_equal_nucleotide_frequencies
from phylo_utils.substitution_models.abstract import check_frequencies, check_rates, compute_q_matrix, get_eigen, Eigen, \
    DNAReversibleModel


class GTR(DNAReversibleModel):
    _name = 'GTR'
    def __init__(self, rates=None, freqs=None, scale_q=True):
        """
        Create a GTR model for the given rates and frequencies.
        scale_q controls whether the instantaneous rate (Q) matrix
        should be scaled so that units are measured in average
        substitutions per site.
        """
        if rates is None:
            rates = fixed_equal_nucleotide_rates.copy()
        else:
            rates = np.ascontiguousarray(rates)
        if freqs is None:
            freqs = fixed_equal_nucleotide_frequencies.copy()
        if rates.shape == (self.size, self.size):
            self._rates = check_rates(rates, self.size)
        elif rates.shape == (self.size*(self.size-1)/2,):
            self._rates = check_rates(self.square_matrix(rates), self.size)
        self._freqs = check_frequencies(freqs, self.size)
        self._q_mtx = compute_q_matrix(self._rates, self._freqs, scale_q)
        self.eigen = Eigen(*get_eigen(self._q_mtx, self._freqs))

    def square_matrix(self, uppertri):
        mtx = np.zeros((self.size, self.size))
        mtx[0, 1] = mtx[1, 0] = uppertri[0]  # AC
        mtx[0, 2] = mtx[2, 0] = uppertri[1]  # AG
        mtx[0, 3] = mtx[3, 0] = uppertri[2]  # AT
        mtx[1, 2] = mtx[2, 1] = uppertri[3]  # CG
        mtx[1, 3] = mtx[3, 1] = uppertri[4]  # CT
        mtx[2, 3] = mtx[3, 2] = uppertri[5]  # GT
        return mtx
