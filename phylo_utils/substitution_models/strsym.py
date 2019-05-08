import numpy as np

from phylo_utils.data import fixed_equal_nucleotide_rates
from phylo_utils.substitution_models.abstract import Eigen, \
    DNANonReversibleModel
from phylo_utils.substitution_models.utils import compute_q_matrix, q_to_freqs, get_eigen


class Strsym(DNANonReversibleModel):
    _name = 'STRSYM'

    def __init__(self, rates=None):
        """
        Due to strand symmetry, there are six rate parameters:
        A<=>C == T<=>G
        A<=>G == T<=>C
        A<=>T == T<=>A
        C<=>A == G<=>T
        C<=>G == G<=>C

        These should be provided as a length 6 numpy array, or a list

        :param rates:
        """

        if rates is None:
            rates_m = fixed_equal_nucleotide_rates.copy()

        if len(rates) != 6:
            raise ValueError("Provide a list of 6 rate parameters")

        rates_m = np.zeros((4, 4))

        rates_m[np.array([0, 0, 0, 1, 1, 1]), np.array([1, 2, 3, 0, 2, 3])] = rates
        rates_m[np.array([3, 3, 3, 2, 2, 2]), np.array([2, 1, 0, 3, 1, 0])] = rates

        super(Strsym, self).__init__(rates_m)
        self._q_mtx = compute_q_matrix(self._rates, None)
        self.eigen = Eigen(*get_eigen(self._q_mtx))

    @property
    def freqs(self):
        return q_to_freqs(self._q_mtx)
