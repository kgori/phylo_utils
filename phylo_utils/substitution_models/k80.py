from phylo_utils.data import fixed_equal_nucleotide_frequencies
from phylo_utils.substitution_models.tn93 import TN93


class K80(TN93):
    _name = 'K80'
    _freqs = fixed_equal_nucleotide_frequencies.copy()
    def __init__(self, kappa, scale_q=True):
        super(K80, self).__init__(kappa, kappa, 1, self._freqs, scale_q=scale_q)
