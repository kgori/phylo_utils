from phylo_utils.data import lg_rates, lg_freqs
from phylo_utils.substitution_models.abstract import Eigen, ProteinModel
from phylo_utils.substitution_models.utils import check_frequencies, compute_q_matrix, get_eigen


class LG(ProteinModel):
    _name = 'LG'
    _rates = lg_rates.copy()
    def __init__(self, freqs=None):
        if freqs is None:
            self._freqs = lg_freqs.copy()
        else:
            self._freqs = check_frequencies(freqs, self.size)
        self._q_mtx = compute_q_matrix(self._rates, self._freqs)
        self.eigen = Eigen(*get_eigen(self._q_mtx, self._freqs))
