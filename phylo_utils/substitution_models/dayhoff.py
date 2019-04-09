from phylo_utils.data import dayhoff_rates, dayhoff_freqs
from phylo_utils.substitution_models.abstract import check_frequencies, compute_q_matrix, get_eigen, Eigen, ProteinModel


class Dayhoff(ProteinModel):
    _name = 'Dayhoff'
    _rates = dayhoff_rates.copy()
    def __init__(self, freqs=None):
        if freqs is None:
            self._freqs = dayhoff_freqs.copy()
        else:
            self._freqs = check_frequencies(freqs, self.size)
        self._q_mtx = compute_q_matrix(self._rates, self._freqs)
        self.eigen = Eigen(*get_eigen(self._q_mtx, self._freqs))
