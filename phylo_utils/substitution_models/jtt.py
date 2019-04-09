from phylo_utils.data import jtt_rates, jtt_freqs
from phylo_utils.substitution_models.abstract import check_frequencies, compute_q_matrix, get_eigen, Eigen, ProteinModel


class JTT(ProteinModel):
    _name = 'JTT'
    _rates = jtt_rates.copy()
    def __init__(self, freqs=None):
        if freqs is None:
            self._freqs = jtt_freqs.copy()
        else:
            self._freqs = check_frequencies(freqs, self.size)
        self._q_mtx = compute_q_matrix(self._rates, self._freqs)
        self.eigen = Eigen(*get_eigen(self._q_mtx, self._freqs))
