import numpy as np

from phylo_utils.data import fixed_equal_nucleotide_frequencies
from phylo_utils.substitution_models.abstract import check_frequencies, check_rates, Eigen, DNAReversibleModel


def tn93_q(pi_a, pi_c, pi_g, pi_t, alpha_y, alpha_r, beta, scale):
    pi_y = pi_t + pi_c
    pi_r = pi_a + pi_g
    q = np.ascontiguousarray([
        [-(alpha_r * pi_g + beta * pi_y), beta * pi_c, alpha_r * pi_g, beta * pi_t],
        [beta * pi_a, -(alpha_y * pi_t + beta * pi_r), beta * pi_g, alpha_y * pi_t],
        [alpha_r * pi_a, beta * pi_c, -(alpha_r * pi_a + beta * pi_y), beta * pi_t],
        [beta * pi_a, alpha_y * pi_c, beta * pi_g, -(alpha_y * pi_c + beta * pi_r)]])
    return q / scale


def tn93_scale(pi_a, pi_c, pi_g, pi_t, alpha_y, alpha_r, beta):
    return 2 * (alpha_y*pi_c*pi_t +
                beta*pi_a*pi_t +
                beta*pi_a*pi_c +
                alpha_r*pi_a*pi_g +
                beta*pi_g*pi_t +
                beta*pi_c*pi_g)


def tn93_evecs(pi_a, pi_c, pi_g, pi_t):
    pi_y = pi_t + pi_c
    pi_r = pi_a + pi_g
    return np.ascontiguousarray(
        [[1, -1 / pi_r, pi_g / pi_r, 0],
         [1, 1/pi_y, 0, -pi_t/pi_y],
         [1, -1/pi_r, -pi_a/pi_r, 0],
         [1, 1 / pi_y, 0, pi_c / pi_y]],
        dtype=np.double)


def tn93_ivecs(pi_a, pi_c, pi_g, pi_t):
    pi_y = pi_t + pi_c
    pi_r = pi_a + pi_g
    return np.asfortranarray(
        [[pi_a, pi_c, pi_g, pi_t],
         [-pi_a * pi_y, pi_c * pi_r, -pi_g * pi_y, pi_t * pi_r],
         [1, 0, -1, 0],
         [0, -1, 0, 1]],
        dtype=np.double)


def tn93_evals(pi_a, pi_c, pi_g, pi_t, alpha_y, alpha_r, beta, scale):
    pi_y = pi_t + pi_c
    pi_r = pi_a + pi_g
    return np.ascontiguousarray([0,
                                 -beta,
                                 -(pi_r*alpha_r + pi_y*beta),
                                 -(pi_y*alpha_y + pi_r*beta)], dtype=np.double) / scale


class TN93(DNAReversibleModel):
    _name = 'TN93'
    def __init__(self, alpha_y, alpha_r, beta=1.0, freqs=None, scale_q=True):
        if freqs is None:
            freqs = fixed_equal_nucleotide_frequencies.copy()
        else:
            freqs = check_frequencies(freqs, 4)
        self._freqs = freqs
        scale = tn93_scale(*freqs, alpha_y=alpha_y, alpha_r=alpha_r, beta=beta) if scale_q else 1.0
        mtx = np.array([
            [0, beta, alpha_r, beta],
            [beta, 0, beta, alpha_y],
            [alpha_r, beta, 0, beta],
            [beta, alpha_y, beta, 0]])
        self._rates = check_rates(mtx, 4)
        self._alpha_y = alpha_y
        self._alpha_r = alpha_r
        self._beta = beta

        self._q_mtx = tn93_q(freqs[0], freqs[1], freqs[2], freqs[3], alpha_y, alpha_r, beta, scale)
        self.eigen = Eigen(tn93_evecs(freqs[0], freqs[1], freqs[2], freqs[3]),
                        tn93_evals(freqs[0], freqs[1], freqs[2], freqs[3],
                                      alpha_y, alpha_r, beta, scale),
                        tn93_ivecs(freqs[0], freqs[1], freqs[2], freqs[3]))
