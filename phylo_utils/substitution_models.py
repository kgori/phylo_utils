from __future__ import division
import numpy as np
from abc import ABCMeta

from phylo_utils.data import lg_rates, lg_freqs, wag_rates, wag_freqs, fixed_equal_nucleotide_rates, \
    fixed_equal_nucleotide_frequencies

__all__ = ['LG', 'WAG', 'GTR', 'JC69', 'K80', 'F81', 'F84', 'HKY85', 'TN93']

MIN_BRANCH_LENGTH = 1/2**16
SMALL = 1/2**128

def check_frequencies(freqs, length):
    freqs = np.ascontiguousarray(freqs)
    if len(freqs) != length:
        raise ValueError('Frequencies vector is not the right length (length={})'.format(len(freqs)))
    if not np.allclose(sum(freqs), 1.0, rtol=1e-16):
        raise ValueError('Frequencies do not add to 1.0 within tolerance (sum={})'.format(sum(freqs)))
    return freqs

def check_rates(rates, size):
    rates = np.ascontiguousarray(rates)
    if rates.shape != (size, size):
        raise ValueError('Rate matrix is not the right shape (length={})'.format(rates.shape))
    if not np.allclose(rates, rates.T):
        raise ValueError('Rate matrix is not symmetrical')
    return rates

def impose_min_probs(mtx):
    if np.min(mtx) < SMALL:
        clipped = np.clip(mtx, SMALL, 1.0)
        clipped = clipped / clipped.sum(1)[:,np.newaxis]
        return (clipped + clipped.T) / 2.0
    return mtx


class Model(object):
    __metaclass__ = ABCMeta
    _name = None
    _rates = None
    _freqs = None
    _size = None
    _states = None

    def __init__(self):
        self.eigen = None
        self._q_mtx = None

    @property
    def name(self):
        return self._name

    @property
    def rates(self):
        return self._rates

    @property
    def freqs(self):
        return self._freqs

    @property
    def size(self):
        return self._size

    @property
    def states(self):
        return self._states

    def get_q_matrix(self):
        return self._q_mtx

    def get_b_matrix(self):
        return compute_b_matrix(self.get_q_matrix(), np.sqrt(self.freqs))

    def get_p_matrix(self, t, rates=None):
        """
        P = transition probabilities
        """
        evecs, evals, ivecs = self.eigen.values
        if rates is None:
            return impose_min_probs((evecs * np.exp(evals * t)).dot(ivecs))
            # return (evecs * np.exp(evals * t)).dot(ivecs)
        else:
            return np.stack(
                [impose_min_probs((evecs * np.exp(evals * t * rate)).dot(ivecs)) for rate in rates],
                # [(evecs * np.exp(evals * t * rate)).dot(ivecs) for rate in rates],
                axis=2)

    def get_dp_matrix(self, t, rates=None):
        """
        First derivative of P
        """
        evecs, evals, ivecs = self.eigen.values
        if rates is None:
            return (evecs * evals * np.exp(evals * t)).dot(ivecs)
        else:
            return np.stack([(evecs * evals * np.exp(evals * t * rate)).dot(ivecs) for rate in rates],
                            axis=2)

    def get_d2p_matrix(self, t, rates=None):
        """
        Second derivative of P
        """
        evecs, evals, ivecs = self.eigen.values
        if rates is None:
            return (evecs * evals * evals * np.exp(evals * t)).dot(ivecs)
        else:
            return np.stack([(evecs * evals * evals * np.exp(evals * t * rate)).dot(ivecs) for rate in rates], axis=2)


def compute_q_matrix(rates, freqs, scale=True):
    """
    Computes the instantaneous rate matrix (Q matrix)
    from substitution rates and equilibrium frequencies.
    Values are scaled s.t. units are in expected substitutions
    per site (E(sps)) - see
    https://en.wikipedia.org/wiki/Models_of_DNA_evolution#Scaling_of_branch_lengths.
    Scaling factor = -∑π_i*q_ii
    """
    if freqs is None:
        q = rates
    else:
        q = rates.dot(np.diag(freqs))
    q.flat[::len(freqs)+1] -= q.sum(1)
    if scale:
        scale_factor = -np.diag(q).dot(freqs)
        q /= scale_factor # scale so lengths are in E(sps)
    return q


def compute_b_matrix(q_matrix, sqrtfreqs):
    """
    Computes a matrix (B matrix), similar to Q - i.e. with
    the same eigenvalues.
    B is symmetric if Q is reversible, allowing the use of more
    stable numerical eigen decomposition routines on B than on Q.
    """
    return np.diag(sqrtfreqs).dot(q_matrix).dot(np.diag(1/sqrtfreqs))


def q_to_freqs(q_matrix):
    """
    Compute the equilibrium frequencies from a Q matrix by
    solving Q'r = 0 subject to 1.r = 1
    """
    n = q_matrix.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[0] = 1
    M[1:, :n] = q_matrix.T

    pi, _, _, _ = np.linalg.lstsq(M[:, :n], M[:, n])
    return pi


def get_eigen(q_matrix, freqs=None):
    if freqs is not None:
        rootf = np.sqrt(freqs)
        mtx = compute_b_matrix(q_matrix, rootf)
        evals, r = np.linalg.eigh(mtx)
        evecs = np.diag(1/rootf).dot(r)
        ivecs = r.T.dot(np.diag(rootf))
    else:
        mtx = q_matrix
        evals, evecs = np.linalg.eig(mtx)
        sort_ix = np.argsort(evals)
        evals = evals[sort_ix]
        evecs = evecs[:, sort_ix]
        ivecs = np.linalg.inv(evecs)
    return (np.ascontiguousarray(evecs),
            np.ascontiguousarray(evals),
            np.asfortranarray(ivecs))


class Eigen(object):
    __slots__ = ['evals', 'evecs', 'ivecs']
    def __init__(self, evecs, evals, ivecs):
        self.evecs = evecs
        self.evals = evals
        self.ivecs = ivecs

    @property
    def values(self):
        return self.evecs, self.evals, self.ivecs


class ProteinModel(Model):
    __metaclass__ = ABCMeta
    _name = 'GenericProtein'
    _size = 20
    _states = ['A', 'R', 'N', 'D', 'C',
               'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P',
               'S', 'T', 'W', 'Y', 'V']
    def __init__(self, rates, freqs):
        self._rates = check_rates(rates, self.size)
        self._freqs = check_frequencies(freqs, self.size)
        self._q_mtx = compute_q_matrix(self._rates, self._freqs)
        self.eigen = Eigen(*get_eigen(self._q_mtx, self._freqs))


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


class WAG(ProteinModel):
    _name = 'WAG'
    _rates = wag_rates.copy()
    def __init__(self, freqs=None):
        if freqs is None:
            self._freqs = wag_freqs.copy()
        else:
            self._freqs = check_frequencies(freqs, self.size)
        self._q_mtx = compute_q_matrix(self._rates, self._freqs)
        self.eigen = Eigen(*get_eigen(self._q_mtx, self._freqs))


class DNAReversibleModel(Model):
    _name = 'GenericReversibleDNA'
    _size = 4
    _states = ['A', 'C', 'G', 'T']
    def __init__(self, rates, freqs):
        self._rates = check_rates(rates, self.size)
        self._freqs = check_frequencies(freqs, self.size)

    def get_q_matrix(self):
        return self._q_mtx


class DNANonReversibleModel(Model):
    _name = 'GenericNonReversibleDNA'
    _size = 4
    _states = ['A', 'C', 'G', 'T']
    def __init__(self, rates):
        if not rates.shape == (self.size, self.size):
            raise ValueError('DNA rate matrix is not 4x4')
        self._rates = rates

    def get_q_matrix(self):
        return self._q_mtx


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

    def get_p_matrix(self, t):
        e1 = 0.25 + 0.75*np.exp(-4*t/3.)
        e2 = 0.25 - 0.25*np.exp(-4*t/3.)
        return impose_min_probs(np.array([[e1, e2, e2, e2],
                                          [e2, e1, e2, e2],
                                          [e2, e2, e1, e2],
                                          [e2, e2, e2, e1]]))
        # return np.array([[e1, e2, e2, e2], [e2, e1, e2, e2], [e2, e2, e1, e2], [e2, e2, e2, e1]])

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
    def __init__(self, alpha_y, alpha_r, beta, freqs=None, scale_q=True):
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


class K80(TN93):
    _name = 'K80'
    _freqs = fixed_equal_nucleotide_frequencies.copy()
    def __init__(self, kappa, scale_q=True):
        super(K80, self).__init__(kappa, kappa, 1, self._freqs, scale_q=scale_q)


class F81(TN93):
    _name = 'F81'
    def __init__(self, freqs, scale_q=True):
        super(F81, self).__init__(1, 1, 1, freqs, scale_q=scale_q)


class F84(TN93):
    _name = 'F84'
    def __init__(self, kappa, freqs, scale_q=True):
        alpha_y = 1+kappa/(freqs[1]+freqs[3])
        alpha_r = 1+kappa/(freqs[0]+freqs[2])
        super(F84, self).__init__(alpha_y, alpha_r, 1, freqs, scale_q=scale_q)


class HKY85(TN93):
    _name = 'HKY85'
    def __init__(self, kappa, freqs, scale_q=True):
        super(HKY85, self).__init__(kappa, kappa, 1, freqs, scale_q=scale_q)
