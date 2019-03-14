from abc import ABCMeta

import numpy as np
from scipy import linalg as LA

MIN_BRANCH_LENGTH = 1/2**16
SMALL = 1/2**128


def check_frequencies(freqs, length):
    freqs = np.ascontiguousarray(freqs)
    if len(freqs) != length:
        raise ValueError('Frequencies vector is not the right length (length={})'.format(len(freqs)))
    if np.min(freqs) < 0:
        raise ValueError('Frequencies vector contains negative values')
    if not np.allclose(sum(freqs), 1.0, rtol=1e-16):
        raise ValueError('Frequencies do not add to 1.0 within tolerance (sum={})'.format(sum(freqs)))
    return freqs


def check_rates(rates, size, symmetry=True):
    rates = np.ascontiguousarray(rates)
    if rates.shape != (size, size):
        raise ValueError('Rate matrix is not the right shape (length={})'.format(rates.shape))
    if np.min(rates) < 0:
        raise ValueError('Rate matrix contains negative values')
    if symmetry and not np.allclose(rates, rates.T):
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

    def q(self):
        return self._q_mtx

    def b(self):
        return compute_b_matrix(self.q(), np.sqrt(self.freqs))

    def p(self, t, rates=None):
        """
        P = transition probabilities after time period t
        :param rates = list/array of floats or None. If not None, probabilities
        are calculated separately for t * rate, for each rate, and stacked
        along the last axis of a multidimensional array.
        """
        if rates is None:
            return self.eigen.exp(t)
        else:
            return np.stack([self.eigen.exp(t * rate) for rate in rates], axis=2)

    def dp_dt(self, t, rates=None):
        """
        First derivative of P w.r.t t
        """
        if rates is None:
            return self.eigen.fn_apply(lambda x: x * np.exp(x * t))
        else:
            return np.stack([self.eigen.fn_apply(lambda x: x * np.exp(x * t * rate)) for rate in rates], axis=2)

    def d2p_dt2(self, t, rates=None):
        """
        Second derivative of P w.r.t t
        """
        if rates is None:
            return self.eigen.fn_apply(lambda x: x * x * np.exp(x * t))
        else:
            return np.stack([self.eigen.fn_apply(lambda x: x * x * np.exp(x * t * rate)) for rate in rates], axis=2)

    def detailed_balance(self):
        """
        Check if the model satisfies detailed balance (π_i * q_ij == π_j * q_ji)
        :return: True if detailed balance holds, otherwise False
        """
        m = self.q().T * self.freqs
        return np.allclose(m, m.T)


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
        q = rates.copy()
    else:
        q = rates.dot(np.diag(freqs))
    assert q.shape[0] == q.shape[1], 'Q is not square'
    q.flat[::q.shape[0]+1] -= q.sum(1)
    if scale:
        if freqs is None:
            freqs = q_to_freqs(q)
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

    pi, _, _, _ = np.linalg.lstsq(M[:, :n], M[:, n], rcond=None)
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

    def exp(self, t=1.0):
        """
        Compute matrix exponential using eigenvalues
        Shorthand for Eigen.fn_apply(lambda x: np.exp(x * t))
        :return:
        """
        return (self.evecs * np.exp(self.evals * t)).dot(self.ivecs)

    def reconstitute(self):
        """
        Reconstitute the original matrix from its eigenvalue
        decomposition.
        Shorthand for Eigen.fn_apply(lambda x: x)
        :return:
        """
        return (self.evecs * self.evals).dot(self.ivecs)

    def fn_apply(self, fn):
        """
        Apply a function to the eigenvalues, then reconstitute
        :param fn: a function, lambda, or callable object
        :return:
        """
        return (self.evecs * fn(self.evals)).dot(self.ivecs)


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


class DNAReversibleModel(Model):
    _name = 'GenericReversibleDNA'
    _size = 4
    _states = ['A', 'C', 'G', 'T']
    def __init__(self, rates, freqs):
        self._rates = check_rates(rates, self.size)
        self._freqs = check_frequencies(freqs, self.size)

    def q(self):
        return self._q_mtx


class DNANonReversibleModel(Model):
    _name = 'GenericNonReversibleDNA'
    _size = 4
    _states = ['A', 'C', 'G', 'T']
    def __init__(self, rates):
        self._rates = check_rates(rates, self.size, symmetry=False)

    def q(self):
        return self._q_mtx

    def p(self, t, rates = None):
        q = self.q()
        if rates is None:
            return LA.expm(q * t)
        else:
            return np.stack([LA.expm(q * rate * t) for rate in rates], axis=2)

    def dp_dt(self, t, rates = None):
        q = self.q()
        if rates is None:
            return q.dot(LA.expm(q * t))
        else:
            return np.stack([q.dot(LA.expm(q * rate * t)) for rate in rates], axis=2)

    def d2p_dt2(self, t, rates = None):
        q = self.q()
        if rates is None:
            return q.dot(q).dot(LA.expm(q * t))
        else:
            return np.stack([q.dot(q).dot(LA.expm(q * rate * t)) for rate in rates], axis=2)
