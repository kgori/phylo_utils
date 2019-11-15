from abc import ABCMeta

import numpy as np

from phylo_utils.substitution_models.utils import compute_b_matrix, check_frequencies, check_rates, compute_q_matrix, \
    get_eigen, expm

MIN_BRANCH_LENGTH = 1/2**16


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
            return np.stack([self.eigen.exp(t * rate) for rate in rates], axis=0)

    def dp_dt(self, t, rates=None):
        """
        First derivative of P w.r.t t
        """
        if rates is None:
            return self.eigen.fn_apply(lambda x: x * np.exp(x * t))
        else:
            return np.stack([self.eigen.fn_apply(lambda x: x * np.exp(x * t * rate)) for rate in rates], axis=0)

    def d2p_dt2(self, t, rates=None):
        """
        Second derivative of P w.r.t t
        """
        if rates is None:
            return self.eigen.fn_apply(lambda x: x * x * np.exp(x * t))
        else:
            return np.stack([self.eigen.fn_apply(lambda x: x * x * np.exp(x * t * rate)) for rate in rates], axis=0)

    def detailed_balance(self):
        """
        Check if the model satisfies detailed balance (π_i * q_ij == π_j * q_ji)
        :return: True if detailed balance holds, otherwise False
        """
        m = self.q().T * self.freqs
        return np.allclose(m, m.T)


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
        if t < 1e-8:
            return (self.evecs * np.expm1(self.evals * t)).dot(self.ivecs) + np.eye(self.evals.shape[0])
        else:
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

    def __repr__(self):
        s = 'Protein model: {}\n'.format(self._name)
        s += 'Freqs:        {}\n'.format(self._freqs)
        return s


class DNAReversibleModel(Model):
    _name = 'GenericReversibleDNA'
    _size = 4
    _states = ['A', 'C', 'G', 'T']
    def __init__(self, rates, freqs):
        self._rates = check_rates(rates, self.size)
        self._freqs = check_frequencies(freqs, self.size)

    def q(self):
        return self._q_mtx

    def __repr__(self):
        s = 'DNA reversible model: {}\n'.format(self._name)
        s += 'Rel. rates: {}\n'.format(self._rates[np.array([0,0,0,1,1,2]), np.array([1,2,3,2,3,3])])
        s += 'Freqs:      {}\n'.format(self._freqs)
        return s


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
            return expm(q * t)
        else:
            return np.stack([expm(q * rate * t) for rate in rates], axis=0)

    def dp_dt(self, t, rates = None):
        q = self.q()
        if rates is None:
            return q.dot(expm(q * t))
        else:
            return np.stack([q.dot(expm(q * rate * t)) for rate in rates], axis=0)

    def d2p_dt2(self, t, rates = None):
        q = self.q()
        if rates is None:
            return q.dot(q).dot(expm(q * t))
        else:
            return np.stack([q.dot(q).dot(expm(q * rate * t)) for rate in rates], axis=0)

    def __repr__(self):
        s = 'DNA reversible model: {}\n'.format(self._name)
        s += 'Rel. rates: {}\n'.format(self._rates[np.array([0,0,0,1,1,1,2,2,2,3,3,3]),
                                                   np.array([1,2,3,0,2,3,0,1,3,0,1,2])])
        s += 'Freqs:      {}\n'.format(self.freqs)
        return s