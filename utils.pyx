import numpy as np
cimport numpy as np
cimport cython

__all__ = ['discrete_gamma', 'likvec']

cdef extern from "discrete_gamma.h":
    int DiscreteGamma(double* freqK, double* rK, double alpha, double beta, int K, int UseMedian)

def _discrete_gamma(np.ndarray[np.double_t,ndim=1] freqK, np.ndarray[np.double_t,ndim=1] rK, alpha, beta, K, UseMedian):
    return DiscreteGamma(<double*>freqK.data, <double*>rK.data, alpha, beta, K, UseMedian)

def discrete_gamma(alpha, ncat, median_rates=False):
    """
    Generates rates for discrete gamma distribution,
    for `ncat` categories. By default calculates mean rates,
    can also calculate median rates by setting median_rates=True.
    Phylogenetic context, so assumes that gamma parameters 
    alpha == beta, so that expectation of gamma dist. is 1.
    
    C source code taken from PAML.
    
    Usage:
    rates = discrete_gamma(0.5, 5)  # Mean rates (see Fig 4.9, p.118, Ziheng's book on lizards)
    >>> array([ 0.02121238,  0.15548577,  0.46708288,  1.10711735,  3.24910162])
    """

    weights = np.zeros(ncat, dtype=np.double)
    rates = np.zeros(ncat, dtype=np.double)
    _ = _discrete_gamma(weights, rates, alpha, alpha, ncat, <int>median_rates)
    return rates

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _lnlmv(double[:,::1] probs, double[:,::1] partials, double[:,::1] return_value) nogil:
    cdef size_t i, j, k
    cdef double entry
    sites = partials.shape[0]
    states = partials.shape[1]
    for i in xrange(sites):
        for j in xrange(states):
            entry = 0
            for k in xrange(states):
                entry += probs[j, k] * partials[i, k]
            return_value[i, j] = entry
    return 0

def likvec(probs, partials):
    sites, states = partials.shape
    r = np.empty((sites,states))
    _lnlmv(probs, 
         partials,
         r)
    return r