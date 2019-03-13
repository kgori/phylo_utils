# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
import numpy as np
cimport numpy as np

#########################################################
# Code from PAML to generate rates for gamma distribution
#########################################################

cdef extern from "c_discrete_gamma.h":
    int DiscreteGamma(double* freqK, double* rK, double alpha, double beta, int K, int UseMedian) nogil
    double LnGamma(double x) nogil
    double QuantileChi2(double prob, double v) nogil
    double IncompleteGamma(double x, double alpha, double ln_gamma_alpha) nogil

cdef double _ln_gamma(double x) nogil:
    return LnGamma(x)

cdef double _quantile_gamma(double prob, double alpha, double beta) nogil:
    return QuantileChi2(prob, 2.0 * (alpha)) / (2.0 * (beta))

cdef double _incomplete_gamma(double x, double alpha, double ln_gamma_alpha) nogil:
    return IncompleteGamma(x, alpha, ln_gamma_alpha)

cpdef int _discrete_gamma(double[:] freqK, double[:] rK, double alpha, double beta, int K, int UseMedian) nogil:
    return DiscreteGamma(&freqK[0], &rK[0], alpha, beta, K, UseMedian)

def discrete_gamma(double alpha, int ncat, int median_rates=False):
    """
    Generates rates for discrete gamma distribution,
    for `ncat` categories. By default calculates mean rates,
    can also calculate median rates by setting median_rates=True.
    Phylogenetic context, so assumes that gamma parameters
    alpha == beta, so that expectation of gamma dist. is 1.

    C source code taken from PAML.

    Usage:
    rates = discrete_gamma(0.5, 5)  # Mean rates (see Fig 4.9, p.118, Ziheng's Molecular Evolution)
    >>> array([ 0.02121238,  0.15548577,  0.46708288,  1.10711735,  3.24910162])
    """
    weights = np.zeros(ncat, dtype=np.double)
    rates = np.zeros(ncat, dtype=np.double)
    _ = _discrete_gamma(weights, rates, alpha, alpha, ncat, <int>median_rates)
    return rates
