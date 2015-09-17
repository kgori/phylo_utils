import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, log

__all__ = ['discrete_gamma', 'likvec', 'likvec2']

cdef extern from "discrete_gamma.h":
    int DiscreteGamma(double* freqK, double* rK, double alpha, double beta, int K, int UseMedian) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
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
    rates = discrete_gamma(0.5, 5)  # Mean rates (see Fig 4.9, p.118, Ziheng's book on lizards)
    >>> array([ 0.02121238,  0.15548577,  0.46708288,  1.10711735,  3.24910162])
    """
    weights = np.zeros(ncat, dtype=np.double)
    rates = np.zeros(ncat, dtype=np.double)
    _ = _discrete_gamma(weights, rates, alpha, alpha, ncat, <int>median_rates)
    return rates

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _partials_one_term(double[:,::1] probs, double[:,::1] partials, double[:,::1] return_value) nogil:
    """ Cython implementation of single term of Eq (2), Yang (2000) """
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _partials(double[:,::1] probs1, double[:,::1] probs2, double[:,::1] partials1,
                    double[:,::1] partials2, double[:,::1] out_buffer) nogil:
    """ Cython implementation of Eq (2), Yang (2000) """
    cdef size_t i, j, k
    cdef double entry1, entry2
    sites = partials1.shape[0]
    states = partials1.shape[1]
    for i in xrange(sites):
        for j in xrange(states):
            entry1 = 0
            entry2 = 0
            for k in xrange(states):
                entry1 += probs1[j, k] * partials1[i, k]
                entry2 += probs2[j, k] * partials2[i, k]
            out_buffer[i, j] = entry1 * entry2
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _scaled_partials(double[:,::1] probs1, double[:,::1] probs2, double[:,::1] partials1,
                    double[:,::1] partials2, double[::1] scale_buffer, double[:,::1] out_buffer) nogil:
    """ Cython implementation of Eq (2), Yang (2000) """
    cdef size_t i, j, k
    cdef double entry1, entry2, l, lmax
    sites = partials1.shape[0]
    states = partials1.shape[1]
    for i in xrange(sites):
        lmax = 0
        for j in xrange(states):
            entry1 = 0
            entry2 = 0
            for k in xrange(states):
                entry1 += probs1[j, k] * partials1[i, k]
                entry2 += probs2[j, k] * partials2[i, k]
            l = entry1 * entry2
            if l > lmax: 
                lmax = l
            out_buffer[i, j] = l
        for k in xrange(states):
            out_buffer[i, k] /= lmax
        scale_buffer[i] = log(lmax)
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _single_site_lik_derivs(double[:,::1] evecs, double[::1] evals, double[::1,:] ivecs, 
                                  double[::1] pi, double t, 
                                  double[::1] partials_a, double[::1] partials_b,
                                  double[::1] out) nogil:
    """
    Compute sitewise values of log-likelihood and derivatives 
    - equations (10) & (11) from Yang (2000)
    """
    cdef size_t w, a, b, k  # loop indices
    cdef double f, fp, f2p  # values to return
    cdef double abuf, apbuf, a2pbuf, bbuf, bpbuf, b2pbuf, tmp1, tmp2  # buffers store partial results of sums
    states = partials_a.shape[0]

    f = 0
    fp = 0
    f2p = 0
    for a in xrange(states):
        abuf = 0
        apbuf = 0
        a2pbuf = 0
        for b in xrange(states):
            bbuf = 0
            bpbuf = 0
            b2pbuf = 0
            for k in xrange(states):
                tmp1 = evecs[a, k] * ivecs[k, b] * exp(evals[k]*t)
                tmp2 = tmp1 * evals[k]
                bbuf += tmp1
                bpbuf += tmp2
                b2pbuf += tmp2 * evals[k]
            abuf += bbuf * partials_b[b]
            apbuf += bpbuf * partials_b[b]
            a2pbuf += b2pbuf * partials_b[b]
        f += pi[a] * abuf * partials_a[a]
        fp += pi[a] * apbuf * partials_a[a]
        f2p += pi[a] * a2pbuf * partials_a[a]
    if f < 1e-320: # numerical stability issues, clamp to a loggable value
        f = 1e-320
    out[0] = f
    out[1] = fp
    out[2] = f2p
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _single_site_lik(double[:,::1] evecs, double[::1] evals, double[::1,:] ivecs, 
                           double[::1] pi, double t, double[::1] partials_a,
                           double[::1] partials_b, double[::1] out) nogil:
    """
    Compute sitewise values of log-likelihood
    """
    cdef size_t i, j, k
    cdef double f, s, sb
    states = partials_a.shape[0]

    f = 0
    for a in xrange(states):
        sb = 0
        for b in xrange(states):
            s = 0
            for k in xrange(states):
                s += evecs[a, k] * ivecs[k, b] * exp(evals[k]*t)
            sb += s * partials_b[b]
        f += pi[a] * sb * partials_a[a]
    if f < 1e-320: # numerical stability issues, clamp to a loggable value
        f = 1e-320
    out[0] = f
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _sitewise_lik_derivs(double[:,::1] evecs, double[::1] evals, double[::1,:] ivecs, 
                               double[::1] pi, double t, 
                               double[:,::1] partials_a, double[:,::1] partials_b,
                               double[:,::1] out) nogil:
    """
    Compute sitewise values of log-likelihood and derivatives 
    - equations (10) & (11) from Yang (2000)
    """
    cdef size_t site  # loop indices
    sites = partials_a.shape[0]
    states = partials_a.shape[1]

    for site in xrange(sites):
        _single_site_lik_derivs(evecs, evals, ivecs, pi, t, partials_a[site], partials_b[site], out[site])
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int _sitewise_lik(double[:,::1] evecs, double[::1] evals, double[::1,:] ivecs, 
                        double[::1] pi, double t, double[:,::1] partials_a,
                        double[:,::1] partials_b, double[:,::1] out) nogil:
    """
    Compute sitewise values of log-likelihood - equation (10) from Yang (2000)
    """
    cdef size_t site
    sites = partials_a.shape[0]
    states = partials_a.shape[1]

    for site in xrange(sites):
        _single_site_lik(evecs, evals, ivecs, pi, t, partials_a[site], partials_b[site], out[site])
    return 0

def likvec_1desc(probs, partials):
    """
    Compute the vector of partials for a single descendant
    The partials vector for a node is the product of these vectors
    for all descendants
    If the node has exactly 2 descendants, then likvec_2desc will
    compute this product directly, and be faster
    One half of Equation (2) from Yang (2000)
    """
    sites, states = partials.shape
    r = np.empty((sites,states))
    _partials_one_term(probs,
         partials,
         r)
    return r

def likvec_2desc(probs1, probs2, partials1, partials2):
    """
    Compute the product of vectors of partials for two descendants,
    i.e. the partials for a node with two descendants
    Equation (2) from Yang (2000)
    """
    if not partials1.shape == partials2.shape or not probs1.shape == probs2.shape: raise ValueError('Mismatched arrays')
    sites, states = partials1.shape
    r = np.empty((sites,states))
    _partials(probs1, probs2, partials1, partials2, r)
    return r

def likvec_2desc_scaled(probs1, probs2, partials1, partials2):
    """
    Compute the product of vectors of partials for two descendants,
    i.e. the partials for a node with two descendants
    Equation (2) from Yang (2000)
    """
    if not partials1.shape == partials2.shape or not probs1.shape == probs2.shape: raise ValueError('Mismatched arrays')
    sites, states = partials1.shape
    r = np.empty((sites,states))
    s = np.empty(sites)
    _scaled_partials(probs1, probs2, partials1, partials2, s, r)
    return r, s

def sitewise_lik_derivs(evecs, evals, ivecs, freqs, t, partials_a, partials_b):
    sites = partials_a.shape[0]
    r = np.empty((sites, 3))
    _sitewise_lik_derivs(evecs, evals, ivecs, freqs, t, partials_a, partials_b, r)
    return r

def sitewise_lik(evecs, evals, ivecs, freqs, t, partials_a, partials_b):
    sites = partials_a.shape[0]
    r = np.empty((sites, 1))
    _sitewise_lik(evecs, evals, ivecs, freqs, t, partials_a, partials_b, r)
    return r
