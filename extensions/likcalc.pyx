# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log

__all__ = ['discrete_gamma', 'likvec', 'likvec2']

cdef double SCALE_THRESHOLD = 1.0 / (2.0**128)
cdef double LOG_SCALE_VALUE = log(SCALE_THRESHOLD)

cdef extern from "discrete_gamma.h":
    int DiscreteGamma(double* freqK, double* rK, double alpha, double beta, int K, int UseMedian) nogil

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

cpdef int _partials(double[:,::1] probs1, double[:,::1] probs2, double[:,::1] partials1,
                    double[:,::1] partials2, double[:,::1] out_buffer) nogil:
    """ Cython implementation of Eq (2), Yang (2000) """
    cdef size_t i, j, k
    cdef double entry1, entry2
    cdef size_t sites = partials1.shape[0]
    cdef size_t states = partials1.shape[1]
    for i in xrange(sites):
        for j in xrange(states):
            entry1 = 0
            entry2 = 0
            for k in xrange(states):
                entry1 += probs1[j, k] * partials1[i, k]
                entry2 += probs2[j, k] * partials2[i, k]
            out_buffer[i, j] = entry1 * entry2
    return 0

cpdef int _scaled_partials(double[:,::1] probs1, double[:,::1] probs2, double[:,::1] partials1,
                    double[:,::1] partials2, int[::1] scale_buffer, double[:,::1] out_buffer) nogil:
    """
    _scaled_partials(double[:,::1] probs1, double[:,::1] probs2, double[:,::1] partials1,
                    double[:,::1] partials2, int[::1] scale_buffer, double[:,::1] out_buffer)

    Cython implementation of Eq (2), Yang (2000)

    Calculate partials array at parent node, given descendents' probability matrices and
    partials arrays. Does scaling to avoid underflows.
    """
    cdef size_t i, j, k
    cdef double entry1, entry2
    cdef size_t sites = partials1.shape[0]
    cdef size_t states = partials1.shape[1]
    cdef int do_scaling  # if all values are < SCALE_THRESHOLD, do scaling
    # The way we decide to do scaling is, for each site, we commit to
    # do scaling, but change our mind if any value is above the threshold.
    # If *all* entries are below the threshold, then divide all entries
    # by the threshold value. The scale buffer keeps count of how many
    # times this has been done for each site (so it can be undone).
    for i in xrange(sites):
        do_scaling = 1
        for j in xrange(states):
            entry1 = 0
            entry2 = 0
            for k in xrange(states):
                entry1 += probs1[j, k] * partials1[i, k]
                entry2 += probs2[j, k] * partials2[i, k]
            out_buffer[i, j] = entry1 * entry2
            if out_buffer[i, j] > SCALE_THRESHOLD:
                do_scaling = 0

        # scaling
        if do_scaling == 1:
            for k in xrange(states):
                out_buffer[i, k] /= SCALE_THRESHOLD
            scale_buffer[i] = 1  # should this be += 1 ? - no, accumulation is done over the tree

    return do_scaling

cpdef int _single_site_lik_derivs(double[:,::1] probs, double[:,::1] dprobs, double[:,::1] d2probs,
                                  double[::1] pi, double[::1] partials_a, double[::1] partials_b,
                                  double[::1] out) nogil:
    """
    Compute sitewise values of log-likelihood and derivatives
    - equations (10) & (11) from Yang (2000)
    """
    cdef size_t a, b  # loop indices
    cdef double f, fp, f2p  # values to return
    cdef double abuf, apbuf, a2pbuf# buffers store partial results of sums
    cdef size_t states = partials_a.shape[0]
    cdef int retval = 0

    f = 0
    fp = 0
    f2p = 0

    for a in xrange(states):
        abuf = 0
        apbuf = 0
        a2pbuf = 0
        for b in xrange(states):
            abuf += probs[a, b] * partials_b[b]
            apbuf += dprobs[a, b] * partials_b[b]
            a2pbuf += d2probs[a, b] * partials_b[b]
        f += pi[a] * partials_a[a] * abuf
        fp += pi[a] * partials_a[a] * apbuf
        f2p += pi[a] * partials_a[a] * a2pbuf

    if f < 1e-320: # numerical stability issues, clamp to a loggable value
        f = 1e-320 # (but this should never be needed with proper scaling)
        retval = 1
    out[0] = log(f) # requires scaling
    out[1] = fp / f # scale free
    out[2] = ((f * f2p) - (fp * fp)) / (f * f) # scale free
    return retval

cpdef int _single_site_lik(double[:,::1] probs,
                           double[::1] pi, double[::1] partials_a,
                           double[::1] partials_b, double[::1] out) nogil:
    """
    Compute sitewise values of log-likelihood
    """
    cdef size_t a, b
    cdef double f, abuf
    cdef size_t states = partials_a.shape[0]
    cdef int retval = 0

    f = 0
    for a in xrange(states):
        abuf = 0
        for b in xrange(states):
            abuf += probs[a, b] * partials_b[b]
        f += pi[a] * partials_a[a] * abuf
    if f < 1e-320: # numerical stability issues, clamp to a loggable value
        f = 1e-320 # (but this should never be needed with proper scaling)
        retval = 1
    out[0] = log(f)
    return 1

cpdef int _sitewise_lik_derivs(double[:,::1] probs, double[:,::1] dprobs, double[:,::1] d2probs,
                               double[::1] pi, double[:,::1] partials_a, double[:,::1] partials_b,
                               double[:,::1] out) nogil:
    """
    Compute sitewise values of log-likelihood and derivatives
    - equations (10) & (11) from Yang (2000)
    """
    cdef size_t site  # loop indices
    cdef size_t sites = partials_a.shape[0]
    cdef size_t states = partials_a.shape[1]
    cdef int retval = 0

    for site in xrange(sites):
        if _single_site_lik_derivs(probs, dprobs, d2probs, pi, partials_a[site], partials_b[site], out[site]) > 0:
            retval = 1
    return 0

cpdef int _sitewise_lik(double[:,::1] probs,
                        double[::1] pi, double[:,::1] partials_a,
                        double[:,::1] partials_b, double[:,::1] out) nogil:
    """
    Compute sitewise values of log-likelihood - equation (10) from Yang (2000)
    """
    cdef size_t site
    cdef size_t sites = partials_a.shape[0]
    cdef size_t states = partials_a.shape[1]

    for site in xrange(sites):
        _single_site_lik(probs, pi, partials_a[site], partials_b[site], out[site])
    return 0

cpdef int _weighted_choice(int[::1] choices, double[::1] weights) nogil:
    """
    Return a random choice from int array 'choices', proportionally
    weighted by 'weights'
    """
    cdef size_t l = weights.shape[0]
    cdef double total = 0.0
    for i in xrange(l):
        total += weights[i]
    cdef double r = total * rand()/RAND_MAX
    cdef double upto = 0

    for i in xrange(l):
        upto += weights[i]
        if upto > r:
            return choices[i]
    return 0

cpdef int _weighted_choices(int[::1] choices, double[::1] weights, int[::1] output) nogil:
    """
    Fill output array with weighted choices from 'choices' array, proportionally
    weighted by weights
    """
    cdef size_t nchoices = weights.shape[0]
    cdef size_t nsites = output.shape[0]
    cdef double total = 0.0
    cdef size_t i, j
    for i in xrange(nchoices):
        total += weights[i]
    cdef double r, upto

    for i in xrange(nsites):
        r = total * rand()/RAND_MAX
        upto = 0
        for j in xrange(nchoices):
            upto += weights[j]
            if upto > r:
                output[i] = choices[j]
                break
    return 0

cpdef int _evolve_states(int[::1] all_states, int[::1] parent_states, int[::1] categories, double[:,:,::1] probs, int[::1] child_states) nogil:
    """
    Evolve states from parent to child according to 'probs'
    """
    cdef size_t nsites = parent_states.shape[0]
    cdef size_t i
    for i in xrange(nsites):
        child_states[i] = _weighted_choice(all_states, probs[categories[i], parent_states[i], :])
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
    r = np.empty_like(partials)
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
    r = np.empty_like(partials1)
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
    r = np.empty_like(partials1)
    s = np.zeros(sites, dtype=np.intc)
    _scaled_partials(probs1, probs2, partials1, partials2, s, r)
    return r, s

def sitewise_lik_derivs(probs, dprobs, d2probs, freqs, partials_a, partials_b):
    sites = partials_a.shape[0]
    r = np.empty((sites, 3))
    check = _sitewise_lik_derivs(probs, dprobs, d2probs, freqs, partials_a, partials_b, r)
    if check == 1:
        print 'Scaling error encountered! Used hack!'
    return r

def sitewise_lik(probs, freqs, partials_a, partials_b):
    sites = partials_a.shape[0]
    r = np.empty((sites, 1))
    check = _sitewise_lik(probs, freqs, partials_a, partials_b, r)
    if check == 1:
        print 'Scaling error encountered! Used hack!'
    return r

def get_scale_threshold():
    return SCALE_THRESHOLD

def get_log_scale_value():
    return LOG_SCALE_VALUE

cdef int _simplex_encode(double[::1] p, double[::1] theta):
    """
    Convert vector p (length N) to vector theta (length N-1)
    p is constrained to sum to 1, theta is not
    """
    cdef size_t i, N=p.shape[0]
    cdef double y=1.0
    for i in range(N-1):
        theta[i] = p[i] / y
        y -= p[i]
    return 0

def simplex_encode(p):
    theta = np.zeros(p.size - 1)
    _simplex_encode(p, theta)
    return theta

cdef int _simplex_decode(double[::1] theta, double[::1] p):
    cdef size_t i, N=theta.shape[0]
    cdef double x=1.0
    for i in range(N):
        p[i] = theta[i]*x
        x *= 1.0 - theta[i]
    p[N] = x
    return 0

def simplex_decode(theta):
    p = np.zeros(theta.size + 1)
    _simplex_decode(theta, p)
    return p

def logit(p):
    return np.log(p/(1-p))

def expit(p):
    return 1/(1+np.exp(-p))

def transform_params(p):
    return logit(simplex_encode(p))

def decode_params(q):
    return simplex_decode(expit(q))
