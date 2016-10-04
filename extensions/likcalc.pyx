# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX
from libc.stdio cimport printf
from libc.math cimport log, fabs
cimport openmp
from cython.parallel import prange, parallel

__all__ = ['discrete_gamma', 'likvec', 'likvec2']

cdef double UPPER_SCALE_THRESHOLD = 1.0 / (2.0**32)
cdef double LOWER_SCALE_THRESHOLD = 1.0 / (2.0**128)
cdef double LOG_UPPER_SCALE_THRESHOLD = log(UPPER_SCALE_THRESHOLD)
cdef double LOG_LOWER_SCALE_THRESHOLD = log(LOWER_SCALE_THRESHOLD)


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

cdef int _scaled_partials_parallel(double[:,::1] probs1, double[:,::1] probs2, double[:,::1] partials1,
                    double[:,::1] partials2, double[::1] scale_buffer, double[:,::1] out_buffer, long threads) nogil:
    """
    _scaled_partials(double[:,::1] probs1, double[:,::1] probs2, double[:,::1] partials1,
                    double[:,::1] partials2, int[::1] scale_buffer, double[:,::1] out_buffer)

    Cython implementation of Eq (2), Yang (2000)

    Calculate partials array at parent node, given descendents' probability matrices and
    partials arrays. Does scaling to avoid underflows.
    """
    cdef int i
    cdef size_t j, k
    cdef double entry1, entry2
    cdef size_t sites = partials1.shape[0]
    cdef size_t states = partials1.shape[1]
    cdef int do_scaling
    cdef int min_scaling_breached
    cdef double max_entry, tmp
    # if all values are < SCALE_THRESHOLD, do scaling
    # The way we decide to do scaling is, for each site, we commit to
    # do scaling, but change our mind if any value is above the threshold.
    # If *all* entries are below the threshold, then divide all entries
    # by the threshold value. The scale buffer keeps count of how many
    # times this has been done for each site (so it can be undone).
    with parallel(num_threads=threads):
        for i in prange(sites, schedule='static'):
            max_entry = 0.0
            do_scaling = 1
            min_scaling_breached = 0
            for j in xrange(states):
                entry1 = 0
                entry2 = 0
                for k in xrange(states):
                    entry1 = entry1 + probs1[j, k] * partials1[i, k]
                    entry2 = entry2 + probs2[j, k] * partials2[i, k]
                tmp = entry1 * entry2
                # printf("entry1,entry2,product=%f,%f,%f\n",entry1,entry2,out_buffer[i,j])
                if tmp > max_entry:
                    max_entry = out_buffer[i, j]
                if tmp > UPPER_SCALE_THRESHOLD:
                    do_scaling = 0
                if tmp < LOWER_SCALE_THRESHOLD:
                    min_scaling_breached = 1
                out_buffer[i, j] = tmp

            # scaling
            if (min_scaling_breached or do_scaling):
                for k in xrange(states):
                    # printf(" buffer,max=%f,%f\n", out_buffer[i,k],max_entry);
                    out_buffer[i, k] /= max_entry
                    # printf(" scaled=%f\n", out_buffer[i,k]);
                scale_buffer[i] = log(max_entry)  # should this be += 1 ? - no, accumulation is done over the tree
                # printf(" scalebuf=%f\n", scale_buffer[i]);

    return do_scaling

cdef int _scaled_partials_sequential(double[:,::1] probs1, double[:,::1] probs2, double[:,::1] partials1,
                    double[:,::1] partials2, double[::1] scale_buffer, double[:,::1] out_buffer) nogil:
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
    cdef int do_scaling
    cdef int min_scaling_breached
    cdef double max_entry
    # if all values are < SCALE_THRESHOLD, do scaling
    # The way we decide to do scaling is, for each site, we commit to
    # do scaling, but change our mind if any value is above the threshold.
    # If *all* entries are below the threshold, then divide all entries
    # by the threshold value. The scale buffer keeps count of how many
    # times this has been done for each site (so it can be undone).
    for i in xrange(sites):
        max_entry = 0.0
        do_scaling = 1
        min_scaling_breached = 0
        for j in xrange(states):
            entry1 = 0
            entry2 = 0
            for k in xrange(states):
                entry1 = entry1 + probs1[j, k] * partials1[i, k]
                entry2 = entry2 + probs2[j, k] * partials2[i, k]
            out_buffer[i, j] = entry1 * entry2
            if out_buffer[i, j] > max_entry:
                max_entry = out_buffer[i, j]
            if out_buffer[i, j] > UPPER_SCALE_THRESHOLD:
                do_scaling = 0
            if out_buffer[i, j] < LOWER_SCALE_THRESHOLD:
                min_scaling_breached = 1

        # scaling
        if (min_scaling_breached or do_scaling):
            for k in xrange(states):
                out_buffer[i, k] /= max_entry
            scale_buffer[i] = log(max_entry)  # should this be += 1 ? - no, accumulation is done over the tree

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

cdef int _sitewise_lik_derivs_parallel(double[:,::1] probs, double[:,::1] dprobs, double[:,::1] d2probs,
                                  double[::1] pi, double[:,::1] partials_a, double[:,::1] partials_b,
                                  double[:,::1] out, long threads) nogil:
    """
    Compute sitewise values of log-likelihood and derivatives
    - equations (10) & (11) from Yang (2000)
    """
    cdef int site  # omp_parallel loop indices
    cdef size_t sites = partials_a.shape[0]
    cdef size_t states = partials_a.shape[1]
    cdef size_t a, b # loop indices
    cdef double abuf, apbuf, a2pbuf, pi_a_times_partials  # buffers store partial results of sums
    cdef double f, fp, f2p  # temporaries to transfer to result after loops

    with parallel(num_threads=threads):
        for site in prange(sites, schedule='static'):

            f = 0
            fp = 0
            f2p = 0

            for a in xrange(states):
                abuf = 0
                apbuf = 0
                a2pbuf = 0
                pi_a_times_partials = pi[a] * partials_a[site, a]
                for b in xrange(states):
                    abuf = abuf + probs[a, b] * partials_b[site, b]
                    apbuf = apbuf + dprobs[a, b] * partials_b[site, b]
                    a2pbuf = a2pbuf + d2probs[a, b] * partials_b[site, b]
                f = f + pi_a_times_partials * abuf
                fp = fp + pi_a_times_partials * apbuf
                f2p = f2p + pi_a_times_partials * a2pbuf

            out[site, 0] = log(f) # requires scaling
            out[site, 1] = fp / f # scale free
            out[site, 2] = ((f * f2p) - (fp ** 2)) / (f ** 2) # scale free

    return 0

cdef int _sitewise_lik_derivs_sequential(double[:,::1] probs, double[:,::1] dprobs, double[:,::1] d2probs,
                                  double[::1] pi, double[:,::1] partials_a, double[:,::1] partials_b,
                                  double[:,::1] out) nogil:
    """
    Compute sitewise values of log-likelihood and derivatives
    - equations (10) & (11) from Yang (2000)
    """
    cdef int site  # omp_parallel loop indices
    cdef size_t sites = partials_a.shape[0]
    cdef size_t states = partials_a.shape[1]
    cdef size_t a, b # loop indices
    cdef double abuf, apbuf, a2pbuf  # buffers store partial results of sums
    cdef double f, fp, f2p

    for site in xrange(sites):

        f = 0
        fp = 0
        f2p = 0

        for a in xrange(states):
            abuf = 0
            apbuf = 0
            a2pbuf = 0
            for b in xrange(states):
                abuf += probs[a, b] * partials_b[site, b]
                apbuf += dprobs[a, b] * partials_b[site, b]
                a2pbuf += d2probs[a, b] * partials_b[site, b]
            f += pi[a] * partials_a[site, a] * abuf
            fp += pi[a] * partials_a[site, a] * apbuf
            f2p += pi[a] * partials_a[site, a] * a2pbuf

#             if out[site, 0] < 1e-320: # numerical stability issues, clamp to a loggable value
#                 out[site, 0] = 1e-320 # (but this should never be needed with proper scaling)
#                 retval = 1

        out[site, 0] = log(f) # requires scaling
        out[site, 1] = fp / f # scale free
        out[site, 2] = ((f * f2p) - (fp ** 2)) / (f ** 2) # scale free

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

def likvec_2desc_scaled(probs1, probs2, partials1, partials2, threads=1):
    """
    Compute the product of vectors of partials for two descendants,
    i.e. the partials for a node with two descendants
    Equation (2) from Yang (2000)
    """
    if not partials1.shape == partials2.shape or not probs1.shape == probs2.shape:
        raise ValueError('Mismatched arrays')
    sites, states = partials1.shape
    r = np.empty_like(partials1)
    s = np.zeros(sites, dtype=np.double)
    assert threads > 0
    if threads > 1:
        _scaled_partials_parallel(probs1, probs2, partials1, partials2, s, r, threads)
    else:
        _scaled_partials_sequential(probs1, probs2, partials1, partials2, s, r)
    return r, s

def sitewise_lik_derivs(probs, dprobs, d2probs, freqs, partials_a, partials_b, threads=1):
    sites = partials_a.shape[0]
    r = np.empty((sites, 3))
    assert threads > 0
    if threads > 1:
        _sitewise_lik_derivs_parallel(probs, dprobs, d2probs, freqs, partials_a, partials_b, r, threads)
    else:
        _sitewise_lik_derivs_sequential(probs, dprobs, d2probs, freqs, partials_a, partials_b, r)
    return r

def sitewise_lik(probs, freqs, partials_a, partials_b):
    sites = partials_a.shape[0]
    r = np.empty((sites, 1))
    check = _sitewise_lik(probs, freqs, partials_a, partials_b, r)
    if check == 1:
        print 'Scaling error encountered! Used hack!'
    return r

def get_scale_threshold():
    return UPPER_SCALE_THRESHOLD

def get_log_scale_value():
    return LOG_UPPER_SCALE_THRESHOLD

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


##############
# OPTIMISATION
##############

cdef int ITMAX = 100
cdef double CGOLD = 0.3819660112501051
cdef double ZEPS=1.0e-10
cdef double TINY = 1e-15


cpdef double quad_interp(double p, double q, double r, double fp, double fq, double fr):
    """
    cpdef double quad_interp(double p, double q, double r, double fp, double fq, double fr)

    Quadratic interpolation of (p, fp), (q, fq), (r, fr) -
    Fits a parabola to the three points and returns the abcissa
    of the turning point of the parabola

    Turning point
    """
    cdef double pfp, pfq, pfr, qfp, qfq, qfr, rfp, rfq, rfr
    cdef double num, div  # Numerator and divisor of extremum point equation E = b/2a
    pfp, pfq, pfr = p*fp, p*fq, p*fr
    qfp, qfq, qfr = q*fp, q*fq, q*fr
    rfp, rfq, rfr = r*fp, r*fq, r*fr
    num = ( -r*rfp + q*qfp - p*pfq + r*rfq - q*qfr + p*pfr )
    div = (qfp - rfp + rfq - pfq + pfr - qfr)
    if div < 0 and -div < TINY:
        div = -TINY
    elif div >= 0 and div < TINY:
        div = TINY
    extremum = num / (2 * div)
    return extremum

cdef void brent(double ax, double bx, double cx, f, double tol, double[:] out):
    """
    brent(double ax, double bx, double cx, f, double tol, double[:] out)

    Given a function f, and given a bracketing triplet of abscissas ax, bx, cx (such that bx is
    between ax and cx, and f(bx) is less than both f(ax) and f(cx)), this routine isolates
    the minimum to a fractional precision of about tol using Brent's method. The out array is
    filled with the x, f(x) and the number of iterations.
    """

    cdef int iter
    cdef double a, b, d, etemp, fu, fv, fw, fx, p, q, r, tol1, tol2, u, v, w, x, xm
    cdef double e = 0.0

    a, b = (ax, cx) if (ax < cx) else (cx, ax)

    x = w = v = bx
    fw = fv = fx = f(x)
    for iter in xrange(1, ITMAX+1): # Main loop
        xm = 0.5*(a + b)
        tol1 = tol*fabs(x) + ZEPS
        tol2 = 2.0*tol1
        if (fabs(x - xm) <= (tol2 - 0.5*(b - a))): # Test for done here.
            out[0] = x
            out[1] = fx
            out[2] = iter
            return

        if (fabs(e) > tol1):  # Construct a trial parabolic fit
            r = (x - w)*(fx - fv)
            q = (x - v)*(fx - fw)
            p = (x - v)*q - (x - w)*r
            q = 2.0*(q - r)
            if (q > 0.0):
                p = -p

            q = fabs(q)
            etemp = e
            e = d
            if fabs(p) >= fabs(0.5*q*etemp) or p <= q*(a - x) or p >= q*(b - x):
                e = (a-x) if (x >= xm) else (b-x)
                d = CGOLD*e
            # The above conditions determine the acceptability of the parabolic fit. Here we
            # take the golden section step into the larger of the two segments.
            else:
                d = p / q  # Take the parabolic step
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if (xm-x) >=0 else -tol1


        else:
            e = (a-x) if (x >= xm) else (b-x)
            d = CGOLD*e

        if fabs(d) >= tol1:
            u = x+d
        else:
            u = x + (tol1 if d >= 0 else -tol1)

        fu = f(u) # This is the one function evaluation per iteration

        if (fu <= fx):   # Now decide what to do with our
            if (u >= x): # function evaluation
                a = x
            else:
                b = x
            v, w, x = w, x, u  # Housekeeping follows:
            fv, fw, fx = fw, fx, fu

        else:
            if (u < x):
                a = u
            else:
                b = u
            if (fu <= fw or w == x):
                v = w
                w = u
                fv = fw
                fw = fu

            elif (fu <= fv or v == x or v == w):
                v = u
                fv = fu
            # Done with housekeeping. Back for
            # another iteration.

    out[0] = x
    out[1] = f(x)
    out[2] = ITMAX+1
    return


cpdef void dbrent(double ax, double bx, double cx, f, df, double tol, double[:] out):
    """
    Given a function f and its derivative function df, and given a bracketing triplet of abscissas ax,
    bx, cx [such that bx is between ax and cx, and f(bx) is less than both f(ax) and f(cx)],
    this routine isolates the minimum to a fractional precision of about tol using a modification of
    Brent's method that uses derivatives. The abscissa of the minimum is returned as xmin, and
    the minimum function value is returned as dbrent, the returned function value.
    """

    assert out.size >= 3
    cdef int iter,ok1,ok2 # Will be used as flags for whether proposed steps are acceptable or not.
    cdef double a, b, d, d1, d2, du, dv, dw, dx, e=0.0
    cdef double fu, fv, fw, fx, olde, tol1, tol2, u, u1, u2, v, w, x, xm

    # Comments following will point out only differences from the routine brent. Read that
    # routine first.
    a, b = (ax, cx) if (ax < cx) else (cx, ax)

    x = w = v = bx  # All our housekeeping chores are doubled
    fw=fv=fx=f(x)   # by the necessity of moving
    dw=dv=dx=df(x)  # derivative values around as well
                    # as function values.

    for iter in range(1, ITMAX):
        xm = 0.5*(a+b)
        tol1 = tol*fabs(x)+ZEPS
        tol2 = 2.0*tol1
        if fabs(x-xm) <= (tol2-0.5*(b-a)):
            out[0] = x
            out[1] = fx
            out[2] = iter
            return

        if fabs(e) > tol1:
            d1=2.0*(b-a) # Initialize these d’s to an out-of-bracket
            d2=d1        # value.

            if dw != dx:
                d1=(w-x)*dx/(dx-dw) # Secant method with one point.
            if dv != dx:
                d2=(v-x)*dx/(dx-dv) # And the other.

            # Which of these two estimates of d shall we take? We will insist that they be within
            # the bracket, and on the side pointed to by the derivative at x:

            u1=x+d1;
            u2=x+d2;

            ok1 = (a-u1)*(u1-b) > 0.0 and dx*d1 <= 0.0;
            ok2 = (a-u2)*(u2-b) > 0.0 and dx*d2 <= 0.0;
            olde=e # Movement on the step before last.
            e=d
            if ok1 or ok2: # Take only an acceptable d, and if
                           # both are acceptable, then take
                           # the smallest one.

                if ok1 and ok2:
                    d = d1 if fabs(d1) < fabs(d2) else d2

                elif ok1:
                    d=d1

                else:
                    d=d2
                if fabs(d) <= fabs(0.5*olde):
                    u=x+d
                    if u-a < tol2 or b-u < tol2:
                        d= tol1 if (xm-x) >= 0 else -tol1

                else: # Bisect, not golden section.
                    e = a-x if dx >= 0 else b-x
                    d=0.5*e
                    # Decide which segment by the sign of the derivative.

            else:
                e = a-x if dx >= 0.0 else b-x
                d=0.5*e

        else:
            e = a-x if dx >= 0.0 else b-x
            d = 0.5*e

        if fabs(d) >= tol1:
            u = x + d
            fu = f(u)
        else:
            u = x + (tol if d >=0 else -tol)
            fu = f(u)
            if fu > fx: # If the minimum step in the downhill
                        # direction takes us uphill, then
                        # we are done.
                out[0] = x
                out[1] = fx
                out[2] = iter
                return

        du=df(u) # Now all the housekeeping, sigh.
        if (fu <= fx):
            if (u >= x):
                a=x
            else:
                b=x;
            v,fv,dv = w,fw,dw
            w,fw,dw = x,fx,dx
            x,fx,dx = u,fu,du
        else:
            if (u < x):
                a=u
            else:
                b=u
            if (fu <= fw or w == x):
                v,fv,dv = w,fw,dw
                w,fw,dw = u,fu,du
            elif (fu < fv or v == x or v == w):
                v,fv,dv = u,fu,du

    out[0] = x
    out[1] = f(x)
    out[2] = ITMAX+1


def dbrent_wrap(double guess, double lbracket, double rbracket, fn, dfn, tol=1.5e-8):
    """
    dbrent_wrap(double guess, double lbracket, double rbracket, fn, dfn, tol=1.5e-8)
    """
    out = np.zeros(3, dtype=np.double)
    dbrent(lbracket, rbracket, guess, fn, dfn, tol, out)
    return out

def brent_wrap(double guess, double lbracket, double rbracket, fn, tol=1.5e-8):
    """
    brent_wrap(double guess, double lbracket, double rbracket, fn, tol=1.5e-8)
    """
    out = np.zeros(3, dtype=np.double)
    brent(lbracket, rbracket, guess, fn, tol, out)
    return out