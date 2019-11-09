import os
os.environ['NUMBA_ENABLE_AVX'] = '1'
#os.environ['NUMBA_NUM_THREADS'] = '2'
import numba
import numpy as np
from math import log
SCALE_THRESHOLD = 1.0 / 2.0**128

# numba vectorized version, should broadcast automatically over N datapoints
@numba.guvectorize([
    ('void(double[:,:,::1], double[:,:,::1], double[:,::1], double[:,::1], double[::1], double[::1], double[::1], double[:,::1])')],
    '(ncat,nstate,nstate), (ncat,nstate,nstate), (ncat,nstate), (ncat,nstate), (ncat), (ncat), (ncat) -> (ncat,nstate)',
    nopython=True, forceobj=False, target='parallel') 
def clv(p1, p2, clv1, clv2, scaler_a, scaler_b, cml_scaler, out):
    """
        Compute the conditional likelihood vector at the parent of nodes 'a' and 'b'.

        The information at 'a' and 'b' is introduced via parameters probs_a and probs_b,
        which are the Markov transition probabilities from 'a' and 'b' to the parent
        (given the branch lengths), and clv_a and clv_b, which are the conditional likelihood
        vectors at 'a' and 'b'.

        The computation automatically vectorises over broadcast dimension (i.e. multiple sites inputs)
        'out' argument will be automatically allocated if omitted, or can be provided for reuse

        :param p1: matrix of transition probabilities for branch from A to parent node, for each rate class. Must be c-contiguous and shaped as (ncat, nstates, nstates).
        :param p2: matrix of transition probabilities for branch from B to parent node, for each rate class. Must be c-contiguous and shaped as (ncat, nstates, nstates).
        :param clv1: vector of sitewise conditional likelihoods at descendant node A, for each rate class. Must be c-contiguous and shaped as ([nsites], ncat, nstates).
        :param clv2: vector of sitewise conditional likelihoods at descendant node B, for each rate class. Must be c-contiguous and shaped as ([nsites], ncat, nstates).
        :param scaler_a: sitewise log scale values at A. Must be c-contiguous and shaped as (ncat).
        :param scaler_b: sitewise log scale values at B. Must be c-contiguous and shaped as (ncat).
        :param cml_scaler: sitewise log scale values computed for current (parent) node. Must be c-contiguous and shaped as (ncat).
        :param out: destination array for computed values. Must be c-contiguous and same shape as clv1 and clv2.
        """
    for cat in range(p1.shape[0]):
        out[cat] = np.dot(p1[cat], clv1[cat]) * np.dot(p2[cat], clv2[cat])
        m = np.max(out[cat])

        if m < SCALE_THRESHOLD and m > 0:
            cml_scaler[cat] = scaler_a[cat] + scaler_b[cat] + np.log(m)
            out[cat] /= m

        else:
            cml_scaler[cat] = scaler_a[cat] + scaler_b[cat]

    return


@numba.guvectorize(['void(double[:,:,::1], double[::1], double[::1], double[::1], double[::1], double[::1], double[::1])'],
                   '(m,n,n),(n),(n),(n),(),()->(m)', nopython=True)
def lnl_branch_derivs(probs, pi, partials_a, partials_b, scale_a, scale_b, out):
    f = np.sum(np.dot(probs[0], partials_a) * partials_b * pi)
    fp = np.sum(np.dot(probs[1], partials_a) * partials_b * pi)
    f2p = np.sum(np.dot(probs[2], partials_a) * partials_b * pi)
    out[0] = log(f) + scale_a[0] + scale_b[0]
    out[1] = fp / f
    out[2] = ((f2p * f) - (fp * fp)) / (f * f)


@numba.guvectorize(['void(double[:,::1], double[::1], double[::1], double[::1], double[::1], double[::1], double[::1])'],
                   '(n,n),(n),(n),(n),(),()->()', nopython=True)
def lnl_branch(probs, pi, partials_a, partials_b, scale_a, scale_b, out):
    """
        Compute log-likelihood across a branch between A and B. CLVs at either end of the branch
        have already been computed and are passed as `clv_a` and `clv_b`.
        Any scaling values existing at A or B are passed as `scale_a` and `scale_b`

        :param probs: tensor of probabilities and first and second derivatives, w.r.t branch length between A and B
        :param pi: vector of equilibrium base frequencies
        :param clv_a: sitewise conditional likelihoods at A
        :param clv_b: sitewise conditional likelihoods at B
        :param scale_a: sitewise log scale value at A
        :param scale_b: sitewise log scale values at B
        :param out: Will be filled in as a sitewise vector of log-likelihood and first and second derivatives.
        Created if not passed.
        :return: out unless out was supplied
        """
    f = np.sum(np.dot(probs, partials_a) * partials_b * pi)
    out[0] = log(f) + scale_a[0] + scale_b[0]


@numba.guvectorize(['void(double[:], double[:,:], double[:], double[:])'],
                   '(nstate),(ncat,nstate),(ncat)->(ncat)', nopython=True)
def lnl_node(pi, partials, scale, out):
    for cat in range(partials.shape[0]):
        f = np.sum(partials[cat] * pi)
        out[cat] = (log(f) + scale[cat] if f > 0 else -np.inf)
