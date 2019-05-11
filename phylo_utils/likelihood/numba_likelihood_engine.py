import os
os.environ['NUMBA_ENABLE_AVX'] = '1'
#os.environ['NUMBA_NUM_THREADS'] = '2'
import numba
import numpy as np
from math import log
SCALE_THRESHOLD = np.finfo('float').eps

@numba.guvectorize([('void(double[:,:,:], double[:,:,:],'
                     ' double[:,:], double[:,:],'
                     ' double[:], double[:],'
                     ' double[:], double[:,:])')],
                   ('(n,n,m),(n,n,m),'
                    '(n,m),(n,m),'
                    '(m),(m),'
                    '(m)->(n,m)'), nopython=True)
def clv(probs_a, probs_b,
        clv_a, clv_b,
        scale_a, scale_b,
        scale, out):
    """
    Compute the conditional likelihood vector at the parent of nodes 'a' and 'b'.

    The information at 'a' and 'b' is introduced via parameters probs_a and probs_b,
    which are the Markov transition probabilities from 'a' and 'b' to the parent
    (given the branch lengths), and clv_a and clv_b, which are the conditional likelihood
    vectors at 'a' and 'b'.

    The computation automatically vectorises over broadcast dimension (i.e. multiple sites inputs)
    'out' argument will be automatically allocated if omitted, or can be provided for reuse

    :param probs_a: matrix of transition probabilities for branch from A to parent node, for each rate class
    :param probs_b: matrix of transition probabilities for branch from B to parent node, for each rate class
    :param clv_a: vector of sitewise conditional likelihoods at descendant node A, for each rate class
    :param clv_b: vector of sitewise conditional likelihoods at descendant node B, for each rate class
    :param scale_a: sitewise log scale values at A
    :param scale_b: sitewise log scale values at B
    :param scale: sitewise log scale values computed for current (parent) node
    :param out:
    """
    for k in range(probs_a.shape[2]):
        tmp = np.dot(probs_a[:, :, k], clv_a[:, k]) * np.dot(probs_b[:, :, k], clv_b[:, k])
        m = np.max(tmp)
        if m < SCALE_THRESHOLD and m > 0:
            scale[k] = np.log(m) + scale_a[k] + scale_b[k]
            for j in range(tmp.size):
                out[j, k] = tmp[j] / m
        else:
            scale[k] = scale_a[k] + scale_b[k]
            for j in range(tmp.size):
                out[j, k] = tmp[j]


@numba.guvectorize(['void(double[:,:,:], double[:], double[:], double[:], double[:], double[:], double[:])'],
                   '(m,n,n),(n),(n),(n),(),()->(m)', nopython=True)
def lnl_branch_derivs(probs, pi, partials_a, partials_b, scale_a, scale_b, out):
    f = np.sum(np.dot(probs[0], partials_a) * partials_b * pi)
    fp = np.sum(np.dot(probs[1], partials_a) * partials_b * pi)
    f2p = np.sum(np.dot(probs[2], partials_a) * partials_b * pi)
    out[0] = log(f) + scale_a[0] + scale_b[0]
    out[1] = fp / f
    out[2] = ((f2p * f) - (fp * fp)) / (f * f)


@numba.guvectorize(['void(double[:,:], double[:], double[:], double[:], double[:], double[:], double[:])'],
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
                   '(n),(n,k),(k)->(k)', nopython=True)
def lnl_node(pi, partials, scale, out):
    for k in range(partials.shape[1]):
        f = np.sum(partials[:, k] * pi)
        out[k] = (log(f) + scale[k] if f > 0 else -np.inf)


def make_multithread(inner_func, numthreads):
    """
    Run the given function inside *numthreads* threads, splitting its
    arguments into equal-sized chunks.
    """
    def func_mt(*args):
        length = len(args[0])

        chunklen = (length + numthreads - 1) // numthreads
        # Create argument tuples for each input chunk
        chunks = [[arg[i * chunklen:(i + 1) * chunklen] for arg in args]
                  for i in range(numthreads)]
        # Spawn one thread per chunk
        threads = [threading.Thread(target=inner_func, args=chunk)
                   for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    return func_mt
