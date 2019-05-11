import numpy as np
SCALE_THRESHOLD = np.finfo('float').eps

def clv(probs_a, probs_b,
        clv_a, clv_b,
        scale_a, scale_b,
        scale, out=None):
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
    return_out = False
    if out is None:
        out = np.zeros_like(clv_a)
        return_out = True
    for site in range(clv_a.shape[0]):
        np.einsum('ijz,ikz,jz,kz->iz', probs_a, probs_b, clv_a[site], clv_b[site], out=out[site])
        m = np.max(out[site], 0)
        m[m >= SCALE_THRESHOLD] = 1
        scale[site, :] = np.log(m) + scale_a[site] + scale_b[site]
        out[site] /= m
    if return_out: return out

def lnl_branch_derivs(probs, pi, clv_a, clv_b, scale_a, scale_b, out):
    f = np.sum(np.dot(probs[0], clv_a) * clv_b * pi)
    fp = np.sum(np.dot(probs[1], clv_a) * clv_b * pi)
    f2p = np.sum(np.dot(probs[2], clv_a) * clv_b * pi)
    out[0] = np.log(f) + scale_a[0] + scale_b[0]
    out[1] = fp / f
    out[2] = ((f2p * f) - (fp * fp)) / (f * f)

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

def lnl_node(pi, partials, scale, out=None):
    return_out = False
    if out is None:
        out = np.zeros_like(scale)
        return_out = True
    for site in range(partials.shape[0]):
        for k in range(partials.shape[2]):
            f = np.sum(partials[site, :, k] * pi)
            out[site, k] = np.log(f) + scale[site, k]
    if return_out: return out
