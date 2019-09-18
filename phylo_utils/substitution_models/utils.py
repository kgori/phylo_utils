import numpy as np

SMALL = 1/2**128

def compute_b_matrix(q_matrix, sqrtfreqs):
    """
    Computes a matrix (B matrix), similar to Q - i.e. with
    the same eigenvalues.
    B is symmetric if Q is reversible, allowing the use of more
    stable numerical eigen decomposition routines on B than on Q.
    """
    return np.diag(sqrtfreqs).dot(q_matrix).dot(np.diag(1/sqrtfreqs))


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


def expm(matrix):
    """
    Using scaling and squaring technique with Taylor approximation
    (implementation based on RevBayes)
    """
    s = 8
    scale = 1.0 / 2**s
    p = matrix * scale
    p_2 = p.dot(p)
    p_3 = p.dot(p_2)
    p_4 = p.dot(p_3)

    p += np.eye(p.shape[0]) + p_2 / 2.0 + p_3 / 6.0 + p_4 / 24.0
    for _ in range(s):
        p = p.dot(p)
    return p

