import numpy as np

def get_q_matrix(rates, freqs):
    # TODO: cythonize
    q = rates.dot(np.diag(freqs))
    q.flat[::len(freqs)+1] -= q.sum(1)
    q /= (-(np.diag(q)*freqs).sum())
    return q

def get_b_matrix(q_matrix, sqrtfreqs):
    # TODO: cythonize
    return np.diag(sqrtfreqs).dot(q_matrix).dot(np.diag(1/sqrtfreqs))

def get_eigen(q_matrix, freqs=None):
    if freqs is not None:
        rootf = np.sqrt(freqs)
        mtx = get_b_matrix(q_matrix, rootf)
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


class Eigen(object):
    def __init__(self, qmatrix, freqs=None):
        evecs, evals, ivecs = get_eigen(qmatrix, freqs)
        self.evecs = evecs
        self.evals = evals
        self.ivecs = ivecs

    @property
    def values(self):
        return self.evecs, self.evals, self.ivecs


class TransitionMatrix(object):
    def __init__(self, model):
        self.model = model
        self.q_mtx = get_q_matrix(model.rates, model.freqs)
        self.freqs = model.freqs
        self.eigen = Eigen(self.q_mtx, model.freqs)
        self.size = len(model.freqs)

    def get_q_matrix(self):
        return self.q_mtx

    def get_p_matrix(self, t):
        """
        P = transition probabilities
        """
        # if t < 1e-8: return np.eye(self.eigen.evals.shape[0])
        evecs, evals, ivecs = self.eigen.values
        return (evecs*np.exp(evals*t)).dot(ivecs)

    def get_dp_matrix(self, t):
        """
        First derivative of P
        """
        evecs, evals, ivecs = self.eigen.values
        return (evecs*evals*np.exp(evals*t)).dot(ivecs)

    def get_d2p_matrix(self, t):
        """
        Second derivative of P
        """
        evecs, evals, ivecs = self.eigen.values
        return (evecs*evals*evals*np.exp(evals*t)).dot(ivecs)
