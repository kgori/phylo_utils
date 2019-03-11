from scipy.special import gammaincinv, gammainc
import numpy as np

def discrete_gamma(ncat, alpha, beta=None):
    """
    Compute discretised gamma distribution (pure python)
    :param ncat: (int) number of categories
    :param alpha: (float) alpha parameter of gamma distribution
    (beta is fixed so that the mean is 1)
    :return: numpy array of discrete rates (length=ncat)
    """
    if beta is None:
        beta = alpha
    mean = alpha / beta
    cutpoints = np.arange(0, ncat + 1, dtype=np.double) / ncat
    quantiles = gammaincinv(alpha, cutpoints) # don't divide by beta, saves multiplying in next step
    cdfs = gammainc(alpha + 1, quantiles)
    return ncat * mean * np.diff(cdfs)
