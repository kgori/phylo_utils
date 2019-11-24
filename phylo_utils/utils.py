import copy

import numpy as np
import scipy.stats as ss
from scipy.integrate import quad


def setup_logger():
    import logging
    logger = logging.getLogger(__name__)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger


def discretize(alpha, ncat, dist=ss.gamma):
    """
    Return discretized approximation to a probability distribution
    Supported distributions are all strictly positive, fixed to have mean=1:
    Gamma, LogNormal, Exponential (quantitatively the same as Gamma(alpha=1).
    Note that the Exponential distribution has no free parameter, so alpha
    has no effect.
    :param alpha: The free parameter of the distribution
    :param ncat: Number of discrete categories to use
    :param dist: Probability distribution
    :return: Numpy array of ncat values with mean=1
    """
    if dist == ss.gamma:
        dist = dist(alpha, scale=1 / alpha)
    elif dist == ss.lognorm:
        dist = dist(s=alpha, loc=0, scale=np.exp(-0.5 * alpha**2))
    elif dist == ss.expon():
        dist = dist()
    assert dist.mean() == 1, 'Distribution mean is {}'.format(dist.mean())
    quantiles = dist.ppf(np.arange(0, ncat + 1) / ncat)
    rates = np.zeros(ncat, dtype=np.double)
    for i in range(ncat-1):
        rates[i] = ncat * quad(lambda x: x * dist.pdf(x),
                               quantiles[i], quantiles[i+1])[0]
    return rates


def deepcopy_tree(tree):
    clone = copy.deepcopy(tree)
    clone.deroot()
    clone.resolve_polytomies()
    return clone