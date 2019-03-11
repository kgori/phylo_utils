import numpy as np
from phylo_utils.discrete_gamma import discrete_gamma

class RateModel(object):
    pass


class GammaRateModel(RateModel):
    def __init__(self, ncat, alpha=1.0):
        self.ncat = ncat
        self.alpha = float(alpha)
        self.__weights = np.array([1.0 / ncat] * ncat)

    def __repr__(self):
        return "GammaRateModel(ncat={},alpha={})".format(self.ncat, self.alpha)

    def __str__(self):
        return '\n'.join(
            [self.__repr__(),
             "weights={}".format(self.weights),
             "rates={}".format(self.rates)])

    @property
    def weights(self):
        return self.__weights

    @property
    def rates(self):
        return self.__rates

    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha):
        self.__alpha = alpha
        self.__rates = discrete_gamma(alpha, self.ncat)


class UniformRateModel(RateModel):
    def __init__(self):
        self.ncat = 1
        self.__weights = np.array([1.0])
        self.__rates = np.array([1.0])

    def __repr__(self):
        return "UniformRateModel()"

    @property
    def weights(self):
        return self.__weights

    @property
    def rates(self):
        return self.__rates
