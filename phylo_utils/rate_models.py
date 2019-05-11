import numpy as np
from phylo_utils.discrete_gamma import discrete_gamma

class RateModel():

    @property
    def weights(self):
        return self._weights

    @property
    def rates(self):
        return self._rates


class GammaRateModel(RateModel):
    def __init__(self, ncat, alpha=1.0):
        self.ncat = ncat
        self.alpha = float(alpha)
        self._weights = np.array([1.0 / ncat] * ncat)

    def __repr__(self):
        return "GammaRateModel(ncat={},alpha={})".format(self.ncat, self.alpha)

    def __str__(self):
        return '\n'.join(
            [self.__repr__(),
             "weights={}".format(self.weights),
             "rates={}".format(self.rates)])

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        self._rates = discrete_gamma(alpha, self.ncat)


class UniformRateModel(RateModel):
    def __init__(self):
        self.ncat = 1
        self._weights = np.array([1.0])
        self._rates = np.array([1.0])

    def __repr__(self):
        return "UniformRateModel()"


class InvariantSitesModel(RateModel):
    def __init__(self, pinvar):
        self.pinvar = pinvar
        self.ncat = 2

    def __repr__(self):
        return "InvariantSitesModel(pinvar={})".format(self.pinvar)

    def __str__(self):
        return '\n'.join(
            [self.__repr__(),
             "weights={}".format(self.weights),
             "rates={}".format(self.rates)])

    @property
    def pinvar(self):
        return self._pinvar

    @pinvar.setter
    def pinvar(self, pinvar):
        if not 0 <= pinvar < 1:
            raise ValueError("pinvar must be in the range [0, 1)")
        self._pinvar = pinvar
        self._weights = np.array([pinvar, 1 - pinvar])
        self._rates = np.array([0, 1 / (1 - pinvar)])


class InvariantGammaModel(RateModel):
    def __init__(self, pinvar, n_gamma_cat, alpha=1.0):
        if not 0 <= pinvar < 1:
            raise ValueError("pinvar must be in the range [0, 1)")
        if not 0.001 <= alpha:
            raise ValueError("alpha must be greater than 0.001")
        self.ncat = n_gamma_cat + 1
        self._pinvar = pinvar
        self._alpha = float(alpha)
        rates, weights = self._compute_rates_and_weights(pinvar, n_gamma_cat, alpha)
        self._rates = rates
        self._weights = weights

    def _compute_rates_and_weights(self, pinvar, ncat, alpha):
        gamma_rates = discrete_gamma(alpha, ncat)
        gamma_weights = np.ones(ncat) / ncat
        rates = np.hstack([0, gamma_rates / (1 - pinvar)])
        weights = np.hstack([pinvar, gamma_weights * (1 - pinvar)])
        return rates, weights

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if not 0.001 <= alpha:
            raise ValueError("alpha must be greater than 0.001")
        self._alpha = alpha
        rates, weights = self._compute_rates_and_weights(self.pinvar, self.ncat-1, alpha)
        self._rates = rates
        self._weights = weights

    @property
    def pinvar(self):
        return self._pinvar

    @ pinvar.setter
    def pinvar(self, pinvar):
        if not 0 <= pinvar < 1:
            raise ValueError("pinvar must be in the range [0, 1)")
        self._pinvar = pinvar
        rates, weights = self._compute_rates_and_weights(pinvar, self.ncat - 1, self.alpha)
        self._rates = rates
        self._weights = weights
