from phylo_utils.substitution_models.tn93 import TN93


class F84(TN93):
    _name = 'F84'
    def __init__(self, kappa, freqs, scale_q=True):
        alpha_y = 1+kappa/(freqs[1]+freqs[3])
        alpha_r = 1+kappa/(freqs[0]+freqs[2])
        super(F84, self).__init__(alpha_y, alpha_r, 1, freqs, scale_q=scale_q)
