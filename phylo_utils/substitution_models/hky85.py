from phylo_utils.substitution_models.tn93 import TN93


class HKY85(TN93):
    _name = 'HKY85'
    def __init__(self, kappa, freqs, scale_q=True):
        super(HKY85, self).__init__(kappa, kappa, 1, freqs, scale_q=scale_q)
