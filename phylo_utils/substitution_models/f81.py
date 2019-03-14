from phylo_utils.substitution_models.tn93 import TN93


class F81(TN93):
    _name = 'F81'
    def __init__(self, freqs, scale_q=True):
        super(F81, self).__init__(1, 1, 1, freqs, scale_q=scale_q)
