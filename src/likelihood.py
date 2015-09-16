import numpy as np
from .extensions import utils
from .markov import TransitionMatrix

def setup_logger():
    import logging
    logger = logging.getLogger()
    for handler in logger.handlers:
        logger.removeHandler(handler)
    ch=logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()


class PairLikelihood(object):
    """
    See Yang, (2000) "Maximum Likelihood Estimation on Large Phylogenies and Analysis of
    Adaptive Evolution in Human Influenza Virus A", J. Mol. Evol.
    """
    def __init__(self, transmat, edgelen):
        self.transmat = transmat
        self.update_transmat(edgelen)

    def update_transmat(self, edgelen):
        """ Update transition probabilities for new branch lengths """
        self.p = self.transmat.get_p_matrix(edgelen)
        self.dp = self.transmat.get_dp_matrix(edgelen)
        self.d2p = self.transmat.get_d2p_matrix(edgelen)

    def _f(self, partial_a, partial_b, freqs):
        return (freqs*self.p*partial_a*partial_b[:,np.newaxis]).sum()

    def _df(self, partial_a, partial_b, freqs):
        return (freqs*self.dp*partial_a*partial_b[:,np.newaxis]).sum()

    def _d2f(self, partial_a, partial_b, freqs):
        return (freqs*self.d2p*partial_a*partial_b[:,np.newaxis]).sum()

    def calculate(self, partials_a, partials_b, freqs):
        lnl = 0
        dlnl = 0
        d2lnl = 0
        for sa, sb in zip(sites_a, sites_b):
            f = self._f(sa, sb, freqs)
            df = self._df(sa, sb, freqs)
            d2f = self._d2f(sa, sb, freqs)
            lnl += np.log(f)
            dlnl += df / f
            d2lnl += (f * d2f - df * df) / (f * f)
        return lnl, dlnl, d2lnl


class Lnl(object):
    """
    Calculate and store the transition probabilities
    """
    def __init__(self, transmat):
        self.transmat = transmat
        self.partials = None

    def update_transition_probabilities(self, len1, len2):
        self.probs1 = self.transmat.get_p_matrix(len1)
        self.probs2 = self.transmat.get_p_matrix(len2)

    def compute_partials(self, partials1, partials2, len1, len2):
        """ Update partials at this node """
        self.partials = utils.likvec_2desc(self.probs1, self.probs2, partials1, partials2)

    def compute_root_likelihood(self, partials, brlen, derivatives=False):
        """ Calculate the likelihood with this node at root """
        probs = self.transmat.get_p_matrix(brlen)


class Likelihood(object):
    """
    See Yang, (2000) "Maximum Likelihood Estimation on Large Phylogenies and Analysis of
    Adaptive Evolution in Human Influenza Virus A", J. Mol. Evol.
    """
    def __init__(self, transmat, edgelen_left, edgelen_right=0):
        """ Initialise object with TransitionMatrix and a branch lengths.
        """
        self.transmat = transmat
        self.update_transmat(edgelen_left, edgelen_right)
        self.size = transmat.size

    def update_transmat(self, edgelen_left, edgelen_right=0):
        """ Update transition probabilities for new branch lengths """
        self.p_left = self.transmat.get_p_matrix(edgelen_left)
        self.p_right = self.transmat.get_p_matrix(edgelen_right)
        self.dp = self.transmat.get_dp_matrix(edgelen_left+edgelen_right)
        self.d2p = self.transmat.get_d2p_matrix(edgelen_left+edgelen_right)

    def _likvec(self, partial_a, partial_b):
        """ Calculate the likelihood vector for a site """
        return (self.p_left*partial_a).sum(1) * (self.p_right*partial_b).sum(1)

    def _f(self, partial_a, partial_b, freqs):
        """ Calculate the root likelihood for a site """
        vec = self._likvec(partial_a, partial_b)
        return (vec*freqs).sum()

    def _df(self, partial_a, partial_b, freqs):
        """ Calculate the root first derivative of the likelihood for a site """
        return (freqs*self.dp*partial_a*partial_b[:,np.newaxis]).sum()

    def _d2f(self, partial_a, partial_b, freqs):
        """ Calculate the root second derivative of the likelihood for a site """
        return (freqs*self.d2p*partial_a*partial_b[:,np.newaxis]).sum()

    def _calculate_no_derivatives(self, sites_a, sites_b, pi):
        lnl = 0
        for sa, sb in zip(sites_a, sites_b):
            lnl += self._f(sa, sb, pi)
        return lnl

    def calculate(self, sites_a, sites_b, pi, derivatives=False):
        """ Calculate log likelihood and first and second derivatives over all sites.
        Sites need to be given as partials (i.e. conditional probability vectors) """
        if derivatives:
            return self._calculate_with_derivatives(sites_a, sites_b, pi)
        else:
            return self._calculate_no_derivatives(sites_a, sites_b, pi)

    def _calculate_with_derivatives(self, sites_a, sites_b, pi):
        lnl = 0
        dlnl = 0
        d2lnl = 0
        for sa, sb in zip(sites_a, sites_b):
            f = self._f(sa, sb, pi)
            df = self._df(sa, sb, pi)
            d2f = self._d2f(sa, sb, pi)
            lnl += np.log(f)
            dlnl += df / f
            d2lnl += (f * d2f - df * df) / (f * f)
        return lnl, dlnl, d2lnl


class OptWrapper(object):
    """
    Wrapper for use with scipy optimiser (e.g. brenth/brentq)
    """
    def __init__(self, likelihood, sites_a, sites_b, freqs):
        self.lik = likelihood
        self.sites_a = sites_a
        self.sites_b = sites_b
        self.freqs = freqs
        self.updated = None

    def update(self, brlen):
        if self.updated == brlen:
            return
        else:
            self.updated = brlen
            self.lik.update_transmat(brlen)
            self.lnl, self.dlnl, self.d2lnl = self.lik.calculate(self.sites_a, self.sites_b, self.freqs)

    def get_dlnl(self, brlen):
        self.update(brlen)
        return self.dlnl

    def get_d2lnl(self, brlen):
        self.update(brlen)
        return self.d2lnl

    def __str__(self):
        return 'Branch length={}, Variance={}, Likelihood+derivatives = {} {} {}'.format(self.updated, -1/self.d2lnl, self.lnl, self.dlnl, self.d2lnl)


def optimise(likelihood, partials_a, partials_b, frequencies, min_brlen=0.00001, max_brlen=10, verbose=True):
    """
    Optimise ML distance between two partials. min and max set brackets
    likelihood = Likelihood or PairLikelihood object. PairLikelihood is slightly faster.
    """
    from scipy.optimize import brenth
    wrapper = OptWrapper(likelihood, partials_a, partials_b, frequencies)
    brlen = 0.5
    n=brenth(wrapper.get_dlnl, min_brlen, max_brlen)
    if verbose:
        print(wrapper)
    return n, -1/wrapper.get_d2lnl(n)

if __name__ == '__main__':
    kappa = 1
    k80 = np.array([[0,kappa,1,1],[kappa,0,1,1],[1,1,0,kappa],[1,1,kappa,0]], dtype=np.float)
    k80f = np.array([0.25,0.25,0.25,0.25])
    ed = EigenDecomp(get_q_matrix(k80, k80f), k80f)
    tm = TransitionMatrix(ed)
    lk = PairLikelihood(tm, 0)

    # Simulated data from K80, kappa=1, distance = 0.8
    sites_a = seq_to_partials('ACCCTCCGCGTTGGGTAGTCCTAGGCCCAATGGCGTTTATGCCTCGATTTTTAGTTCTACCGTCCCTACAGATGGATGCCGTCGCATAGACACTGTCAATTCCATTCGGCAGGCTTCACACTGTTGCATTTTCATTTTGTACACGGTACCAACATAGGAGTGCTGTATTGCTATATTTCCAGTACACGGCGTTGAGTCGGATGGAAACGCCGGCGGAAGACAGCTTGGCGGGTCTTCACGCATCACCGCGGGGTCTGAAAGGTATTATCGCTGCTTAAATCAGACCGGTCAAGCTTCCTGGCGGAAGGCGGCAAGGTCCAGCCACAGCATGCTTATTCCTTGTCACGCCGGGTGGAAATCTAGAGCGTCCGGTGGACACAGAGTGATTTTGTACGGGGGGTTCCATACCAGGACATTAGGGTCGGTTTACGGTCTGAGATGTATGTTGCCTTGCGGTCGACGAGCACTGATTCCCCTGAACTTCGTAAGACACATATAGTTTTAATGAAATCCCCAAAACGAGCATGGTTTCAGTATACGCGACAACTTAGGATACAACATACTGAACCAGTCCGCATTGAGGTGCCAATCAAACGGGACCGGGACTGATAAGTATAAAATAGGTTTCCCTGTCCTCTACCTACGTTATCCTCGCGTCGATTTTGATTCTTACCAAGACTGCTAATCAGGCCCTGTGGCCTGCATGTCACCATGTCAGCGTGTTTGGCTAAATTCACGGGATTGGCCTTACCGACTTACATCAGTATTTCATACATAGTTACTCGAGTTTAACGTTGACAGTTAGTCCCATGATACGGCAAAGCCTGGTTCGGCGGATTTCCGAGTACAGCATCTTCGCCCCCGAGATTGCCGCCAATGGACACCCTCCTGAGATGCAGATATGAGTGTTTTTGACACTCTGAGGCTGAGATCCTCACACTTCCGGAGCTTCCGCGATAGTCACGTGGTTATTAGACTTACGGCAGGAAAAATCATGTTA', alphabet='dna')
    sites_b = seq_to_partials('AAGCTCCGCGTAAGCTAACGACCAGTCAGCTAGGTTTAGTGCCACCAGTATGGCTAGTTCCGGAGGGCAAACCGGATGCTACCGATTGGTCACCCTCAGGGTGATTTCGCAGGGCGCTCACTTATTCCTTTTAAATCCTGCCAACAGACTAAGAAAGTTGTACGGTATTCCTATATCTTCAGTACTGCTCTTGGCCGTGCATGTAGCCGAACGACGAGGACGGTACATGAGTTTCTCACCAATTACAGGCGGTTCCATTAGGCAGTAGCTGCGGTTAGTTCATACTGCTAAAGAATCTTCTTGGAACGTGCCAAGGACCAGTCACACACATGTTGTAGTCCCTCATCGTGGTAGGCGTTCCAGACCGTCCGTGGTACACATACCAAATTTCGTACCGGCTGACTCAAAGCGGGAGTTCGCATGATACCAGGGAACGAGATGTTCAAAACGATCAGGTAGTGCCGCCATCTTTCAGGTTCTTTCGTTTCGTCCTATGATACTTGAGTAGCGGTCAAACGAAGCTCGTAGGTGACAGTTACGAGACATGCTGGGATGCAACATACTTTCGCAGTTAGCTAGTAGGTACCTATCTAGCGAATCGAGCTAGGATACCCTGATTATGCTTGTCTCCGTCCTCTTACTATGATCTCCTCGCGTGGTTTTTGCTGCTTAACCGTTGTGCCGTATAAAACAAGAGGCGGGAGTTTAGCTGTGGGAACTTCGTAGACCTTGTAAGCTGGATAGGCCCGTCCGTCGTAATTAATTACCTAAAAGAGAGTCAAACAAGCTTAAGTCGCCGAGTTAGTCGGATAAGAAGCCATTCTCTGGTCCGCCAACCTTCCCATGCCAGTACGGTTGCCGAGGTCCATTCGGTGACTGTGGGATAACCGTTGCCGGAGCTATGAGATCCATTACAACTCTGCGCCTAGGATGTTAACTCTACCGAAGTTTGCGACCCCGGAACCTGTAAATTGTCCTTAGGGTCGTAACATTTTCAAGC', alphabet='dna')

    optimise(lk, sites_a, sites_b, k80f)

    # Example from Section 4.2 of Ziheng's book - his value for node 6 is wrong!
np.set_printoptions(precision=6)
kappa = 2
k80 = np.array([[0,kappa,1,1],[kappa,0,1,1],[1,1,0,kappa],[1,1,kappa,0]], dtype=np.float)
k80f = np.array([0.25,0.25,0.25,0.25])
ed = EigenDecomp(get_q_matrix(k80, k80f), k80f)
tm = TransitionMatrix(ed)

partials_1 = np.array([1, 0, 0, 0], dtype=np.float)
partials_2 = np.array([0, 1, 0, 0], dtype=np.float)
partials_3 = np.array([0, 0, 1, 0], dtype=np.float)
partials_4 = np.array([0, 1, 0, 0], dtype=np.float)
partials_5 = np.array([0, 1, 0, 0], dtype=np.float)

lik_inner = Likelihood(tm, 0.1, 0.1)
lik_outer = Likelihood(tm, 0.2, 0.2)
lik_inner_outer = Likelihood(tm, 0.1, 0.2)

partials_7 = lik_outer._likvec(partials_1, partials_2)
# scale all the vectors!

#log_scale_factor = 0
#scale_factor = partials_7.max()
#partials_7 /= scale_factor
logger.info('partials_7: {}'.format(partials_7))
#logger.info('Scale factor 7 = {}, log scale_factor 7 = {}'.format(scale_factor, np.log(scale_factor)))
#log_scale_factor += np.log(scale_factor)
# logger.info('Cumulative scale factor = {}'.format(log_scale_factor))

partials_6 = lik_inner_outer._likvec(partials_7, partials_3)

# scale_factor = partials_6.max()
# partials_6 /= scale_factor
logger.info('partials_6: {}'.format(partials_6))
# logger.info('Scale factor 6 = {}, log scale_factor 6 = {}'.format(scale_factor, np.log(scale_factor)))
# log_scale_factor += np.log(scale_factor)
# logger.info('Cumulative scale factor = {}'.format(log_scale_factor))

partials_8 = lik_outer._likvec(partials_4, partials_5)
logger.info('partials_8: {}'.format(partials_6))
# scale_factor = partials_8.max()
# partials_8 /= scale_factor
# logger.info('Scale factor 8 = {}, log scale_factor 8 = {}'.format(scale_factor, np.log(scale_factor)))
# log_scale_factor += np.log(scale_factor)
# logger.info('Cumulative scale factor = {}'.format(log_scale_factor))

partials_0 = lik_inner._likvec(partials_6, partials_8)


base_likelihood = np.log((k80f*partials_0).sum())
logger.info('Likelihood = {}'.format(base_likelihood))
# base_likelihood += log_scale_factor
# logger.info('rescaled likelihood = {}'.format(base_likelihood))
