import numpy as np
import likcalc
from markov import TransitionMatrix
import dendropy as dpy

def setup_logger():
    import logging
    logger = logging.getLogger(__name__)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    ch=logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()


class Leaf(object):
    """ Object to store partials at a leaf """
    def __init__(self, partials):
        self.set_partials(partials)

    def set_partials(self, partials):
        """ Set the partials at this node """
        self.partials = np.ascontiguousarray(partials, dtype=np.double)


class LnlModel(object):
    """
    Attaches to a node. Calculates and stores partials (conditional likelihood vectors),
    and transition probabilities.
    """
    def __init__(self, transmat):
        self.transmat = transmat
        self.partials = None
        self.sitewise = None
        self.scale_buffer = None

    def update_transition_probabilities(self, len1, len2):
        self.probs1 = self.transmat.get_p_matrix(len1)
        self.probs2 = self.transmat.get_p_matrix(len2)

    def set_partials(self, partials):
        """ Set the partials at this node """
        self.partials = np.ascontiguousarray(partials)

    def compute_partials(self, lnlmodel1, lnlmodel2, scale=False):
        """ Update partials at this node """
        if scale:
            self.partials, self.scale_buffer = likcalc.likvec_2desc_scaled(self.probs1, self.probs2, lnlmodel1.partials, lnlmodel2.partials)
        else:
            self.partials = likcalc.likvec_2desc(self.probs1, self.probs2, lnlmodel1.partials, lnlmodel2.partials)

    def compute_edge_sitewise_likelihood(self, lnlmodel, brlen, derivatives=False):
        """ Calculate the likelihood with this node at root - 
        returns array of [f, f', f''] values, where fs are unscaled unlogged likelihoods, and
        f' and f'' are unconverted partial derivatives.
        Logging, scaling and conversion are done in compute_likelihood """
        probs = self.transmat.get_p_matrix(brlen)

        if derivatives:
            dprobs = self.transmat.get_dp_matrix(brlen)
            d2probs = self.transmat.get_d2p_matrix(brlen)
            self.sitewise = likcalc.sitewise_lik_derivs(probs, dprobs, d2probs, self.transmat.freqs, self.partials, lnlmodel.partials)
        else:
            self.sitewise = likcalc.sitewise_lik(probs, self.transmat.freqs, self.partials, lnlmodel.partials)

    def compute_likelihood(self, lnlmodel, brlen, derivatives=False, accumulated_scale_buffer=None):
        self.compute_edge_sitewise_likelihood(lnlmodel, brlen, derivatives)
        f = self.sitewise[:, 0]
        if accumulated_scale_buffer is not None:
            lnl = (np.log(f) + accumulated_scale_buffer).sum()
        else:
            lnl = np.log(f).sum()
        if derivatives:
            fp = self.sitewise[:, 1]
            f2p = self.sitewise[:, 2]
            dlnl = (fp/f).sum()
            d2lnl = (((f*f2p)-(fp*fp))/(f*f)).sum()
            return lnl, dlnl, d2lnl
        else:
            return lnl


class RunOnTree(object):
    def __init__(self, transition_matrix, partials_dict, scale_freq=20):
        # Initialise leaves
        self.leaf_models = {}
        for (leafname, partials) in partials_dict.items():
            model = LnlModel(transition_matrix)
            model.set_partials(partials)
            self.leaf_models[leafname] = model

        self.nsites = partials_dict.values()[0].shape[0]
        self.tm = transition_matrix
        self.internal_node_counter = 0
        self.accumulated_scale_buffer = None
        self.scale_freq = scale_freq

    def set_tree(self, tree):
        #self.tree = dpy.Tree.get_from_string(tree, 'newick', preserve_underscores=True)
        #self.tree.resolve_polytomies() # Require strictly binary tree, including root node
        self.tree = tree
        for leaf in self.tree.leaf_nodes():
            leaf.model = self.leaf_models[leaf.taxon.label]

    def update_transition_matrix(self, tm):
        self.tm = tm
        for leaf in self.tree.leaf_nodes():
            leaf.model.transmat = tm

    def run(self, derivatives=False):
        self.internal_node_counter = 0
        self.accumulated_scale_buffer = np.zeros(self.nsites)
        for node in self.tree.postorder_internal_node_iter():
            self.internal_node_counter += 1
            children = node.child_nodes()
            node.model = LnlModel(self.tm)
            l1, l2 = [ch.edge.length for ch in children]
            node.model.update_transition_probabilities(l1,l2)
            model1, model2 = [ch.model for ch in node.child_nodes()]
            if self.internal_node_counter % self.scale_freq == 0 and not node == self.tree.seed_node:
                node.model.compute_partials(model1, model2, True)
                self.accumulated_scale_buffer += node.model.scale_buffer
            else:
                node.model.compute_partials(model1, model2, False)
        ch1, ch2 = self.tree.seed_node.child_nodes()[:2]
        return ch1.model.compute_likelihood(ch2.model, ch1.edge.length + ch2.edge_length, derivatives, self.accumulated_scale_buffer)

    def get_sitewise_likelihoods(self):
        ch = self.tree.seed_node.child_nodes()[0]
        return np.log(ch.model.sitewise[:, 0]) + self.accumulated_scale_buffer


class Mixture(object):
    def __init__(self):
        pass

    def mix_likelihoods(self, sw_lnls):
        ma = sw_lnls.max(1)[:,np.newaxis]
        wa = sw_lnls + self.logweights
        return np.log(np.exp(wa-ma).sum(1))[:,np.newaxis] + ma

    def mix_likelihoods2(self, sw_lnls):
        mb = sw_lnls.max(1)[:,np.newaxis]
        vb = np.exp(sw_lnls - mb)
        cb = (self.weights * vb)
        return np.log(cb.sum(1))[:, np.newaxis] + mb


class GammaMixture(Mixture):
    def __init__(self, alpha, ncat):
        self.ncat = ncat
        self.rates = likcalc.discrete_gamma(alpha, ncat)
        self.weights = np.array([1.0/ncat] * ncat)
        self.logweights = np.log(self.weights)

    def update_alpha(self, alpha):
        self.rates = likcalc.discrete_gamma(alpha, self.ncat)
        self.set_tree(self.tree)

    def update_transition_matrix(self, tm):
        for runner in self.runners:
            runner.update_transition_matrix(tm)

    def init_models(self, tm, partials_dict, scale_freq=20):
        self.runners = []
        for cat in xrange(self.ncat):
            runner = RunOnTree(tm, partials_dict, scale_freq)
            self.runners.append(runner)

    def set_tree(self, tree):
        self.tree = tree
        for cat in xrange(self.ncat):
            t = dpy.Tree.get_from_string(tree, 'newick', preserve_underscores=True)
            t.resolve_polytomies()
            t.scale_edges(self.rates[cat])
            self.runners[cat].set_tree(t)

    def run(self):
        for runner in self.runners:
            runner.run()

    def get_sitewise_likelihoods(self):
        swlnls = np.empty((self.runners[0].nsites, self.ncat))
        for cat in xrange(self.ncat):
            swlnls[:,cat] = self.runners[cat].get_sitewise_likelihoods()
        return swlnls

    def get_likelihood(self):
        self.run()
        sw_lnls_per_class = self.get_sitewise_likelihoods()
        sw_lnls = self.mix_likelihoods(sw_lnls_per_class)
        return sw_lnls.sum()  


class OptWrapper(object):
    """
    Wrapper for use with scipy optimiser (e.g. brenth/brentq)
    """
    def __init__(self, tm, partials1, partials2, initial_brlen=1.0):
        self.root = LnlModel(tm)
        self.leaf = Leaf(partials2)
        self.root.set_partials(partials1)
        self.updated = None
        self.update(initial_brlen)

    def update(self, brlen):
        if self.updated == brlen:
            return
        else:
            self.updated = brlen
            self.lnl, self.dlnl, self.d2lnl = self.root.compute_likelihood(self.leaf, brlen, derivatives=True)

    def get_dlnl(self, brlen):
        self.update(brlen)
        return self.dlnl

    def get_d2lnl(self, brlen):
        self.update(brlen)
        return self.d2lnl

    def __str__(self):
        return 'Branch length={}, Variance={}, Likelihood+derivatives = {} {} {}'.format(self.updated, -1/self.d2lnl, self.lnl, self.dlnl, self.d2lnl)


def optimise(likelihood, partials_a, partials_b, min_brlen=0.00001, max_brlen=10, verbose=True):
    """
    Optimise ML distance between two partials. min and max set brackets
    """
    from scipy.optimize import brenth
    wrapper = OptWrapper(likelihood, partials_a, partials_b, (min_brlen+max_brlen)/2.)
    brlen = 0.5
    n=brenth(wrapper.get_dlnl, min_brlen, max_brlen)
    if verbose:
        logger.info(wrapper)
    return n, -1/wrapper.get_d2lnl(n)
