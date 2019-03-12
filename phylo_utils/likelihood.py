import numpy as np
# from . import likcalc
import dendropy as dpy
from scipy.optimize import minimize


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


logger = setup_logger()


class Leaf(object):
    """ Object to store partials at a leaf """

    def __init__(self, partials):
        self.set_partials(partials)

    def set_partials(self, partials):
        """ Set the partials at this node """
        self.partials = np.ascontiguousarray(partials, dtype=np.double)


class LnlNode(object):
    """
    Attaches to a node. Calculates and stores partials (conditional likelihood vectors),
    and transition probabilities.
    """

    def __init__(self, subst_model):
        self.subst_model = subst_model
        self.partials = None
        self.sitewise = None
        self.scale_buffer = None

    def update_transition_probabilities(self, len1, len2):
        self.probs1 = self.subst_model.p(len1)
        self.probs2 = self.subst_model.p(len2)

    def set_partials(self, partials):
        """ Set the partials at this node """
        partials = np.asarray(partials, dtype=np.double)
        if partials.ndim == 1:
            self.partials = np.ascontiguousarray(partials[np.newaxis], dtype=np.double)
        else:
            self.partials = np.ascontiguousarray(partials, dtype=np.double)

    def compute_partials(self, lnlmodel1, lnlmodel2, scale=True):
        """ Update partials at this node """
        if scale:
            self.partials, self.scale_buffer = likcalc.likvec_2desc_scaled(self.probs1, self.probs2, lnlmodel1.partials,
                                                                           lnlmodel2.partials)
        else:
            self.partials = likcalc.likvec_2desc(self.probs1, self.probs2, lnlmodel1.partials, lnlmodel2.partials)

    def compute_edge_sitewise_likelihood(self, lnlmodel, brlen, derivatives=False):
        """ Calculate the likelihood with this node at root - 
        returns array of [f, f', f''] values, where fs are unscaled unlogged likelihoods, and
        f' and f'' are unconverted partial derivatives.
        Logging, scaling and conversion are done in compute_likelihood """
        probs = self.subst_model.p(brlen)

        if derivatives:
            dprobs = self.subst_model.dp_dt(brlen)
            d2probs = self.subst_model.d2p_dt2(brlen)
            self.sitewise = likcalc.sitewise_lik_derivs(probs, dprobs, d2probs, self.subst_model.freqs, self.partials,
                                                        lnlmodel.partials)
        else:
            self.sitewise = likcalc.sitewise_lik(probs, self.subst_model.freqs, self.partials, lnlmodel.partials)

    def compute_likelihood(self, lnlmodel, brlen, derivatives=False, accumulated_scale_buffer=None):
        self.compute_edge_sitewise_likelihood(lnlmodel, brlen, derivatives)
        swlnl = self.sitewise[:, 0]
        if accumulated_scale_buffer is not None:
            lnl = (swlnl + accumulated_scale_buffer).sum()
        else:
            lnl = swlnl.sum()
        if derivatives:
            dlnl = self.sitewise[:, 1].sum()
            d2lnl = self.sitewise[:, 2].sum()
            return lnl, dlnl, d2lnl
        else:
            return lnl


class LnlModel(object):
    def __init__(self, subst_model, partials_dict):
        # Initialise leaves
        self.leaf_models = {}
        for (leafname, partials) in partials_dict.items():
            model = LnlNode(subst_model)
            model.set_partials(partials)
            self.leaf_models[leafname] = model

        self.nsites = next(iter(partials_dict.values())).shape[0]
        self.subst_model = subst_model
        self.accumulated_scale_buffer = None

    def set_tree(self, tree):
        # self.tree = dpy.Tree.get_from_string(tree, 'newick', preserve_underscores=True)
        # self.tree.resolve_polytomies() # Require strictly binary tree, including root node
        self.tree = tree
        for leaf in self.tree.leaf_nodes():
            leaf.model = self.leaf_models[leaf.taxon.label]

    def update_subst_model(self, subst_model):
        self.subst_model = subst_model
        for leaf in self.tree.leaf_nodes():
            leaf.model.subst_model = subst_model

    def run(self, derivatives=False):
        self.accumulated_scale_buffer = np.zeros(self.nsites, dtype=np.double)
        for node in self.tree.postorder_internal_node_iter():
            children = node.child_nodes()
            node.model = LnlNode(self.subst_model)
            l1, l2 = [ch.edge.length for ch in children]
            node.model.update_transition_probabilities(l1, l2)
            model1, model2 = [ch.model for ch in node.child_nodes()]
            node.model.compute_partials(model1, model2, True)
            if node is not self.tree.seed_node:
                self.accumulated_scale_buffer += node.model.scale_buffer
        ch1, ch2 = self.tree.seed_node.child_nodes()[:2]
        return ch1.model.compute_likelihood(ch2.model, ch1.edge.length + ch2.edge_length,
                                            derivatives, self.accumulated_scale_buffer)

    def get_sitewise_likelihoods(self):
        ch = self.tree.seed_node.child_nodes()[0]
        scaler = np.zeros_like(ch.model.sitewise)
        scaler[:, 0] = self.accumulated_scale_buffer# * likcalc.get_log_scale_value()
        return ch.model.sitewise + scaler

    # def get_sitewise_fval(self):
    #     ch = self.tree.seed_node.child_nodes()[0]
    #     return ch.model.sitewise[:, 0]


class Mixture(object):
    def mix_likelihoods(self, sw_lnls):
        ma = sw_lnls.max(1)[:, np.newaxis]
        wa = sw_lnls + self.logweights
        return np.log(np.exp(wa - ma).sum(1))[:, np.newaxis] + ma


class GammaMixture(Mixture):
    def __init__(self, alpha, ncat):
        self.ncat = ncat
        self.rates = likcalc.discrete_gamma(alpha, ncat)
        self.weights = np.array([1.0 / ncat] * ncat)
        self.logweights = np.log(self.weights)

    def update_alpha(self, alpha):
        self.rates = likcalc.discrete_gamma(alpha, self.ncat)
        self.set_tree(self.tree)

    def update_substitution_model(self, tm):
        for runner in self.runners:
            runner.update_subst_model(tm)

    def init_models(self, tm, partials_dict):
        self.runners = []
        for cat in range(self.ncat):
            runner = LnlModel(tm, partials_dict)
            self.runners.append(runner)

    def set_tree(self, tree):
        self.tree = tree
        for cat in range(self.ncat):
            t = dpy.Tree.get_from_string(tree, 'newick', preserve_underscores=True)
            t.resolve_polytomies()
            t.scale_edges(self.rates[cat])
            self.runners[cat].set_tree(t)

    def run(self, derivatives=False):
        for runner in self.runners:
            runner.run(derivatives)

    def get_sitewise_likelihoods(self):
        swlnls = np.empty((self.runners[0].nsites, self.ncat))
        for cat in range(self.ncat):
            swlnls[:, cat] = self.runners[cat].get_sitewise_likelihoods()[:,0]
        return swlnls

    def get_scale_bufs(self):
        scale_bufs = np.array([model.accumulated_scale_buffer for model in self.runners]).T
        return scale_bufs

    def get_sitewise_fvals(self):
        swfvals = np.empty((self.runners[0].nsites, self.ncat))
        for cat in range(self.ncat):
            swfvals[:, cat] = self.runners[cat].get_sitewise_fval()
        return swfvals

    def get_likelihood(self):
        return self.mix_likelihoods(self.get_sitewise_likelihoods()).sum()


class OptWrapper(object):
    """
    Wrapper for use with scipy optimiser (e.g. brenth/brentq)
    """

    def __init__(self, tm, partials1, partials2, initial_brlen=1.0):
        self.root = LnlNode(tm)
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
        return 'Branch length={}, Variance={}, Likelihood+derivatives = {} {} {}'.format(self.updated, -1 / self.d2lnl,
                                                                                         self.lnl, self.dlnl,
                                                                                         self.d2lnl)


def optimise(likelihood, partials_a, partials_b, min_brlen=0.00001, max_brlen=10, verbose=True):
    """
    Optimise ML distance between two partials. min and max set brackets
    """
    from scipy.optimize import brenth
    wrapper = OptWrapper(likelihood, partials_a, partials_b, (min_brlen + max_brlen) / 2.)
    brlen = 0.5
    n = brenth(wrapper.get_dlnl, min_brlen, max_brlen)
    if verbose:
        logger.info(wrapper)
    return n, -1 / wrapper.get_d2lnl(n)


class BranchLengthOptimiser(object):
    """
    Wrapper for use with scipy optimiser (e.g. brenth/brentq)
    """

    def __init__(self, node1, node2, initial_brlen=1.0):
        self.root = node1
        self.desc = node2
        self.updated = None
        self.__call__(initial_brlen)

    def __call__(self, brlen):
        if self.updated != brlen:
            self.updated = brlen
            self.lnl, self.dlnl, self.d2lnl = self.root.compute_likelihood(self.desc, brlen, derivatives=True)
        return self.lnl, self.dlnl, self.d2lnl

    def get_lnl(self, brlen):
        return self.__call__(brlen)[0]

    def get_dlnl(self, brlen):
        return np.array([self.__call__(brlen)[1]])

    def get_d2lnl(self, brlen):
        return np.array([self.__call__(brlen)[2]])

    def get_negative_lnl(self, brlen):
        return -self.__call__(max(0,brlen))[0]

    def get_negative_dlnl(self, brlen):
        return -self.__call__(max(0,brlen))[1]

    def get_negative_d2lnl(self, brlen):
        return -self.__call__(max(0,brlen))[2]

    def __str__(self):
        return 'Branch length={}, Variance={}, Likelihood+derivatives = {} {} {}'.format(self.updated, -1 / self.d2lnl,
                                                                                         self.lnl, self.dlnl,
                                                                                         self.d2lnl)

def brent_optimise(node1, node2, min_brlen=0.00001, max_brlen=10, verbose=True):
    """
    Optimise ML distance between two partials. min and max set brackets
    """
    from scipy.optimize import minimize_scalar
    wrapper = BranchLengthOptimiser(node1, node2, (min_brlen + max_brlen) / 2.)
    n = minimize_scalar(lambda x: -wrapper(x)[0], method='brent', bracket=(min_brlen, max_brlen))['x']
    if verbose:
        logger.info(wrapper)
    return n, -1 / wrapper.get_d2lnl(n)


def scipy_optimise(node1, node2, min_brlen=0.00001, max_brlen=10, verbose=True, method='l-bfgs-b'):
    init_x = (min_brlen + max_brlen) / 2.
    wrapper = BranchLengthOptimiser(node1, node2, init_x)
    opt = minimize(wrapper.get_negative_lnl, init_x,
                   jac=wrapper.get_negative_dlnl,
                   hess=wrapper.get_negative_d2lnl,
                   method=method)  # minimise the negative => maximise
    n = opt['x']
    return n, -1 / wrapper.get_d2lnl(n)
