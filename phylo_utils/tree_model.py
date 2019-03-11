from phylo_utils.likelihood.numba_likelihood_engine import clv, lnl_node
from .traversal import Traversal
from .utils import deepcopy_tree, seq_to_partials

import numpy as np
from scipy.misc import logsumexp

class TreeModel(object):
    """
    Initialise and maintain data structures needed
    to compute likelihood on tree

    Structure:
    N leaf nodes
    N-1 inner nodes (including root)
    2N-2 branches

    Models:
    List of substitution models, plus index to associate 2N-2 branches to models
    Gamma distributed rates multipliers

    Data:
    s*(2n-1)*c*k double entries for clvs
    s*(2n-1)*k entries for scaling vectors
    (2n-2) branch lengths
      (c=alphabet size, k=number of gamma cats,
       s=number of sites, n=number of sequences)

    Requirements:
    Sequence alignment,
    Tree,
    Substitution model(s),
    Rates model
    """
    alignment = None

    def set_alignment(self, alignment, alphabet, compress=True):
        """
        Assumes biopython alignment
        """
        if tuple([int(v) for v in np.version.version.split('.')]) < (1, 13, 0):
            compress = False
        _alignment = np.stack(
            [seq_to_partials(str(seq.seq), alphabet) for seq in alignment])
        self.names = {seq.name: i for i, seq in enumerate(alignment)}
        n_sites = _alignment.shape[1]
        if compress:
            self.alignment, self.inverse_index, self.siteweights = np.unique(_alignment,
                                                                             return_inverse=True, return_counts=True,
                                                                             axis=1)
        else:
            self.alignment = _alignment
            self.siteweights = np.ones(n_sites, dtype=np.int)
            self.inverse_index = np.arange(n_sites)

    def get_empirical_freqs(self, pseudocount=None, include_ambiguous=False):
        if self.alignment is not None:
            counts = (
                self.alignment.sum((0, 1))
                if include_ambiguous
                else self.alignment[np.where(self.alignment.sum(2) < self.alignment.shape[2])].sum(0)
            )

            if pseudocount is not None:
                try:
                    counts += np.array(pseudocount)
                except TypeError:
                    print("Log: pseudocount {} caused Type error. Carrying on "
                          "without pseudocount.".format(pseudocount))
                except ValueError:
                    print("Log: pseudocount {} caused Value error (probably the"
                          " wrong length). Carrying on without pseudocount.".format(pseudocount))
            return counts / counts.sum()
        else:
            print('Log: No alignment set')

    def set_substitution_model(self, model):
        """
        TODO: extend to multiple models + index list
        """
        self.substitution_model = model

    def set_rate_model(self, rate_model):
        self.rate_model = rate_model

    def set_tree(self, dpytree):
        self.tree = deepcopy_tree(dpytree)
        self.traversal = Traversal(self.tree)

    def initialise(self):
        """
        Allocate numpy arrays to store all partial likelihoods
        """
        # TODO: assertions that all data are inplace, names match between
        #       alignment and tree, etc...
        n_leaves, n_sites, n_chars = self.alignment.shape
        n_cat = self.rate_model.ncat

        self.partials = np.zeros((n_sites, (2 * n_leaves - 2), n_chars, n_cat),
                                 dtype=np.double)
        self.scale = np.zeros((n_sites, (2 * n_leaves - 2), n_cat),
                              dtype=np.double)
        self.root_partials = np.zeros((n_sites, n_chars, n_cat),
                                      dtype=np.double)
        self.root_scale = np.zeros((n_sites, n_cat),
                                   dtype=np.double)

        # transfer alignment tip partials to partials array
        for name in self.traversal.names:
            tree_index = self.traversal.names[name]
            alignment_index = self.names[name]
            for cat in range(n_cat):
                self.partials[:, tree_index, :, cat] = self.alignment[alignment_index]
                self.scale[:, tree_index, cat] = 0

        self.compute_partials()

    def compute_partials(self):
        """
        Do 1 postorder traversal and compute partials (CLVs) at internal nodes
        """
        for instruction in self.traversal.postorder_traversal:
            PAR, CH1, CH2 = instruction
            brlen1 = self.traversal.brlens[(PAR, CH1)]
            brlen2 = self.traversal.brlens[(PAR, CH2)]
            prob1 = self.substitution_model.get_p_matrix(brlen1, self.rate_model.rates)
            prob2 = self.substitution_model.get_p_matrix(brlen2, self.rate_model.rates)

            clv1 = self.partials[:, CH1, :, :]
            clv2 = self.partials[:, CH2, :, :]
            scale1 = self.scale[:, CH1, :]
            scale2 = self.scale[:, CH2, :]
            scalep = self.scale[:, PAR, :]
            self.partials[:, PAR, :, :] = clv(prob1, prob2, clv1, clv2, scale1, scale2, scalep)

    def compute_partials_at_edge(self, node_a, node_b):
        """
        Place the root at the edge between a and b, and compute partials
        Note: the values are only valid if the CLVs at a and b are valid!
        """
        # Final computation at root
        try:
            length = self.traversal.brlens[node_a, node_b]
        except KeyError:
            raise ValueError('There is no edge connecting nodes {} and {}'.format(node_a, node_b))

        prob1 = self.substitution_model.get_p_matrix(0, self.rate_model.rates)
        prob2 = self.substitution_model.get_p_matrix(length, self.rate_model.rates)
        clv1 = self.partials[:, node_a, :, :]
        clv2 = self.partials[:, node_b, :, :]
        scale1 = self.scale[:, node_a, :]
        scale2 = self.scale[:, node_b, :]
        scalep = self.root_scale[:, :]
        clv(prob1, prob2, clv1, clv2, scale1, scale2, scalep,
            self.root_partials[:, :, :])
        return self.root_partials, self.root_scale

    def compute_likelihood_at_edge(self, node_a, node_b):
        """
        Place the root at the edge between a and b, and compute the sitewise likelihoods
        Note: the values are only valid if the CLVs at a and b are valid!
        """
        self.compute_partials_at_edge(node_a, node_b)
        swlnls = lnl_node(self.substitution_model.freqs,
                          self.root_partials, self.root_scale)
        swlnls = logsumexp(swlnls + np.log(self.rate_model.weights), axis=1)
        return swlnls[self.inverse_index]
