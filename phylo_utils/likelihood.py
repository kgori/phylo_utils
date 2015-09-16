import numpy as np
import likcalc
from markov import TransitionMatrix
import dendropy as dpy

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
        evecs, evals, ivecs = self.transmat.eigen.values
        if derivatives:
            self.sitewise = likcalc.sitewise_lik_derivs(evecs, evals, ivecs, self.transmat.freqs, brlen, self.partials, lnlmodel.partials)
        else:
            self.sitewise = likcalc.sitewise_lik(evecs, evals, ivecs, self.transmat.freqs, brlen, self.partials, lnlmodel.partials)

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
    def __init__(self, tree, transition_matrix, scale_freq=20):
        self.tree = dpy.Tree.get_from_string(tree, 'newick')
        self.tree.resolve_polytomies()

        # Initialise models
        self.num_leaves = len(self.tree.leaf_nodes())
        self.num_inner = len(self.tree.internal_nodes())
        self.leaf_models = [Leaf() for _ in xrange(self.num_leaves)]
        self.inner_models = [LnlModel(transition_matrix) for _ in xrange(self.num_inner)]

        self.tm = transition_matrix
        self.internal_node_counter = 0
        self.accumulated_scale_buffer = None
        self.scale_freq = scale_freq

    def attach_models_to_tree(self, partials_dict):
        pass
        
    def init_leaves(self, partials_dict):
        self.internal_node_counter = 0
        example_leaf = None
        for i, leaf in enumerate(self.tree.leaf_nodes()):
            taxon = leaf.taxon.label
            leaf.model = Leaf(partials_dict[taxon])
        self.nsites = leaf.model.partials.shape[0]

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
        m = sw_lnls.max(1)[:,np.newaxis]
        v = np.exp(sw_lnls - m)
        c = (self.weights * v)
        return np.log(c.sum(1))[:, np.newaxis] + m


class GammaMixture(Mixture):
    def __init__(self, alpha, ncat):
        self.ncat = ncat
        self.rates = likcalc.discrete_gamma(alpha, ncat)
        self.weights = np.array([1.0/ncat] * ncat)

    def add_tree(self, tree, tm, scale_freq=20):
        self.runners = []
        for cat in range(self.ncat):
            t = dpy.Tree.get_from_string(tree, 'newick')
            t.resolve_polytomies()
            t.scale_edges(self.rates[cat])
            self.runners.append(RunOnTree(t.as_newick_string()+';', tm))

    def init_leaves(self, partials_dict):
        for runner in self.runners:
            runner.init_leaves(partials_dict)

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
        print(wrapper)
    return n, -1/wrapper.get_d2lnl(n)

if __name__ == '__main__':
    from seq_to_partials import dna_charmap, protein_charmap, seq_to_partials

    #####################################
    # Pairwise distance optimiser demo:

    kappa = 1
    k80 = np.array([[0,kappa,1,1],[kappa,0,1,1],[1,1,0,kappa],[1,1,kappa,0]], dtype=np.float)
    k80f = np.array([0.25,0.25,0.25,0.25])
    tm = TransitionMatrix(k80, k80f)
    # Simulated data from K80, kappa=1, distance = 0.8
    sites_a = seq_to_partials('ACCCTCCGCGTTGGGTAGTCCTAGGCCCAATGGCGTTTATGCCTCGATTTTTAGTTCTACCGTCCCTACAGATGGATGCCGTCGCATAGACACTGTCAATTCCATTCGGCAGGCTTCACACTGTTGCATTTTCATTTTGTACACGGTACCAACATAGGAGTGCTGTATTGCTATATTTCCAGTACACGGCGTTGAGTCGGATGGAAACGCCGGCGGAAGACAGCTTGGCGGGTCTTCACGCATCACCGCGGGGTCTGAAAGGTATTATCGCTGCTTAAATCAGACCGGTCAAGCTTCCTGGCGGAAGGCGGCAAGGTCCAGCCACAGCATGCTTATTCCTTGTCACGCCGGGTGGAAATCTAGAGCGTCCGGTGGACACAGAGTGATTTTGTACGGGGGGTTCCATACCAGGACATTAGGGTCGGTTTACGGTCTGAGATGTATGTTGCCTTGCGGTCGACGAGCACTGATTCCCCTGAACTTCGTAAGACACATATAGTTTTAATGAAATCCCCAAAACGAGCATGGTTTCAGTATACGCGACAACTTAGGATACAACATACTGAACCAGTCCGCATTGAGGTGCCAATCAAACGGGACCGGGACTGATAAGTATAAAATAGGTTTCCCTGTCCTCTACCTACGTTATCCTCGCGTCGATTTTGATTCTTACCAAGACTGCTAATCAGGCCCTGTGGCCTGCATGTCACCATGTCAGCGTGTTTGGCTAAATTCACGGGATTGGCCTTACCGACTTACATCAGTATTTCATACATAGTTACTCGAGTTTAACGTTGACAGTTAGTCCCATGATACGGCAAAGCCTGGTTCGGCGGATTTCCGAGTACAGCATCTTCGCCCCCGAGATTGCCGCCAATGGACACCCTCCTGAGATGCAGATATGAGTGTTTTTGACACTCTGAGGCTGAGATCCTCACACTTCCGGAGCTTCCGCGATAGTCACGTGGTTATTAGACTTACGGCAGGAAAAATCATGTTA', alphabet='dna')
    sites_b = seq_to_partials('AAGCTCCGCGTAAGCTAACGACCAGTCAGCTAGGTTTAGTGCCACCAGTATGGCTAGTTCCGGAGGGCAAACCGGATGCTACCGATTGGTCACCCTCAGGGTGATTTCGCAGGGCGCTCACTTATTCCTTTTAAATCCTGCCAACAGACTAAGAAAGTTGTACGGTATTCCTATATCTTCAGTACTGCTCTTGGCCGTGCATGTAGCCGAACGACGAGGACGGTACATGAGTTTCTCACCAATTACAGGCGGTTCCATTAGGCAGTAGCTGCGGTTAGTTCATACTGCTAAAGAATCTTCTTGGAACGTGCCAAGGACCAGTCACACACATGTTGTAGTCCCTCATCGTGGTAGGCGTTCCAGACCGTCCGTGGTACACATACCAAATTTCGTACCGGCTGACTCAAAGCGGGAGTTCGCATGATACCAGGGAACGAGATGTTCAAAACGATCAGGTAGTGCCGCCATCTTTCAGGTTCTTTCGTTTCGTCCTATGATACTTGAGTAGCGGTCAAACGAAGCTCGTAGGTGACAGTTACGAGACATGCTGGGATGCAACATACTTTCGCAGTTAGCTAGTAGGTACCTATCTAGCGAATCGAGCTAGGATACCCTGATTATGCTTGTCTCCGTCCTCTTACTATGATCTCCTCGCGTGGTTTTTGCTGCTTAACCGTTGTGCCGTATAAAACAAGAGGCGGGAGTTTAGCTGTGGGAACTTCGTAGACCTTGTAAGCTGGATAGGCCCGTCCGTCGTAATTAATTACCTAAAAGAGAGTCAAACAAGCTTAAGTCGCCGAGTTAGTCGGATAAGAAGCCATTCTCTGGTCCGCCAACCTTCCCATGCCAGTACGGTTGCCGAGGTCCATTCGGTGACTGTGGGATAACCGTTGCCGGAGCTATGAGATCCATTACAACTCTGCGCCTAGGATGTTAACTCTACCGAAGTTTGCGACCCCGGAACCTGTAAATTGTCCTTAGGGTCGTAACATTTTCAAGC', alphabet='dna')

    optimise(tm, sites_a, sites_b)

    ############################################################################
    # Example from Section 4.2 of Ziheng's book - his value for node 6 is wrong!
    np.set_printoptions(precision=6)
    kappa = 2
    k80 = np.array([[0,kappa,1,1],[kappa,0,1,1],[1,1,0,kappa],[1,1,kappa,0]], dtype=np.float)
    k80f = np.array([0.25,0.25,0.25,0.25])
    tm = TransitionMatrix(k80, k80f)

    partials_1 = np.ascontiguousarray(np.array([[1, 0, 0, 0]], dtype=np.float))
    partials_2 = np.ascontiguousarray(np.array([[0, 1, 0, 0]], dtype=np.float))
    partials_3 = np.ascontiguousarray(np.array([[0, 0, 1, 0]], dtype=np.float))
    partials_4 = np.ascontiguousarray(np.array([[0, 1, 0, 0]], dtype=np.float))
    partials_5 = np.ascontiguousarray(np.array([[0, 1, 0, 0]], dtype=np.float))

    partials_dict = {'1': partials_1,
                     '2': partials_2,
                     '3': partials_3,
                     '4': partials_4,
                     '5': partials_5}
    

    t = '(((1:0.2,2:0.2)7:0.1,3:0.2)6:0.1,(4:0.2,5:0.2)8:0.1)0;'
    t = '((1:0.2,2:0.2):0.1,3:0.2,(4:0.2,5:0.2):0.2);'
    runner = RunOnTree(t, tm)
    runner.init_leaves(partials_dict)
    print runner.run(True)
    print runner.get_sitewise_likelihoods()
    
    gamma = GammaMixture(0.4, 4)
    gamma.add_tree(t, tm, scale_freq=3)
    gamma.init_leaves(partials_dict)
    print gamma.get_likelihood()
    print gamma.get_sitewise_likelihoods()

    kappa = 1
    k80 = np.array([[0,kappa,1,1],[kappa,0,1,1],[1,1,0,kappa],[1,1,kappa,0]], dtype=np.float)
    k80f = np.array([0.25,0.25,0.25,0.25])
    tm = TransitionMatrix(k80, k80f)
    
    partials_dict = {'1': seq_to_partials('ACCCT'),
                 '2': seq_to_partials('TCCCT'),
                 '3': seq_to_partials('TCGGT'),
                 '4': seq_to_partials('ACCCA'),
                 '5': seq_to_partials('CCCCC')}

    gamma = GammaMixture(.02, 4)
    gamma.add_tree(t, tm, scale_freq=3)
    gamma.init_leaves(partials_dict)
    print gamma.get_likelihood()
    print gamma.get_sitewise_likelihoods()
    print gamma.get_sitewise_likelihoods().sum(0)

