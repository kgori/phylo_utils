import numpy as np

from phylo_utils.utils import get_postorder_traversal, get_optimising_traversal, get_branch_lengths


class Traversal(object):
    def __init__(self, tree):
        # TODO: assert is a rooted dendropy tree

        # Collect tree info

        # index the nodes
        i = 0
        self.node_dict = {}
        self.names = {}
        for node in tree.postorder_node_iter():
            if node is tree.seed_node: continue
            self.node_dict[node] = i
            if node.is_leaf():
                self.names[node.taxon.label] = i
            i += 1

        nleaves = len(self.names)
        self.root_edge = tuple([self.node_dict[n] for n in tree.seed_node.child_nodes()])
        self.brlens = get_branch_lengths(self.node_dict)

        # Allocate empty arrays
        postorder_traversal = np.zeros((nleaves - 2, 3), dtype=np.int)
        optimising_traversal = np.zeros((3 * nleaves - 5, 5), dtype=np.int)

        # Fill the arrays
        self.postorder_traversal = get_postorder_traversal(tree, postorder_traversal,
                                                           self.node_dict)
        self.optimising_traversal = get_optimising_traversal(tree, optimising_traversal,
                                                             self.node_dict)


