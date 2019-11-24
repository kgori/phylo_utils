import itertools

import numpy as np


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


def get_grandparent(tree, node):
    candidate = node.parent_node.parent_node
    if candidate is tree.seed_node:
        return get_sibling(node.parent_node)
    return candidate


def preorder_self_exclude(node):
    return itertools.chain(*(child.preorder_iter() for child in node.child_nodes()))


def label_tree(tree):
    c = 0
    for leaf in tree.leaf_node_iter():
        leaf.index = c
        c += 1

    for inner_node in tree.postorder_internal_node_iter(exclude_seed_node=True):
        inner_node.index = c
        c += 1

    def edge_filter(edge):
        return edge.head_node is not tree.seed_node and edge.tail_node is not tree.seed_node

    c = 0
    for edge in tree.postorder_edge_iter(filter_fn=edge_filter):
        edge.index = c
        c += 1


def get_node_dict(tree):
    return dict(zip(itertools.chain(*(tree.leaf_node_iter(),
                                      tree.postorder_internal_node_iter(exclude_seed_node=True))),
                    itertools.count()))


def get_edge_dict(tree):
    d = {}
    c = 0
    for edge in tree.postorder_edge_iter(
            filter_fn=lambda x: x.head_node is not tree.seed_node and x.tail_node is not tree.seed_node):
        d[edge] = c
        c += 1

    left, right = tree.seed_node.child_nodes()
    root_edge_length = left.edge_length + right.edge_length
    left.edge_length = root_edge_length
    right.edge_length = 0
    tree.seed_node.edge_length = 0
    d[left.edge] = c
    return d


def get_connections(tree):
    def _filter(x):
        return (x.head_node is not tree.seed_node and
                x.tail_node is not tree.seed_node)

    d = {}
    for edge in tree.postorder_edge_iter(filter_fn=_filter):
        d[edge] = c


def get_sibling(node):
    for candidate in node.parent_node.child_node_iter():
        if candidate is not node:
            return candidate


def get_postorder_traversal(tree, descriptor, node_dict):
    # Number of operations (N leaves):
    # (N-2) inner node partials calculations
    for i, node in enumerate(tree.postorder_internal_node_iter(exclude_seed_node=True)):
        NODi = node_dict[node]
        CH1i, CH2i = [node_dict[child] for child in node.child_node_iter()]
        descriptor[i] = NODi, CH1i, CH2i
    return descriptor


class BranchLengths(dict):
    def __getitem__(self, key):
        val = self.get(key)
        if val is None:
            val = self.get(key[::-1])
        if val is not None:
            return val
        else:
            raise KeyError(key)


def get_optimising_traversal(tree, descriptor, node_dict):
    # Number of operations (N leaves):
    #    (N-2)  inner node partial calculations
    #  + (2N-3) branches to optimise           \
    #                                          | compressed together into
    #                                          | 2N-3 operations
    #   [(2N-4)]rerooting partial calculations /
    #  = (5N-9) operations on 3N-5 rows
    # So, descriptor must have 3N-5 rows
    # First optimize between LEFT and RIGHT,
    # then describe remaining traversal

    def _traverse(node, index_gen):
        NODi = node_dict[node]

        if not node in (LEFT, RIGHT):
            # Book-keeping
            PAR = node.parent_node
            PARi = node_dict[PAR]
            if PAR == LEFT:
                GPA = RIGHT
            elif PAR == RIGHT:
                GPA = LEFT
            else:
                GPA = PAR.parent_node
            GPAi = node_dict[GPA]
            SIB = get_sibling(node)
            SIBi = node_dict[SIB]

            # Write operations to descriptor
            # prepare partials (index 0,1,2)
            # and optimize (index 3,4)
            descriptor[next(index_gen)] = PARi, SIBi, GPAi, NODi, PARi

        # Recursion
        if not node.is_leaf():
            CH1, CH2 = node.child_node_iter()
            CH1i = node_dict[CH1]
            CH2i = node_dict[CH2]
            _traverse(CH1, index_gen)
            _traverse(CH2, index_gen)
            # Reorient partials back towards root
            descriptor[next(index_gen)] = NODi, CH1i, CH2i, -1, -1

    DUMMY = tree.seed_node
    LEFT, RIGHT = DUMMY.child_nodes()
    descriptor[0] = -1, -1, -1, node_dict[LEFT], node_dict[RIGHT]
    indexer = itertools.count(start=1)
    for node in LEFT, RIGHT:
        _traverse(node, indexer)

    return descriptor


def get_branch_lengths(node_dict):
    brlens = BranchLengths()
    for node in node_dict:
        if not node.parent_node.parent_node:
            neighbour = get_sibling(node)
            length = max(n.edge_length for n in node.parent_node.child_nodes())
        else:
            neighbour = node.parent_node
            length = node.edge_length

        brlens[tuple(sorted((node_dict[node], node_dict[neighbour])))] = length
    return brlens