import itertools

import numpy as np

def label_nodes(tree):
    T = len(tree.leaf_nodes())
    tree.is_rooted = len(tree.seed_node.child_nodes())==2

    N = 2 * T - (1 if tree.is_rooted else 2)
    E = 2 * T - (2 if tree.is_rooted else 3)

    leaf_node_index = 0
    inner_node_index = T
    leaf_edge_index = 0
    inner_edge_index = T
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            node.index = leaf_node_index
            node.edge_index = leaf_edge_index
            leaf_node_index += 1
            leaf_edge_index += 1
        else:
            node.index = inner_node_index
            inner_node_index += 1
            if not node is tree.seed_node:
                node.edge_index = inner_edge_index
                inner_edge_index += 1

    if tree.is_rooted:
        tree.seed_node.edge_index = None
    else:
        # tree.seed_node.edge_index = max(tree.seed_node.child_nodes(), key=lambda x: x.edge_index).edge_index
        tree.seed_node.edge_index = tree.seed_node.child_nodes()[0].edge_index

    tree.is_labelled = True

def generate_postorder_traversal(tree):
    ensure_labelled(tree)

    descriptor = []
    for node in tree.postorder_internal_node_iter():
        children = node.child_nodes()
        if node is tree.seed_node and not tree.is_rooted:
            children = [child for child in children if child.edge_index != node.edge_index]
        descriptor.append(
            (node.index, children[0].index, children[0].edge_index, children[1].index, children[1].edge_index))

    return descriptor

def generate_reverse_levelorder_traversal(tree):
    ensure_labelled(tree)

    descriptor = []
    reverse_levelorder = list(tree.levelorder_node_iter(filter_fn=lambda x: x.is_internal()))[::-1]
    for node in reverse_levelorder:
        children = node.child_nodes()
        if node is tree.seed_node and not tree.is_rooted:
            children = [child for child in children if child.edge_index != node.edge_index]
        descriptor.append(
            (node.index, children[0].index, children[0].edge_index, children[1].index, children[1].edge_index))

    return descriptor


def generate_optimising_traversal(tree):
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
    # This traversal is a combined preorder (in edges)
    # and postorder (in nodes) traversal.
    # Every edge passed in preorder triggers an update
    # of partials of the tail node, and an optimisation
    # of the edge, and every internal node passed in postorder
    # triggers a partials update (no optimisation)
    ensure_labelled(tree)

    descriptor = []

    def _traverse(node, descriptor):
        parent = node.parent_node
        siblings = [sib for sib in parent.child_nodes() if sib is not node]
        if parent is tree.seed_node:
            if tree.is_rooted:
                assert len(siblings) == 1
                grandparent_index = -1
                grandparent_edge = -1
                sibling = siblings[0]
            else:
                assert len(siblings) == 2
                sibling = siblings[0]
                grandparent_index = siblings[1].index
                grandparent_edge = siblings[1].edge_index
        else:
            assert len(siblings) == 1
            grandparent_index = parent.parent_node.index
            grandparent_edge = parent.edge_index
            sibling = siblings[0]

        # Descriptor format: 8 indices:
        #   1=Node at which to update partials, through the following:
        #   2=Child 1's partials
        #   3=Child 1's edge
        #   4=Child 2's partials
        #   5=Child 2's edge
        #   6 and 7=Nodes between which the branch to be optimised exists
        #   8=Optimised branch's edge index
        descriptor.append((parent.index, sibling.index, sibling.edge_index, grandparent_index, grandparent_edge,
                           parent.index, node.index, node.edge_index))

        # Recursion
        if node.is_internal():
            child1, child2 = node.child_node_iter()
            _traverse(child1, descriptor)
            _traverse(child2, descriptor)
            # Reorient partials back towards root
            descriptor.append((node.index, child1.index, child1.edge_index, child2.index, child2.edge_index,
                               -1, -1, -1))
        return

    if tree.is_rooted:
        for child_node in tree.seed_node.child_nodes():
            _traverse(child_node, descriptor)

    else:
        left, middle, right = tree.seed_node.child_nodes()
        descriptor.append((-1, -1, -1, -1, -1,
                           tree.seed_node.index, left.index, left.edge_index))
        for child in left.child_nodes():
            _traverse(child, descriptor)
        _traverse(middle, descriptor)
        _traverse(right, descriptor)

    return descriptor


def ensure_labelled(tree):
    if not hasattr(tree, 'is_labelled'):
        label_nodes(tree)
    elif not tree.is_labelled:
        label_nodes(tree)


class Traversal(object):
    def __init__(self, tree):
        # TODO: assert is a rooted dendropy tree

        label_nodes(tree)
        nleaves = len(tree.leaf_nodes())
        nedges = 2 * nleaves - (2 if tree.is_rooted else 3)

        # index the nodes
        self.node_dict = {}
        self.names = {}
        self.brlens = np.zeros(nedges)
        for node in tree.postorder_node_iter():
            if node is tree.seed_node: continue
            self.brlens[node.edge_index] = node.edge_length
            self.node_dict[node] = node.index
            if node.is_leaf():
                self.names[node.taxon.label] = node.index

        nleaves = len(self.names)
        self.root_edge = tuple([self.node_dict[n] for n in tree.seed_node.child_nodes()])

        # Generate the traversals
        self.postorder_traversal = np.array(generate_postorder_traversal(tree))
        self.optimising_traversal = np.array(generate_optimising_traversal(tree))
        self.reverse_levelorder = np.array(generate_reverse_levelorder_traversal(tree))


def get_sibling(node):
    for candidate in node.parent_node.child_node_iter():
        if candidate is not node:
            return candidate


class BranchLengths(dict):
    def __getitem__(self, key):
        val = self.get(key)
        if val is None:
            val = self.get(key[::-1])
        if val is not None:
            return val
        else:
            raise KeyError(key)


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