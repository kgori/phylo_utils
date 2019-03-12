from __future__ import print_function
from phylo_utils.utils import seq_to_partials
from phylo_utils.substitution_models import K80, WAG, LG
from phylo_utils.likelihood import optimise, GammaMixture, LnlModel, LnlNode, Leaf, brent_optimise
import numpy as np
import dendropy as dpy

#####################################
# Pairwise distance optimiser demo:

kappa = 0.7345
k80 = K80(kappa)
probs = k80.p(1)
dprobs = k80.dp_dt(1)
d2probs = k80.d2p_dt2(1)
# Simulated data from K80, kappa=1, distance = 0.8
sites_a = seq_to_partials('ACCCTCCGCGTTGGGTAGTCCTAGGCCCAATGGCGTTTATGCCTCGATTTTTAGTTCTACCGTCCCTACAGATGGATGCCGTCGCATAGACACTGTCAATTCCATTCGGCAGGCTTCACACTGTTGCATTTTCATTTTGTACACGGTACCAACATAGGAGTGCTGTATTGCTATATTTCCAGTACACGGCGTTGAGTCGGATGGAAACGCCGGCGGAAGACAGCTTGGCGGGTCTTCACGCATCACCGCGGGGTCTGAAAGGTATTATCGCTGCTTAAATCAGACCGGTCAAGCTTCCTGGCGGAAGGCGGCAAGGTCCAGCCACAGCATGCTTATTCCTTGTCACGCCGGGTGGAAATCTAGAGCGTCCGGTGGACACAGAGTGATTTTGTACGGGGGGTTCCATACCAGGACATTAGGGTCGGTTTACGGTCTGAGATGTATGTTGCCTTGCGGTCGACGAGCACTGATTCCCCTGAACTTCGTAAGACACATATAGTTTTAATGAAATCCCCAAAACGAGCATGGTTTCAGTATACGCGACAACTTAGGATACAACATACTGAACCAGTCCGCATTGAGGTGCCAATCAAACGGGACCGGGACTGATAAGTATAAAATAGGTTTCCCTGTCCTCTACCTACGTTATCCTCGCGTCGATTTTGATTCTTACCAAGACTGCTAATCAGGCCCTGTGGCCTGCATGTCACCATGTCAGCGTGTTTGGCTAAATTCACGGGATTGGCCTTACCGACTTACATCAGTATTTCATACATAGTTACTCGAGTTTAACGTTGACAGTTAGTCCCATGATACGGCAAAGCCTGGTTCGGCGGATTTCCGAGTACAGCATCTTCGCCCCCGAGATTGCCGCCAATGGACACCCTCCTGAGATGCAGATATGAGTGTTTTTGACACTCTGAGGCTGAGATCCTCACACTTCCGGAGCTTCCGCGATAGTCACGTGGTTATTAGACTTACGGCAGGAAAAATCATGTTA', alphabet='dna')
sites_b = seq_to_partials('AAGCTCCGCGTAAGCTAACGACCAGTCAGCTAGGTTTAGTGCCACCAGTATGGCTAGTTCCGGAGGGCAAACCGGATGCTACCGATTGGTCACCCTCAGGGTGATTTCGCAGGGCGCTCACTTATTCCTTTTAAATCCTGCCAACAGACTAAGAAAGTTGTACGGTATTCCTATATCTTCAGTACTGCTCTTGGCCGTGCATGTAGCCGAACGACGAGGACGGTACATGAGTTTCTCACCAATTACAGGCGGTTCCATTAGGCAGTAGCTGCGGTTAGTTCATACTGCTAAAGAATCTTCTTGGAACGTGCCAAGGACCAGTCACACACATGTTGTAGTCCCTCATCGTGGTAGGCGTTCCAGACCGTCCGTGGTACACATACCAAATTTCGTACCGGCTGACTCAAAGCGGGAGTTCGCATGATACCAGGGAACGAGATGTTCAAAACGATCAGGTAGTGCCGCCATCTTTCAGGTTCTTTCGTTTCGTCCTATGATACTTGAGTAGCGGTCAAACGAAGCTCGTAGGTGACAGTTACGAGACATGCTGGGATGCAACATACTTTCGCAGTTAGCTAGTAGGTACCTATCTAGCGAATCGAGCTAGGATACCCTGATTATGCTTGTCTCCGTCCTCTTACTATGATCTCCTCGCGTGGTTTTTGCTGCTTAACCGTTGTGCCGTATAAAACAAGAGGCGGGAGTTTAGCTGTGGGAACTTCGTAGACCTTGTAAGCTGGATAGGCCCGTCCGTCGTAATTAATTACCTAAAAGAGAGTCAAACAAGCTTAAGTCGCCGAGTTAGTCGGATAAGAAGCCATTCTCTGGTCCGCCAACCTTCCCATGCCAGTACGGTTGCCGAGGTCCATTCGGTGACTGTGGGATAACCGTTGCCGGAGCTATGAGATCCATTACAACTCTGCGCCTAGGATGTTAACTCTACCGAAGTTTGCGACCCCGGAACCTGTAAATTGTCCTTAGGGTCGTAACATTTTCAAGC', alphabet='dna')

optimise(k80, sites_a, sites_b)

root = LnlNode(k80)
root.set_partials(sites_a)
leaf = Leaf(sites_b)
# newton_optimise(root, leaf, tolerance=1e-6)
brent_optimise(root, leaf)
# res=scipy_optimise(root, leaf, method='l-bfgs-b')

# newton1d_optimise(root, leaf)

############################################################################
# Example from Section 4.2 of Ziheng's book - his value for node 6 is wrong!
np.set_printoptions(precision=8)
kappa = 2
k80 = K80(kappa)

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

t = '((1:0.2,2:0.2):0.1,3:0.2,(4:0.2,5:0.2):0.2);'
tree = dpy.Tree.get(schema='newick', data=t)
tree.resolve_polytomies()
# runner = LnlModel(k80, partials_dict)
# runner.set_tree(tree)
# print(runner.run(True))
# print(runner.get_sitewise_likelihoods())
#
# gamma = GammaMixture(400, 4)
# gamma.init_models(k80, partials_dict)
# gamma.set_tree(t)
# gamma.run()
# print(gamma.get_likelihood())
# print(gamma.get_sitewise_likelihoods())
#
kappa = 1
k80 = K80(kappa)

partials_dict = {'1': seq_to_partials('ACCCT'),
                 '2': seq_to_partials('TCCCT'),
                 '3': seq_to_partials('TCGGT'),
                 '4': seq_to_partials('ACCCA'),
                 '5': seq_to_partials('CCCCC')}

gamma = GammaMixture(0.03, 4)
gamma.init_models(k80, partials_dict)
gamma.set_tree(t)
gamma.run()
print(gamma.get_likelihood())
print(gamma.get_scale_bufs())
print(gamma.get_sitewise_likelihoods())
print(gamma.get_sitewise_likelihoods().sum(0))

# gamma.update_alpha(1.0)
# gamma.run()
# print(gamma.get_likelihood())
# print(gamma.get_sitewise_likelihoods())
# print(gamma.get_sitewise_likelihoods().sum(0))
#
# gamma.update_substitution_model(K80(3))
# gamma.run()
# print(gamma.get_likelihood())
# print(gamma.get_sitewise_likelihoods())
# print(gamma.get_sitewise_likelihoods().sum(0))
#
# t = '((a:10, b:10.):10., c:10.);'
# partials_dict = {'a': seq_to_partials('A'),
#                  'b': seq_to_partials('T'),
#                  'c': seq_to_partials('G')}
#
# tree = dpy.Tree.get(schema='newick', data=t)
# tree.resolve_polytomies()
# gamma = GammaMixture(0.03, 4)
# gamma.init_models(k80, partials_dict)
# gamma.set_tree(t)
# gamma.run()
# print(gamma.get_scale_bufs())
# print(gamma.get_likelihood())
# print(gamma.get_sitewise_likelihoods())

