from __future__ import print_function
from phylo_utils.seq_to_partials import dna_charmap, protein_charmap, seq_to_partials
from phylo_utils.models import K80, WAG, LG
from phylo_utils.likelihood import optimise, GammaMixture, LnlModel
import numpy as np
import dendropy as dpy

#####################################
# Pairwise distance optimiser demo:

kappa = 0.7345
k80 = K80(kappa)
# Simulated data from K80, kappa=1, distance = 0.8
sites_a = seq_to_partials('ACCCTCCGCGTTGGGTAGTCCTAGGCCCAATGGCGTTTATGCCTCGATTTTTAGTTCTACCGTCCCTACAGATGGATGCCGTCGCATAGACACTGTCAATTCCATTCGGCAGGCTTCACACTGTTGCATTTTCATTTTGTACACGGTACCAACATAGGAGTGCTGTATTGCTATATTTCCAGTACACGGCGTTGAGTCGGATGGAAACGCCGGCGGAAGACAGCTTGGCGGGTCTTCACGCATCACCGCGGGGTCTGAAAGGTATTATCGCTGCTTAAATCAGACCGGTCAAGCTTCCTGGCGGAAGGCGGCAAGGTCCAGCCACAGCATGCTTATTCCTTGTCACGCCGGGTGGAAATCTAGAGCGTCCGGTGGACACAGAGTGATTTTGTACGGGGGGTTCCATACCAGGACATTAGGGTCGGTTTACGGTCTGAGATGTATGTTGCCTTGCGGTCGACGAGCACTGATTCCCCTGAACTTCGTAAGACACATATAGTTTTAATGAAATCCCCAAAACGAGCATGGTTTCAGTATACGCGACAACTTAGGATACAACATACTGAACCAGTCCGCATTGAGGTGCCAATCAAACGGGACCGGGACTGATAAGTATAAAATAGGTTTCCCTGTCCTCTACCTACGTTATCCTCGCGTCGATTTTGATTCTTACCAAGACTGCTAATCAGGCCCTGTGGCCTGCATGTCACCATGTCAGCGTGTTTGGCTAAATTCACGGGATTGGCCTTACCGACTTACATCAGTATTTCATACATAGTTACTCGAGTTTAACGTTGACAGTTAGTCCCATGATACGGCAAAGCCTGGTTCGGCGGATTTCCGAGTACAGCATCTTCGCCCCCGAGATTGCCGCCAATGGACACCCTCCTGAGATGCAGATATGAGTGTTTTTGACACTCTGAGGCTGAGATCCTCACACTTCCGGAGCTTCCGCGATAGTCACGTGGTTATTAGACTTACGGCAGGAAAAATCATGTTA', alphabet='dna')
sites_b = seq_to_partials('AAGCTCCGCGTAAGCTAACGACCAGTCAGCTAGGTTTAGTGCCACCAGTATGGCTAGTTCCGGAGGGCAAACCGGATGCTACCGATTGGTCACCCTCAGGGTGATTTCGCAGGGCGCTCACTTATTCCTTTTAAATCCTGCCAACAGACTAAGAAAGTTGTACGGTATTCCTATATCTTCAGTACTGCTCTTGGCCGTGCATGTAGCCGAACGACGAGGACGGTACATGAGTTTCTCACCAATTACAGGCGGTTCCATTAGGCAGTAGCTGCGGTTAGTTCATACTGCTAAAGAATCTTCTTGGAACGTGCCAAGGACCAGTCACACACATGTTGTAGTCCCTCATCGTGGTAGGCGTTCCAGACCGTCCGTGGTACACATACCAAATTTCGTACCGGCTGACTCAAAGCGGGAGTTCGCATGATACCAGGGAACGAGATGTTCAAAACGATCAGGTAGTGCCGCCATCTTTCAGGTTCTTTCGTTTCGTCCTATGATACTTGAGTAGCGGTCAAACGAAGCTCGTAGGTGACAGTTACGAGACATGCTGGGATGCAACATACTTTCGCAGTTAGCTAGTAGGTACCTATCTAGCGAATCGAGCTAGGATACCCTGATTATGCTTGTCTCCGTCCTCTTACTATGATCTCCTCGCGTGGTTTTTGCTGCTTAACCGTTGTGCCGTATAAAACAAGAGGCGGGAGTTTAGCTGTGGGAACTTCGTAGACCTTGTAAGCTGGATAGGCCCGTCCGTCGTAATTAATTACCTAAAAGAGAGTCAAACAAGCTTAAGTCGCCGAGTTAGTCGGATAAGAAGCCATTCTCTGGTCCGCCAACCTTCCCATGCCAGTACGGTTGCCGAGGTCCATTCGGTGACTGTGGGATAACCGTTGCCGGAGCTATGAGATCCATTACAACTCTGCGCCTAGGATGTTAACTCTACCGAAGTTTGCGACCCCGGAACCTGTAAATTGTCCTTAGGGTCGTAACATTTTCAAGC', alphabet='dna')

optimise(k80, sites_a, sites_b)

############################################################################
# Example from Section 4.2 of Ziheng's book - his value for node 6 is wrong!
np.set_printoptions(precision=6)
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
tree = dpy.Tree.get(schema='newick', data='(((1:0.2,2:0.2)7:0.1,3:0.2)6:0.1,(4:0.2,5:0.2)8:0.1)0;')
tree = dpy.Tree.get(schema='newick', data=t)
tree.resolve_polytomies()
runner = LnlModel(k80, partials_dict)
runner.set_tree(tree)
print(runner.run(True))
print(runner.get_sitewise_likelihoods())

gamma = GammaMixture(400, 4)
gamma.init_models(k80, partials_dict)
gamma.set_tree(t)
gamma.run()
print(gamma.get_likelihood())
print(gamma.get_sitewise_likelihoods())

kappa = 2
k80 = K80(kappa)

partials_dict = {'1': seq_to_partials('ACCCT'),
                 '2': seq_to_partials('TCCCT'),
                 '3': seq_to_partials('TCGGT'),
                 '4': seq_to_partials('ACCCA'),
                 '5': seq_to_partials('CCCCC')}

gamma = GammaMixture(.03, 4)
gamma.init_models(k80, partials_dict)
gamma.set_tree(t)
gamma.run()
print(gamma.get_likelihood())
print(gamma.get_sitewise_likelihoods())
print(gamma.get_sitewise_likelihoods().sum(0))

gamma.update_alpha(1.0)
gamma.run()
print(gamma.get_likelihood())
print(gamma.get_sitewise_likelihoods())
print(gamma.get_sitewise_likelihoods().sum(0))

gamma.update_substitution_model(K80(3))
gamma.run()
print(gamma.get_likelihood())
print(gamma.get_sitewise_likelihoods())
print(gamma.get_sitewise_likelihoods().sum(0))
