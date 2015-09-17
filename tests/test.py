from phylo_utils.seq_to_partials import dna_charmap, protein_charmap, seq_to_partials
from phylo_utils.models import K80
from phylo_utils.markov import TransitionMatrix
from phylo_utils.likelihood import optimise, GammaMixture, RunOnTree
import numpy as np
#####################################
# Pairwise distance optimiser demo:

kappa = 1
k80 = K80(kappa)
tm = TransitionMatrix(k80)
# Simulated data from K80, kappa=1, distance = 0.8
sites_a = seq_to_partials('ACCCTCCGCGTTGGGTAGTCCTAGGCCCAATGGCGTTTATGCCTCGATTTTTAGTTCTACCGTCCCTACAGATGGATGCCGTCGCATAGACACTGTCAATTCCATTCGGCAGGCTTCACACTGTTGCATTTTCATTTTGTACACGGTACCAACATAGGAGTGCTGTATTGCTATATTTCCAGTACACGGCGTTGAGTCGGATGGAAACGCCGGCGGAAGACAGCTTGGCGGGTCTTCACGCATCACCGCGGGGTCTGAAAGGTATTATCGCTGCTTAAATCAGACCGGTCAAGCTTCCTGGCGGAAGGCGGCAAGGTCCAGCCACAGCATGCTTATTCCTTGTCACGCCGGGTGGAAATCTAGAGCGTCCGGTGGACACAGAGTGATTTTGTACGGGGGGTTCCATACCAGGACATTAGGGTCGGTTTACGGTCTGAGATGTATGTTGCCTTGCGGTCGACGAGCACTGATTCCCCTGAACTTCGTAAGACACATATAGTTTTAATGAAATCCCCAAAACGAGCATGGTTTCAGTATACGCGACAACTTAGGATACAACATACTGAACCAGTCCGCATTGAGGTGCCAATCAAACGGGACCGGGACTGATAAGTATAAAATAGGTTTCCCTGTCCTCTACCTACGTTATCCTCGCGTCGATTTTGATTCTTACCAAGACTGCTAATCAGGCCCTGTGGCCTGCATGTCACCATGTCAGCGTGTTTGGCTAAATTCACGGGATTGGCCTTACCGACTTACATCAGTATTTCATACATAGTTACTCGAGTTTAACGTTGACAGTTAGTCCCATGATACGGCAAAGCCTGGTTCGGCGGATTTCCGAGTACAGCATCTTCGCCCCCGAGATTGCCGCCAATGGACACCCTCCTGAGATGCAGATATGAGTGTTTTTGACACTCTGAGGCTGAGATCCTCACACTTCCGGAGCTTCCGCGATAGTCACGTGGTTATTAGACTTACGGCAGGAAAAATCATGTTA', alphabet='dna')
sites_b = seq_to_partials('AAGCTCCGCGTAAGCTAACGACCAGTCAGCTAGGTTTAGTGCCACCAGTATGGCTAGTTCCGGAGGGCAAACCGGATGCTACCGATTGGTCACCCTCAGGGTGATTTCGCAGGGCGCTCACTTATTCCTTTTAAATCCTGCCAACAGACTAAGAAAGTTGTACGGTATTCCTATATCTTCAGTACTGCTCTTGGCCGTGCATGTAGCCGAACGACGAGGACGGTACATGAGTTTCTCACCAATTACAGGCGGTTCCATTAGGCAGTAGCTGCGGTTAGTTCATACTGCTAAAGAATCTTCTTGGAACGTGCCAAGGACCAGTCACACACATGTTGTAGTCCCTCATCGTGGTAGGCGTTCCAGACCGTCCGTGGTACACATACCAAATTTCGTACCGGCTGACTCAAAGCGGGAGTTCGCATGATACCAGGGAACGAGATGTTCAAAACGATCAGGTAGTGCCGCCATCTTTCAGGTTCTTTCGTTTCGTCCTATGATACTTGAGTAGCGGTCAAACGAAGCTCGTAGGTGACAGTTACGAGACATGCTGGGATGCAACATACTTTCGCAGTTAGCTAGTAGGTACCTATCTAGCGAATCGAGCTAGGATACCCTGATTATGCTTGTCTCCGTCCTCTTACTATGATCTCCTCGCGTGGTTTTTGCTGCTTAACCGTTGTGCCGTATAAAACAAGAGGCGGGAGTTTAGCTGTGGGAACTTCGTAGACCTTGTAAGCTGGATAGGCCCGTCCGTCGTAATTAATTACCTAAAAGAGAGTCAAACAAGCTTAAGTCGCCGAGTTAGTCGGATAAGAAGCCATTCTCTGGTCCGCCAACCTTCCCATGCCAGTACGGTTGCCGAGGTCCATTCGGTGACTGTGGGATAACCGTTGCCGGAGCTATGAGATCCATTACAACTCTGCGCCTAGGATGTTAACTCTACCGAAGTTTGCGACCCCGGAACCTGTAAATTGTCCTTAGGGTCGTAACATTTTCAAGC', alphabet='dna')

optimise(tm, sites_a, sites_b)

############################################################################
# Example from Section 4.2 of Ziheng's book - his value for node 6 is wrong!
np.set_printoptions(precision=6)
kappa = 2
k80 = K80(kappa)
tm = TransitionMatrix(k80)

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
runner = RunOnTree(tm, partials_dict)
runner.set_tree(t)
print runner.run(True)
print runner.get_sitewise_likelihoods()

gamma = GammaMixture(400, 4)
gamma.init_models(tm, partials_dict, scale_freq=3)
gamma.set_tree(t)
print gamma.get_likelihood()
print gamma.get_sitewise_likelihoods()

kappa = 2
k80 = K80(kappa)
tm = TransitionMatrix(k80)

partials_dict = {'1': seq_to_partials('ACCCT'),
                 '2': seq_to_partials('TCCCT'),
                 '3': seq_to_partials('TCGGT'),
                 '4': seq_to_partials('ACCCA'),
                 '5': seq_to_partials('CCCCC')}

gamma = GammaMixture(.03, 4)
gamma.init_models(tm, partials_dict, scale_freq=200)
gamma.set_tree(t)
print gamma.get_likelihood()
print gamma.get_sitewise_likelihoods()
print gamma.get_sitewise_likelihoods().sum(0)

gamma.update_alpha(1.0)
print gamma.get_likelihood()
print gamma.get_sitewise_likelihoods()
print gamma.get_sitewise_likelihoods().sum(0)

gamma.update_transition_matrix(TransitionMatrix(K80(3)))
print gamma.get_likelihood()
print gamma.get_sitewise_likelihoods()
print gamma.get_sitewise_likelihoods().sum(0)
