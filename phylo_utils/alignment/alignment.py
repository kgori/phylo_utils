# Utilities for reading alignments and converting to numpy arrays

import numpy as np
from Bio import AlignIO
from Bio.Alphabet.IUPAC import IUPACAmbiguousDNA
from functools import reduce

from phylo_utils.alignment.charmaps import dna_charmap, protein_charmap, binary_charmap
from phylo_utils.alignment.alphabets import DNA, PROTEIN, BINARY
from phylo_utils.utils import *


logger = setup_logger()

def read_alignment(filename, format, alphabet=None):
    alignment = next(AlignIO.parse(filename, format))
    return alignment

def sample_characters(alignment, seqlen=1000, nseq=100):
    charsample = set(''.join(str(seq.seq[:seqlen]) for seq in alignment[:nseq]))
    return charsample

def guess_alphabet(charsample):
    return PROTEIN if len(charsample - set(IUPACAmbiguousDNA.letters) - set('-')) > 0 else DNA

def seq_to_partials(seq, alphabet):
    if alphabet == DNA:
        charmap = dna_charmap
    elif alphabet == PROTEIN:
        charmap = protein_charmap
    elif alphabet == BINARY:
        charmap = binary_charmap
    else:
        logger.warn("Unrecognised alphabet. Trying DNA")
        charmap = DNA

    return np.ascontiguousarray([charmap[char] for char in seq])


def alignment_to_numpy(alignment, alphabet, compress=True):
    if tuple([int(v) for v in np.version.version.split('.')]) < (1, 13, 0):
        compress = False
    _alignment = np.stack(
        [seq_to_partials(str(seq.seq), alphabet) for seq in alignment])
    names = {seq.name: i for i, seq in enumerate(alignment)}
    n_sites = _alignment.shape[1]
    if compress:
        alignment, inverse_index, siteweights = np.unique(_alignment,
                                                          return_inverse=True,
                                                          return_counts=True,
                                                          axis=1)
    else:
        alignment = _alignment
        siteweights = np.ones(n_sites, dtype=np.int)
        inverse_index = np.arange(n_sites)

    return alignment, siteweights, inverse_index, names

def invariant_sites(alignment):
    """
    Create an index array of the invariant sites in the alignment
    :param alignment: Alignment data in numpy array form
    :return: numpy boolean array where True indicates an invariant site
    """
    return [np.any(reduce(np.logical_and, alignment[:, i, :], np.ones(alignment.shape[2], )))
            for i in range(alignment.shape[1])]
