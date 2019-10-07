import phylo_utils as phy
from phylo_utils.alignment.alignment import read_alignment
from phylo_utils.alignment import alphabets
import dendropy as dpy
import numpy as np

import argparse
import os
import sys

from pyparsing import Word, alphas, nums, Combine, Optional, Suppress, Group, delimitedList

def parse_cli():
    parser = argparse.ArgumentParser('phy - calculate likelihood of an alignment given a phylogenetic model')
    parser.add_argument('-t', '--tree', type=str, help='File path to a tree in newick format')
    parser.add_argument('-s', '--alignment', type=str, help='File path to a tree in fasta format')
    parser.add_argument('-m', '--model', type=str, help='Model specification string',
                        default='JC')
    return parser.parse_args()


def validate_args(args):
    if args.tree is None:
        raise ValueError('Tree file not specified')
    if args.alignment is None:
        raise ValueError('Alignment file not specified')
    if not os.path.exists(args.tree):
        raise FileNotFoundError('Tree file {} does not exist'.format(args.tree))
    if not os.path.exists(args.alignment):
        raise FileNotFoundError('Alignment file {} does not exist'.format(args.alignment))


def read_tree(filename):
    try:
        return dpy.Tree.get_from_path(filename, schema='newick', preserve_underscores=True)
    except ValueError as e:
        sys.stderr.write('Dendropy returned an error: {}\n'.format(e))
        sys.exit(1)


def parse_model_string(model_string):
    Model = Word(alphas).setResultsName('subs_model')
    Real = Combine(Word(nums) + '.' + Word(nums))
    CommaSepReals = Group(delimitedList(Real, delim=','))
    BraceParams = Optional(Suppress('{') + CommaSepReals + Suppress('}'))
    Freqs = Word('F', max=1)
    RateModel = Word('G', max=1).setResultsName('rate_model') + Word(nums).setResultsName('rate_cats')
    Plus = Suppress('+')

    parser = Model +\
        BraceParams.setResultsName('model_params') +\
        Optional(Plus + (Freqs + BraceParams.setResultsName('freq_params') |\
                         RateModel + BraceParams.setResultsName('rate_param'))) +\
        Optional(Plus + (Freqs + BraceParams.setResultsName('freq_params') |\
                         RateModel + BraceParams.setResultsName('rate_param')))
    return parser.parseString(model_string)


def look_up_model(model):
    models = {
        'GTR': phy.substitution_models.GTR,
        'HKY': phy.substitution_models.HKY85,
        'HKY85': phy.substitution_models.HKY85,
        'K80': phy.substitution_models.K80,
        'F81': phy.substitution_models.F81,
        'F84': phy.substitution_models.F84,
        'JC': phy.substitution_models.JC69,
        'JC69': phy.substitution_models.JC69,
        'TN93': phy.substitution_models.TN93,
        'LG': phy.substitution_models.LG,
        'WAG': phy.substitution_models.WAG,
        'JTT': phy.substitution_models.JTT,
        'Dayhoff': phy.substitution_models.Dayhoff
    }
    try:
        return models[model]
    except ValueError as e:
        sys.stderr.write('ERROR: Unrecognised model {}\n'.format(model))
        sys.stderr.write('Valid options are {}\n'.format(', '.join(models.keys())))
        sys.exit(1)


def unpack(param_vals, numtype=float):
    if param_vals is not None:
        vals = [numtype(val) for val in param_vals[0]]
        if len(vals) == 1:
            return vals[0]
        else:
            return vals
    else:
        return param_vals


if __name__ == '__main__':
    args = parse_cli()

    # Check files defined in cli exist
    try:
        validate_args(args)
    except ValueError as e:
        sys.stderr.write('ERROR: {}\n'.format(e))
        sys.exit(1)
    except FileNotFoundError as e:
        sys.stderr.write('ERROR: {}\n'.format(e))
        sys.exit(1)

    tree = read_tree(args.tree)
    aln = read_alignment(args.alignment, 'fasta')

    model_desc = parse_model_string(args.model)

    subs_model_name = model_desc['subs_model']
    if subs_model_name is None:
        sys.stderr.write('No substitution model specified\n')
        sys.exit(1)

    if subs_model_name in ["JTT", "Dayhoff", "WAG", "LG"]:
        alphabet = alphabets.PROTEIN
    else:
        alphabet = alphabets.DNA

    subs_model_class = look_up_model(subs_model_name)

    subs_params = unpack(model_desc.get('model_params'), float)

    freq_params = unpack(model_desc.get('freq_params'), float)

    subs_model = subs_model_class(rates=subs_params, freqs=freq_params)

    rate_model_name = model_desc.get('rate_model')
    if rate_model_name == 'G':
        ncat = unpack(model_desc.get('rate_cats'), int)
        alpha = unpack(model_desc.get('rate_param'), float)
        rate_model = phy.rate_models.GammaRateModel(ncat if ncat is not None else 4,
                                                    alpha if alpha is not None else 0.5)

    else:
        rate_model = phy.rate_models.UniformRateModel()

    tm = phy.tree_model.TreeModel()
    tm.set_tree(tree)
    tm.set_alignment(aln, alphabet)
    tm.set_rate_model(rate_model)
    tm.set_substitution_model(subs_model)
    tm.initialise()
    print('lnL = {}'.format(tm.compute_likelihood_at_edge(*tm.traversal.root_edge).sum()))

