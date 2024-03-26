import time, numpy, torch, mpmath
# import numpy
from itertools import pairwise
from tqdm import tqdm
from symb_xai.model.transformer import bert_base_uncased_model
from symb_xai.lrp.symbolic_xai import BERTSymbXAI
from symb_xai.utils import powerset, Query
from random import shuffle
from transformers import BertTokenizer

def approx_query_search(explainer,
                        tokens,
                        maxorder,
                        norm_mode='occlusion',
                        maxvalsnum=1,
                        minvalsnum=0,
                        verbose=True):
    attr = {}
    outvals = {}

    if verbose: start_t = time.time()
    for opt_fct in [max]*maxvalsnum + [min]*minvalsnum:
        all_vals = {f'{order}-order': {} for o in range(1,maxorder +1)}
        fixed_feats = ()
        for order in range(1,maxorder+1):
            if norm_mode == 'significance':
                eta = 2^( order - len(tokens) )
            elif norm_mode == 'occlusion':
                eta = 1
            else:
                raise Exception(f'{norm_mode} is not implemented at the moment')

            for I in range(1, len(tokens) -1):
                if I in fixed_feats: continue
                curr_feats = fixed_feats + (I,)
                # check repetition of previous searches:
                rep_bool = [(set(curr_feats) == set(prev_feats)) for prev_feats in outvals.keys()]
                if any(rep_bool): continue

                all_vals[f'{order}-order'][curr_feats] = explainer.symb_and(list(curr_feats))

            odict = all_vals[f'{order}-order']
            outvals[opt_fct(odict, key=odict.get)] = opt_fct(odict.values())
            fixed_feats = opt_fct(odict, key=odict.get)

    if verbose:
        ftime = time.time()-start_t
        print(f'calculation took {round(ftime, 4)} sec.')
        return outvals, all_vals, ftime
    else:
        return outvals

def exhaustive_query_search(
                        explainer,
                        tokens,
                        maxorder,
                        norm_mode='occlusion',
                        verbose=True):
    all_vals = {}
    outvals = {}
    if verbose: start_t = time.time()

    for order in range(1,maxorder +1):
        # Make the feat-grid:
        grid = [numpy.arange(0, len(tokens))]*(order)
        grid = numpy.meshgrid(*grid)
        grid = [I.flatten() for I in grid]
        all_feats = [feats for feats in zip(*grid) if all(a < b for a, b in pairwise(feats))]
        # print(all_feats)
        if norm_mode == 'significance-2':
            eta = 2^( order - len(tokens) )
        elif norm_mode == 'occlusion':
            eta = 1
        else:
            raise Exception(f'{norm_mode} is not implemented at the moment')

        for curr_feats in tqdm(all_feats):
            all_vals[curr_feats] = eta * explainer.symb_and(list(curr_feats))

    outvals[max(all_vals, key=all_vals.get)] = max(all_vals.values())

    if verbose :
        ftime = time.time() - start_t
        print(f'calculation took {round(ftime, 4)} sec.')
        return outvals, all_vals, ftime
    else:
        return outvals

def weight_query_attr_directly(all_exh_outs, modes):
    all_outs = {mode: {} for mode in modes}
    for mode in modes:
        if mode == 'occlusion':
            alpha = 0.
        elif 'significance' in mode:
            alpha = float(mode.split('-')[1])
        else:
            raise NotImplementedError(f'The weight mode {mode} is not implemented.')

        for key, val in all_exh_outs.items():
            order=len(key)
            all_outs[mode][key] = mpmath.power(2, alpha*(order - 4)) * float(val)
    return all_outs

def weight_query_attr_harsanyi(all_queries,har_div, modes):
    query_attr = {mode: {} for mode in modes}
    for weight_mode in modes:
        for q in tqdm(all_queries, desc=f'Query Attribution of {weight_mode}.'):
            if weight_mode == 'occlusion':
                query_attr[weight_mode][q.hash] = sum([ val for S, val in har_div.items() if q(S) ])

            elif 'shapley':
                attr = 0
                for S, val in har_div.items():
                    weight = sum([int(query(S)) for query in all_queries ])
                    attr += val/weight
                query_attr[weight_mode][q.hash] = attr

            else:
                raise NotImplementedError(f'Weight mode {weight_mode} is not implemented yet.')

    return query_attr

def comp_all_harsanyi_sst(explainer, harsanyi_maxorder=5):

    all_feats = explainer.node_domain
    power_feats = powerset(all_feats, K=harsanyi_maxorder)
    shuffle(power_feats) # only to see better in the tqdm timeline how long it takes.

    har_div = {}

    for S in tqdm(power_feats, desc='Harsanyi Dividends'):
        har_div[tuple(S)] = explainer.harsanyi_div(S)

    return har_div

def setup_queries(  feat_domain,
                    max_and_order,
                    max_setsize,
                    max_indexdist=1,
                    mode='set conjuction'):

    assert mode == 'set conjuction', f'The mode {mode} is not implemented yet.'
    assert sorted(feat_domain) == feat_domain, 'has to be sorted'

    all_sets = powerset(feat_domain , K=max_setsize)

    def check_pairwise_dist(elem_list, max_indexdist):
        if len(elem_list) == 1:
            return True
        else:
            return all([ abs( elem_list[i+1] - elem_list[i]) <= max_indexdist for i in range(len(elem_list)-1) ])

    all_sets = [ fset for fset  in all_sets if check_pairwise_dist(fset, max_indexdist) ]

    query_fct = {}
    query_attr = {}
    # query_str = {}
    queries = []
    for order in range(1,max_and_order +1):
        # Make the feat-grid:
        grid = [numpy.arange(0, len(all_sets))]*(order)
        grid = numpy.meshgrid(*grid)
        grid = [I.flatten() for I in grid]
        all_psets_comb_ids = [feats for feats in zip(*grid) if all(a < b for a, b in pairwise(feats))] # all powerset combination identifiers

        for pset_comb_ids in all_psets_comb_ids:
            # specify the query

            curr_sets = tuple([tuple(all_sets[ids]) for ids in pset_comb_ids])
            bool_q = lambda L, sets=curr_sets : all([ any( [I in L for I in subset ]) for subset in sets ])
            q = Query(bool_fct=bool_q, hash_rep=curr_sets)
            queries.append(q)

            # query_fct[curr_sets] = q

    return queries
