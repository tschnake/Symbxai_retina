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

def comp_all_harsanyi_sst(explainer, harsanyi_maxorder=5, do_shuffling =False):

    all_feats = explainer.node_domain
    power_feats = powerset(all_feats, K=harsanyi_maxorder)
    if do_shuffling: shuffle(power_feats) # only to see better in the tqdm timeline how long it takes.

    har_div = {}

    for S in tqdm(power_feats, desc='Harsanyi Dividends'):
        har_div[tuple(S)] = explainer.harsanyi_div(S)

    return har_div

def setup_queries(  feat_domain,
                    max_and_order,
                    max_setsize= float('inf'),
                    max_indexdist=1,
                    mode='conj. disj. reasonably mixed',
                    neg_tokens=None):

    def check_pairwise_dist(elem_list, max_indexdist):
        if len(elem_list) == 1:
            return True
        else:
            return all([ abs( elem_list[i+1] - elem_list[i]) <= max_indexdist for i in range(len(elem_list)-1) ])

    assert sorted(feat_domain) == feat_domain, 'has to be sorted'
    # make sure max_setsize is properly parsed.
    if max_setsize is None or max_setsize < 0:
        max_setsize= float('inf')

    # neglect tokens we don't want. usually the [cls] and [sep] token in the transormer models
    if neg_tokens is not None:
        feat_domain = [feat for feat in feat_domain if feat not in neg_tokens]

    all_sets = powerset(feat_domain , K=max_setsize)
    all_sets = [ fset for fset  in all_sets if check_pairwise_dist(fset, max_indexdist) ]

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
            if mode == 'set conjuction':
                ...
            elif mode == 'conj. disj. reasonably mixed':
                # Check if sets are pairwise disjoint
                if sum(map(len, curr_sets)) != len(set().union(*curr_sets)): continue
                else: ...

            else:
                raise NotImplementedError(f'The mode {mode} is not implemented yet.')

            # bool_q = lambda L, sets=curr_sets : all([ any( [I in L for I in subset ]) for subset in sets ])
            # def bool_q(L, sets=curr_sets):
                # return all([ any( [I in L for I in subset ]) for subset in sets ])

            # q = Query(bool_fct=bool_q, hash_rep=curr_sets)
            q = Query(hash_rep=curr_sets)
            queries.append(q)

    return queries

def calc_corr(arg):
    ''' helping function for the parallelization'''
    def m(x, w):
        """Weighted Mean"""
        return numpy.sum(x * w) / numpy.sum(w)

    def cov(x, y, w):
        """Weighted Covariance"""
        return numpy.sum(w * (x - m(x, w)) * (y - m(y, w))) / numpy.sum(w)

    def corr(x, y, w):
        """Weighted Correlation"""
        return cov(x, y, w) / numpy.sqrt(cov(x, x, w) * cov(y, y, w))

    q, hars_div, weight_mode, all_queries = arg
    supp = tuple([S for S in hars_div.keys() if q(S)])
    q_vec, val_vec, weight_vec = map(numpy.array, zip(*[(int(q(S)), val, 1) for S, val in hars_div.items() ]))

    q.attribution = corr(q_vec, val_vec, weight_vec)
    q.set_support(supp)

    return q

def calc_attr_supp(arg):
    ''' helping function for the parallelization'''
    q, hars_div, weight_mode, all_queries = arg
    attr = 0
    supp = ()
    for S, val in hars_div.items():
        if q(S):
            if weight_mode == 'occlusion':
                attr += val
            elif weight_mode == 'shapley':
                weight = sum([int(query(S)) for query in all_queries ])
                attr += val/weight
            elif weight_mode == 'occlusion error':
                ...
            else:
                raise NotImplementedError(f'Weight mode {weight_mode} is not implemented yet.')
            supp += (S,)

        elif not q(S) and weight_mode == 'occlusion error':
            attr += val**2

    q.set_support(supp)
    q.attribution = attr
    return q
