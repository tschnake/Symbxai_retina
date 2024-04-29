from itertools import chain, combinations, pairwise
import numpy
# from itertools import pairwise


def powerset(s, K=float('inf'), with_empty=False):
    '''powerset([1,2,3]) --> [[1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]
    if the sets have not more than K elements in it.'''

    pset = chain.from_iterable( combinations(s, r) for r in range(len(s)+1))
    if not with_empty:
        pset =  [list(elem) for elem in pset if list(elem) != []] # no empty set
    pset = [list(elem) for elem in pset if len(elem)<= K ] # No set bigger that K

    return  pset

class Query():
    def __init__(self, bool_fct= None, str_rep=None, hash_rep=None, nb_feats=None):
        self.bool_fct = bool_fct
        self.str_rep  = str_rep
        self.hash = hash_rep
        self.nb_feats = nb_feats
        # maybe explicitly giving the subsets for which this query is true? This speeds up the attribution
        self.support = None


    def __call__(self, feat_set: tuple[int]):
        # if self.bool_fct is not None:
        #     return self.bool_fct(feat_set)
        # elif type(self.hash[0]) == frozenset:
            # We can work with this hash type:
        return all([ any( [(I in feat_set) if I >= 0 else (I + self.nb_feats not in feat_set) for I in subset ]) for subset in self.hash ])
        # else:
        #     raise NotImplementedError

    def get_support(self):
        if self.support is not None:
            return self.support
        else:
            return False

    def set_support(self, sets, node_domain=None, max_order=5 ):
        self.support = sets
        # elif type(self.hash) == frozenset and type(self.hash[0]) == frozenset:
        #     return None
        #     self.support = []
        #     grid = numpy.meshgrid(*self.hash)
        #     grid = [I.flatten() for I in grid]
        #     pset_combis = [feats for feats in zip(*grid) if all(a < b for a, b in pairwise(feats))]
        #
        #     for feat_seq in pset_combis:
        #         all_context = powerset([index for index in node_domain if index not in feat_seq],
        #                                 K=max_order - len(feat_seq),
        #                                 with_empty=True)
        #
        #         for context_feats in all_context:
        #             new_set = feat_seq + tuple(context_feats)
        #             new_set = tuple(sorted(new_set))
        #             self.support.append(new_set)


class FeatSet():
    def __init__(self,sets):
        self.sets = sets
        self.shap_weight = None
    def get(self):
        return sets
    def set_shap_weight(self, value):
        self.shap_weight = value
    def get_shap_weight(self):
        return self.shap_weight
