from itertools import chain, combinations

def powerset(s, K=float('inf')):
    '''powerset([1,2,3]) --> [[1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]
    if the sets have not more than K elements in it.'''

    pset = chain.from_iterable( combinations(s, r) for r in range(len(s)+1))
    pset =  [list(elem) for elem in pset if list(elem) != []] # no empty set
    pset = [elem for elem in pset if len(elem)<= K ] # No set bigger that K

    return  pset

class Query():
    def __init__(self, bool_fct= None, str_rep=None, hash_rep=None):
        self.bool_fct = bool_fct
        self.str_rep  = str_rep
        self.hash = hash_rep
        # maybe explicitly giving the subsets for which this query is true? This speeds up the attribution

    def __call__(self, feat_set: tuple[int]):
        return self.bool_fct(feat_set)
