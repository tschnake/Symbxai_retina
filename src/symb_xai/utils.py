from itertools import chain, combinations

def powerset(s):
    "powerset([1,2,3]) --> [[1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]"
    out = chain.from_iterable( combinations(s, r) for r in range(len(s)+1))
    return [list(elem) for elem in out if list(elem) != []] # no empty set
