from enum import IntEnum


class Mode(IntEnum):
    SUBSET = 0
    AND = 1
    OR = 2
    NOT = 3


# -------------------
def run_query(rels, subgraph, mode=0):
    modes = [mode for mode in Mode]
    assert mode in modes, f'Mode "{mode}" is not implemented.'

    r = None
    if mode == Mode.SUBSET:
        r = subset_query(rels, subgraph)
    elif mode == Mode.AND:
        r = and_query(rels, subgraph)
    elif mode == Mode.OR:
        r = or_query(rels, subgraph)
    elif mode == Mode.NOT:
        r = not_query(rels, subgraph)

    return r


# Queries.
# -------------------
def and_query(rels, subgraph):
    r = 0.
    for w, rel in rels.items():
        if all([token in w for token in subgraph]):
            r += rel
    return r


# -------------------
def or_query(rels, subgraph):
    r = 0.
    for w, rel in rels.items():
        if any([token in w for token in subgraph]):
            r += rel
    return r


# -------------------
def subset_query(rels, subgraph):
    r = 0.
    for w, rel in rels.items():
        if all([token in subgraph for token in w]):
            r += rel
    return r


# -------------------
def not_query(rels, subgraph):
    r = 0.
    for w, rel in rels.items():
        if all([token not in w for token in subgraph]):
            r += rel
    return r
