

def symb_xai(rels, feats, mode='subset'):
    assert mode in ['subset', 'and', 'or', 'not'], f'Mode "{mode}" is not implemented.'
    r = 0.
    for w, rel in rels.items():
        if mode == 'subset' and all([token in feats for token in w ]):
            r += rel
        elif mode == 'and' and all([ token in w for token in feats ]):
            r += rel
        elif mode == 'or' and any([ token in w for token in feats]):
            r += rel
        elif mode == 'not' and all([ token not in w for token in feats]):
            r += rel
    return r
