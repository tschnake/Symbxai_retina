
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

def create_text_heat_map(text, heat,  scal_val=1., font_size=20, break_at=6, scale_by_max=False):
    assert len(text) == len(heat), 'Sorry, the text and heat values needs to have same length.'

    if scale_by_max:
        scal_val /= max([abs(float(heat[i].sum())) for i in range(heat.shape[0])])

    # Adjust the heat:
    heat = [scal_val*h for h in heat]

    # clipping
    heat = [-1. if h< -1. else h for h in heat]
    heat = [1. if h > 1. else h for h in heat ]

    output = '<p><center>'

    for i in range(len(text)):

        if text[i] not in [',', '.', ';', ':', "n't", "'s"]:
            output += ' '
        if text[i] in ["n't"]:
            output +=  '<span style="font-size:30%;">&nbsp;</span>' # small space
        if heat[i] <= 0:
            output += '<text style="background-color:rgba(0, 0, 255,{});font-weight: bold; font-size:{}px; font-family: Courier;">{}</text>'.format(
                -heat[i],
                font_size,
                text[i]
            )
        else:
            output += '<text style="background-color:rgba(255, 0, 0,{});font-weight: bold;font-size:{}px; font-family: Courier;">{}</text>'.format(
                heat[i],
                font_size,
                text[i]
            )

        if i % break_at == 0 and i != 0:
            output+='<br>'

    output += '</center></p>'

    return output

def make_color(rel, scaling=1.):
    # scaling
    rel*= scaling
    # clipping
    if rel > 1:
        rel = 1.
    if rel<-1:
        rel = -1.

    if rel<0: # blue
        color = ( 1.-abs(rel), 1.-abs(rel), 1. )
    else: # red
        color = (1,  1.-abs(rel), 1.-abs(rel) )

    return color
