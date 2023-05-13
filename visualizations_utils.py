import matplotlib.pyplot as plt
import networkx as nx
import copy
import dgl
from networkx.drawing.nx_pydot import graphviz_layout
from xai_utils import make_color



def rescale_score_by_abs (score, max_score, min_score):
    """
    Normalize the relevance value (=score), accordingly to the extremal relevance values (max_score and min_score), 
    for visualization with a diverging colormap.
    i.e. rescale positive relevance to the range [0.5, 1.0], and negative relevance to the range [0.0, 0.5],
    using the highest absolute relevance for linear interpolation.
    """
    
    # CASE 1: positive AND negative scores occur --------------------
    if max_score>0 and min_score<0:
    
        if max_score >= abs(min_score):   # deepest color is positive
            if score>=0:
                return 0.5 + 0.5*(score/max_score)
            else:
                return 0.5 - 0.5*(abs(score)/max_score)

        else:                             # deepest color is negative
            if score>=0:
                return 0.5 + 0.5*(score/abs(min_score))
            else:
                return 0.5 - 0.5*(score/min_score)   
    
    # CASE 2: ONLY positive scores occur -----------------------------       
    elif max_score>0 and min_score>=0: 
        if max_score == min_score:
            return 1.0
        else:
            return 0.5 + 0.5*(score/max_score)
    
    # CASE 3: ONLY negative scores occur -----------------------------
    elif max_score<=0 and min_score<0: 
        if max_score == min_score:
            return 0.0
        else:
            return 0.5 - 0.5*(score/min_score)    
  
      
def getRGB (c_tuple):
    return "#%02x%02x%02x"%(int(c_tuple[0]*255), int(c_tuple[1]*255), int(c_tuple[2]*255))

     
def span_word (word, score, colormap):
    return "<span style=\"background-color:"+getRGB(colormap(score))+"\">"+word+"</span>"


def html_heatmap (words, scores, cmap_name="bwr"):
    """
    Return word-level heatmap in HTML format,
    with words being the list of words (as string),
    scores the corresponding list of word-level relevance values,
    and cmap_name the name of the matplotlib diverging colormap.
    """
    
    colormap  = plt.get_cmap(cmap_name)
     
    assert len(words)==len(scores)
    max_s     = max(scores)
    min_s     = min(scores)
    
    output_text = ""
    
    for idx, w in enumerate(words):
        score       = rescale_score_by_abs(scores[idx], max_s, min_s)
        output_text = output_text + span_word(w, score, colormap) + " "
    
    return output_text + "\n"

def make_text_string(lsent):
    sentence = ''
    for i, token in enumerate(lsent):
        if token not in [',', '.', ';', ':', "n't", "'s", "''", "'d", "'re", "'m'"] and i != 0 and sentence[-2:] != "``" :
            sentence += ' '
        sentence += token
        
    return sentence

def vis_tree_heat(tree, node_heat, vocab_words, node_labels=None, save_dir=None, word_dist=50, node_size=2000):
    plt.figure(figsize=[12,7])

    G = dgl.to_networkx(tree)
    G = G.to_undirected()
    pos = graphviz_layout(G, prog="dot")

    # nodes
    nx.draw_networkx_nodes(G, pos, 
                           node_color = node_heat,
                           edgecolors = 'black', 
                           node_size=node_size)

    # labels
    if node_labels is not None:
        nx.draw_networkx_labels(G, pos, 
                            labels =node_labels,
                           font_size=16)

    # edges
    nx.draw_networkx_edges(G,pos)
    
    # words
    mask = tree.ndata['mask']
    leaf_nodes = mask.nonzero().squeeze().numpy()
    pos_ = copy.copy(pos)
    
    for ln in leaf_nodes: pos_[ln] = (pos_[ln][0], pos_[ln][1] -word_dist)
    
    input_ids = tree.ndata['x']
    nx.draw_networkx_labels(G, pos_, 
                            labels = {ln: vocab_words[idw] for ln, idw in zip(leaf_nodes, input_ids[mask == 1])},
                           font_size=16)

    plt.axis('off')
    plt.tight_layout()
    plt.margins(y=.1, x=.0)
    if save_dir is not None:
        plt.savefig(save_dir, transparent=True)
    plt.show()