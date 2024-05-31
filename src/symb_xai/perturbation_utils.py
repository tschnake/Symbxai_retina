import numpy as np
import torch
from tqdm import tqdm

# from symb_xai.lrp.symbolic_xai import BERTSymbXAI


def get_node_ordering(explainer,
                    attribution_method,
                    auc_task,
                    perturbation_type,
                    verbose=False):
# model, sample, tokenizer,

    # We implement not the exaustive search, but the local best guess search
    def local_best_guess_search(explainer, set_attribution_fct, auc_task):
        node_heat = np.zeros(explainer.num_nodes)
        growing_node_set = []

        # We want to see the development of the iteration when verbose is True
        if verbose:
            loop_wrapper = lambda iter_range: tqdm(iter_range, desc=f'{attribution_method}-{perturbation_type}-{auc_task}')
        else:
            loop_wrapper = lambda iter_range: iter_range

        for synth_heat in loop_wrapper(range(explainer.num_nodes, 0, -1)):
            # Test which node in this iteration is the most promising
            mask_val = -float('inf') if auc_task == 'maximize' else float('inf')
            local_node_heat = [ set_attribution_fct( growing_node_set + [node_id] ) if node_id not in growing_node_set else mask_val for node_id in explainer.node_domain  ]
            if auc_task == 'minimize':
                winning_node_id = np.argmin(local_node_heat)
            elif auc_task == 'maximize':
                winning_node_id = np.argmax(local_node_heat)
            else:
                raise NotImplementedError

            node_heat[winning_node_id] = synth_heat # This is a synthetic heat, so it's just the ordering we want to have
            growing_node_set.append(winning_node_id)

        return node_heat

    # Generate node heat
    if attribution_method == 'random':
        node_heat = np.random.rand(len(explainer.node_domain))

    elif attribution_method == 'LRP':
        node_heat = explainer.node_relevance()

    elif attribution_method == 'SHAP':
        ...
    elif attribution_method == 'PredDiff':
        node_heat = []
        for node_id in explainer.node_domain:
            curr_rel = explainer.subgraph_relevance(explainer.node_domain) - explainer.subgraph_relevance([ ids for ids in explainer.node_domain if ids != node_id] )
            node_heat.append(curr_rel)
        node_heat = np.array(node_heat)

    elif attribution_method == 'SymbXAI':
        if perturbation_type == 'removal':
            set_attribution_fct = lambda S: explainer.subgraph_relevance( [ idn for idn in explainer.node_domain if idn not in S ])
        elif perturbation_type == 'generation':
            set_attribution_fct = lambda S: explainer.subgraph_relevance(S)
        else:
            raise NotImplementedError

        node_ordering = np.argsort(-local_best_guess_search(explainer, set_attribution_fct, auc_task))

    else:
        raise NotImplementedError

    if attribution_method in ['random', 'LRP', 'PredDiff', 'SHAP']: # all first order methods just generate node heat
        if perturbation_type == 'removal' and auc_task == 'minimize':
            node_ordering = np.argsort( -node_heat)
        elif perturbation_type == 'removal' and auc_task == 'maximize':
            node_ordering = np.argsort( node_heat)
        elif perturbation_type == 'generation' and auc_task == 'minimize':
            node_ordering = np.argsort( node_heat)
        elif perturbation_type == 'generation' and auc_task == 'maximize':
            node_ordering = np.argsort( -node_heat)
        else:
            raise NotImplementedError

    return node_ordering
