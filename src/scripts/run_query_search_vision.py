# Import packages
import click, ast, torch, time, transformers, numpy, pickle, multiprocessing
from copy import copy
from itertools import pairwise
from tqdm import tqdm
import numpy as np
import cv2
from copy import deepcopy

# ViT model
from symb_xai.lrp.symbolic_xai import ViTSymbolicXAI
from symb_xai.model.vision_transformer import ModifiedViTForImageClassification

# Dataset
from symb_xai.dataset.part_imagenet import PartImageNetDataset

from symb_xai.model.utils import get_masked_patch_ids
from symb_xai.query_search.utils import comp_all_harsanyi_sst, setup_queries,  calc_attr_supp, calc_corr, calc_weights, calc_cov
from symb_xai.utils import powerset
from utils import PythonLiteralOption
import transformers


# Get arguments
@click.command()
@click.option('--sample_range',
                cls=PythonLiteralOption,
                default='[16224]',
                help='List of sample ids in the dataset.')
@click.option('--max_and_order',
                type=int,
                default=1,
                help='Maximum order of conjunctive (AND) operation.')
@click.option('--logfolder',
                type=str,
                default='/home/farnoush//MyGiT/symbolicXAI/local_experiments/logs/query_auto_search/',
                help='Folder where we safe the logs.')
@click.option('--resultfolder',
                type=str,
                default='/home/farnoush//MyGiT/symbolicXAI/local_experiments/intermediate_results/query_search_algo/',
                help='Folder where we intermediate results are safed.')
@click.option('--datamode',
                type=str,
                default='part_imagenet',
                help='What dataset to load')
@click.option('--weight_mode',
              type=str,
              default="'occlusion'",
              help='A list of identifiers that describe how to weight the attributions')
@click.option('--harsanyi_maxorder',
              type=int,
              default=3,
              help='Specifies until what order the Harsanyi dividends will be computed.')
@click.option('--query_mode',
              type=str,
              default='conj. disj. (neg. disj.) reasonably mixed',
              help='What kind of queries will be attributed.')
@click.option('--max_setsize',
              type=int,
              default=3,
              help='Maximum size of feature sets in the queries. Similar to: the maximum order of the OR operation.')
@click.option('--nb_cores',
              type=int,
              default=1,
              help='Number of cores to use. If 1: do not use multiprocessing.')
@click.option('--attribution_mode',
              type=str,
              default='corr(q,f)',
              help='The type of mode to attribute. We have "attribution" and "corr(q,f)".')
@click.option('--load_harsanyi',
                is_flag=True,
                help='Flag to denote if we load the harsanyi dividends from data.')
@click.option('--neg_tokens_hars_bool',
                is_flag=True,
                help='Flag to denote if we neglect the unwanted tokens in the harsanyi dividends as well.')


def main(
    sample_range,
    max_and_order,
    logfolder,
    resultfolder,
    datamode,
    weight_mode,
    harsanyi_maxorder,
    query_mode,
    max_setsize,
    nb_cores,
    attribution_mode,
    load_harsanyi,
    neg_tokens_hars_bool):

    # Setup and load model
    processor = transformers.AutoImageProcessor.from_pretrained("WinKawaks/vit-tiny-patch16-224")
    model = transformers.AutoModelForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224")
    
    model.eval()
    
    model.vit.embeddings.patch_embeddings.requires_grad = False
    model.vit.embeddings.patch_embeddings.requires_grad = False
    
    for name, param in model.named_parameters():
        if name.endswith('embed'):
            param.requires_grad = False
    
    pretrained_embeddings = model.vit.embeddings

    # Setup and load dataset
    val_set = PartImageNetDataset(data_path="/home/space/datasets/PartImageNet_OOD/", mode='test', get_masks=True, image_size=(224, 224),
                 evaluate=True)

    for ids in sample_range:
        logfile_str = logfolder  + f'sst_search_exhaustive_sample-{ids}_max_and_order-{max_and_order}_datamode-{datamode}.txt'

        with open(logfile_str, 'w') as outfile:
            cout = f'\n ---------------------------- sample {ids} ----------------------------'
            print(cout, file=outfile)

            inputs, _, landmarks = val_set.__getitem__(ids)
            image = inputs.permute(1, 2, 0).numpy()

            logits = model(inputs.unsqueeze(0)).logits
            prediction = logits.argmax()
            targets = torch.eye(1000, dtype=inputs.dtype)[prediction]

            masks = []
            patch_ids = []
            object_tokens = []
            for idx in range(landmarks.shape[0]):
                if landmarks[idx].sum() > 0:
                    patch_id = get_masked_patch_ids(image, landmarks[idx].numpy().astype(np.uint8), (16, 16))
                    patch_ids.append(patch_id)
                    object_tokens.extend(patch_id)
                    masks.append(landmarks[idx].numpy().astype(np.uint8))

            explainer = ViTSymbolicXAI(
                    model=deepcopy(model),
                    embeddings=pretrained_embeddings,
                    sample=inputs.unsqueeze(0),
                    target=targets,
                    scal_val=1.,
                    use_lrp_layers=True
                )

            # Setup other variables
            file_name_harsanyi_divs = resultfolder + f'harsanyi_div_maxorder-{harsanyi_maxorder}_sampleid-{ids}_datamode-{datamode}.pkl'
            file_name_all_queries = resultfolder  + f'all_queries-{ids}_max_and_order-{max_and_order}_datamode-{datamode}_attribution_mode-{attribution_mode}_query_mode-{query_mode}_harsanyi_maxorder-{harsanyi_maxorder}.pkl'

            ########################################
            ## Compute or load the Harsanyi dividends
            #########################################
            neg_tokens_ids = list(set(explainer.node_domain) - set(object_tokens))
            
            if load_harsanyi:
                hars_div = pickle.load(open(file_name_harsanyi_divs, 'rb'))
                print('loaded Harsanyi dividends from file', file=outfile)
            else:
                start = time.time()
                hars_div = comp_all_harsanyi_sst(explainer, harsanyi_maxorder=harsanyi_maxorder, neg_tokens=None if not neg_tokens_hars_bool else neg_tokens_ids)
                pickle.dump(hars_div, open(file_name_harsanyi_divs, 'wb'))
                print(f'computing the harsanyi dividends took {time.time() - start} seconds', file=outfile)

            ########################################
            ## Setup and attribute the queries
            #########################################
            # 1) Setup queries
            all_queries = setup_queries(explainer.node_domain,
                                        tokens,
                                        max_and_order,
                                        max_setsize=max_setsize,
                                        max_indexdist=1,
                                        mode=query_mode,
                                        neg_tokens=neg_tokens_ids )
            # 2) Setup weight function
            weight_vec = calc_weights(weight_mode, hars_div, all_queries)

            # 3) Attribution
            start = time.time()
            if attribution_mode == 'corr(q,f)':
                calculation_fct = calc_corr
            elif attribution_mode == 'cov(q,f)':
                calculation_fct = calc_cov
            elif attribution_mode =='supp(q)':
                 calculation_fct = calc_attr_supp
            else:
                raise NotImplementedError

            if nb_cores >1:
                pool = multiprocessing.Pool(nb_cores)
                args = [(q, hars_div, weight_vec) for q in all_queries]
                all_queries = pool.map(calculation_fct, args)
                pool.close()
            else:
                for q in tqdm(all_queries, desc=f'Query Attribution of {attribution_mode}.'):
                   q = calculation_fct((q, hars_div, weight_vec))

            print('Attribution of the queries with' if nb_cores>1 else 'without', f'parallization it took {time.time() - start} seconds.', file=outfile)
            with open(file_name_all_queries, 'wb') as queryfile:
                pickle.dump(all_queries, queryfile)


if __name__ == '__main__':
    main()
    



            
