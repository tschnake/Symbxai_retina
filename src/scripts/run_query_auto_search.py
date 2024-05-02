
# Import packages
import click, ast, torch, time, transformers, numpy, pickle, multiprocessing
from copy import copy
from itertools import pairwise
from tqdm import tqdm


from symb_xai.dataset.utils import load_sst_treebank
from symb_xai.lrp.symbolic_xai import BERTSymbXAI
from symb_xai.model.transformer import bert_base_uncased_model
# from symb_xai.model.utils import load_pretrained_weights
from symb_xai.query_search.utils import comp_all_harsanyi_sst, setup_queries,  calc_attr_supp, calc_corr, calc_weights

from transformers import BertTokenizer
# from dgl.data import SSTDataset
from datasets import load_dataset

from symb_xai.visualization.utils import html_heatmap, make_text_string

from symb_xai.utils import powerset
# from symb_xai.lrp.symbolic_xai import attribute

# Preliminary functions
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        # Either we denote a range, or a list with precise samples:
        # 1) Range:
        if 'range' in value:
            idx_open = value.find('(')
            idx_close = value.find(')')
            # Get the range input
            range_input = value[idx_open + 1:idx_close].split(',')
            # Make it to numbers:
            range_input = [int(num) for num in range_input]
            try :
                return list(range(*range_input))
            except:
                raise click.BadParameter(value)
        else:
            try:
                return ast.literal_eval(value)
            except:
                raise click.BadParameter(value)

# Get arguments
@click.command()
@click.option('--sample_range',
                cls=PythonLiteralOption,
                default='[0]',
                help='List of sample ids in the dataset.')
@click.option('--max_and_order',
                type=int,
                default=1,
                help='Maximum order of conjunctive (AND) operation.')
@click.option('--logfolder',
                type=str,
                default='/Users/thomasschnake/Research/Projects/symbolic_xai/local_experiments/logs/query_auto_search/',
                help='Folder where we safe the logs.')
@click.option('--resultfolder',
                type=str,
                default='/Users/thomasschnake/Research/Projects/symbolic_xai/local_experiments/intermediate_results/query_search_algo/',
                help='Folder where we intermediate results are safed.')
@click.option('--datamode',
                type=str,
                default='sst_treebank',
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
def main(sample_range,
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
        load_harsanyi):

    ##########################
    ## Fix input variables
    ###########################
    neg_tokens = ['[CLS]', '[SEP]', ',', '.', '_', '-', "'" ]

    ##########################
    ## Setup model and dataset
    ###########################
    sst_model = bert_base_uncased_model(
        pretrained_model_name_or_path='textattack/bert-base-uncased-SST-2'
        )

    sst_model.eval()
    pretrained_embeddings = sst_model.bert.embeddings
    tokenizer = transformers.BertTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

    if datamode == 'sst_huggingface':
        sst_dataset = load_dataset("sst2", "default")

    elif datamode == 'sst_treebank':
            sst_dataset = load_sst_treebank(sample_range, mode='train')
    else:
        raise NotImplementedError

    for ids in sample_range:
        logfile_str = logfolder  + f'sst_search_exhaustive_sample-{ids}_max_and_order-{max_and_order}_datamode-{datamode}.txt'

        with open(logfile_str, 'w') as outfile:
            cout = f'\n ---------------------------- sample {ids} ----------------------------'
            print(cout, file=outfile)

            sentence = sst_dataset['validation']['sentence'][ids]
            print(sentence, file=outfile)

            ##########################
            ## Process sample specific data
            ###########################
            target_value = sst_dataset['validation']['label'][ids]
            sample = tokenizer(sentence, return_tensors="pt")
            tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'].squeeze())

            logits = sst_model(**sample).logits
            prediction = logits.argmax()
            if target_value != prediction:
                cout = '\nwe skip this sample'
                cout+= '\n-------------------------------------------------------------------'
                print(cout, file=outfile)
                continue

            cout='\ntarget and predicion is '+ ('negative' if target_value == 0 else 'positive')
            print(cout, file=outfile)

            if prediction.item() == 0:
                target = torch.tensor([1,-1])
            else:
                target = torch.tensor([-1,1])

            # Setup explainer:
            explainer = BERTSymbXAI(sample=sample,
                                    target=target,
                                    model=sst_model,
                                    embeddings=pretrained_embeddings)

            neg_tokens_ids = [idn for idn, tok in enumerate(tokens) if tok in neg_tokens]

            print('', file=outfile)

            # Setup other variables
            file_name_harsanyi_divs = resultfolder + f'harsanyi_div_maxorder-{harsanyi_maxorder}_sampleid-{ids}_datamode-{datamode}.pkl'
            file_name_all_queries = resultfolder  + f'all_queries-{ids}_max_and_order-{max_and_order}_datamode-{datamode}_attribution_mode-{attribution_mode}_query_mode-{query_mode}_harsanyi_maxorder-{harsanyi_maxorder}.pkl'
            if max_setsize > len(tokens):
                max_setsize = len(tokens)
            ########################################
            ## Compute or load the Harsanyi dividends
            #########################################
            if load_harsanyi:
                hars_div = pickle.load(open(file_name_harsanyi_divs, 'rb'))
                print('loaded Harsanyi dividends from file', file=outfile)
            else:
                start = time.time()
                hars_div = comp_all_harsanyi_sst(explainer, harsanyi_maxorder=harsanyi_maxorder) #, neg_tokens=neg_tokens_ids)
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
            else:
                 calculation_fct = calc_attr_supp

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

# ddddddddd
#             all_queries = setup_queries(explainer.node_domain,
#                                 max_and_order,
#                                 max_setsize=max_setsize,
#                                 max_indexdist=1,
#                                 mode=query_mode,
#                                 neg_tokens=[0,len(explainer.node_domain) -1 ])
#
#             all_weighted_outs = weight_query_attr_harsanyi(all_queries,hars_div, weight_modes)




if __name__ == '__main__':
    main()
