
# Import packages
import click, ast, torch, time, transformers, numpy, pickle
from copy import copy
from symb_xai.dataset.utils import load_sst_treebank

from itertools import pairwise
from tqdm import tqdm

from symb_xai.lrp.symbolic_xai import TransformerSymbXAI, BERTSymbXAI
from symb_xai.model.transformer import tiny_transformer_with_3_layers, bert_base_uncased_model
from symb_xai.model.utils import load_pretrained_weights
from symb_xai.query_search.utils import approx_query_search, exhaustive_query_search, comp_all_harsanyi_sst, weight_query_attr_directly, weight_query_attr_harsanyi, setup_queries

from transformers import BertTokenizer
from dgl.data import SSTDataset
from datasets import load_dataset

from symb_xai.visualization.utils import html_heatmap, make_text_string

from symb_xai.utils import powerset
from symb_xai.lrp.symbolic_xai import attribute

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
@click.option('--weight_modes',
              cls=PythonLiteralOption,
              default="['occlusion']",
              help='A list of identifiers that describe how to weight the attributions')
@click.option('--comp_mode',
              type=str,
              default='harsanyi',
              help='The mode how to compute the attributions. Either "directly" or "harsanyi".')
@click.option('--harsanyi_maxorder',
              type=int,
              default=3,
              help='Specifies until what order the Harsanyi dividends will be computed.')
@click.option('--query_mode',
              type=str,
              default='conj. disj. reasonably mixed',
              help='What kind of queries will be attributed.')
@click.option('--max_setsize',
              type=int,
              default=3,
              help='Maximum size of feature sets in the queries. Similar to: the maximum order of the OR operation.')
def main(sample_range,
        max_and_order,
        logfolder,
        resultfolder,
        datamode,
        weight_modes,
        comp_mode,
        harsanyi_maxorder,
        query_mode,
        max_setsize):

    ### Set up data and model:
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

    # order = 3
    for ids in sample_range:
        # verbose = True
        logfile_str = logfolder  + f'sst_search_exhaustive_sample-{ids}_max_and_order-{max_and_order}_datamode-{datamode}.txt'

        # all_outvals = {}

        # for i in  intersting_samples + list(range(35, 50)): #12, 13, 14, 15, 16, 17, 18]:
        with open(logfile_str, 'w') as outfile:
            cout = f'\n ---------------------------- sample {ids} ----------------------------'
            print(cout, file=outfile)

            sentence = sst_dataset['validation']['sentence'][ids]
            print(sentence, file=outfile)

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

            target = torch.tensor([-1,1])

            explainer = BERTSymbXAI(sample=sample,
                                    target=target,
                                    model=sst_model,
                                    embeddings=pretrained_embeddings)

            print('', file=outfile)
            if comp_mode == 'directly':
                resultfile_str = resultfolder  + f'sst_search_exhaustive_sample-{ids}_max_and_order-{max_and_order}_datamode-{datamode}.pkl'
                outvals, all_vals, ftime = exhaustive_query_search(explainer,
                                                               tokens,
                                                               maxorder=max_and_order,
                                                               # maxvalsnum=maxvalsnum,
                                                               #  minvalsnum=minvalsnum,
                                                               verbose=True)
                # all_outvals[ids] = outvals

                all_weighted_outs = weight_query_attr_directly(all_vals, weight_modes)

            elif comp_mode == 'harsanyi':
                # First, setup and save the harsanyi dividends
                file_name_harsanyi_divs = f'harsanyi_div_maxorder-{harsanyi_maxorder}_sampleid-{ids}_datamode-{datamode}.pkl'
                resultfile_str = resultfolder  + f'sst_search_exhaustive_sample-{ids}_max_and_order-{max_and_order}_datamode-{datamode}_comp_mode-{comp_mode}_harsanyi_maxorder-{harsanyi_maxorder}.pkl'
                har_div = comp_all_harsanyi_sst(explainer, harsanyi_maxorder=harsanyi_maxorder)
                pickle.dump(har_div, open(resultfolder + file_name_harsanyi_divs, 'wb'))

                # Second, setup and attribute the queries
                all_queries = setup_queries(explainer.node_domain,
                                    max_and_order,
                                    max_setsize=max_setsize,
                                    max_indexdist=1,
                                    mode=query_mode,
                                    neg_tokens=[0,len(explainer.node_domain) -1 ])

                all_weighted_outs = weight_query_attr_harsanyi(all_queries,har_div, weight_modes)
            else:
                raise NotImplementedError

            # for feats, val in outvals.items():
            #     cout = ' & '.join([tokens[I] for I in feats]) + '\t---> ' + str(round(val.item(), 3))
            #     if len(feats) == max_and_order:
            #         cout += '\n'
            #     print(cout, file=outfile)
            # cout = '-------------------------------------------------------------------'
            # print(cout, file=outfile)

        with open(resultfile_str, 'wb') as resultfile:
            pickle.dump(all_weighted_outs, resultfile)

if __name__ == '__main__':
    main()
