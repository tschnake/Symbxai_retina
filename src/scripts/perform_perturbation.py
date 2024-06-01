import click, torch
from symb_xai.lrp.symbolic_xai import BERTSymbXAI
from symb_xai.model.transformer import bert_base_uncased_model
import transformers

from symb_xai.perturbation_utils import get_node_ordering
from symb_xai.visualization.utils import make_text_string
from utils import PythonLiteralOption


import fcntl

import pickle

def save_to_file(output_sequence,
                param,
                attribution_method,
                filename,
                default_dict):
    with open(filename, 'ab+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, 0)
        try:
            existing_data = pickle.load(f)
        except EOFError:
            existing_data = default_dict
        existing_data[param][attribution_method].append(output_sequence)
        f.seek(0)
        f.truncate()
        pickle.dump(existing_data, f)
        fcntl.flock(f, fcntl.LOCK_UN)


@click.command()
@click.option('--sample_range',
                cls=PythonLiteralOption,
                default='[0]',
                help='List of sample ids in the dataset.')
@click.option('--data_mode',
              type=str,
              default='sst',
              help='Parameter for the data we should be using.')
@click.option('--result_dir',
              type=str,
              default='/Users/thomasschnake/Research/Projects/symbolic_xai/local_experiments/intermediate_results/',
              help='The directory in which we save the results.')
# @click.option('--create_data_file',
#                 is_flag=True,
#                 help='A flag that specifies whether the result file should be created from scratch. \
#                 Otherwise just append the values')

def main(sample_range,
         data_mode,
         result_dir,
         # create_data_file
         ):

    if data_mode == 'sst':
        from symb_xai.dataset.utils import load_sst_treebank
        # load model
        model = bert_base_uncased_model(
            pretrained_model_name_or_path='textattack/bert-base-uncased-SST-2' )
        model.eval()
        # pretrained_embeddings = model.bert.embeddings
        tokenizer = transformers.BertTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
        # load data
        dataset = load_sst_treebank(sample_range, verbose=False)['validation']
    else:
        raise NotImplementedError(f'data mode {data_mode} does not exist')

    # perform perturbation

    target_mask=torch.tensor([-1,1])

    attribution_methods = ['SymbXAI', 'LRP', 'PredDiff','random',  ]
    # auc_task =  'minimize' # 'maximize' #
    # perturbation_type =    'removal' #  'generation' #
    optimize_parameter = [('minimize', 'removal'), ('maximize', 'removal') , ('minimize', 'generation'), ('maximize', 'generation')]
    filename = f'perturbation_results_{data_mode}.pkl'

    default_dict = { param: {attribution_method: [] for attribution_method in attribution_methods} for param in optimize_parameter}

    # if create_data_file:
    #
    #     with open(result_dir + filename, 'w') as f:
    #         json.dump(all_output_sequences, f)


    for attribution_method in attribution_methods:
        for auc_task, perturbation_type in optimize_parameter:
            for sample_id in dataset['sentence'].keys():

                ## Preprocess input
                sentence, label = dataset['sentence'][sample_id], dataset['label'][sample_id]

                sample = tokenizer(sentence, return_tensors="pt")
                tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'].squeeze())
                target_class = model(**sample)['logits'].argmax().item()
                # output_mask = torch.tensor([0,0]); output_mask[target_class]=1
                output_mask = torch.tensor([-1,1])

                ### Generate node ordering
                explainer = BERTSymbXAI(sample=sample,
                                    target=target_mask,
                                    model=model,
                                    embeddings=model.bert.embeddings)
                node_ordering = get_node_ordering(explainer, attribution_method, auc_task, perturbation_type, verbose=True)

                ### Create purturbation curve
                gliding_subset_ids = []
                if perturbation_type == 'removal':
                    output_sequence = [(model(**sample)['logits']*output_mask).sum().item()]
                elif perturbation_type == 'generation':
                    output_sequence = [(model(**tokenizer('', return_tensors="pt"))['logits']*output_mask).sum().item()]

                for node_id in node_ordering:
                    gliding_subset_ids.append(node_id)
                    if perturbation_type == 'removal':
                        input_ids = [ids for ids in range(len(tokens)) if ids not in gliding_subset_ids]
                        input_ids = sorted(input_ids)

                    elif perturbation_type == 'generation':
                        input_ids = gliding_subset_ids
                        input_ids = sorted(input_ids)
                    else:
                        raise NotImplementedError(f'perturbation type -{perturbation_type}- does not exist')
                    # make into a model input
                    new_sentence = make_text_string([ tokens[ids] for ids in input_ids])
                    new_sample  = tokenizer(new_sentence, return_tensors="pt")

                    # save alternative output
                    output_sequence.append((model(**new_sample)['logits']*output_mask).sum().item())

                # save the results in files
                save_to_file(output_sequence,
                            (auc_task, perturbation_type),
                            attribution_method,
                            result_dir + filename,
                            default_dict)

                # all_output_sequences[(auc_task, perturbation_type)][attribution_method].append(output_sequence)
    print('ok1')
if __name__ == '__main__':
    main()