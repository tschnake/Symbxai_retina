import click, torch, torchvision, transformers

from symb_xai.lrp.symbolic_xai import BERTSymbXAI
from symb_xai.lrp.symbolic_xai import ViTSymbolicXAI

from symb_xai.model.transformer import bert_base_uncased_model


from symb_xai.perturbation_utils import get_node_ordering
from symb_xai.visualization.utils import make_text_string, remove_patches
from utils import PythonLiteralOption
from tqdm import tqdm


import fcntl
from filelock import FileLock

import pickle


def create_mapping(shift=1):
    mapping = {}
    new_id = 0

    for i in range(7):
        for j in range(7):
            # Calculate the top-left corner index of the 2x2 patch in the 14x14 grid
            top_left = (i * 2) * 14 + (j * 2) + shift
            # The four indices of the 2x2 patch
            patch_indices = [
                top_left,
                top_left + 1,
                top_left + 14,
                top_left + 14 + 1
            ]
            # Add the mapping to the dictionary
            mapping[new_id] = patch_indices
            new_id += 1

    return mapping


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
@click.option('--auc_task',
            type=str,
            default='',
            help="it can be 'minimize' or 'maximize'")
@click.option('--perturbation_type',
            type=str,
            default='',
            help="it can be 'removal' or 'generation'")
@click.option('--data_dir',
                type=str,
                default='/Users/thomasschnake/Research/Projects/symbolic_xai/datasets/fer_images/train/')
# @click.option('--create_data_file',
#                 is_flag=True,
#                 help='A flag that specifies whether the result file should be created from scratch. \
#                 Otherwise just append the values')

def main(sample_range,
         data_mode,
         result_dir,
         auc_task,
         perturbation_type,
         data_dir
         # create_data_file
         ):

    if data_mode == 'sst':
        from symb_xai.dataset.utils import load_sst_treebank
        # load data
        dataset = load_sst_treebank(sample_range, verbose=False)['validation']
        # load model
        model = bert_base_uncased_model(
            pretrained_model_name_or_path='textattack/bert-base-uncased-SST-2' )
        model.eval()
        tokenizer = transformers.BertTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
        input_type = 'sentence'

    elif data_mode == 'imdb': # Load IMDB data and model
        from symb_xai.dataset.utils import load_imdb_dataset
        # Load the dataset
        dataset = load_imdb_dataset(sample_range)

        # Load the model and tokenizer
        model = bert_base_uncased_model(
                pretrained_model_name_or_path="textattack/bert-base-uncased-imdb" )
        model.eval()
        tokenizer = transformers.BertTokenizer.from_pretrained("textattack/bert-base-uncased-imdb", local_files_only=True)
        input_type = 'sentence'

    elif data_mode == 'fer':
        ## Load model
        processor = transformers.AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
        model = transformers.AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
        model.eval()

        from symb_xai.dataset.utils import load_fer_dataset
        dataset = load_fer_dataset(sample_range, processor, data_dir)

        input_type = 'image'

    else:
        raise NotImplementedError(f'data mode {data_mode} does not exist')

    # Perform perturbation
    attribution_methods = [ 'random', 'SymbXAI', 'LRP', 'PredDiff' ]

    if auc_task and perturbation_type:
        optimize_parameter = [(auc_task, perturbation_type)]
        save_seperatly = True
    else:
        optimize_parameter = [('minimize', 'removal'), ('maximize', 'removal') , ('minimize', 'generation'), ('maximize', 'generation')]
        save_seperatly = False
    print('doing', data_mode, sample_range)

    for sample_id in dataset[input_type].keys():
        went_through = 0
        output_dict_curves = {param: {attribution_method: {} for attribution_method in attribution_methods} for param in optimize_parameter }
        output_dict_orderings = {param: {attribution_method: {} for attribution_method in attribution_methods} for param in optimize_parameter }

        for attribution_method in attribution_methods:
            for auc_task, perturbation_type in optimize_parameter:
                ## Preprocess input
                if data_mode in ['sst', 'imdb']: ## NLP
                    sentence, label = dataset[input_type][sample_id], dataset['label'][sample_id]

                    sample = tokenizer(sentence, return_tensors="pt")
                    tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'].squeeze())
                    if len(tokens) > 256: continue
                    # target_class = model(**sample)['logits'].argmax().item()
                    # output_mask = torch.tensor([0,0]); output_mask[target_class]=1
                    output_mask = torch.tensor([-1,1])

                    ### Generate node ordering
                    explainer = BERTSymbXAI(sample=sample,
                                        target=output_mask,
                                        model=model,
                                        embeddings=model.bert.embeddings)

                    model_output = lambda sample: (model(**sample)['logits']*output_mask).sum().item()
                    empty_sample = tokenizer('', return_tensors="pt")
                    node_mapping_model_patches = {i:[i] for i in explainer.node_domain}
                    node_mapping_image_patches = {i:[i] for i in explainer.node_domain}

                elif data_mode == 'fer': ## Vision
                    sample, label = dataset[input_type][sample_id], dataset['label'][sample_id]
                    output_mask = torch.eye(7, dtype=sample.dtype)[label]

                    explainer = ViTSymbolicXAI(
                                model=model,
                                embeddings=model.vit.embeddings,
                                sample=sample.unsqueeze(0),
                                target=output_mask
                                        )
                    model_output = lambda sample: (model(sample.unsqueeze(0)).logits*output_mask).sum().item()
                    empty_sample = torch.zeros(sample.shape)+.5
                    node_mapping_model_patches = create_mapping(shift=1)
                    node_mapping_image_patches = create_mapping(shift=0)
                # Compute the node ordering.
                ## This might take some time...
                node_ordering = get_node_ordering(explainer,
                                                    attribution_method,
                                                    auc_task,
                                                    perturbation_type,
                                                    verbose=True,
                                                    node_mapping=node_mapping_model_patches,
                                                    add_cls= True if data_mode=='fer' else False)

                output_dict_orderings[(auc_task, perturbation_type)][attribution_method][sample_id] = node_ordering
                ### Create purturbation curve
                gliding_subset_ids = []
                if perturbation_type == 'removal':
                    output_sequence = [model_output(sample)]
                elif perturbation_type == 'generation':
                    output_sequence = [model_output(empty_sample)]

                for node_id in tqdm(node_ordering, desc=f'flip {attribution_method}'):
                    gliding_subset_ids += node_mapping_image_patches[node_id]
                    if perturbation_type == 'removal':
                        input_ids = [ids for ids in explainer.node_domain if ids not in gliding_subset_ids]
                        input_ids = sorted(input_ids)

                    elif perturbation_type == 'generation':
                        input_ids = gliding_subset_ids
                        input_ids = sorted(input_ids)
                    else:
                        raise NotImplementedError(f'perturbation type -{perturbation_type}- does not exist')
                    # make into a model input
                    if data_mode in ['sst', 'imdb']:
                        new_sentence = make_text_string([ tokens[ids] for ids in input_ids])
                        new_sample  = tokenizer(new_sentence, return_tensors="pt")
                    elif data_mode == 'fer':
                        new_sample = remove_patches(sample, [ids for ids in explainer.node_domain if ids not in input_ids])
                    else:
                        raise NotImplementedError


                    # save alternative output
                    output_sequence.append(model_output(new_sample))

                output_dict_curves[(auc_task, perturbation_type)][attribution_method][sample_id] = output_sequence

                # save the results in files
                # save_to_file(output_sequence,
                #             (auc_task, perturbation_type),
                #             attribution_method,
                #             sample_id,
                #             result_dir + filename,
                #             default_dict)

                went_through +=1

        if save_seperatly:
            filename_curves = f'perturbation_results_{data_mode}_{sample_id}_{auc_task}_{perturbation_type}_curves.pkl'
            filename_orderings = f'perturbation_results_{data_mode}_{sample_id}_{auc_task}_{perturbation_type}_orderings.pkl'
        else:
            filename_curves = f'perturbation_results_{data_mode}_{sample_id}_curves.pkl'
            filename_orderings = f'perturbation_results_{data_mode}_{sample_id}_orderings.pkl'

        with open(result_dir + filename_curves,'wb') as f:
            pickle.dump(output_dict_curves, f)
        with open(result_dir + filename_orderings,'wb') as f:
            pickle.dump(output_dict_orderings, f)

        if went_through > 0:
            print('ok', went_through, 'times for', sample_id)
        else:
            print('skipped', sample_id)

if __name__ == '__main__':
    main()
