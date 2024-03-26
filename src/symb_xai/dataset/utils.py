from copy import copy
from symb_xai.visualization.utils import  make_text_string
from dgl.data import SSTDataset

def load_sst_treebank(sample_range, mode='train'):
    # from dgl.data import SSTDataset
    dataset = SSTDataset(mode=mode)
    vocab_words = list(dataset.vocab.keys())
    sst_dataset = {'validation': {
            'sentence': {},
            'label'   : {}
    }}
    # Preprocess the data into the desired format
    for sid in copy(sample_range):
        tree = dataset[sid]
        input_ids = tree.ndata['x']
        mask = tree.ndata['mask']
        lsent =  [vocab_words[idw] for idw in input_ids[mask == 1]]

        if int(tree.ndata['y'][0].item()) == 2:
            print(f'we skip sample {sid}')
            sample_range.remove(sid)
            continue
        target = int(tree.ndata['y'][0]>2)
        sentence =  make_text_string(lsent)

        sst_dataset['validation']['sentence'][sid] = sentence
        sst_dataset['validation']['label'][sid]    = target
    return sst_dataset
