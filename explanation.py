from itertools import product
from torch import Tensor
from models import ModifiedBertForSequenceClassification
from transformers import BertForSequenceClassification, BertTokenizer, AutoTokenizer, AutoModelForPreTraining


def gamma(
        gam: float = 0.0
) -> Tensor:

    def modify_parameters(parameters: Tensor):
        return parameters + (gam * parameters.clamp(min=0))

    return modify_parameters

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

def lrp(model, x, target, indices, pretrained_embeddings):
    A = {}
    
    hidden_states = pretrained_embeddings(input_ids=x['input_ids'], token_type_ids=x['token_type_ids'])
    A['hidden_states'] = hidden_states
    attn_input = hidden_states
    
    if model.order == 'higher':
        M = torch.eye(hidden_states.shape[1])
        k, l = indices
        Mk = M[k].unsqueeze(0).unsqueeze(2)
        Ml = M[l].unsqueeze(0).unsqueeze(2)
        Mj = M[0].unsqueeze(0).unsqueeze(2)
        
    n = len(model.bert.encoder.layer)
    for i, layer in enumerate(model.bert.encoder.layer):
            attn_inputdata = attn_input.data
            attn_inputdata.requires_grad_(True) 
                        
            A['attn_input_{}_data'.format(i)] = attn_inputdata
            A['attn_input_{}'.format(i)] = attn_input
            
            if model.order == 'higher':
                if i == 0:
                    output = model.bert.encoder.layer[i](A['attn_input_{}_data'.format(i)], Mk)
                elif i == 1:
                    output = model.bert.encoder.layer[i](A['attn_input_{}_data'.format(i)], Ml) 
                else:
                    output = model.bert.encoder.layer[i](A['attn_input_{}_data'.format(i)], Mj)
            else:
                output = model.bert.encoder.layer[i](A['attn_input_{}_data'.format(i)])
            attn_input = output
            
    outputdata = output.data
    outputdata.requires_grad_(True)
    
    pooled = model.bert.pooler(outputdata)
    pooleddata = pooled.data
    pooleddata.requires_grad_(True) 
        
    logits = model.classifier(pooleddata)
    pred = torch.argmax(logits)
    Rout = (logits * target).sum()
    
    Rout.backward()
    ((pooleddata.grad)*pooled).sum().backward()
                
    Rpool = ((outputdata.grad)*output)
    R_ = Rpool
    
    R_all = []
    for i, layer in list(enumerate(model.bert.encoder.layer))[::-1]:
        R_.sum().backward()
        R_grad = A['attn_input_{}_data'.format(i)].grad
        R_attn =  (R_grad)*A['attn_input_{}'.format(i)]
        
        R_ = R_attn
        R_all.append(R_.sum(2).detach().cpu().numpy().squeeze().sum())

    R = R_.sum(2).detach().cpu().numpy()
    
    return R, pred, Rout, R_all

class HOExplainer:
    def __init__(self, model):
        self.pretrained_embeddings = model.bert.embeddings
        self.modified_model = ModifiedBertForSequenceClassification(model,
                                                       self.pretrained_embeddings,
                                                       order='higher')
        self.modified_model.eval()

        self.tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
        self.UNK_IDX = self.tokenizer.unk_token_id  # an out-of-vocab token
        
    def setup_sample(self,sentence, target):
        self.sentence = sentence
        self.target = target
        self.x = self.tokenizer(sentence, return_tensors="pt")
        self.words = self.tokenizer.convert_ids_to_tokens(self.x['input_ids'].squeeze())
        
        return True
    
    def explain(self):
        rels = {}
        combinations = set(product(range(len(self.words)),repeat = 3))
        for (j, k, l) in combinations:
            indices = (k, l)
            R, pred, logit, R_all = lrp(self.modified_model, self.x, self.target, indices, self.pretrained_embeddings)
            rels[(j, k, l)] = R.squeeze()[j].sum()

        return rels