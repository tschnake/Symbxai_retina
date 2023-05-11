import torch
from transformers import BertForSequenceClassification, BertTokenizer, AutoTokenizer, AutoModelForPreTraining
import explanation
from base_models import stabilize, modified_layer, ModifiedLinear, ModifiedLayerNorm, ModifiedAct, ModifiedTanh
import numpy as np
import math
from torch.nn.modules import Module
from torch import Tensor
from torch import nn as nn


class BertSelfAttention(nn.Module):
    def __init__(self, self_attention):
        super(BertSelfAttention, self).__init__()
        self.query = self_attention.query
        self.key = self_attention.key
        self.value = self_attention.value
        self.dropout = self_attention.dropout   
        self.num_attention_heads = self_attention.num_attention_heads
        self.attention_head_size = self_attention.attention_head_size
        self.all_head_size = self_attention.all_head_size
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states):
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores).detach()
        
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = context_layer
        
        return outputs
        
class BertSelfOutput(nn.Module): 
    def __init__(self, self_output):
        super(BertSelfOutput, self).__init__()
        self.dense = self_output.dense
        self.LayerNorm = self_output.LayerNorm
        self.dropout = self_output.dropout
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)                
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
        
    
class BertAttention(nn.Module):
    def __init__(self, attention):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(attention.self)
        self.output = BertSelfOutput(attention.output)
        
    def forward(self, hidden_states):
        self_output = self.self(hidden_states)
        attention_output = self.output(self_output, hidden_states)
        
        return attention_output
    
class BertLayer(nn.Module):
    def __init__(self, layer):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(layer.attention)
        
    def forward(self, hidden_states):
        hidden_states = self.attention(hidden_states)
        
        return hidden_states

        
class BertEncoder(nn.Module):
    def __init__(self, encoder):
        super(BertEncoder, self).__init__()
        layers = []
        
        for i, layer in enumerate(encoder.layer[:3]):
            layers.append(BertLayer(layer))
        self.layer = nn.ModuleList(layers)
        
    def forward(self, hidden_states):        
        for i, layer in enumerate(self.layer):
            hidden_states = layer(hidden_states)      
        return hidden_states
    
class BertPooler(nn.Module):
    def __init__(self, pooler):
        super(BertPooler, self).__init__()
        self.dense = pooler.dense
        self.activation = pooler.activation
        
    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        
        return pooled_output

class BertModel(nn.Module):
    def __init__(self, bert, embeddings):
        super(BertModel, self).__init__()
        self.embeddings = embeddings
        self.encoder = BertEncoder(bert.encoder)
        self.pooler = BertPooler(bert.pooler)
        
    def forward(self, x):
        hidden_states = self.embeddings(
            input_ids=x['input_ids'],
            token_type_ids=x['token_type_ids']
        )    
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.pooler(hidden_states)
        
        return hidden_states
    
class BertForSequenceClassification(nn.Module):
    def __init__(self, bert_classification, embeddings):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel(bert_classification.bert, embeddings)
        self.dropout = bert_classification.dropout
        self.classifier = bert_classification.classifier
        
    def forward(self, x):
        hidden_states = self.bert(x)
        hidden_states = self.classifier(hidden_states)
        
        return hidden_states
    
    
class ModifiedBertSelfAttention(nn.Module):
    def __init__(self, self_attention):
        super(ModifiedBertSelfAttention, self).__init__()
        self.query = ModifiedLinear(fc=self_attention.query, transform=explanation.gamma())
        self.key = ModifiedLinear(fc=self_attention.key, transform=explanation.gamma())
        self.value = ModifiedLinear(fc=self_attention.value, transform=explanation.gamma())
        
        self.dropout = self_attention.dropout   
        self.num_attention_heads = self_attention.num_attention_heads
        self.attention_head_size = self_attention.attention_head_size
        self.all_head_size = self_attention.all_head_size
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states):
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores).detach()
        
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = context_layer
        
        return outputs
        
class ModifiedBertSelfOutput(nn.Module): 
    def __init__(self, self_output):
        super(ModifiedBertSelfOutput, self).__init__()
        self.dense = ModifiedLinear(fc=self_output.dense, transform=explanation.gamma())
        self.LayerNorm = ModifiedLayerNorm(norm_layer=self_output.LayerNorm,
                                           normalized_shape=self_output.dense.weight.shape[1])
        self.dropout = self_output.dropout
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)                
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
        
    
class ModifiedBertAttention(nn.Module):
    def __init__(self, attention):
        super(ModifiedBertAttention, self).__init__()
        self.self = ModifiedBertSelfAttention(attention.self)
        self.output = ModifiedBertSelfOutput(attention.output)
        
    def forward(self, hidden_states):
        self_output = self.self(hidden_states)
        attention_output = self.output(self_output, hidden_states)
        
        return attention_output
        
        
class ModifiedBertIntermediate(nn.Module):
    def __init__(self, intermediate):
        super(ModifiedBertIntermediate, self).__init__()
        self.dense = ModifiedLinear(fc=intermediate.dense, transform=explanation.gamma())
        self.intermediate_act_fn = ModifiedAct(intermediate.intermediate_act_fn)
        
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states  

class ModifiedBertOutput(nn.Module):
    def __init__(self, output):
        super(ModifiedBertOutput, self).__init__()
        self.dense = ModifiedLinear(fc=output.dense, transform=explanation.gamma())
        self.LayerNorm = ModifiedLayerNorm(norm_layer=output.LayerNorm,
                                           normalized_shape=output.dense.weight.shape[1])
        self.dropout = output.dropout
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)                
        hidden_states = self.LayerNorm(hidden_states + input_tensor) 

        return hidden_states
    
class ModifiedBertLayer(nn.Module):
    def __init__(self, layer):
        super(ModifiedBertLayer, self).__init__()
        self.attention = ModifiedBertAttention(layer.attention)
        
    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        
        return hidden_states

# -------- Second-Order --------
class ModifiedBertLayerHigherOrder(nn.Module):
    def __init__(self, layer):
        super(ModifiedBertLayerHigherOrder, self).__init__()
        self.attention = ModifiedBertAttention(layer.attention)
        
    def forward(self, hidden_states, mask):
        attention_output = self.attention(hidden_states)
        attention_output = attention_output * mask + attention_output.data * (1 - mask)
        return attention_output
# -----------------------------
        
class ModifiedBertEncoder(nn.Module):
    def __init__(self, encoder, order):
        super(ModifiedBertEncoder, self).__init__()
        self.order = order
        layers = []
        
        n = len(encoder.layer)
        for i, layer in enumerate(encoder.layer):
            if order == 'higher' and (i == 0 or i == 1 or i==2):
                # Mask layers 0 and 1.
                layers.append(ModifiedBertLayerHigherOrder(layer))
            else:
                layers.append(ModifiedBertLayer(layer))
        self.layer = nn.ModuleList(layers)
        
    def forward(self, hidden_states):        
        for i, layer in enumerate(self.layer):
            hidden_states = layer(hidden_states)      
        return hidden_states
    
class ModifiedBertPooler(nn.Module):
    def __init__(self, pooler):
        super(ModifiedBertPooler, self).__init__()
        self.dense = ModifiedLinear(fc=pooler.dense, transform=explanation.gamma())
        self.activation = ModifiedTanh(pooler.activation)
        
    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        
        return pooled_output

class ModifiedBertModel(nn.Module):
    def __init__(self, bert, embeddings, order):
        super(ModifiedBertModel, self).__init__()
        self.embeddings = embeddings
        self.encoder = ModifiedBertEncoder(bert.encoder, order)
        self.pooler = ModifiedBertPooler(bert.pooler)
        
    def forward(self, x):
        hidden_states = self.embeddings(x)    
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.pooler(hidden_states)
        
        return hidden_states
    
class ModifiedBertForSequenceClassification(nn.Module):
    def __init__(self, bert_classification, embeddings, order='higher'):
        super(ModifiedBertForSequenceClassification, self).__init__()
        self.bert = ModifiedBertModel(bert_classification.bert, embeddings, order)
        self.dropout = bert_classification.dropout
        self.classifier = ModifiedLinear(fc=bert_classification.classifier, transform=explanation.gamma())
        self.order = order
        
    def forward(self, x):
        hidden_states = self.bert(x)
        hidden_states = self.classifier(hidden_states)
        
        return hidden_states