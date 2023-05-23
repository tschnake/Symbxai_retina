import torch
from transformers import BertForSequenceClassification
import math
from torch import nn as nn
from lrp.rules import gamma
from lrp.core import ModifiedLinear, ModifiedLayerNorm, ModifiedAct


# ------ Tiny transformer with 3 layers ------

class SelfAttention(nn.Module):
    def __init__(self, self_attention):
        super(SelfAttention, self).__init__()
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

        attention_probs = nn.Softmax(dim=-1)(attention_scores)


        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = context_layer

        return outputs


class SelfOutput(nn.Module):
    def __init__(self, self_output):
        super(SelfOutput, self).__init__()
        self.dense = self_output.dense
        self.LayerNorm = self_output.LayerNorm
        self.dropout = self_output.dropout

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Attention(nn.Module):
    def __init__(self, attention):
        super(Attention, self).__init__()
        self.self = SelfAttention(attention.self)
        self.output = SelfOutput(attention.output)

    def forward(self, hidden_states):
        self_output = self.self(hidden_states)
        attention_output = self.output(self_output, hidden_states)

        return attention_output


class Layer(nn.Module):
    def __init__(self, layer):
        super(Layer, self).__init__()
        self.attention = Attention(layer.attention)

    def forward(self, hidden_states):
        hidden_states = self.attention(hidden_states)

        return hidden_states


class Encoder(nn.Module):
    def __init__(self, encoder):
        super(Encoder, self).__init__()
        layers = []

        for i, layer in enumerate(encoder.layer[:3]):
            layers.append(Layer(layer))
        self.layer = nn.ModuleList(layers)

    def forward(self, hidden_states):
        for i, layer in enumerate(self.layer):
            hidden_states = layer(hidden_states)
        return hidden_states


class Pooler(nn.Module):
    def __init__(self, pooler):
        super(Pooler, self).__init__()
        self.dense = pooler.dense
        self.activation = pooler.activation

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class Model(nn.Module):
    def __init__(self, bert, embeddings):
        super(Model, self).__init__()
        self.embeddings = embeddings
        self.encoder = Encoder(bert.encoder)
        self.pooler = Pooler(bert.pooler)

    def forward(self, x):
        hidden_states = self.embeddings(
            input_ids=x['input_ids'],
            token_type_ids=x['token_type_ids']
        )
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.pooler(hidden_states)

        return hidden_states


class TinyTransformerForSequenceClassification(nn.Module):
    def __init__(self, bert_classification, name):
        super(TinyTransformerForSequenceClassification, self).__init__()
        self.name = name
        self.bert = Model(bert_classification.bert, bert_classification.bert.embeddings)
        self.dropout = bert_classification.dropout
        self.classifier = bert_classification.classifier

    def forward(self, x):
        hidden_states = self.bert(x)
        hidden_states = self.classifier(hidden_states)

        return hidden_states


# ------ LRP for tiny transformer with 3 layers ------

class ModifiedSelfAttention(nn.Module):
    def __init__(self, self_attention):
        super(ModifiedSelfAttention, self).__init__()
        self.query = ModifiedLinear(fc=self_attention.query, transform=gamma())
        self.key = ModifiedLinear(fc=self_attention.key, transform=gamma())
        self.value = ModifiedLinear(fc=self_attention.value, transform=gamma())

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


class ModifiedSelfOutput(nn.Module):
    def __init__(self, self_output):
        super(ModifiedSelfOutput, self).__init__()
        self.dense = ModifiedLinear(fc=self_output.dense, transform=gamma())
        self.LayerNorm = ModifiedLayerNorm(norm_layer=self_output.LayerNorm,
                                           normalized_shape=self_output.dense.weight.shape[1])
        self.dropout = self_output.dropout

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class ModifiedAttention(nn.Module):
    def __init__(self, attention):
        super(ModifiedAttention, self).__init__()
        self.self = ModifiedSelfAttention(attention.self)
        self.output = ModifiedSelfOutput(attention.output)

    def forward(self, hidden_states):
        self_output = self.self(hidden_states)
        attention_output = self.output(self_output, hidden_states)

        return attention_output


class ModifiedLayer(nn.Module):
    def __init__(self, layer):
        super(ModifiedLayer, self).__init__()
        self.attention = ModifiedAttention(layer.attention)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        return attention_output


# -------- Second-Order --------
class ModifiedLayerHigherOrder(nn.Module):
    def __init__(self, layer):
        super(ModifiedLayerHigherOrder, self).__init__()
        self.attention = ModifiedAttention(layer.attention)

    def forward(self, hidden_states, mask):
        attention_output = self.attention(hidden_states)
        attention_output = attention_output * mask + attention_output.data * (1 - mask)
        return attention_output


# -----------------------------

class ModifiedEncoder(nn.Module):
    def __init__(self, encoder, order):
        super(ModifiedEncoder, self).__init__()
        self.order = order
        self.ho_layers = [0, 1]

        layers = []
        for i, layer in enumerate(encoder.layer):
            if order == 'higher' and i in self.ho_layers:
                layers.append(ModifiedLayerHigherOrder(layer))
            else:
                layers.append(ModifiedLayer(layer))
        self.layer = nn.ModuleList(layers)

    def forward(self, hidden_states, masks):
        for i, layer in enumerate(self.layer):
            if self.order == 'higher':
                for j in self.ho_layers:
                    hidden_states = layer(hidden_states, masks[j])
            else:
                hidden_states = layer(hidden_states)
        return hidden_states


class ModifiedPooler(nn.Module):
    def __init__(self, pooler):
        super(ModifiedPooler, self).__init__()
        self.dense = ModifiedLinear(fc=pooler.dense, transform=gamma())
        self.activation = ModifiedAct(pooler.activation)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class ModifiedModel(nn.Module):
    def __init__(self, bert, order):
        super(ModifiedModel, self).__init__()
        self.encoder = ModifiedEncoder(bert.encoder, order)
        self.pooler = ModifiedPooler(bert.pooler)

    def forward(self, x, masks):
        hidden_states = self.encoder(x, masks)
        hidden_states = self.pooler(hidden_states)

        return hidden_states


class ModifiedTinyTransformerForSequenceClassification(nn.Module):
    def __init__(self, bert_classification, order='higher'):
        super(ModifiedTinyTransformerForSequenceClassification, self).__init__()
        self.bert = ModifiedModel(bert_classification.bert, order)
        self.classifier = ModifiedLinear(fc=bert_classification.classifier, transform=gamma())
        self.order = order

    def forward(self, embeddings, masks=None):
        hidden_states = self.bert(embeddings, masks)
        hidden_states = self.classifier(hidden_states)

        return hidden_states


def tiny_transformer_with_3_layers(
        pretrained_model_name_or_path
):
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path
    )

    return TinyTransformerForSequenceClassification(
        model,
        "tiny_transformer_with_3_layers"
    )

