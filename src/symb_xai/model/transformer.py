import torch
from transformers import BertForSequenceClassification
import math
from torch import nn as nn
from ..lrp.rules import gamma
from ..lrp.core import ModifiedLinear, ModifiedLayerNorm, ModifiedAct


# ------ Tiny transformer with 3 layers ------

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if self.elementwise_affine:
            if self.elementwise_affine:
                self.weight = nn.Parameter(torch.Tensor(normalized_shape))
                self.bias = nn.Parameter(torch.Tensor(normalized_shape))
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        input_norm = (x - mean) / (std + self.eps)
        return input_norm


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(
            in_features=768,
            out_features=768
        )
        self.key = nn.Linear(
            in_features=768,
            out_features=768
        )
        self.value = nn.Linear(
            in_features=768,
            out_features=768
        )

        self.num_attention_heads = 12
        self.attention_head_size = int(768 / 12)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)


        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = context_layer

        return outputs


class SelfOutput(nn.Module):
    def __init__(self):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(
            in_features=768,
            out_features=768
        )
        self.LayerNorm = LayerNorm(
            normalized_shape=768,
            eps=1e-12,
            elementwise_affine=True
        )

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self.self = SelfAttention()
        self.output = SelfOutput()

    def forward(self, hidden_states):
        self_output = self.self(hidden_states)
        attention_output = self.output(self_output, hidden_states)

        return attention_output


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        layers = []

        for _ in range(3):
            layers.append(Layer())
        self.layer = nn.ModuleList(layers)

    def forward(self, hidden_states):
        for i, layer in enumerate(self.layer):
            hidden_states = layer(hidden_states)
        return hidden_states


class Pooler(nn.Module):
    def __init__(self):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(
            in_features=768,
            out_features=768
        )
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class Model(nn.Module):
    def __init__(self, embeddings):
        super(Model, self).__init__()
        self.embeddings = embeddings
        self.encoder = Encoder()
        self.output = SelfOutput()
        self.pooler = Pooler()

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
        self.bert = Model(embeddings=bert_classification.bert.embeddings)
        self.classifier = nn.Linear(
            in_features=768,
            out_features=2,
            bias=True
        )

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
        self.LayerNorm = ModifiedLayerNorm(norm_layer=self_output.LayerNorm
                                           # normalized_shape=self_output.dense.weight.shape[1]
                                           )

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class ModifiedLayer(nn.Module):
    def __init__(self, layer):
        super(ModifiedLayer, self).__init__()
        self.self = ModifiedSelfAttention(layer.self)
        self.output = ModifiedSelfOutput(layer.output)

    def forward(self, hidden_states):
        attention_output = self.self(hidden_states)
        layer_output = self.output(attention_output, hidden_states)

        return layer_output


class ModifiedEncoder(nn.Module):
    def __init__(self, encoder):
        super(ModifiedEncoder, self).__init__()

        layers = []
        for i, layer in enumerate(encoder.layer):
            layers.append(ModifiedLayer(layer))
        self.layer = nn.ModuleList(layers)

    def forward(self, hidden_states):
        for i, layer in enumerate(self.layer):
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
    def __init__(self, bert):
        super(ModifiedModel, self).__init__()
        self.encoder = ModifiedEncoder(bert.encoder)
        self.pooler = ModifiedPooler(bert.pooler)

    def forward(self, x):
        hidden_states = self.encoder(x)
        hidden_states = self.pooler(hidden_states)

        return hidden_states


class ModifiedTinyTransformerForSequenceClassification(nn.Module):
    def __init__(self, bert_classification, order='higher'):
        super(ModifiedTinyTransformerForSequenceClassification, self).__init__()
        self.bert = ModifiedModel(bert_classification.bert)
        self.classifier = ModifiedLinear(fc=bert_classification.classifier, transform=gamma())

    def forward(self, embeddings):
        hidden_states = self.bert(embeddings)
        hidden_states = self.classifier(hidden_states)

        return hidden_states


def tiny_transformer_with_3_layers(
        pretrained_model_name_or_path
):
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path
    )
    model.bert.embeddings.requires_grad = False
    for name, param in model.named_parameters():
        if name.startswith('embeddings'):
            param.requires_grad = False

    return TinyTransformerForSequenceClassification(
        model,
        "tiny_transformer_with_3_layers"
    )

def bert_base_uncased_model(
        pretrained_model_name_or_path
):
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path
    )
    model.bert.embeddings.requires_grad = False
    for name, param in model.named_parameters():
        if name.startswith('embeddings'):
            param.requires_grad = False

    return model

# ------ LRP for BERT Model ------
class ModifiedBertSelfAttention(nn.Module):
    def __init__(self, self_attention, gam=.09):
        super(ModifiedBertSelfAttention, self).__init__()
        self.query = self_attention.query #ModifiedLinear(fc=self_attention.query, transform=gamma(), gam=gam) ## actually we should modify anything here!
        self.key = self_attention.key #ModifiedLinear(fc=self_attention.key, transform=gamma(),gam=gam)
        self.value = self_attention.value #ModifiedLinear(fc=self_attention.value, transform=gamma(),gam=gam)

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
    def __init__(self, self_output, gam=.09):
        super(ModifiedBertSelfOutput, self).__init__()
        self.dense = self_output.dense #ModifiedLinear(fc=self_output.dense, transform=gamma(), gam=gam)
        self.LayerNorm = ModifiedLayerNorm(norm_layer=self_output.LayerNorm
                                           # normalized_shape=self_output.dense.weight.shape[1]
                                           )


    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class ModifiedBertAttention(nn.Module):
    def __init__(self, attention, gam=.09):
        super(ModifiedBertAttention, self).__init__()
        self.self = ModifiedBertSelfAttention(attention.self)
        self.output = ModifiedBertSelfOutput(attention.output)

    def forward(self, hidden_states):
        self_output = self.self(hidden_states)
        attention_output = self.output(self_output, hidden_states)

        return attention_output


class ModifiedBertIntermediate(nn.Module):
    def __init__(self, intermediate, gam=.15):
        super(ModifiedBertIntermediate, self).__init__()
        self.dense = ModifiedLinear(fc=intermediate.dense, transform=gamma(), gam=gam)
        self.intermediate_act_fn = ModifiedAct(intermediate.intermediate_act_fn)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ModifiedBertOutput(nn.Module):
    def __init__(self, output, gam=.09):
        super(ModifiedBertOutput, self).__init__()
        self.dense = output.dense #ModifiedLinear(fc=output.dense, transform=gamma(),gam=gam)
        self.LayerNorm = ModifiedLayerNorm(norm_layer=output.LayerNorm
                                           # normalized_shape=output.dense.weight.shape[1]
                                           )

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class ModifiedBertLayer(nn.Module):
    def __init__(self, layer, gam=.15):
        super(ModifiedBertLayer, self).__init__()
        self.attention = ModifiedBertAttention(layer.attention)
        self.intermediate = ModifiedBertIntermediate(layer.intermediate, gam=gam)
        self.output = ModifiedBertOutput(layer.output)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        hidden_states = self.output(intermediate_output, attention_output)

        return hidden_states


class ModifiedBertEncoder(nn.Module):
    def __init__(self, encoder, gam=0.15):
        super(ModifiedBertEncoder, self).__init__()
        layers = []
        for i, layer in enumerate(encoder.layer):
            layers.append(ModifiedBertLayer(layer, gam=gam))
        self.layer = nn.ModuleList(layers)

    def forward(self, hidden_states):
        for i, layer in enumerate(self.layer):
            hidden_states = layer(hidden_states)
        return hidden_states


class ModifiedBertPooler(nn.Module):
    def __init__(self, pooler, gam=0.15):
        super(ModifiedBertPooler, self).__init__()
        self.dense = ModifiedLinear(fc=pooler.dense, transform=gamma(), gam=gam)
        self.activation = ModifiedAct(pooler.activation)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class ModifiedBertModel(nn.Module):
    def __init__(self, bert, gam=0.15, add_pooling_layer=True):
        super(ModifiedBertModel, self).__init__()
        self.encoder = ModifiedBertEncoder(bert.encoder, gam=gam)
        self.add_pooling_layer = add_pooling_layer
        if add_pooling_layer:
            self.pooler = ModifiedBertPooler(bert.pooler, gam=gam)

    def forward(self, x):
        hidden_states = self.encoder(x)
        if self.add_pooling_layer:
            hidden_states = self.pooler(hidden_states)

        return hidden_states


class ModifiedBertForSequenceClassification(nn.Module):
    def __init__(self, bert_classification, gam=0.15):
        super(ModifiedBertForSequenceClassification, self).__init__()
        self.bert = ModifiedBertModel(bert_classification.bert, gam=gam)
        self.classifier = bert_classification.classifier # ModifiedLinear(fc=bert_classification.classifier, transform=gamma(), gam=.09)

    def forward(self, x):
        hidden_states = self.bert(x)
        hidden_states = self.classifier(hidden_states)

        return hidden_states
