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
        self.LayerNorm = ModifiedLayerNorm(norm_layer=self_output.LayerNorm,
                                           normalized_shape=self_output.dense.weight.shape[1])

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


# ------ LRP for BERT Model ------
class ModifiedBertSelfAttention(nn.Module):
    def __init__(self, self_attention):
        super(ModifiedBertSelfAttention, self).__init__()
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
        self.dense = ModifiedLinear(fc=self_output.dense, transform=gamma())
        self.LayerNorm = ModifiedLayerNorm(norm_layer=self_output.LayerNorm,
                                           normalized_shape=self_output.dense.weight.shape[1])

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
        self.dense = ModifiedLinear(fc=intermediate.dense, transform=gamma())
        self.intermediate_act_fn = ModifiedAct(intermediate.intermediate_act_fn)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ModifiedBertOutput(nn.Module):
    def __init__(self, output):
        super(ModifiedBertOutput, self).__init__()
        self.dense = ModifiedLinear(fc=output.dense, transform=gamma())
        self.LayerNorm = ModifiedLayerNorm(norm_layer=output.LayerNorm,
                                           normalized_shape=output.dense.weight.shape[1])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class ModifiedBertLayer(nn.Module):
    def __init__(self, layer):
        super(ModifiedBertLayer, self).__init__()
        self.attention = ModifiedBertAttention(layer.attention)
        self.intermediate = ModifiedBertIntermediate(layer.intermediate)
        self.output = ModifiedBertOutput(layer.output)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        hidden_states = self.output(intermediate_output, attention_output)

        return hidden_states


class ModifiedBertEncoder(nn.Module):
    def __init__(self, encoder):
        super(ModifiedBertEncoder, self).__init__()
        layers = []
        for i, layer in enumerate(encoder.layer):
            layers.append(ModifiedBertLayer(layer))
        self.layer = nn.ModuleList(layers)

    def forward(self, hidden_states):
        for i, layer in enumerate(self.layer):
            hidden_states = layer(hidden_states)
        return hidden_states


class ModifiedBertPooler(nn.Module):
    def __init__(self, pooler):
        super(ModifiedBertPooler, self).__init__()
        self.dense = ModifiedLinear(fc=pooler.dense, transform=gamma())
        self.activation = ModifiedAct(pooler.activation)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class ModifiedBertModel(nn.Module):
    def __init__(self, bert, add_pooling_layer=True):
        super(ModifiedBertModel, self).__init__()
        self.encoder = ModifiedBertEncoder(bert.encoder)
        self.add_pooling_layer = add_pooling_layer
        if add_pooling_layer:
            self.pooler = ModifiedBertPooler(bert.pooler)

    def forward(self, x):
        hidden_states = self.encoder(x)
        if self.add_pooling_layer:
            hidden_states = self.pooler(hidden_states)

        return hidden_states


class ModifiedBertForSequenceClassification(nn.Module):
    def __init__(self, bert_classification):
        super(ModifiedBertForSequenceClassification, self).__init__()
        self.bert = ModifiedBertModel(bert_classification.bert)
        self.classifier = ModifiedLinear(fc=bert_classification.classifier, transform=gamma())

    def forward(self, x):
        hidden_states = self.bert(x)
        hidden_states = self.classifier(hidden_states)

        return hidden_states


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

class BERTSymbXAI(SymbXAI):
    def __init__(
            self,
            sample,
            target,
            model,
            embeddings,
            scal_val=1.
    ):
        model.zero_grad()

        # Prepare the input embeddings.
        x = embeddings(
            input_ids=sample['input_ids'],
            token_type_ids=sample['token_type_ids']
        )

        # Make the model explainable.
        modified_model = ModifiedBertForSequenceClassification(
            model
        )

        if len(x.shape) >= 3:
            batch_dim = True
            num_tokens = x.shape[1]
        else:
            batch_dim = False
            num_tokens = x.shape[0]

        lamb = torch.ones((num_tokens, num_tokens))
        lamb_last_layer = torch.zeros((num_tokens, num_tokens))

        layers = []
        for layer in modified_model.bert.encoder.layer:
            layers.append(layer)

        def output_module(hidden_states):
            pooled_data = modified_model.bert.pooler(hidden_states)
            logits = modified_model.classifier(pooled_data)
            output = (logits * target).sum().unsqueeze(0).unsqueeze(0)
            return output

        layers.append(output_module)

        lamb_last_layer[0, :] = torch.ones(num_tokens)
        lambs = [lamb for _ in range(len(layers) - 2)] + [lamb_last_layer] + [torch.ones(num_tokens).unsqueeze(0)]

        super().__init__(
            layers,
            x.data,
            num_tokens,
            lambs,
            R_T=None,
            batch_dim=batch_dim,
            scal_val=scal_val
        )

    def subgraph_relevance(
            self,
            subgraph,
            from_walks=False
    ):

        # TODO: Change the code for from_walks=True
        if from_walks:
            if self.walk_rels_tens is None:
                _ = self.walk_relevance(rel_rep='tens')  # Just build the tensor.

            # Transform subgraph which is given by a set of node representations,
            # into a set of node identifications.
            subgraph_idn = [self.node2idn[idn] for idn in subgraph]

            # Define the mask for the subgraph.
            m = torch.zeros((self.walk_rels_tens.shape[0],))
            for ft in subgraph_idn:
                m[ft] = 1
            ms = [m] * self.num_layer

            # Extent the masks by an artificial dimension.
            for dim in range(self.num_layer):
                for unsqu_pos in [0] * (self.num_layer - 1 - dim) + [-1] * dim:
                    ms[dim] = ms[dim].unsqueeze(unsqu_pos)

            # Perform tensor-product.
            m = reduce(lambda x, y: x * y, ms)
            assert self.walk_rels_tens.shape == m.shape, f'R.shape = {self.walk_rels_tens.shape}, m.shape = {m.shape}'

            # Just sum the relevance scores where the mask is non-zero.
            R_subgraph = (self.walk_rels_tens * m).sum()

            return R_subgraph * self.scal_val
        else:
            # Initialize the last relevance.
            curr_subgraph_node = Node(
                0,
                self.lamb_per_layer[self.num_layer - 1],
                None,
                self.R_T[0] if not self.batch_dim else self.R_T[0, 0],
                domain_restrict=None
            )

            for act, layer, layer_id in list(zip(self.xs[:-1], self.layers, range(len(self.layers))))[::-1]:
                # Iterate over the nodes.
                R = self._relprop_standard(act,
                                           layer,
                                           curr_subgraph_node.R,
                                           curr_subgraph_node.node_rep)

                if layer_id == 12:
                    # Create new subgraph nodes.
                    new_node = Node(0,
                                    self.lamb_per_layer[layer_id - 1],
                                    curr_subgraph_node,
                                    R[0] if not self.batch_dim else R[0, 0],
                                    domain_restrict=None
                                    )
                else:
                    # Create new subgraph nodes.
                    new_node = Node(subgraph,
                                    self.lamb_per_layer[layer_id - 1],
                                    curr_subgraph_node,
                                    R[subgraph] if not self.batch_dim else R[0, subgraph],
                                    domain_restrict=None
                                    )

                curr_subgraph_node = new_node

            return curr_subgraph_node.R.sum() * self.scal_val
