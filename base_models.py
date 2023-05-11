from torch.nn.modules import Module
import torch
from torch import nn as nn
from transformers import BertForSequenceClassification, BertTokenizer, AutoTokenizer, AutoModelForPreTraining
import numpy as np
import copy

def stabilize(z):
    return z + ((z == 0.).to(z) + z.sign()) * 1e-6


def modified_layer(
        layer,
        transform
):
    """
    This function creates a copy of a layer and modify
    its parameters based on a transformation function 'transform'.
    -------------------
    :param layer: A layer which its parameters are going to be transformed.
    :param transform: A transformation function.
    :return: A new layer with modified parameters.
    """
    new_layer = copy.deepcopy(layer)

    try:
        new_layer.weight = torch.nn.Parameter(transform(layer.weight.float()))
    except AttributeError as e:
        print(e)

    try:
        new_layer.bias = torch.nn.Parameter(transform(layer.bias.float()))
    except AttributeError as e:
        print(e)

    return new_layer


class ModifiedLinear(Module):
    def __init__(
            self,
            fc,
            transform
    ):
        super(ModifiedLinear, self).__init__()
        self.fc = fc
        
        # TODO: Do not set bias to 0.
        # self.fc.bias = torch.nn.Parameter(torch.zeros(self.fc.bias.shape))
        
        self.transform = transform
        self.modified_fc = modified_layer(layer=fc, transform=transform)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        z = self.fc(x)
        zp = self.modified_fc(x)
        zp = stabilize(zp)
        return (zp.double() * (z.double() / zp.double()).data.double()).float()

    
class ModifiedLayerNorm(Module):
    def __init__(
            self,
            norm_layer,
            normalized_shape,
            eps=1e-12
    ):
        super(ModifiedLayerNorm, self).__init__()
        # TODO: Do not set bias to 0.
        # norm_layer.bias = torch.nn.Parameter(torch.zeros(norm_layer.bias.shape))
        
        self.norm_layer = norm_layer
        self.weight = norm_layer.weight
        self.bias = norm_layer.bias
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:

        z = self.norm_layer(input)
        mean = input.mean(dim=-1, keepdim=True)
        var = torch.var(input, unbiased=False, dim=-1, keepdim=True)
        denominator = torch.sqrt(var + self.eps)
        denominator = denominator.detach()
        zp = ((input - mean) / denominator) * self.weight + self.bias
        zp = stabilize(zp)
        return (zp.double() * (z.double() / zp.double()).data.double()).float()
    
class ModifiedAct(Module):
    def __init__(
        self,
        act
    ):
        super(ModifiedAct, self).__init__()
        self.modified_act = nn.Identity()
        self.act = act
    
    def forward(
        self,
        x
    ):
        z = self.act(x)
        zp = self.modified_act(x)
        zp = stabilize(zp)
        return (zp.double() * (z.double() / zp.double()).data.double()).float()
    
class ModifiedTanh(Module):
    def __init__(
        self,
        act
    ):
        super(ModifiedTanh, self).__init__() 
        self.act = act
        self.modified_act = nn.Identity()
    
    def forward(
        self,
        x
    ):
        z = self.act(x)
        zp = self.modified_act(x)
        zp = stabilize(zp)
        return (zp.double() * (z.double() / zp.double()).data.double()).float()