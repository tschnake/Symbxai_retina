from typing import Any, Tuple
from torch.nn.modules import Module
import torch
from torch import nn as nn
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

    :param layer: a layer which its parameters are going to be transformed.
    :param transform: a transformation function.
    :return: a new layer with modified parameters.
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
            fc: torch.nn.Linear,
            transform: Any,
            zero_bias: bool = False
    ):
        """
        A wrapper to make torch.nn.Linear explainable.
        -------------------

        :param fc: a fully-connected layer (torch.nn.Linear).
        :param transform: a transformation function to modify the layer's parameters.
        :param zero_bias: set the layer's bias to zero. It is useful when checking the conservation property.
        """
        super(ModifiedLinear, self).__init__()
        self.fc = fc

        if zero_bias:
            self.fc.bias = torch.nn.Parameter(torch.zeros(self.fc.bias.shape))
        
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
            norm_layer: torch.nn.LayerNorm,
            normalized_shape: Tuple,
            eps: float = 1e-12,
            zero_bias: bool = False
    ):
        """
        A wrapper to make torch.nn.LayerNorm explainable.
        -------------------

        :param norm_layer: a norm layer (torch.nn.LayerNorm).
        :param normalized_shape:
        :param eps: a value added to the denominator for numerical stability
        :param zero_bias: set the layer's bias to zero. It is useful when checking the conservation property.
        """
        super(ModifiedLayerNorm, self).__init__()

        if zero_bias:
            norm_layer.bias = torch.nn.Parameter(torch.zeros(norm_layer.bias.shape))
        
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
        act: Any
    ):
        """
       A wrapper to make activation layers such as torch.nn.Tanh or torch.nn.ReLU explainable.
       -------------------

       :param act: an activation layer (torch.nn.Tanh or torch.nn.ReLU).
       """
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
