import torch

from typing import List
from torchvision.ops.misc import ConvNormActivation
from torch import nn, Tensor
from models.Attention.CoordAttention import CoordAttention


class MixInvertedResidual(nn.Module):
    def __init__(self, in_channels, ex_channels, out_channels):
        super(MixInvertedResidual, self).__init__()

        layers: List[nn.Module] = []

        # expand
        layers.append(ConvNormActivation(in_channels, ex_channels, kernel_size=1,
                                         activation_layer=torch.nn.Hardswish))
        # depthwise
        layers.append(ConvNormActivation(ex_channels, ex_channels, kernel_size=5, stride=1, groups=ex_channels,
                                         activation_layer=torch.nn.Hardswish))
        # Attention
        layers.append(CoordAttention(ex_channels, ex_channels, reduction=4))
        # project
        layers.append(ConvNormActivation(ex_channels, out_channels, kernel_size=1, activation_layer=None))

        self.block = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        result += input

        return result
