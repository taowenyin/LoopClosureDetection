import torch
import torch.nn as nn

from models.Block.MixInvertedResidual import MixInvertedResidual


class AttentionPool(nn.Module):
    def __init__(self, in_channels=160, ex_channels=960, out_channels=160):
        super(AttentionPool, self).__init__()

        self.block = MixInvertedResidual(in_channels, ex_channels, out_channels)

    def forward(self, x):
        out = self.block(x)

        return out
