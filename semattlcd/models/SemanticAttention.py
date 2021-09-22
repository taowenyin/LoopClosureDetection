import torch
import torch.nn as nn


class SemanticAttention(nn.Module):
    def __init__(self, dim=512):
        super(SemanticAttention, self).__init__()

        self.dim = dim

    def forward(self, x):
        return x
