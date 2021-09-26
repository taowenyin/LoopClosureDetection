import torch
import torch.nn as nn


class SemanticAttention(nn.Module):
    def __init__(self, dim=512):
        super(SemanticAttention, self).__init__()

        self.dim = dim

    def forward(self, x):
        out = torch.flatten(x, start_dim=1)

        return out
