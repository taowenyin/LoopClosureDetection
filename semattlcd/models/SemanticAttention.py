import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAttention(nn.Module):
    def __init__(self, dim, pca_dim):
        super(SemanticAttention, self).__init__()

        self.pca_dim = pca_dim
        self.flatten = nn.Flatten(start_dim=1)
        self.pca_conv = nn.Conv2d(dim, pca_dim, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.pca_conv(x)
        out = self.flatten(out)
        out = F.normalize(out, p=2, dim=1)

        return out
