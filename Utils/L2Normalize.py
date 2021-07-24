import torch.nn.functional as F
import torch.nn as nn


class L2Normalize(nn.Module):
    def __init__(self, dim=1):
        super(L2Normalize, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)