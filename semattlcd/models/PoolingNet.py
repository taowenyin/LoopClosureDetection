import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from semattlcd.models.ShuffleAttention import ShuffleAttention


class PoolingNet(nn.Module):
    def __init__(self, dim, config):
        super(PoolingNet, self).__init__()

        self.config = config

        pca_dim = math.ceil(dim * float(config['pca_rate']))

        if config['arch_type'] == 'mobilenet':
            self.upsample = nn.Conv2d(dim, 256, kernel_size=(1, 1), stride=1, padding=0)
            dim = 256

        self.attention = ShuffleAttention(dim, G=8)
        self.pca_dim = pca_dim
        self.flatten = nn.Flatten(start_dim=1)
        self.pca_conv = nn.Conv2d(dim, pca_dim, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        if self.config['arch_type'] == 'mobilenet':
            x = self.upsample(x)
        x = self.attention(x)
        x = self.pca_conv(x)
        x = self.flatten(x)
        x = F.normalize(x, p=2, dim=1)

        return x
