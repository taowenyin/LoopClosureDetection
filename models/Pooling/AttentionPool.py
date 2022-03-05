import torch.nn as nn

from models.Attention.CoordAttention import CoordAttention


class AttentionPool(nn.Module):
    def __init__(self, in_channels=160, reduction=32):
        """
        注意力池化模型

        :param in_channels: 输入通道数
        :param reduction: 通过缩小率
        """
        super(AttentionPool, self).__init__()

        self.__attention = CoordAttention(in_channels, in_channels, reduction=reduction)
        self.__pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.__flatten = nn.Flatten()

    def forward(self, x):
        out = self.__attention(x)
        out = self.__pooling(out)
        out = self.__flatten(out)

        return out
