import torch
import torch.nn as nn
import numpy as np


class LCDNet(nn.Module):
    def __init__(self, encoding_model, pooling_model, pca_model=None):
        """
        回环检测模型

        :param encoding_model: 编码层模型
        :param pooling_model: 编码层模型
        """

        super(LCDNet, self).__init__()

        self.__encoding_model = encoding_model
        self.__pooling_model = pooling_model
        self.__pca_model = pca_model

    @property
    def encoder(self):
        return self.__encoding_model

    @property
    def pool(self):
        return self.__pooling_model

    def forward(self, x):
        out = self.__encoding_model(x)
        out = self.__pooling_model(out)

        if self.__pca_model is not None:
            out = self.__pca_model(out)

        return out
