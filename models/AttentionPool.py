import torch
import torch.nn as nn
import numpy as np

from models.Block.MixInvertedResidual import MixInvertedResidual
from sklearn.neighbors import NearestNeighbors


class AttentionPool(nn.Module):
    def __init__(self, in_channels=160, ex_channels=960, num_clusters=20, vlad_v2=False):
        """
        注意力池化模型

        :param in_channels: 输入通道数
        :param ex_channels: 逆残差的通道膨胀数
        :param num_clusters: 聚类的数量
        :param vlad_v2: True时表示VLAD V2，否则为V1
        """
        super(AttentionPool, self).__init__()

        self.__block = MixInvertedResidual(in_channels, ex_channels, in_channels)

        self.__alpha = 0
        self.__conv = nn.Conv2d(in_channels, num_clusters, kernel_size=(1, 1), bias=False)
        # 初始化中心点的维度是(num_clusters, encoding_dim)
        self.__centroids = nn.Parameter(torch.rand(num_clusters, in_channels))
        self.__vlad_v2 = vlad_v2

    def init_params(self, clusters, descriptors):
        """
        初始化PatchNetVLAD的参数

        :param clusters: 图像聚类后的中心点
        :param descriptors: 经过BackBone后的图像描述
        """

        if not self.__vlad_v2: # 不是VLAD V2的参数初始化
            # 执行L2范数，并对原始数据进行正则化操作
            clusters_assignment = clusters / np.linalg.norm(clusters, axis=1, keepdims=True)
            # 计算中心点特征和图像特征之间的余弦距离
            cos_dis = np.dot(clusters_assignment, descriptors.T)
            cos_dis.sort(0)
            # 对余弦距离进行降序排列
            cos_dis = cos_dis[::-1, :]
            #
            self.__alpha = (np.log(0.01) / np.mean(cos_dis[0, :] - cos_dis[1, :])).item()
            # 使用聚类后的中心点来初始化中心点参数
            self.__centroids = nn.Parameter(torch.from_numpy(clusters))

            # 初始化卷积权重和偏置
            self.__conv.weight = nn.Parameter(
                torch.from_numpy(self.__alpha * clusters_assignment).unsqueeze(2).unsqueeze(3))
            self.__conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(descriptors)
            del descriptors

            # 通过KNN算法得到与中心点最近的2个图像的索引
            distance_square = np.square(knn.kneighbors(clusters, 2)[1])
            del knn

            self.__alpha = (-1 * np.log(0.01) / np.mean(distance_square[:, 1] - distance_square[:, 0])).item()
            # 使用聚类后的中心点来初始化中心点参数
            self.__centroids = nn.Parameter(torch.from_numpy(clusters))
            del clusters, distance_square

            # 初始化卷积权重和偏置
            self.__conv.weight = nn.Parameter((2.0 * self.__alpha * self.__centroids).unsqueeze(-1).unsqueeze(-1))
            self.__conv.bias = nn.Parameter(-1 * self.__alpha * self.__centroids.norm(dim=1))

    def forward(self, x):
        out = self.__block(x)
        B, C, H, W = out.size()

        out = out.view(B, C, -1)
        out = out.sum(-1)

        return out
