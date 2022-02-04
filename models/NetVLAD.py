import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.neighbors import NearestNeighbors


class NetVLAD(nn.Module):
    def __init__(self, num_clusters, encoding_dim, normalize_input=True, vlad_v2=False):
        """
        NetVLAD模型

        :param num_clusters: 聚类的数量
        :param encoding_dim: 图像特征的编码
        :param vlad_v2: True时表示VLAD V2，否则为V1
        """
        super(NetVLAD, self).__init__()

        self.__num_clusters = num_clusters
        self.__encoding_dim = encoding_dim
        self.__alpha = 0
        self.__normalize_input = normalize_input
        self.__vlad_v2 = vlad_v2

        self.__conv = nn.Conv2d(encoding_dim, num_clusters, kernel_size=(1, 1), bias=vlad_v2)
        # 初始化中心点的维度是(num_clusters, encoding_dim)
        self.__centroids = nn.Parameter(torch.rand(num_clusters, encoding_dim))

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
        B, C = x.shape[:2]

        # 跨通道归一化
        if self.__normalize_input:
            x = F.normalize(x, p=2, dim=1)

        # ======步骤一：NetVLAD的soft-assignment部分======
        # 经过一个1x1的卷积，从（B, C, H, W）->(B, K, HxW)
        soft_assign = self.__conv(x).view(B, self.__num_clusters, -1)
        # 经过Softmax得到soft-assignment
        soft_assign = F.softmax(soft_assign, dim=1)
        # =============================================

        # 从（B, C, H, W）->(B, K, HxW)
        x_flatten = x.view(B, C, -1)

        # 创建用于保存每个聚类残差值的Tensor
        vlad = torch.zeros([B, self.__num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)

        # 循环计算X与每个重点之间的残差，并保存在store_residual中
        for i in range(self.__num_clusters):
            # =================================步骤二：NetVLAD的VLAD core部分==================================
            # 把 (B, C, HxW)的X变为(B, 1, C, HxW)，用于与后续的num_clusters个聚类中心点进行残差计算，其中的1表示就是就是聚类个数
            input_x = x_flatten.unsqueeze(0).permute(1, 0, 2, 3)
            # 取出每个聚类中心，形状为(1, encoding_dim)，把该中心点的形状变为(HxW, 1, encoding_dim)，
            # 再变为(1, encoding_dim, HxW)，再变为(1, 1, encoding_dim, HxW)，与X的形状保持一致
            centroids = self.__centroids[i:i + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)

            # 计算X与每个中心点的残差
            residual = input_x - centroids

            # =====================步骤三：VLAD core与soft-assignment相乘==========================
            # soft-assignment作为α与残差相乘，并且把形状为(B, 1, H, W)的soft_assign变为(B, 1, 1, H, W)，
            # 第1个表示聚类中的一个，第2个1是增加的维度
            soft_assign_ = soft_assign[:, i:i + 1, :].unsqueeze(2)
            residual *= soft_assign_

            # 对所有残差求和，得到全局的VLAD特征
            vlad_ = residual.sum(-1)
            # 保存残差
            vlad[:, i:i + 1, :] = vlad_

        # intra-normalization
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(x.size(0), -1)
        # L2 normalize
        vlad = F.normalize(vlad, p=2, dim=1)

        return vlad


if __name__ == '__main__':
    if torch.cuda.is_available():
        cuda = True
    else:
        cuda = False
    device = torch.device("cuda" if cuda else "cpu")

    data = torch.rand(2, 512, 60, 80).to(device)

    image_clusters = np.random.rand(20, 512).astype(np.float32)
    image_descriptors = np.random.rand(50000, 512).astype(np.float32)

    model = NetVLAD(20, 512)
    model.init_params(image_clusters, image_descriptors)

    model = model.to(device)

    output = model(data)

    print('xx')
