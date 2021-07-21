import torch
import torch.nn as nn
import torch.nn.functional as F


# https://www.di.ens.fr/willow/research/netvlad/
# https://zhuanlan.zhihu.com/p/148401141
# https://zhuanlan.zhihu.com/p/148249219
# https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
# https://blog.csdn.net/Yang_137476932/article/details/105169329


class NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, conv_output_channels=128, alpha=1.0, normalize_input=True):
        """
        NetVLAD模型实现

        :param num_clusters : int 聚类的数量
        :param conv_output_channels: int CNN输出的特征通道数
        :param alpha: float 初始化参数。参数越大，越难聚类
        :param normalize_input: bool 假如为True，那么使用L2正则化
        """
        super(NetVLAD, self).__init__()

        self.num_clusters = num_clusters
        self.conv_output_channels = conv_output_channels
        self.alpha = alpha
        self.normalize_input = normalize_input

        # NetVLAD中的第一个1x1卷积
        self.conv = nn.Conv2d(conv_output_channels, num_clusters, kernel_size=1, bias=True)
        # 每个聚类的中心点
        self.centroids = torch.rand(num_clusters, conv_output_channels)

        # self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            -1 * self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        # 获取Batch和Chanel的大小
        B, C = x.shape[:2]

        if self.normalize_input:
            # 跨层做归一化
            x = F.normalize(x, p=2, dim=1)

        print('x size = {}'.format(x.size()))

        # soft-assignment的过程
        # 把特征图变为BxCxWH
        soft_assign = self.conv(x).view(B, self.num_clusters, -1)
        print('soft_assign size = {}'.format(soft_assign.size()))
        soft_assign = F.softmax(soft_assign, dim=1)
        print('soft_assign softmax size = {}'.format(soft_assign.size()))

        # VLAD core的过程
        # 把x的WxH拉平
        x_flatten = x.view(B, C, -1)
        print('x_flatten size = {}'.format(x_flatten.size()))
        x_flatten = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3)
        print('x_flatten expand size = {}'.format(x_flatten.size()))
        # 对聚类的中心点进行变换
        centroids = self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        print('centroids size = {}'.format(centroids.size()))
        # 计算x到聚类的中心的残差，x-c
        residual = x_flatten - centroids
        print('residual size = {}'.format(residual.size()))
        soft_assign = soft_assign.unsqueeze(2)
        print('soft_assign unsqueeze size = {}'.format(soft_assign.size()))
        residual = residual * soft_assign
        print('residual mul size = {}'.format(residual.size()))
        # 得到VLAD向量
        vlad = residual.sum(dim=-1)
        print('vlad size = {}'.format(vlad.size()))

        # 求intra-normalization
        vlad = F.normalize(vlad, p=2, dim=2)
        # 把特征展平为1维向量
        vlad = vlad.view(B, -1)
        # 求L2正则化
        vlad = F.normalize(vlad, p=2, dim=1)

        return vlad


class EmbedNet(nn.Module):
    def __init__(self, base_model, netvlad):
        """
        嵌入模型
        :param base_model: Module NetVLAD的基础模型
        :param netvlad: Module NetVLAD的模型
        """
        super(EmbedNet, self).__init__()

        self.base_model = base_model
        self.netvlad = netvlad

    def forward(self, x):
        x = self.base_model(x)
        embedded_x = self.netvlad(x)

        return embedded_x


if __name__ == '__main__':
    images = torch.randn(4, 3, 352, 480)
    conv = nn.Conv2d(3, 128, kernel_size=1)
    images = conv(images)

    net = NetVLAD()
    vlad = net(images)

    print('xxx')