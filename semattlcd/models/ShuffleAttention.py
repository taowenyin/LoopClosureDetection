import torch
import torch.nn as nn

from torch.nn import Parameter


class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        # flatten
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # 对输入的特征进行分组，Shape = (B * G, C // G, H, W)
        x = x.view(b * self.G, -1, h, w)

        # 把每一组特征分为2部分，Shape = (B * G, C // G // 2, H, W)
        x_0, x_1 = x.chunk(2, dim=1)

        # 执行通道注意力
        x_channel = self.avg_pool(x_0)  # 每个通道通过AVG Pool变为（1,1），Shape = (b * G, C // G // 2, 1, 1)
        x_channel = self.cweight * x_channel + self.cweight  # 计算通道注意力参数，Shape = (b * G, C // G // 2, 1, 1)
        # 执行通道注意力的点积操作
        x_channel = x_0 * self.sigmoid(x_channel)

        # 执行空间注意力
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


if __name__ == '__main__':
    # input = torch.randn(50, 512, 7, 7)
    # se = ShuffleAttention(channel=512, G=8)
    # output = se(input)
    # print(output.shape)

    # m = nn.AdaptiveAvgPool2d(1)
    # input = torch.randn(1, 64, 8, 9)
    # output = m(input)
    # print(output.shape)

    input = torch.randn(20, 6, 10, 10)
    m = nn.GroupNorm(3, 6)
    # m = nn.GroupNorm(6, 6)
    # m = nn.GroupNorm(1, 6)
    output = m(input)
    print(output.shape)
