import torch.nn as nn


class ConvMixerLayer(nn.Module):
    def __init__(self, dim, kernel_size=9):
        super().__init__()

        # 残差结构
        self.Resnet = nn.Sequential(
            # 定义一个深度卷积，并且通过padding='same'使得输入输出的分辨率相同
            nn.Conv2d(dim, dim, kernel_size=(kernel_size, kernel_size), groups=dim, padding='same'),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        # 定义一个逐点卷积
        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1,1)),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = x + self.Resnet(x)
        x = self.Conv_1x1(x)
        return x
