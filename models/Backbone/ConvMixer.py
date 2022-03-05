import torch.nn as nn

from models.Block.ConvMixerLayer import ConvMixerLayer


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7):
        """
        ConvMixer网络

        :param dim: Patch的通道数
        :param depth: ConvMixerLayer的数量
        :param kernel_size: ConvMixerLayer中卷积核的大小
        :param patch_size: Patch的大小
        """
        super(ConvMixer, self).__init__()
        self.patch_embedding = nn.Sequential(
            # (B,C,H,W)->(B,dim,H/p,H/p)
            nn.Conv2d(3, dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size)),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

        self.ConvMixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.ConvMixer_blocks.append(ConvMixerLayer(dim=dim, kernel_size=kernel_size))

    def forward(self, x):
        # 编码时的卷积
        x = self.patch_embedding(x)
        # 多层ConvMixer_block的计算
        for ConvMixer_block in self.ConvMixer_blocks:
            x = ConvMixer_block(x)

        # (B,dim,H/p,H/p)
        return x
