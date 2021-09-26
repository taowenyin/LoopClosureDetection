import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torchvision import models
from typing import List


class BackBone(nn.Module):
    def __init__(self, config):
        super(BackBone, self).__init__()

        if config['arch_type'] == 'resnet50':
            deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)
        elif config['arch_type'] == 'resnet101':
            deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True)
        elif config['arch_type'] == 'mobilenet':
            deeplab = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
        else:
            deeplab = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)

        # 获取BackBone
        self.backbone = deeplab.backbone

        # 1/4特征的结构和activate layout
        if config['arch_type'] == 'resnet50' or config['arch_type'] == 'resnet101':
            quarter_layers: List[nn.Module] = [
                # 获取1/4特征
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu,
                self.backbone.maxpool,
                self.backbone.layer1,
            ]
            activate_layers: List[nn.Module] = [
                # Low Level要经过1x1的卷积
                nn.Conv2d(256, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ]
        elif config['arch_type'] == 'mobilenet':
            quarter_layers: List[nn.Module] = [
                # 获取1/4特征
                self.backbone['0'],
                self.backbone['1'],
                self.backbone['2'],
                self.backbone['3'],
            ]
            activate_layers: List[nn.Module] = [
                # Low Level要经过1x1的卷积
                nn.Conv2d(24, 24, kernel_size=1, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU()
            ]
        else:
            quarter_layers: List[nn.Module] = [
                # 获取1/4特征
                self.backbone['0'],
                self.backbone['1'],
                self.backbone['2'],
                self.backbone['3'],
            ]
            activate_layers: List[nn.Module] = [
                # Low Level要经过1x1的卷积
                nn.Conv2d(24, 24, kernel_size=1, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU()
            ]

        # 获取1/4特征
        self.quarter_block = nn.Sequential(*(quarter_layers + activate_layers))
        # 获取ASPP
        self.aspp = deeplab.classifier[0]

    def forward(self, x):
        # 经过backbone的Layer1获得1/4的特征
        low_level_features = self.quarter_block(x)

        # 经过backbone和ASPP获取1/8特征
        output = self.backbone(x)
        aspp_output = self.aspp(output["out"])

        # 把ASPP变大2倍，与low_level_features大小相同，为1/4
        aspp_output = F.interpolate(aspp_output, size=(int(math.ceil(x.size()[-2] / 4)), int(math.ceil(x.size()[-1] / 4))),
                          mode='bilinear', align_corners=True)

        # 拼接ASPP和low_level_features后的结果
        output = torch.cat((aspp_output, low_level_features), dim=1)

        return output