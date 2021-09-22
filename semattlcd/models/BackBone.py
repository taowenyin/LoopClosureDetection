import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torchvision import models


class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()

        # 创建DeepLabV3
        deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)

        # 获取BackBone
        self.backbone = deeplab.backbone

        # 获取1/4特征
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1 = self.backbone.layer1

        # 获取ASPP
        self.aspp = deeplab.classifier[0]

        # Low Level要经过1x1的卷积
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # 经过backbone的Layer1获得1/4的特征
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.maxpool(output)
        low_level_features = self.layer1(output)

        # 经过backbone和ASPP获取1/8特征
        output = self.backbone(x)
        aspp_output = self.aspp(output)

        # 把ASPP变大2倍，与low_level_features大小相同，为1/4
        aspp_output = F.interpolate(aspp_output, size=(int(math.ceil(x.size()[-2] / 4)), int(math.ceil(x.size()[-1] / 4))),
                          mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu2(low_level_features)

        # 拼接ASPP和low_level_features后的结果
        output = torch.cat((aspp_output, low_level_features), dim=1)

        return output