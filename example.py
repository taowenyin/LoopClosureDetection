import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from typing import List


if __name__ == '__main__':
    # 创建DeepLabV3
    # deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)
    # deeplab = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True)

    # 获取BackBone
    backbone = deeplab.backbone
    # 获取ASPP
    aspp = deeplab.classifier[0]

    quarter_layers: List[nn.Module] = [
        # 获取1/4特征
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
    ]

    activate_layers: List[nn.Module] = [
        # Low Level要经过1x1的卷积
        nn.Conv2d(256, 256, kernel_size=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU()
    ]

    quarter_block = nn.Sequential(*(quarter_layers + activate_layers))

    images = torch.randn(4, 3, 352, 480)

    out = quarter_block(images)

    print(deeplab)

    print('xxx')
