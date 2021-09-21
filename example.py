import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torchvision import models


if __name__ == '__main__':
    # 创建DeepLabV3
    deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)

    # 获取BackBone
    backbone = deeplab.backbone
    # 获取ASPP
    aspp = deeplab.classifier[0]

    # layers = [backbone, aspp]
    # enc = nn.Sequential(*layers)
    #
    conv1 = deeplab.backbone.conv1
    bn1 = deeplab.backbone.bn1
    relu = deeplab.backbone.relu
    maxpool = deeplab.backbone.maxpool
    layer1 = deeplab.backbone.layer1

    # input = torch.randn(1, 3, 224, 224)
    input = torch.randn(2, 3, 352, 480)

    print(input.shape)
    print('==============================')

    output = backbone(input)['out']
    print(output.shape)
    print('===========backbone==========')

    aspp_output = aspp(output)
    print(aspp_output.shape)
    print('===========aspp1==========')

    aspp_output = F.interpolate(aspp_output, size=(int(math.ceil(input.size()[-2] / 4)), int(math.ceil(input.size()[-1] / 4))),
                                mode='bilinear', align_corners=True)
    print(aspp_output.shape)
    print('===========aspp2==========')

    output = conv1(input)
    output = bn1(output)
    output = relu(output)
    output = maxpool(output)
    output = layer1(output)
    print(output.shape)
    print('=========layer1==============')

    conv2 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
    bn2 = nn.BatchNorm2d(256)
    relu2 = nn.ReLU()

    output = conv2(output)
    output = bn2(output)
    output = relu2(output)
    print(output.shape)
    print('=========conv2==============')

    output = torch.cat((aspp_output, output), dim=1)
    print(output.shape)
    print('=========cat==============')

    # print(enc.backbone)
    # print('==============================')
    # print(enc.classifier[0])
    # print('==============================')
    # print(enc)

    # print(aspp.convs)
    # print(aspp)
    # print(deeplab)