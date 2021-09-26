import torch.nn as nn
import math

from semattlcd.models.BackBone import BackBone
from semattlcd.models.SemanticAttention import SemanticAttention


def get_backend(config):
    # 编码器的输出通道数
    if config['arch_type'] == 'resnet50' or config['arch_type'] == 'resnet101':
        enc_dim = 512
    elif config['arch_type'] == 'mobilenet':
        enc_dim = 280
    else:
        enc_dim = 280
    # 编码器模型
    enc = BackBone(config)

    return enc_dim, enc


def get_model(encoder, encoder_dim, config):
    nn_model = nn.Module()
    nn_model.add_module('encoder', encoder)

    pooling_net = SemanticAttention(encoder_dim, pca_dim=math.ceil(encoder_dim * float(config['pca_rate'])))
    nn_model.add_module('pool', pooling_net)

    return nn_model
