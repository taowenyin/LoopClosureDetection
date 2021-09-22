import torch.nn as nn
from semattlcd.models.BackBone import BackBone
from semattlcd.models.SemanticAttention import SemanticAttention


def get_backend():
    # 编码器的输出通道数
    enc_dim = 512
    # 编码器模型
    enc = BackBone()

    return enc_dim, enc


def get_model(encoder, encoder_dim):
    nn_model = nn.Module()
    nn_model.add_module('encoder', encoder)

    pooling_net = SemanticAttention(dim=encoder_dim)
    nn_model.add_module('pool', pooling_net)

    return nn_model
