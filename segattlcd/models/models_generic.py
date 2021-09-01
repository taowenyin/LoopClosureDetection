import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class Flatten(nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_data):
        return F.normalize(input_data, p=2, dim=self.dim)


def get_backend():
    enc_dim = 512
    enc = models.vgg16()
    layers = list(enc.features.children())[:-2]
    # only train conv5_1, conv5_2, and conv5_3 (leave rest same as Imagenet trained weights)
    for layer in layers[:-5]:
        for p in layer.parameters():
            p.requires_grad = False
    enc = nn.Sequential(*layers)
    return enc_dim, enc


def get_model(encoder, encoder_dim, config, append_pca_layer=False):
    nn_model = nn.Module()
    nn_model.add_module('encoder', encoder)

    if append_pca_layer:
        num_pcs = int(config['num_pcs'])
        encoder_output_dim = encoder_dim

        pca_conv = nn.Conv2d(encoder_output_dim, num_pcs, kernel_size=(1, 1), stride=1, padding=0)
        nn_model.add_module('WPCA', nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)]))

    return nn_model