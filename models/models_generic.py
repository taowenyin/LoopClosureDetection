import torch
import torch.nn as nn
import numpy as np
import h5py
import torch.nn.functional as F

from torchvision.models import vgg16, mobilenet_v3_large, mobilenet_v3_small
from math import ceil
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset.mapillary_sls.MSLS import ImagesFromList, MSLS
from os.path import join, exists
from os import makedirs, remove
from tools import ROOT_DIR
from tqdm import tqdm
from sklearn.cluster import KMeans
from models.PatchNetVLAD import PatchNetVLAD
from models.NetVLAD import NetVLAD
from models.GLAttentionNet import GLAttentionNet
from models.AttentionPool import AttentionPool
from models.ConvMixer import ConvMixer


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


def get_backbone(config):
    """
    获取BackBone的模型

    :type config: 配置文件对象
    :return:
    encoding_model: BackBone的模型
    encoding_dim: BackBone的模型输出维度
    """

    if config['model'].get('backbone') == 'vgg16':
        # 模型的输出维度
        encoding_dim = 512
        # 图像编码模型为VGG-16，并且采用ImageNet的预训练参数
        encoding_model = vgg16(pretrained=True)

        # 获取所有的网络层
        layers = list(encoding_model.features.children())[:-2]
        # 只训练conv5_1, conv5_2, and conv5_3的参数，冻结前面所有曾的参数
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False

        # 重新构建BackBone模型
        encoding_model = nn.Sequential(*layers)
    elif config['model'].get('backbone') == 'mobilenets':
        encoding_dim = 96
        encoding_model = mobilenet_v3_small(pretrained=True)

        layers = list(encoding_model.features.children())[:-1]
        for layer in layers[:-2]:
            for p in layer.parameters():
                p.requires_grad = False

        encoding_model = nn.Sequential(*layers)
    elif config['model'].get('backbone') == 'mobilenetl':
        encoding_dim = 160
        encoding_model = mobilenet_v3_large(pretrained=True)

        layers = list(encoding_model.features.children())[:-2]
        for layer in layers[:-2]:
            for p in layer.parameters():
                p.requires_grad = False

        encoding_model = nn.Sequential(*layers)
    elif config['model'].get('backbone') == 'convmixer':
        encoding_dim = 512
        encoding_model = ConvMixer(512, depth=1, kernel_size=9, patch_size=8)
    else:
        raise ValueError('未知的BackBone类型: {}'.format(config['model'].get('backbone')))

    return encoding_model, encoding_dim


def get_model(encoding_model, encoding_dim, config, append_pca_layer=False) -> GLAttentionNet:
    """
    获取训练模型

    :param encoding_model: BackBone的模型
    :param encoding_dim: BackBone的模型输出维度
    :param config: 训练配置信息
    :param append_pca_layer: 是否添加PCA层
    :return:
    """
    pooling_model = nn.Module()
    pca_model = nn.Module()

    # 数据集名称
    dataset_name = config['dataset'].get('name')

    if config['train'].get('pooling').lower() == 'patchnetvlad':
        pooling_model = PatchNetVLAD(num_clusters=config[dataset_name].getint('num_clusters'),
                                     encoding_dim=encoding_dim,
                                     patch_sizes=config['train'].get('patch_sizes'),
                                    strides=config['train'].get('strides'),
                                    vlad_v2=config['train'].getboolean('vlad_v2'))
    elif config['train'].get('pooling').lower() == 'netvlad':
        pooling_model = NetVLAD(num_clusters=config[dataset_name].getint('num_clusters'),
                                encoding_dim=encoding_dim,
                                vlad_v2=config['train'].getboolean('vlad_v2'))
    elif config['train'].get('pooling').lower() == 'attentionpool':
        pooling_model = AttentionPool(in_channels=encoding_dim)
    elif config['train'].get['pooling'].lower() == 'max':
        global_pool = nn.AdaptiveMaxPool2d((1, 1))
        pooling_model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
    elif config['train'].get['pooling'].pooling.lower() == 'avg':
        global_pool = nn.AdaptiveAvgPool2d((1, 1))
        pooling_model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
    else:
        raise ValueError('未知的Pooling类型: {}'.format(config['train'].get('pooling')))

    if append_pca_layer:
        # PCA后的维度
        num_pcas = config['train'].getint('num_pcas')

        if config['train'].get('pooling').lower() == 'netvlad' or \
                config['train'].get('pooling').lower() == 'patchnetvlad':
            encoding_dim *= config[dataset_name].getint('num_clusters')

        pca_conv = nn.Conv2d(encoding_dim, num_pcas, kernel_size=(1, 1), stride=(1, 1), padding=0)
        pca_model.add_module('WPCA', nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)]))

    return GLAttentionNet(encoding_model, pooling_model, pca_model)


def create_image_clusters(cluster_set, model, encoding_dim, device, config, save_file):
    """
    计算并保存图像聚类信息

    :param cluster_set: 用于计算聚类的数据集
    :param model: 用于提取图像特征的模型
    :param encoding_dim: 图像特征的维度
    :param device: 使用的驱动
    :param config: 配置信息
    :param save_file: 图像特征和聚类的存储路径
    """
    # 获取图像Resize大小
    resize = tuple(map(int, str.split(config['train'].get('resize'), ',')))

    # 数据集名称
    dataset_name = config['dataset'].get('name')

    # 一共要保存的图像特征数
    descriptors_size = 50000
    # 每个图像采集不同位置的特征数
    per_image_sample_count = 100
    # 向上取整后，计算一共要采集多少数据
    image_sample_count = ceil(descriptors_size / per_image_sample_count)

    # 聚类采样的索引
    cluster_sampler = SubsetRandomSampler(np.random.choice(len(cluster_set.db_images_key),
                                                           image_sample_count, replace=False))

    # 创建聚类数据集载入器
    cluster_data_loader = DataLoader(dataset=ImagesFromList(cluster_set.db_images_key,
                                                            transform=MSLS.input_transform(resize)),
                                     batch_size=config['train'].getint('cache_batch_size'), shuffle=False,
                                     sampler=cluster_sampler)

    # 创建保存中心点的文件
    if not exists(join(ROOT_DIR, 'desired/centroids')):
        makedirs(join(ROOT_DIR, 'desired/centroids'))

    # 如果文件存在就删除该文件
    if exists(save_file):
        remove(save_file)

    with h5py.File(save_file, mode='w') as h5_file:
        with torch.no_grad():
            model.eval()
            print('===> 提取图像特征')

            # 在H5文件中创建图像描述
            db_feature = h5_file.create_dataset("descriptors", [descriptors_size, encoding_dim], dtype=np.float32)

            for iteration, (input_data, indices) in enumerate(tqdm(cluster_data_loader, desc='Iter'), 1):
                input_data = input_data.to(device)
                # 使用BackBone提取图像特征，并且形状为
                # (B, C, H, W)->(B, encoding_dim, H, W)->(B, encoding_dim, HxW)->(B, HxW, encoding_dim)，
                # HxW表示不同位置，encoding_dim表示不同位置特征的维度
                image_descriptors = model.encoder(input_data).view(input_data.size(0), encoding_dim, -1).permute(0, 2, 1)
                # 对encoding_dim的图像特征进行L2正则化
                image_descriptors = F.normalize(image_descriptors, p=2, dim=2)

                # 每个图像per_image_sample_count个特征，一共有batch_size个图像，计算有多少个特征作为索引偏移
                batch_index = (iteration - 1) * config['train'].getint('cache_batch_size') * per_image_sample_count
                for ix in range(image_descriptors.size(0)):
                    # 对Batch中的每个图像进行随机位置的采样
                    sample = np.random.choice(image_descriptors.size(1), per_image_sample_count, False)
                    # 设置在H5中保存的索引
                    start_ix = batch_index + ix * per_image_sample_count
                    # 保存每个图像提取到的per_image_sample_count个特征
                    db_feature[start_ix:start_ix + per_image_sample_count, :] = \
                        image_descriptors[ix, sample, :].detach().cpu().numpy()

                # 清空内存
                del input_data, image_descriptors

            # 回收GPU内存
            torch.cuda.empty_cache()

        print('====> 开始进行聚类..')
        # 定义聚类方法KMeans
        kmeans = KMeans(n_clusters=config[dataset_name].getint('num_clusters'), max_iter=100)
        # 拟合图像特征数据
        kmeans.fit(db_feature[...])

        print('====> 保存聚类的中心点 {}'.format(kmeans.cluster_centers_.shape))
        h5_file.create_dataset('centroids', data=kmeans.cluster_centers_)

        print('====> 聚类完成')
