import torch
import argparse
import random
import cfg
import numpy as np
import torch.nn as nn
import train_eval
import cluster
import os

from segattlcd.models.NetVLAD import NetVLAD, EmbedNet
from dataset import Pitts250k, Tokyo247, Tokyo247Query
from Utils.Flatten import Flatten
from Utils.L2Normalize import L2Normalize
from torchvision.models import vgg16, alexnet
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='pytorch-NetVlad')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--pretrained', type=bool, default=True, help='Load pretrained weight')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'cluster'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')

if __name__ == '__main__':
    opt = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 设置固定种子
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    base_model = None
    base_model_name = None
    conv_output_channels = 0
    if cfg.BASE_MODEL_TYPE == cfg.BaseModel.VGG16:
        base_model_name = 'VGG16'
        base_model = vgg16(pretrained=opt.pretrained).features[:29]
        # 只训练最后3个Conv，其他层冻结
        for l in base_model[:-5]:
            for p in l.parameters():
                p.requires_grad = False
        conv_output_channels = 512
    elif cfg.BASE_MODEL_TYPE == cfg.BaseModel.AlexNet:
        base_model_name = 'AlexNet'
        base_model = alexnet(pretrained=opt.pretrained).features[:11]
        # 只训练最后一个Conv5，其他层冻结
        for l in base_model[:-1]:
            for p in l.parameters():
                p.requires_grad = False
        conv_output_channels = 256

    pooling_net = None
    # 如果是聚类，则在BaseModel后添加L2Normalize
    if opt.mode == 'cluster':
        pooling_net = nn.Sequential(
            L2Normalize()
        )
    else:
        if cfg.POOLING_TYPE == cfg.Pooling.NetVLAD:
            pooling_net = NetVLAD(32, conv_output_channels=conv_output_channels)
        elif cfg.POOLING_TYPE == cfg.Pooling.MAX:
            pooling_net = nn.Sequential(
                nn.AdaptiveMaxPool2d((1, 1)),
                Flatten(),
                L2Normalize()
            )
        elif cfg.POOLING_TYPE == cfg.Pooling.AVG:
            pooling_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                L2Normalize()
            )

    # 创建模型对象
    net = EmbedNet(base_model, pooling_net).to(device)

    optimizer = optim.Adam(net.parameters(), lr=cfg.BASE_LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.LR_STEP, gamma=cfg.LR_GAMMA)
    criterion = nn.TripletMarginLoss(margin=cfg.TRIPLET_MARGIN, p=2, reduction='sum').to(device)

    dataset_name = None
    if opt.mode != 'cluster':
        if cfg.DATASET_TYPE == cfg.Dataset.Tokyo247:
            centroids_path = './Datasets/Tokyo247'
            dataset_name = 'Tokyo247'
        elif cfg.DATASET_TYPE == cfg.Dataset.Pitts250k:
            centroids_path = './Datasets/Pitts250k'
            dataset_name = 'Pitts250k'
        else:
            centroids_path = './Datasets/Tokyo247'
            dataset_name = 'Tokyo247'

        centroids_file = os.path.join(centroids_path, 'centroids', base_model_name + '_' + dataset_name +
                                      '_' + str(opt.num_clusters) + '_desc_cen.hdf5')

        if not os.path.exists(centroids_file):
            raise Exception('No centroids file!!!!!!!!!!!!!!1')

    # 载入数据集
    whole_dataset = None
    if cfg.DATASET_TYPE == cfg.Dataset.Tokyo247:
        whole_dataset = Tokyo247('./Datasets', model_type=opt.mode, only_db=(opt.mode == 'cluster'))
    elif cfg.DATASET_TYPE == cfg.Dataset.Pitts250k:
        whole_dataset = Pitts250k('./Datasets', model_type=opt.mode, only_db=(opt.mode == 'cluster'))

    print('====> {} data set: {}'.format(opt.mode, len(whole_dataset)))

    if opt.mode == 'train':
        whole_dataset_train = Tokyo247Query('./Datasets', model_type=opt.mode)
        whole_data_loader = DataLoader(whole_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                       pin_memory=True, num_workers=4, drop_last=True)

        for epoch in range(cfg.EPOCH_NUMBER):
            # 获得损失函数
            loss = train_eval.train_one_epoch(net, whole_data_loader, optimizer, criterion, epoch, device)

            # 更新学习率
            scheduler.step()
    if opt.mode == 'test':
        print('xx')
    elif opt.mode == 'cluster':
        # 计算聚类
        cluster.calculate_clusters(whole_dataset, net, conv_output_channels, base_model_name, opt, device)

    print('xxx')
