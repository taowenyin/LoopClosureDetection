import argparse
import configparser
import os.path
import random
import numpy as np
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import h5py

from os.path import join, isfile, exists
from os import makedirs
from models.models_generic import get_backbone, get_model, create_image_clusters
from shutil import copyfile
from dataset.mapillary_sls.MSLS import MSLS
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm
from datetime import datetime
from tools import ROOT_DIR
from training.train_epoch import train_epoch
from time import sleep


def setup_parallel(rank, world_size):
    """
    分布式训练初始化

    :param rank: 当前进程中GPU的编号
    :param world_size: 总共有多少个GPU
    """
    # 确定可用的GPU，注意这句话一定要放在任何对CUDA的操作之前（和别人公用服务器时使用）
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # 设置两个环境变量，localhost是本地的ip地址，12355是用来通信的端口
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    # 实现GPU的负载均衡
    torch.cuda.set_device(rank)


def parallel_cleanup():
    """
    在所有任务行完以后消灭进程用的。
    """
    dist.destroy_process_group()


def run_parallel(parallel_fn, world_size, model, encoding_dim, config, opt, train_dataset, validation_dataset):
    """
    多进程产生函数。不用这个的话需要在运行训练代码的时候，用'python-m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE train.py'才能启动。

    :param parallel_fn: 分布式的函数
    :param world_size: GPU数量
    :param model: 模型
    :param encoding_dim: 编码器的输出通道数
    :param config: 配置信息
    :param opt: 参数对象
    :param train_dataset: 训练数据集
    :param validation_dataset: 验证数据集
    """
    mp.spawn(parallel_fn,
             args=(world_size, model, encoding_dim, config, opt, train_dataset, validation_dataset),
             nprocs=world_size, join=True)


def main_parallel_train(rank, world_size, model, encoding_dim, config, opt, train_dataset, validation_dataset):
    """
    并行训练模型
    :param rank: 当前进程中GPU的编号
    :param world_size: 总共有多少个GPU
    :param model: 训练模型
    :param config: 配置信息
    :param opt: 参数对象
    :param train_dataset: 训练数据集
    :param validation_dataset: 验证数据集
    """
    setup_parallel(rank, world_size)

    print(f'GPU:{rank}开始运行...')

    # 使用三元损失函数，并使用欧氏距离作为距离函数
    criterion = nn.TripletMarginLoss(margin=config['train'].getfloat('margin') ** 0.5,
                                     p=2, reduction='sum')

    if config['train'].get('optim') == 'ADAM':
        optimizer = optim.Adam(filter(lambda par: par.requires_grad, model.parameters()),
                               lr=config['train'].getfloat('lr'))
    else:
        raise ValueError('未知的优化器: ' + config['train'].get('optim'))

    if rank == 0:
        # 创建TensorBoard的写入对象
        writer = SummaryWriter(log_dir=join(opt.result_dir, 'logs'))
    else:
        writer = None

    # 把模型改成并行模型
    model = DDP(model.to(rank), device_ids=[rank])

    # 开始训练，从opt.start_epoch + 1次开始，到opt.epochs_count次结束
    train_epoch_bar = trange(opt.start_epoch + 1, opt.epochs_count + 1)
    for epoch in train_epoch_bar:
        if rank == 0:
            train_epoch_bar.set_description(f'GPU:{rank},第{epoch}/{opt.epochs_count - opt.start_epoch}次训练周期')

        # 执行一个训练周期
        train_epoch(rank, world_size, train_dataset, model, optimizer,
                    criterion, encoding_dim, epoch, config, writer)

    parallel_cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='移动机器人回环检测模型')

    parser.add_argument('--dataset_root_dir', type=str, default='/mnt/Dataset/Mapillary_Street_Level_Sequences',
                        help='数据集的根目录')
    parser.add_argument('--config_path', type=str, default=join(ROOT_DIR, 'configs'), help='模型训练的配置文件的目录')
    parser.add_argument('--save_checkpoint_path', type=str, default=join(ROOT_DIR, 'desired', 'checkpoint'),
                        help='模型checkpoint的保存目录')
    parser.add_argument('--no_cuda', action='store_true', help='如果使用该参数表示只使用CPU，否则使用GPU')
    parser.add_argument('--resume_file', type=str, help='checkpoint文件的保存路径，用于从checkpoint载入训练参数，再次恢复训练')
    parser.add_argument('--cluster_file', type=str, help='聚类数据的保存路径，恢复训练')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='手动设置迭代开始位置，用于重新开始的训练')
    parser.add_argument('--epochs_count', default=30, type=int, help='模型训练的周期数')
    parser.add_argument('--save_every_epoch', action='store_true', help='是否开启每个EPOCH都进行checkpoint保存')

    opt = parser.parse_args()

    # 配置文件的地址
    config_file = join(opt.config_path, 'train.ini')
    # 读取配置文件
    config = configparser.ConfigParser()
    config.read(config_file)

    # 设置GPU或CPU
    cuda = not opt.no_cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("没有找到GPU，运行时添加参数 --no_cuda")
    # 获取GPU的数量
    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if cuda else "cpu")

    # 固定随机种子
    random.seed(config['train'].getint('seed'))
    np.random.seed(config['train'].getint('seed'))
    torch.manual_seed(config['train'].getint('seed'))
    if cuda:
        torch.cuda.manual_seed(config['train'].getint('seed'))

    # 获得数据集名称
    dataset_name = config['dataset'].get('name')

    print('===> 构建网络模型')

    print('===> 构建基础BackBone模型')
    encoding_model, encoding_dim = get_backbone(config)

    print('===> 载入模型')

    model = get_model(encoding_model, encoding_dim, config, append_pca_layer=config['train'].getboolean('wpca'))
    # 保存的图像特征
    init_cache_file = join(join(ROOT_DIR, 'desired', 'centroids'),
                           config['model'].get('backbone') + '_' +
                           dataset_name + '_' +
                           str(config[dataset_name].getint('num_clusters')) + '_desc_cen.hdf5')

    if opt.cluster_file:
        opt.cluster_file = join(join(ROOT_DIR, 'desired', 'centroids'), opt.cluster_file)

        if isfile(opt.cluster_file):
            if opt.cluster_file != init_cache_file:
                copyfile(opt.cluster_file, init_cache_file)
        else:
            raise FileNotFoundError("=> 在'{}'中没有找到聚类数据".format(opt.cluster_file))
    else:
        print('===> 寻找聚类中心点')

        print('===> 载入聚类数据集')
        train_dataset = MSLS(opt.dataset_root_dir, device=device, config=config, mode='test', cities_list='train',
                             img_resize=tuple(map(int, str.split(config['train'].get('resize'), ','))),
                             batch_size=config['train'].getint('cache_batch_size'))

        print('===> 聚类数据集中的数据数量为: {}'.format(len(train_dataset.db_images_key)))

        model = model.to(device)

        print('===> 计算图像特征并创建聚类文件')
        create_image_clusters(train_dataset, model, encoding_dim, device, config, init_cache_file)

        # 把模型转为CPU模式，用于载入参数
        model = model.to(device='cpu')

    # 打开保存的聚类文件
    with h5py.File(init_cache_file, mode='r') as h5:
        # 获取图像聚类信息
        image_clusters = h5.get('centroids')[:]
        # 获取图像特征信息
        image_descriptors = h5.get('descriptors')[:]

        # 初始化模型参数
        model.pool.init_params(image_clusters, image_descriptors)

        del image_clusters, image_descriptors

        # 回头GPU内存
        torch.cuda.empty_cache()

    print('===> 载入训练和验证数据集')

    train_dataset = MSLS(opt.dataset_root_dir, mode='train', device=device, config=config, cities_list='trondheim',
                         img_resize=tuple(map(int, str.split(config['train'].get('resize'), ','))),
                         negative_size=config['train'].getint('negative_size'),
                         batch_size=config['train'].getint('cache_batch_size'),
                         exclude_panos=config['train'].getboolean('exclude_panos'))

    validation_dataset = MSLS(opt.dataset_root_dir, mode='val', device=device, config=config, cities_list='cph',
                              img_resize=tuple(map(int, str.split(config['train'].get('resize'), ','))),
                              positive_distance_threshold=config['train'].getint('positive_distance_threshold'),
                              batch_size=config['train'].getint('cache_batch_size'),
                              exclude_panos=config['train'].getboolean('exclude_panos'))

    print('===> 训练集中Query的数量为: {}'.format(len(train_dataset.q_seq_idx)))
    print('===> 验证集中Query的数量为: {}'.format(len(validation_dataset.q_seq_idx)))

    # 保存训练参数的路径
    opt.resume_dir = join(ROOT_DIR, 'desired', 'checkpoint')
    # 如果目录不存在就创建目录
    if not exists(opt.resume_dir):
        makedirs(opt.resume_dir)

    # 保存可视化结果的路径
    opt.result_dir = join(ROOT_DIR, 'result',
                          '{}_{}_{}'.format(config['model'].get('backbone'), dataset_name,
                                            config[dataset_name].get('num_clusters')),
                          datetime.now().strftime('%Y_%m_%d_%H:%M:%S'))
    if not exists(opt.result_dir):
        makedirs(opt.result_dir)

    # 执行并行训练任务
    run_parallel(main_parallel_train, world_size=n_gpus, model=model, encoding_dim=encoding_dim, config=config,
                 opt=opt, train_dataset=train_dataset, validation_dataset=validation_dataset)

    # 清空CUDA缓存
    torch.cuda.empty_cache()

    print('训练结束...')
