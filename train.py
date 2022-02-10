import argparse
import configparser
import random
import numpy as np
import torch

from os.path import join, exists
from os import makedirs
from dataset.mapillary_sls.MSLS import MSLS
from datetime import datetime
from tools import ROOT_DIR
from training.parallel_train import run_parallel, main_parallel_train


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
    parser.add_argument('--amp', action='store_true', help='是否开启混合精度训练')

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
    run_parallel(main_parallel_train, world_size=n_gpus, config=config, opt=opt,
                 train_dataset=train_dataset, validation_dataset=validation_dataset)

    # 清空CUDA缓存
    torch.cuda.empty_cache()

    print('训练结束...')
