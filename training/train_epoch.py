import torch
import math
import argparse
import configparser
import h5py
import torch.optim as optim
import torch.nn as nn

from dataset.mapillary_sls.MSLS import MSLS
from configparser import ConfigParser
from tqdm import trange
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import join, exists
from os import makedirs
from tools import ROOT_DIR
from tensorboardX import SummaryWriter
from datetime import datetime
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from tools.parallel import reduce_mean
from torch.cuda.amp import GradScaler, autocast


def train_epoch(rank, world_size, train_dataset: MSLS, model: DistributedDataParallel, optimizer: Optimizer,
                criterion: nn.TripletMarginLoss, encoding_dim: int, epoch_num: int, config: ConfigParser,
                opt, writer: SummaryWriter):
    """
    一次训练的过程

    :param rank: GPU
    :param world_size: 进程数量，通常与GPU数量相同
    :param train_dataset: 训练的数据集
    :param model: 训练的模型
    :param optimizer: 训练的优化器
    :param criterion: 训练的损失函数
    :param encoding_dim: Backbone模型的输出维度
    :param epoch_num: 第几个周期
    :param config: 训练的配置参数
    :param opt: 参数信息
    :param writer: Tensorboard的写入对象
    """

    train_dataset.new_epoch()

    # 使用自动混合精度提高效率
    scaler = GradScaler(enabled=opt.amp)

    # 每个训练周期的损失
    epoch_loss = 0

    # 每个训练周期中，Step的起始索引
    start_iter = 1

    # 计算有多少个Batch
    batch_count = math.ceil(len(train_dataset.q_seq_idx) / config['train'].getint('batch_size'))

    # 获得数据集名称
    dataset_name = config['dataset'].get('name')

    # 迭代每一批Query
    cached_q_count_bar = trange(train_dataset.cached_subset_size)
    for sub_cached_q_iter in cached_q_count_bar:
        cached_q_count_bar.set_description(f'GPU:{rank},第{sub_cached_q_iter}/{train_dataset.cached_subset_size}批Query数据')

        if config['train']['pooling'].lower() == 'netvlad' or config['train']['pooling'].lower() == 'patchnetvlad':
            pooling_dim = encoding_dim * config[dataset_name].getint('num_clusters')
        else:
            pooling_dim = encoding_dim

        # 刷新数据
        # train_dataset.refresh_data(model, pooling_dim)
        train_dataset.refresh_data(None, pooling_dim, rank)

        training_data_sampler = DistributedSampler(train_dataset)
        # 训练数据集的载入器，由于采用多卡训练，因此Shuffle需要设置为False，而Shuffle的工作由DistributedSampler完成
        training_data_loader = DataLoader(dataset=train_dataset, batch_size=config['train'].getint('batch_size'),
                                          shuffle=False, collate_fn=MSLS.collate_fn, sampler=training_data_sampler,
                                          num_workers=world_size, pin_memory=True)

        # 进入训练模式
        model.train()

        training_data_bar = tqdm(training_data_loader, leave=False)
        # 使用相同随机种子
        training_data_sampler.set_epoch(sub_cached_q_iter)
        for iteration, (query, positives, negatives, neg_counts, indices) in enumerate(training_data_bar, start_iter):
            training_data_bar.set_description(f'GPU:{rank},周期{epoch_num}的第{sub_cached_q_iter}批的第{iteration}组训练数据')

            if query is None:
                continue

            # 获取Query的B、C、H、W
            B, C, H, W = query.shape
            # 获取Negatives的(B, negative_size, C, H, W)
            N_B, N_S, N_C, N_H, N_W = negatives.shape

            # 计算所有Query对应的反例数量和
            neg_size = torch.sum(neg_counts)
            # Query和Positives的形状都为(B, C, H, W)，但是Negatives的形状为(B, negative_size, C, H, W)，
            # 因此为了使Query和Positives与Negatives的形状保持统一，需要变换Negatives的维度
            negatives = negatives.view(-1, N_C, N_H, N_W)
            # 把Query、Positives和Negatives在第一个维度进行拼接，合并成一个Tensor
            data_input = torch.cat([query, positives, negatives])
            data_input = data_input.to(rank)

            with autocast(enabled=opt.amp):
                # 对数据使用BackBone提取图像特征
                data_encoding = model.module.encoder(data_input)
                # 经过池化后的数据
                pooling_data = model.module.pool(data_encoding)

                patch_loss = 0
                if config['train']['pooling'].lower() == 'patchnetvlad':
                    # =======================================
                    # 计算Patch VLAD特征的损失
                    # =======================================
                    patch_poolings = pooling_data[0]
                    # 读取每个Patch Pooling
                    for i in range(len(patch_poolings)):
                        patch_pooling = patch_poolings[i]
                        patch_pooling_q, patch_pooling_p, patch_pooling_n = torch.split(patch_pooling, [B, B, neg_size])
                        loss_i = 0
                        for i, neg_count in enumerate(neg_counts):
                            for n in range(neg_count):
                                neg_ix = (torch.sum(neg_counts[:i]) + n).item()
                                loss = criterion(patch_pooling_q[i: i + 1],
                                                 patch_pooling_p[i: i + 1],
                                                 patch_pooling_n[neg_ix:neg_ix + 1])
                                loss_i += loss
                        # 计算每个Patch的平均Loss
                        patch_loss += (loss_i / neg_size)

                    # 计算所有Patch Size的平均Loss
                    patch_loss = (patch_loss / len(patch_poolings))

                    # 清空内存
                    del patch_pooling_q, patch_pooling_p, patch_pooling_n

                    global_pooling = pooling_data[1]
                else:
                    global_pooling = pooling_data

                # =======================================
                # 计算Global VLAD特征的损失
                # =======================================
                # 对每个Query、Positive、Negative组成的三元对象进行Loss计算，由于每个Query对应的Negative数量不同，所以需要这样计算
                global_loss = 0

                # 把Pooling的数据分为Query、正例和负例
                global_pooling_q, global_pooling_p, global_pooling_n = torch.split(global_pooling, [B, B, neg_size])

                for i, neg_count in enumerate(neg_counts):
                    for n in range(neg_count):
                        neg_ix = (torch.sum(neg_counts[:i]) + n).item()
                        global_loss += criterion(global_pooling_q[i: i + 1],
                                                 global_pooling_p[i: i + 1],
                                                 global_pooling_n[neg_ix:neg_ix + 1])

                # 对损失求平均
                global_loss = (global_loss / neg_size)

                if config['train']['pooling'].lower() == 'patchnetvlad':
                    total_loss = global_loss + patch_loss
                else:
                    total_loss = global_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # total_loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            # 收集所有GPU上的Loss，并求平均
            total_loss = reduce_mean(total_loss, world_size)

            del data_input, data_encoding, pooling_data, global_pooling_q, global_pooling_p, global_pooling_n
            del query, positives, negatives

            # 只在GPU:0上进行计算
            if rank == 0:
                batch_loss = total_loss.item()
                epoch_loss += batch_loss

                if iteration % 50 == 0 or batch_count <= 10:
                    print("==> 训练周期[{}]({}/{}): Loss: {:.4f}".format(epoch_num, iteration, batch_count, batch_loss))
                    writer.add_scalar('训练损失', batch_loss, ((epoch_num - 1) * batch_count) + iteration)
                    writer.add_scalar('训练的负例数', neg_size, ((epoch_num - 1) * batch_count) + iteration)

        training_data_bar.set_description('')

        start_iter += len(training_data_loader)
        del training_data_loader, total_loss
        # 回头GPU内存
        torch.cuda.empty_cache()

    if rank == 0:
        # 计算平均损失
        avg_loss = epoch_loss / batch_count
        print("===> 第 {} 个周期完成，平均损失: {:.4f}".format(epoch_num, avg_loss))
        writer.add_scalar('训练的平均损失', avg_loss, epoch_num)


if __name__ == '__main__':
    from models.models_generic import get_backbone, get_model

    parser = argparse.ArgumentParser(description='Train Epoch')

    parser.add_argument('--dataset_root_dir', type=str, default='/mnt/Dataset/Mapillary_Street_Level_Sequences',
                        help='Root directory of dataset')
    parser.add_argument('--config_path', type=str, default=join(ROOT_DIR, 'configs'), help='模型训练的配置文件的目录。')
    parser.add_argument('--no_cuda', action='store_true', help='如果使用该参数表示只使用CPU，否则使用GPU。')

    opt = parser.parse_args()

    config_file = join(opt.config_path, 'train.ini')
    config = configparser.ConfigParser()
    config.read(config_file)

    dataset_name = config['dataset'].get('name')

    cuda = not opt.no_cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("没有找到GPU，运行时添加参数 --no_cuda")

    device = torch.device("cuda" if cuda else "cpu")

    encoding_model, encoding_dim = get_backbone(config)
    model = get_model(encoding_model, encoding_dim, config,
                      append_pca_layer=config['train'].getboolean('wpca'))

    init_cache_file = join(join(ROOT_DIR, 'desired', 'centroids'),
                           config['model'].get('backbone') + '_' +
                           dataset_name + '_' +
                           str(config[dataset_name].getint('num_clusters')) + '_desc_cen.hdf5')
    # 打开保存的聚类文件
    with h5py.File(init_cache_file, mode='r') as h5:
        # 获取图像聚类信息
        image_clusters = h5.get('centroids')[:]
        # 获取图像特征信息
        image_descriptors = h5.get('descriptors')[:]

        # 初始化模型参数
        model.pool.init_params(image_clusters, image_descriptors)

        del image_clusters, image_descriptors

    # 保存可视化结果的路径
    opt.result_dir = join(ROOT_DIR, 'result',
                          '{}_{}_{}'.format(config['model'].get('backbone'), dataset_name,
                                            config[dataset_name].get('num_clusters')),
                          datetime.now().strftime('%Y_%m_%d'))
    if not exists(opt.result_dir):
        makedirs(opt.result_dir)

    # 创建TensorBoard的写入对象
    writer = SummaryWriter(log_dir=join(opt.result_dir, datetime.now().strftime('%H:%M:%S')))

    train_dataset = MSLS(opt.dataset_root_dir,
                         mode='train',
                         device=device,
                         config=config,
                         cities_list='trondheim',
                         img_resize=tuple(map(int, str.split(config['train'].get('resize'), ','))),
                         negative_size=config['train'].getint('negative_size'),
                         batch_size=config['train'].getint('cache_batch_size'),
                         exclude_panos=config['train'].getboolean('exclude_panos'))

    optimizer = optim.Adam(filter(lambda par: par.requires_grad, model.parameters()),
                           lr=config['train'].getfloat('lr'))

    criterion = nn.TripletMarginLoss(margin=config['train'].getfloat('margin') ** 0.5,
                                     p=2, reduction='sum').to(device)

    model = model.to(device)
    train_epoch(train_dataset, model, optimizer, criterion, encoding_dim, device, 0, config, writer)
