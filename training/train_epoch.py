import torch
import math
import torch.nn as nn

from dataset.mapillary_sls.MSLS import MSLS
from configparser import ConfigParser
from tqdm import trange
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from tools.parallel import reduce_mean


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

    # 每个训练周期的损失
    epoch_loss = 0

    # 每个训练周期中，Step的起始索引
    start_iter = 1

    # 获得数据集名称
    dataset_name = config['dataset'].get('name')
    model_name = config['model'].get('backbone')
    batch_size = config[model_name].getint('batch_size')

    # 计算有多少个Batch
    batch_count = math.ceil(len(train_dataset.q_seq_idx) / batch_size)

    # 迭代每一批Query
    cached_q_count_bar = trange(train_dataset.cached_subset_size)
    for sub_cached_q_iter in cached_q_count_bar:
        cached_q_count_bar.set_description(f'GPU:{rank},第{sub_cached_q_iter}/{train_dataset.cached_subset_size}批Query数据')

        if config['train'].get('pooling').lower() == 'netvlad':
            pooling_dim = encoding_dim * config[dataset_name].getint('num_clusters')
        else:
            pooling_dim = encoding_dim

        # 刷新数据
        # train_dataset.refresh_data(model, pooling_dim)
        train_dataset.refresh_data(None, pooling_dim, rank)

        training_data_sampler = DistributedSampler(train_dataset)
        # 训练数据集的载入器，由于采用多卡训练，因此Shuffle需要设置为False，而Shuffle的工作由DistributedSampler完成
        training_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
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
            Q_B, Q_C, Q_H, Q_W = query.shape
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

            # 对数据使用BackBone提取图像特征
            data_encoding = model.module.encoder(data_input)
            # 经过池化后的数据
            pooling_data = model.module.pool(data_encoding)

            # =======================================
            # 计算Global VLAD特征的损失
            # =======================================
            # 把Pooling的数据分为Query、正例和负例
            global_pooling_q, global_pooling_p, global_pooling_n = torch.split(pooling_data, [Q_B, Q_B, neg_size])

            optimizer.zero_grad()
            # 对每个Query、Positive、Negative组成的三元对象进行Loss计算，由于每个Query对应的Negative数量不同，所以需要这样计算
            loss = 0
            for i, neg_count in enumerate(neg_counts):
                for n in range(neg_count):
                    neg_ix = (torch.sum(neg_counts[:i]) + n).item()
                    loss += criterion(global_pooling_q[i: i + 1],
                                      global_pooling_p[i: i + 1],
                                      global_pooling_n[neg_ix:neg_ix + 1])

            # 对损失求平均
            loss = (loss / neg_size.float().to(rank))
            loss.backward()
            optimizer.step()

            # 收集所有GPU上的Loss，并求平均
            mean_loss = reduce_mean(loss, world_size)

            del data_input, data_encoding, pooling_data, global_pooling_q, global_pooling_p, global_pooling_n
            del query, positives, negatives

            # 只在GPU:0上进行计算
            if rank == 0:
                batch_loss = mean_loss.item()
                epoch_loss += batch_loss

                if iteration % 50 == 0 or batch_count <= 10:
                    print("==> 训练周期[{}]({}/{}): Loss: {:.4f}".format(epoch_num, iteration, batch_count, batch_loss))
                    writer.add_scalar('训练损失', batch_loss, ((epoch_num - 1) * batch_count) + iteration)
                    writer.add_scalar('训练的负例数', neg_size, ((epoch_num - 1) * batch_count) + iteration)

        training_data_bar.set_description('')

        start_iter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    if rank == 0:
        # 计算平均损失
        avg_loss = epoch_loss / batch_count
        print("===> 第 {} 个周期完成，平均损失: {:.4f}".format(epoch_num, avg_loss))
        writer.add_scalar('训练的平均损失', avg_loss, epoch_num)


if __name__ == '__main__':
    print('')
