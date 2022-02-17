import torch
import numpy as np

from tqdm import tqdm
from dataset.mapillary_sls.MSLS import MSLS, ImagesFromList
from models.GLAttentionNet import GLAttentionNet
from configparser import ConfigParser
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.neighbors import NearestNeighbors


def validation(rank, world_size, eval_set: MSLS, model: GLAttentionNet, encoder_dim: int, device: int,
               config: ConfigParser, writer, epoch_num: int):
    """
    模型的验证

    :param rank: GPU
    :param world_size: 进程数量，通常与GPU数量相同
    :param eval_set: 验证的数据集
    :param model: 验证的模型
    :param encoder_dim: Backbone模型的输出维度
    :param device: 训练使用的设备
    :param config: 训练的配置参数
    :param writer: Tensorboard的写入对象
    :param epoch_num: 当前所在的训练周期数
    """

    eval_set_queries = ImagesFromList(eval_set.q_images_key, eval_set.img_transform)
    eval_set_dbs = ImagesFromList(eval_set.db_images_key, transform=eval_set.img_transform)

    eval_data_loader_queries_sampler = DistributedSampler(eval_set_queries)
    eval_data_loader_queries = DataLoader(dataset=eval_set_queries,
                                          batch_size=config['train'].getint('cache_batch_size'),
                                          shuffle=False, sampler=eval_data_loader_queries_sampler,
                                          num_workers=world_size, pin_memory=True)

    eval_data_loader_dbs_sampler = DistributedSampler(eval_set_dbs)
    eval_data_loader_dbs = DataLoader(dataset=eval_set_dbs,
                                      batch_size=config['train'].getint('cache_batch_size'),
                                      shuffle=False, sampler=eval_data_loader_dbs_sampler,
                                      num_workers=world_size, pin_memory=True)

    # 获得数据集名称
    dataset_name = config['dataset'].get('name')
    cache_batch_size = config['train'].getint('cache_batch_size')

    model.eval()
    with torch.no_grad:
        print('====> 提取验证集特征中...')

        if config['train']['pooling'].lower() == 'netvlad' or config['train']['pooling'].lower() == 'patchnetvlad':
            pooling_dim = encoder_dim * config[dataset_name].getint('num_clusters')
        else:
            pooling_dim = encoder_dim

        q_feature = torch.zeros(len(eval_set_queries), pooling_dim).to(device)
        db_feature = torch.zeros(len(eval_set_dbs), pooling_dim).to(device)

        # 获取验证集Query的VLAD特征
        eval_q_data_bar = tqdm(enumerate(eval_data_loader_queries),
                               leave=True, total=len(eval_set_queries) // cache_batch_size)
        for i, (data, idx) in eval_q_data_bar:
            eval_q_data_bar.set_description('[{}/{}]计算验证集Query的特征...'.format(i, eval_q_data_bar.total))
            image_descriptors = model.encoder(data.to(device))
            pool_descriptors = model.pool(image_descriptors)
            # 如果是PatchNetVLAD那么只是用Global VLAD
            if config['train'].get('pooling') == 'patchnetvlad':
                pool_descriptors = pool_descriptors[1]
            q_feature[i * cache_batch_size: (i + 1) * cache_batch_size, :] = pool_descriptors

            # 清空GPU存储
            del data, image_descriptors, pool_descriptors

        # 获取验证集Database的VLAD特征
        eval_db_data_bar = tqdm(enumerate(eval_data_loader_dbs),
                                leave=True, total=len(eval_set_dbs) // cache_batch_size)
        for i, (data, idx) in eval_db_data_bar:
            eval_db_data_bar.set_description('[{}/{}]计算验证集Database的特征...'.format(i, eval_db_data_bar.total))
            image_descriptors = model.encoder(data.to(device))
            vlad_descriptors = model.pool(image_descriptors)
            # 如果是PatchNetVLAD那么只是用Global VLAD
            if config['train'].get('pooling') == 'patchnetvlad':
                vlad_descriptors = vlad_descriptors[1]
            db_feature[i * cache_batch_size: (i + 1) * cache_batch_size, :] = vlad_descriptors

            # 清空GPU存储
            del data, image_descriptors, vlad_descriptors

        del eval_data_loader_queries, eval_data_loader_dbs
        # 回收GPU内存
        torch.cuda.empty_cache()

    print('===> 构建验证集的最近邻')

    # 对Database进行拟合
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(db_feature.cpu().numpy())

    print('====> 计算召回率 @ N')
    n_values = [1, 5, 10, 20, 50, 100]

    # 计算最近邻
    predictions = np.square(knn.kneighbors(q_feature.cpu().numpy(), max(n_values))[1])

    # 得到所有正例
    gt = eval_set.all_positive_indices

    # 保存不同N的正确率
    correct_at_n = np.zeros(len(n_values))

    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # 只要在gt中存在pred预测的索引，那么correct_at_n就增加1
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_n[i:] += 1
                break

    # 计算召回率
    recall_at_n = correct_at_n / len(eval_set.q_seq_idx)

    # 保存所有召回率
    all_recalls = {}
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        print("====> 召回率@{}: {:.4f}".format(n, recall_at_n[i]))
        writer.add_scalar('验证集的召回率@{}'.format(str(n)), recall_at_n[i], epoch_num)

    return all_recalls
