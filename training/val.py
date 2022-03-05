import torch
import numpy as np
import faiss

from tqdm import tqdm
from dataset.mapillary_sls.MSLS import MSLS, ImagesFromList
from torch.nn.parallel import DistributedDataParallel as DDP
from configparser import ConfigParser
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.neighbors import NearestNeighbors
from tensorboardX import SummaryWriter


def validation(rank, world_size, eval_set: MSLS, model: DDP, encoder_dim: int,
               config: ConfigParser, writer: SummaryWriter, epoch_num: int):
    """
    模型的验证

    :param rank: GPU
    :param world_size: 进程数量，通常与GPU数量相同
    :param eval_set: 验证的数据集
    :param model: 验证的模型
    :param encoder_dim: Backbone模型的输出维度
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
    with torch.no_grad():
        print('====> 提取验证集特征中...')

        if config['train'].get('pooling').lower() == 'netvlad':
            pooling_dim = encoder_dim * config[dataset_name].getint('num_clusters')
        else:
            pooling_dim = encoder_dim

        q_feature = torch.zeros(len(eval_set_queries), pooling_dim, device=rank)
        db_feature = torch.zeros(len(eval_set_dbs), pooling_dim, device=rank)

        for feature, data_loader in zip([q_feature, db_feature], [eval_data_loader_queries, eval_data_loader_dbs]):
            eval_data_bar = tqdm(data_loader, leave=True, total=len(data_loader.dataset) // cache_batch_size)
            for i, (data, idx) in enumerate(eval_data_bar, start=1):
                eval_data_bar.set_description('[{}/{}]计算验证集Query的特征...'.format(i, eval_data_bar.total))

                image_descriptors = model.module.encoder(data.to(rank))
                pool_descriptors = model.module.pool(image_descriptors)

                # 保存图片，并且一定要按idx索引的顺序放，不是按照i索引的顺序放
                feature[idx, :] = pool_descriptors

                del data, image_descriptors, pool_descriptors

    del eval_data_loader_queries, eval_data_loader_dbs

    print('===> 构建验证集的最近邻')
    print('====> 计算召回率 @ N')
    n_values = [1, 5, 10, 20, 50, 100]

    # 得到所有正例
    gt = eval_set.all_positive_indices

    # 保存预测的结果
    predictions = None
    # 开始索引
    start_db_index = start_q_index = 0
    for cities_size, (db_size, q_size) in enumerate(zip(eval_set.db_images_list, eval_set.q_images_size_list)):
        faiss_index = faiss.IndexFlatL2(pooling_dim)
        faiss_index.add(db_feature.cpu().numpy()[start_db_index:start_db_index + db_size, :])
        distances, predictions_indexs = faiss_index.search(q_feature.cpu().numpy()[
                                                           start_q_index:start_q_index + q_size, :],
                                                           max(n_values))
        # 保存预测结果
        if cities_size == 0:
            predictions = predictions_indexs
        else:
            predictions = np.vstack((predictions, predictions_indexs))
        # 更新开始索引
        start_db_index += db_size
        start_q_index += q_size

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

