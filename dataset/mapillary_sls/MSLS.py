import math
import random
import pandas as pd
import numpy as np
import sys
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse
import configparser
import torch
import h5py

from torch.utils.data import Dataset
from tqdm import tqdm
from os.path import join
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from tools import ROOT_DIR

default_cities = {
    'train': ['trondheim', 'london', 'boston', 'melbourne', 'amsterdam', 'helsinki',
              'tokyo', 'toronto', 'saopaulo', 'moscow', 'zurich', 'paris', 'bangkok',
              'budapest', 'austin', 'berlin', 'ottawa', 'phoenix', 'goa', 'amman', 'nairobi', 'manila'],
    'val': ['cph', 'sf'],
    'test': ['miami', 'athens', 'buenosaires', 'stockholm', 'bengaluru', 'kampala']
}


class ImagesFromList(Dataset):
    def __init__(self, images, transform):
        self.images = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = [Image.open(im) for im in self.images[idx].split(",")]
        except:
            img = [Image.open(self.images[0])]
        img = [self.transform(im) for im in img]

        if len(img) == 1:
            img = img[0]

        return img, idx


class MSLS(Dataset):
    def __init__(self, root_dir, device, config, mode='train', cities_list=None, img_resize=(480, 640),
                 negative_size=5, positive_distance_threshold=10, negative_distance_threshold=25,
                 cached_queries=1000, cached_negatives=1000, batch_size=24, task='im2im', sub_task='all',
                 seq_length=1, exclude_panos=True, positive_sampling=True):
        """
        Mapillary Street-level Sequences数据集的读取

        task（任务）：im2im（图像到图像）, seq2seq（图像序列到图像序列）, seq2im（图像序列到图像）, im2seq（图像到图像序列）

        sub_task（子任务）：all，s2w（summer2winter），w2s（winter2summer），o2n（old2new），n2o（new2old），d2n（day2night），n2d（night2day）

        :param root_dir: 数据集的路径
        :param device: 数据运行的设备
        :param config: 配置信息
        :param mode: 数据集的模式[train, val, test]
        :param cities_list: 城市列表
        :param img_resize: 图像大小
        :param negative_size: 每个正例对应的反例个数
        :param positive_distance_threshold: 正例的距离阈值
        :param negative_distance_threshold: 反例的距离阈值，在该距离之内认为是非反例，之外才属于反例，同时正例要在正例阈值内才算正例，正例阈值和负例阈值之间属于非负例
        :param cached_queries: 每次缓存的Query总数，即每个完整的EPOCH中，数据的总量，和Batch Size不同
        :param cached_negatives: 每次缓存的负例总数，即每个完整的EPOCH中，数据的总量，和Batch Size不同
        :param batch_size: 每批数据的大小
        :param task: 任务类型 [im2im, seq2seq, seq2im, im2seq]
        :param sub_task: 任务类型 [all, s2w, w2s, o2n, n2o, d2n, n2d]
        :param seq_length: 不同任务的序列长度
        :param exclude_panos: 是否排除全景图像
        :param positive_sampling: 是否进行正采样
        """
        super().__init__()

        if cities_list in default_cities:
            self.__cities_list = default_cities[cities_list]
        elif cities_list is None:
            self.__cities_list = default_cities[mode]
        else:
            self.__cities_list = cities_list.split(',')

        # 筛选后的Query图像
        self.__q_images_key = []
        # 筛选后的Database图像
        self.__db_images_key = []
        # Query的序列索引
        self.__q_seq_idx = []
        # positive的序列索引
        self.__p_seq_idx = []
        # 不是负例的索引
        self.__non_negative_indices = []
        # 路边的数据
        self.__sideways = []
        # 晚上的数据
        self.__night = []

        # 三元数据
        self.__triplets_data = []

        self.__mode = mode
        self.__device = device
        self.__config = config
        self.__sub_task = sub_task
        self.__exclude_panos = exclude_panos
        self.__negative_size = negative_size
        self.__positive_distance_threshold = positive_distance_threshold
        self.__negative_distance_threshold = negative_distance_threshold
        self.__cached_queries = cached_queries
        self.__cached_negatives = cached_negatives
        self.__batch_size = batch_size

        # 记录当前EPOCH调用数据集自己的次数，也就是多少个cached_queries数据
        self.__current_subset = 0

        # 得到图像转换对象
        self.__img_transform = MSLS.input_transform(img_resize)

        # 把所有数据分为若干批，每批数据的集合
        self.__cached_subset_idx = []

        # 所有Query对应的正例索引
        self.__all_positive_indices = []
        # 每批cached_queries个数据，提供有多少批数据
        self.__cached_subset_size = 0

        # 根据任务类型得到序列长度
        if task == 'im2im':  # 图像到图像
            seq_length_q, seq_length_db = 1, 1
        elif task == 'seq2seq':  # 图像序列到图像序列
            seq_length_q, seq_length_db = seq_length, seq_length
        elif task == 'seq2im':  # 图像序列到图像
            seq_length_q, seq_length_db = seq_length, 1
        else:  # im2seq 图像到图像序列
            seq_length_q, seq_length_db = 1, seq_length

        # 载入数据
        load_data_bar = tqdm(self.__cities_list)
        for city in load_data_bar:
            load_data_bar.set_description('=====> 载入{}数据'.format(city))

            # 根据城市获得数据文件夹名称
            subdir = 'test' if city in default_cities['test'] else 'train_val'

            # 保存没有正例的图像数
            non_positive_q_seq_keys_count = 0
            # 保存有正例的图像数
            has_positive_q_seq_keys_count = 0
            # 保存数据集的个数
            q_seq_keys_count = 0

            # 获取到目前为止用于索引的城市图像的长度
            _lenQ = len(self.__q_images_key)
            _lenDb = len(self.__db_images_key)

            # 读取训练集或验证集数据集
            if self.__mode in ['train', 'val']:
                # 载入Query数据
                q_data = pd.read_csv(join(root_dir, subdir, city, 'query', 'postprocessed.csv'), index_col=0)
                q_data_raw = pd.read_csv(join(root_dir, subdir, city, 'query', 'raw.csv'), index_col=0)

                # 读取数据集数据
                db_data = pd.read_csv(join(root_dir, subdir, city, 'database', 'postprocessed.csv'), index_col=0)
                db_data_raw = pd.read_csv(join(root_dir, subdir, city, 'database', 'raw.csv'), index_col=0)

                # 根据任务把数据变成序列
                q_seq_keys, q_seq_idxs = self.__rang_to_sequence(q_data, join(root_dir, subdir, city, 'query'),
                                                                 seq_length_q)
                db_seq_keys, db_seq_idxs = self.__rang_to_sequence(db_data, join(root_dir, subdir, city, 'database'),
                                                                   seq_length_db)
                q_seq_keys_count = len(q_seq_keys)

                # 如果验证集，那么需要确定子任务的类型
                if self.__mode in ['val']:
                    q_idx = pd.read_csv(join(root_dir, subdir, city, 'query', 'subtask_index.csv'), index_col=0)
                    db_idx = pd.read_csv(join(root_dir, subdir, city, 'database', 'subtask_index.csv'), index_col=0)

                    # 从所有序列数据中根据符合子任务的中心索引找到序列数据帧
                    val_frames = np.where(q_idx[self.__sub_task])[0]
                    q_seq_keys, q_seq_idxs = self.__data_filter(q_seq_keys, q_seq_idxs, val_frames)

                    val_frames = np.where(db_idx[self.__sub_task])[0]
                    db_seq_keys, db_seq_idxs = self.__data_filter(db_seq_keys, db_seq_idxs, val_frames)

                # 筛选出不同全景的数据
                if self.__exclude_panos:
                    panos_frames = np.where((q_data_raw['pano'] == False).values)[0]
                    # 从Query数据中筛选出不是全景的数据
                    q_seq_keys, q_seq_idxs = self.__data_filter(q_seq_keys, q_seq_idxs, panos_frames)

                    panos_frames = np.where((db_data_raw['pano'] == False).values)[0]
                    # 从Query数据中筛选出不是全景的数据
                    db_seq_keys, db_seq_idxs = self.__data_filter(db_seq_keys, db_seq_idxs, panos_frames)

                # 删除重复的idx
                unique_q_seq_idxs = np.unique(q_seq_idxs)
                unique_db_seq_idxs = np.unique(db_seq_idxs)

                # 如果排除重复后没有数据，那么就下一个城市
                if len(unique_q_seq_idxs) == 0 or len(unique_db_seq_idxs) == 0:
                    continue

                # 保存筛选后的图像
                self.__q_images_key.extend(q_seq_keys)
                self.__db_images_key.extend(db_seq_keys)

                # 从原数据中筛选后数据
                q_data = q_data.loc[unique_q_seq_idxs]
                db_data = db_data.loc[unique_db_seq_idxs]

                # 获取图像的UTM坐标
                utm_q = q_data[['easting', 'northing']].values.reshape(-1, 2)
                utm_db = db_data[['easting', 'northing']].values.reshape(-1, 2)

                # 获取Query图像的Night状态、否是Sideways，以及图像索引
                night, sideways, index = q_data['night'].values, \
                                         (q_data['view_direction'] == 'Sideways').values, \
                                         q_data.index

                # 创建最近邻算法，使用暴力搜索法
                neigh = NearestNeighbors(algorithm='brute')
                # 对数据集进行拟合
                neigh.fit(utm_db)
                # 在Database中找到符合positive_distance_threshold要求的Query数据的最近邻数据的索引
                positive_distance, positive_indices = neigh.radius_neighbors(utm_q, self.__positive_distance_threshold)
                # 保存所有正例索引
                self.__all_positive_indices.extend(positive_indices)

                # 训练模式下，获取负例索引
                if self.__mode == 'train':
                    negative_distance, negative_indices = neigh.radius_neighbors(
                        utm_q, self.__negative_distance_threshold)

                # 查看每个Seq的正例
                for q_seq_key_idx in range(len(q_seq_keys)):
                    # 返回每个序列的帧集合
                    q_frame_idxs = self.__seq_idx_2_frame_idx(q_seq_key_idx, q_seq_idxs)
                    # 返回q_frame_idxs在unique_q_seq_idxs中的索引集合
                    q_uniq_frame_idx = self.__frame_idx_2_uniq_frame_idx(q_frame_idxs, unique_q_seq_idxs)
                    # 返回序列Query中序列对应的正例索引
                    positive_uniq_frame_idxs = np.unique([p for pos in positive_indices[q_uniq_frame_idx] for p in pos])

                    # 查询的序列Query至少要有一个正例
                    if len(positive_uniq_frame_idxs) > 0:
                        # 获取正例所在的序列索引，并去除重复的索引
                        positive_seq_idx = np.unique(self.__uniq_frame_idx_2_seq_idx(
                            unique_db_seq_idxs[positive_uniq_frame_idxs], db_seq_idxs))

                        # todo 不知道是什么意思
                        self.__p_seq_idx.append(positive_seq_idx + _lenDb)
                        self.__q_seq_idx.append(q_seq_key_idx + _lenQ)

                        # 在训练的时候需要根据两个阈值找到正例和负例
                        if self.__mode == 'train':
                            # 找到不是负例的数据
                            n_uniq_frame_idxs = np.unique(
                                [n for nonNeg in negative_indices[q_uniq_frame_idx] for n in nonNeg])
                            # 找到不是负例所在的序列索引，并去除重复的索引
                            n_seq_idx = np.unique(
                                self.__uniq_frame_idx_2_seq_idx(unique_db_seq_idxs[n_uniq_frame_idxs], db_seq_idxs))

                            # 保存数据
                            self.__non_negative_indices.append(n_seq_idx + _lenDb)

                            # todo 不知道是什么意思
                            if sum(night[np.in1d(index, q_frame_idxs)]) > 0:
                                self.__night.append(len(self.__q_seq_idx) - 1)
                            if sum(sideways[np.in1d(index, q_frame_idxs)]) > 0:
                                self.__sideways.append(len(self.__q_seq_idx) - 1)

                        has_positive_q_seq_keys_count += 1
                    else:
                        non_positive_q_seq_keys_count += 1

                print('\n=====> {}训练数据中，有正例的[{}/{}]个，无正例的[{}/{}]个'.format(
                    city,
                    has_positive_q_seq_keys_count,
                    q_seq_keys_count,
                    non_positive_q_seq_keys_count,
                    q_seq_keys_count))

            # 读取测试集数据集，GPS/UTM/Pano都不可用
            elif self.__mode in ['test']:
                # 载入对应子任务的图像索引
                q_idx = pd.read_csv(join(root_dir, subdir, city, 'query', 'subtask_index.csv'), index_col=0)
                db_idx = pd.read_csv(join(root_dir, subdir, city, 'database', 'subtask_index.csv'), index_col=0)

                # 根据任务把数据变成序列
                q_seq_keys, q_seq_idxs = self.__rang_to_sequence(q_idx, join(root_dir, subdir, city, 'query'),
                                                                 seq_length_q)
                db_seq_keys, db_seq_idxs = self.__rang_to_sequence(db_idx, join(root_dir, subdir, city, 'database'),
                                                                   seq_length_db)

                # 从所有序列数据中根据符合子任务的中心索引找到序列数据帧
                val_frames = np.where(q_idx[self.__sub_task])[0]
                q_seq_keys, q_seq_idxs = self.__data_filter(q_seq_keys, q_seq_idxs, val_frames)

                val_frames = np.where(db_idx[self.__sub_task])[0]
                db_seq_keys, db_seq_idxs = self.__data_filter(db_seq_keys, db_seq_idxs, val_frames)

                # 保存筛选后的图像
                self.__q_images_key.extend(q_seq_keys)
                self.__db_images_key.extend(db_seq_keys)

                # 添加Query索引
                self.__q_seq_idx.extend(list(range(_lenQ, len(q_seq_keys) + _lenQ)))

        # 如果选择了城市、任务和子任务的组合，其中没有Query和Database图像，则退出。
        if len(self.__q_images_key) == 0 or len(self.__db_images_key) == 0:
            print('退出...')
            print('如果选择了城市、任务和子任务的组合，其中没有Query和Database图像，则退出')
            print('如果选择了城市、任务和子任务的组合，其中没有Query和Database图像，则退出')
            print('尝试选择不同的子任务或其他城市')
            sys.exit()

        self.__q_seq_idx = np.asarray(self.__q_seq_idx)
        self.__q_images_key = np.asarray(self.__q_images_key)
        self.__db_images_key = np.asarray(self.__db_images_key)
        self.__p_seq_idx = np.asarray(self.__p_seq_idx, dtype=object)
        self.__non_negative_indices = np.asarray(self.__non_negative_indices, dtype=object)
        self.__sideways = np.asarray(self.__sideways)
        self.__night = np.asarray(self.__night)

        if self.__mode in ['train']:
            # 计算Query采样时的权重，即晚上和路边权重高，容易被采到
            if positive_sampling:
                self.__calc_sampling_weights()
            else:
                self.__weights = np.ones(len(self.__q_seq_idx)) / float(len(self.__q_seq_idx))

    def __getitem__(self, index):
        # 获取对应的数据和标签
        triplet, target = self.__triplets_data[index]

        # 获取Query、Positive和Negative的索引
        q_idx = triplet[0]
        p_idx = triplet[1]
        n_idx = triplet[2:]

        # 返回图像信息
        query = self.__img_transform(Image.open(self.__q_images_key[q_idx]))
        positive = self.__img_transform(Image.open(self.__db_images_key[p_idx]))
        negatives = torch.stack([self.__img_transform(Image.open(self.__db_images_key[idx])) for idx in n_idx], 0)

        return query, positive, negatives, [q_idx, p_idx] + n_idx

    def __len__(self):
        return len(self.__triplets_data)

    def __calc_sampling_weights(self):
        """
        计算数据权重
        """
        # 计算Query大小
        N = len(self.__q_seq_idx)

        # 初始化权重都为1
        self.__weights = np.ones(N)

        # 夜间或侧面时权重更高
        if len(self.__night) != 0:
            self.__weights[self.__night] += N / len(self.__night)
        if len(self.__sideways) != 0:
            self.__weights[self.__sideways] += N / len(self.__sideways)

        # 打印权重信息
        print("#侧面 [{}/{}]; #夜间; [{}/{}]".format(len(self.__sideways), N, len(self.__night), N))
        print("正面和白天的权重为{:.4f}".format(1))
        if len(self.__night) != 0:
            print("正面且夜间的权重为{:.4f}".format(1 + N / len(self.__night)))
        if len(self.__sideways) != 0:
            print("侧面且白天的权重为{:.4f}".format(1 + N / len(self.__sideways)))
        if len(self.__sideways) != 0 and len(self.__night) != 0:
            print("侧面且夜间的权重为{:.4f}".format(1 + N / len(self.__night) + N / len(self.__sideways)))

    def __seq_idx_2_frame_idx(self, q_seq_key, q_seq_keys):
        """
        把序列索引转化为帧索引

        :param q_seq_key: 序列索引
        :param q_seq_keys: 序列集合
        :return: 索引对应的序列集合
        """
        return q_seq_keys[q_seq_key]

    def __frame_idx_2_uniq_frame_idx(self, frame_idx, uniq_frame_idx):
        """
        获取frame_idx在uniq_frame_idx中的索引列表

        :param frame_idx: 一个序列的帧ID
        :param uniq_frame_idx: 所有帧ID
        :return: 获取frame_idx在uniq_frame_idx中的索引列表
        """

        # 在不重复的数据帧列表uniq_frame_idx中找到要找的数据帧frame_idx，并产生对应的Mask
        frame_mask = np.in1d(uniq_frame_idx, frame_idx)

        # 返回frame_idx在uniq_frame_idx中的索引
        return np.where(frame_mask)[0]

    def __uniq_frame_idx_2_seq_idx(self, frame_idxs, seq_idxs):
        """
        返回图像帧对应的序列索引

        :param frame_idxs: 图像帧
        :param seq_idxs: 序列索引
        :return: 图像正所在的序列索引
        """

        # 在序列索引列表seq_idxs中找到要找的数据帧frame_idxs，并产生对应的Mask
        mask = np.in1d(seq_idxs, frame_idxs)
        # 把Mask重新组织成seq_idxs的形状
        mask = mask.reshape(seq_idxs.shape)

        # 得到序列的索引
        return np.where(mask)[0]

    def __rang_to_sequence(self, data, path, seq_length):
        """
        把数组变为序列

        :param data: 表型数据
        :param path: 数据地址
        :param seq_length: 序列长度
        """
        # 去读序列信息
        seq_info = pd.read_csv(join(path, 'seq_info.csv'), index_col=0)

        # 图像序列的名称和图像序列的索引
        seq_keys, seq_idxs = [], []

        for idx in data.index:
            # 边界的情况
            if idx < (seq_length // 2) or idx >= (len(seq_info) - seq_length // 2):
                continue

            # 计算当前序列数据帧的周边帧
            seq_idx = np.arange(-seq_length // 2, seq_length // 2) + 1 + idx
            # 获取一个序列帧
            seq = seq_info.iloc[seq_idx]

            # 一个序列必须是具有相同的序列键值（即sequence_key相同），以及连续的帧（即frame_number之间的差值为1）
            if len(np.unique(seq['sequence_key'])) == 1 and (seq['frame_number'].diff()[1:] == 1).all():
                seq_key = ','.join([join(path, 'images', key + '.jpg') for key in seq['key']])

                # 保存图像序列的名称
                seq_keys.append(seq_key)
                # 保存图像序列的索引
                seq_idxs.append(seq_idx)

        return seq_keys, np.asarray(seq_idxs)

    def __data_filter(self, seq_keys, seq_idxs, center_frame_condition):
        """
        根据序列中心点索引筛选序列

        :param seq_keys: 序列Key值
        :param seq_idxs: 序列索引
        :param center_frame_condition: 条件筛选的中心帧
        :return: 返回筛选后的Key和Idx
        """
        keys, idxs = [], []
        for key, idx in zip(seq_keys, seq_idxs):
            # 如果序列的中间索引在中心帧中，那么就把Key和Idx放入数组中
            if idx[len(idx) // 2] in center_frame_condition:
                keys.append(key)
                idxs.append(idx)
        return keys, np.asarray(idxs)

    @staticmethod
    def input_transform(resize=(480, 640)):
        """
        对图像进行转换

        :param resize: 转换后的图像大小
        :return: 返回转换对象
        """

        if resize[0] > 0 and resize[1] > 0:
            return transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    @staticmethod
    def collate_fn(batch):
        """
        从三元数据列表中创建mini-batch

        :param batch: batch数据
        :return: query: 数据的形状为(batch_size, 3, h, w); positive: 数据的形状为(batch_size, 3, h, w); negatives: 数据的形状为(batch_size, n, 3, h, w)，n表示反例的个数
        """
        # 对Batch中所有的数据进行检查
        batch = list(filter(lambda x: x is not None, batch))

        if len(batch) == 0:
            return None, None, None, None, None

        query, positive, negatives, indices = zip(*batch)

        query = data.dataloader.default_collate(query)
        positive = data.dataloader.default_collate(positive)
        negative_counts = data.dataloader.default_collate([x.shape[0] for x in negatives])
        negatives = torch.cat([torch.unsqueeze(x, 0) for x in negatives], 0)
        indices = torch.from_numpy(np.asarray(indices))

        return query, positive, negatives, negative_counts, indices

    @property
    def db_images_key(self):
        return self.__db_images_key

    @property
    def q_seq_idx(self):
        return self.__q_seq_idx

    @property
    def cached_subset_size(self):
        return self.__cached_subset_size

    @property
    def q_images_key(self):
        return self.__q_images_key

    @property
    def img_transform(self):
        return self.__img_transform

    @property
    def all_positive_indices(self):
        return self.__all_positive_indices

    def new_epoch(self):
        """
        每一个EPOCH都需要运行改程序，主要作用是把数据分为若干批，每一批再通过循环输出模型
        """

        # 通过向上取整后，计算一共有多少批Query数据
        self.__cached_subset_size = math.ceil(len(self.__q_seq_idx) / self.__cached_queries)

        ##################### 在验证机或测试集上使用
        # 构建所有数据集的索引数组
        q_seq_idx_array = np.arange(len(self.__q_seq_idx))

        # 使用采样方式对Query数据集进行采样
        q_seq_idx_array = random.choices(q_seq_idx_array, self.__weights, k=len(q_seq_idx_array))

        # 把随机采样的Query数据集分为cached_subset_size份
        self.__cached_subset_idx = np.array_split(q_seq_idx_array, self.__cached_subset_size)
        #######################

        # 重置子集的计数
        self.__current_subset = 0

    def refresh_data(self, model=None, output_dim=None, rank=0):
        """
        刷新数据，原因是每个EPOCH都不是取全部数据，而是一部分数据，即cached_queries多的数据，所以要刷新数据，来获取新数据

        :param model: 如果网络已经存在，那么使用该网络对图像进行特征提取，用于验证集或测试集
        :param output_dim: 网络输出的维度
        :param rank: GPU
        """
        # 清空数据
        self.__triplets_data.clear()

        # ========================================================================
        # 如果模型不存在，那么使用UTM来计算图像拍摄的位置，然后获取Query的正例和反例
        # ========================================================================

        if model is None:
            # 随机从q_seq_idx中采样cached_queries长度的数据索引
            q_choice_idxs = np.random.choice(len(self.__q_seq_idx), self.__cached_queries, replace=False)

            for q_choice_idx in q_choice_idxs:
                # 读取随机采样的Query索引
                q_idx = self.__q_seq_idx[q_choice_idx]
                # 读取随机采样的Query的正例索引，并随机从Query的正例中选取1个正例
                p_idx = np.random.choice(self.__p_seq_idx[q_choice_idx], size=1)[0]

                while True:
                    # 从数据库中随机读取negative_num个反例
                    n_idxs = np.random.choice(len(self.__db_images_key), self.__negative_size)

                    # Query的negative_distance_threshold距离外才被认为是负例，而negative_distance_threshold内认为是正例或非负例，
                    # 下面的判断是为了保证选择负例不在negative_distance_threshold范围内
                    if sum(np.in1d(n_idxs, self.__non_negative_indices[q_choice_idx])) == 0:
                        break

                # 创建三元数据和对应的标签
                triplet = [q_idx, p_idx, *n_idxs]
                target = [-1, 1] + [0] * len(n_idxs)

                self.__triplets_data.append((triplet, target))

            # 子数据集调用次数+1
            self.__current_subset += 1

            return

        # ========================================================================
        # 如果模型存在，那么使用模型对图像进行特征提取，然后计算特征之间距离来获取Query的正例和反例
        # ========================================================================

        # 判断当前读取的数据集批次是否为最后一批数据
        if self.__current_subset >= len(self.__cached_subset_idx):
            print('重置数据集批次...')
            self.__current_subset = 0

        # 得到当前批次的Query索引
        q_choice_idxs = np.asarray(self.__cached_subset_idx[self.__current_subset])

        # 得到Query的正例索引
        p_idxs = np.unique([i for idx in self.__p_seq_idx[q_choice_idxs] for i in idx])

        # 从所有数据中选出cached_negatives个负例
        n_idxs = np.random.choice(len(self.__db_images_key), self.__cached_negatives, replace=False)

        # 确保选出的负例中没有正例
        n_idxs = n_idxs[np.in1d(n_idxs,
                                np.unique([i for idx in self.__non_negative_indices[q_choice_idxs] for i in idx]),
                                invert=True)]

        # 构建Query、Positive、negative数据载入器
        opt = {'batch_size': self.__batch_size, 'shuffle': False}
        q_loader = torch.utils.data.DataLoader(ImagesFromList(self.__q_images_key[q_choice_idxs],
                                                              transform=self.__img_transform), **opt)
        p_loader = torch.utils.data.DataLoader(ImagesFromList(self.__db_images_key[p_idxs],
                                                              transform=self.__img_transform), **opt)
        n_loader = torch.utils.data.DataLoader(ImagesFromList(self.__db_images_key[n_idxs],
                                                              transform=self.__img_transform), **opt)

        model.eval()
        with torch.no_grad():
            q_vectors = torch.zeros(len(q_choice_idxs), output_dim).to(self.__device)
            p_vectors = torch.zeros(len(p_idxs), output_dim).to(self.__device)
            n_vectors = torch.zeros(len(n_idxs), output_dim).to(self.__device)

            batch_size = opt['batch_size']

            print('===> 开始计算Query、Positive、Negative的VLAD特征')

            # 获取Query的VLAD特征
            q_data_bar = tqdm(enumerate(q_loader), total=len(q_choice_idxs) // batch_size, leave=True)
            for i, (data, idx) in q_data_bar:
                q_data_bar.set_description('[{}/{}]计算Batch Query的特征...'.format(i, q_data_bar.total))
                image_descriptors = model.encoder(data.to(self.__device))
                vlad_descriptors = model.pool(image_descriptors)
                # 如果是PatchNetVLAD那么只是用Global VLAD
                if self.__config['train'].get('pooling') == 'patchnetvlad':
                    vlad_descriptors = vlad_descriptors[1]
                q_vectors[i * batch_size: (i + 1) * batch_size, :] = vlad_descriptors

                # 清空GPU存储
                del data, image_descriptors, vlad_descriptors

            del q_loader

            # 获取Positive的VLAD特征
            p_data_bar = tqdm(enumerate(p_loader), total=len(p_idxs) // batch_size, leave=True)
            for i, (data, idx) in p_data_bar:
                p_data_bar.set_description('[{}/{}]计算Batch Positive的特征...'.format(i, p_data_bar.total))
                image_descriptors = model.encoder(data.to(self.__device))
                vlad_descriptors = model.pool(image_descriptors)
                # 如果是PatchNetVLAD那么只是用Global VLAD
                if self.__config['train'].get('pooling') == 'patchnetvlad':
                    vlad_descriptors = vlad_descriptors[1]
                p_vectors[i * batch_size: (i + 1) * batch_size, :] = vlad_descriptors

                # 清空GPU存储
                del data, image_descriptors, vlad_descriptors

            del p_loader

            # 获取Negative的VLAD特征
            n_data_bar = tqdm(enumerate(n_loader), total=len(n_idxs) // batch_size, leave=True)
            for i, (data, idx) in n_data_bar:
                n_data_bar.set_description('[{}/{}]计算Batch Negative的特征...'.format(i, n_data_bar.total))
                image_descriptors = model.encoder(data.to(self.__device))
                vlad_descriptors = model.pool(image_descriptors)
                # 如果是PatchNetVLAD那么只是用Global VLAD
                if self.__config['train'].get('pooling') == 'patchnetvlad':
                    vlad_descriptors = vlad_descriptors[1]
                n_vectors[i * batch_size: (i + 1) * batch_size, :] = vlad_descriptors

                # 清空GPU存储
                del data, image_descriptors, vlad_descriptors

            del n_loader

            # 回收GPU内存
            torch.cuda.empty_cache()

        print('===> VLAD特征计算完成，搜索负例中...')

        # 计算Query与Positive的余弦距离
        p_cos_dis = torch.mm(q_vectors, p_vectors.t())
        # 对余弦距离按照降序进行排序
        p_cos_dis, p_cos_dis_rank = torch.sort(p_cos_dis, dim=1, descending=True)

        # 计算Query与Negative的余弦距离
        n_cos_dis = torch.mm(q_vectors, n_vectors.t())
        # 对余弦距离按照降序进行排序
        n_cos_dis, n_cos_dis_rank = torch.sort(n_cos_dis, dim=1, descending=True)

        p_cos_dis, p_cos_dis_rank = p_cos_dis.cpu().numpy(), p_cos_dis_rank.cpu().numpy()
        n_cos_dis, n_cos_dis_rank = n_cos_dis.cpu().numpy(), n_cos_dis_rank.cpu().numpy()

        for q in range(len(q_choice_idxs)):
            q_idx = q_choice_idxs[q]

            # 找到正例的索引
            cached_p_idx = np.where(np.in1d(p_idxs, self.__p_seq_idx[q_idx]))

            p_idx = np.where(np.in1d(p_cos_dis_rank[q, :], cached_p_idx))

            # 得到最近的正例，[q, p_idx]表示第q行，第p_idx列，
            # 但这个p_idx有好几个，并且[q, p_idx]返回的依然是原数组的维度
            closest_positive = p_cos_dis[q, p_idx][0][0]

            # 得到所有负例的距离
            dis_negative = n_cos_dis[q, :]

            # 计算最近距离与负例距离之间的差值，理论上应该都为负值
            distance = closest_positive - dis_negative

            # 如果距离差值大于0，说明就是错误
            error_negative = (distance > 0)

            # 如果正确的负例数小于需要的负例数，那么就跳过该Query
            if np.sum(error_negative) < self.__negative_size:
                continue

            # 获取距离最大的__negative_size个负例
            hardest_negative_idx = np.argsort(distance)[:self.__negative_size]
            # 获取Hardest Negative索引
            cached_hardest_negative = n_cos_dis_rank[q, hardest_negative_idx]

            # 找出最近的正例索引
            cached_p_idx = p_cos_dis_rank[q, p_idx][0][0]

            # 还原为原始的图像索引
            q_idx = self.__q_seq_idx[q_idx]
            p_idx = p_idxs[cached_p_idx]
            hardest_neg = n_idxs[cached_hardest_negative]

            # 打包三元对象
            triplet = [q_idx, p_idx, *hardest_neg]
            target = [-1, 1] + [0] * len(hardest_neg)

            self.__triplets_data.append((triplet, target))

        # 子数据集调用次数+1
        self.__current_subset += 1

        print(f'===> GPU:{rank},数据集建立完毕...')


if __name__ == '__main__':
    from models.models_generic import get_backbone, get_model

    parser = argparse.ArgumentParser(description='MSLS Database')

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

    train_dataset = MSLS(opt.dataset_root_dir,
                         mode='train',
                         device=device,
                         config=config,
                         cities_list='trondheim',
                         img_resize=tuple(map(int, str.split(config['train'].get('resize'), ','))),
                         negative_size=config['train'].getint('negative_size'),
                         batch_size=config['train'].getint('cache_batch_size'),
                         exclude_panos=config['train'].getboolean('exclude_panos'))

    train_dataset.new_epoch()

    if config['train']['pooling'].lower() == 'netvlad' or config['train']['pooling'].lower() == 'patchnetvlad':
        encoding_dim *= config[dataset_name].getint('num_clusters')

    model = model.to(device)

    train_dataset.refresh_data(model, encoding_dim)

    print('xx')
