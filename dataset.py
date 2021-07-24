import os
import numpy as np
import torchvision.transforms as transforms
import h5py
import torch

from torch.utils.data import Dataset
from scipy.io import loadmat
from collections import namedtuple
from PIL import Image
from sklearn.neighbors import NearestNeighbors


# 定义从MATLAB中读取的数据
DBStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
                                   'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])


class Tokyo247(Dataset):
    def __init__(self, data_path=None, model_type='train', only_db=False):
        self.struct_dir = os.path.join(data_path, 'Tokyo247')
        self.image_path = os.path.join(data_path, 'Tokyo247/data')

        if not os.path.exists(self.struct_dir):
            raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Tokyo247 dataset')

        if model_type == 'test':
            db_file = os.path.join(self.struct_dir, 'tokyo247.mat')
            if not os.path.exists(db_file):
                raise FileNotFoundError('tokyo247.mat is hardcoded, please adjust to point to tokyo247.mat')
        elif model_type == 'val':
            db_file = os.path.join(self.struct_dir, 'tokyoTM_val.mat')
            if not os.path.exists(db_file):
                raise FileNotFoundError('tokyoTM_val.mat is hardcoded, please adjust to point to tokyoTM_val.mat')
        else:
            db_file = os.path.join(self.struct_dir, 'tokyoTM_train.mat')
            if not os.path.exists(db_file):
                raise FileNotFoundError('tokyoTM_train.mat is hardcoded, please adjust to point to tokyoTM_train.mat')

        # 解析MATLAB数据结构
        self.dbStruct = self.parse_dbStruct(db_file,  (model_type == 'test' or model_type == 'cluster'))
        # 获取图片的路径
        self.images = [os.path.join(self.image_path, dbIm) for dbIm in self.dbStruct.dbImage]
        if not only_db:
            # 增加Query图片的路径
            self.images += [os.path.join(self.image_path, qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        # 获取文件路径和文件名
        file_path, file_name = os.path.split(self.images[index])
        img = self.image_transform(Image.open(self.images[index]))

        return img, index, file_name

    def __len__(self):
        return len(self.images)

    # 解析MATLAB文件
    def parse_dbStruct(self, file_path, is_test_cluster=False):
        # 载入MATLAB文件
        mat = loadmat(file_path)
        # 获取m文件中的对象
        matStruct = mat['dbStruct'].item()

        # 标记概述及类型
        whichSet = matStruct[0].item()
        # 获取图像信息
        dbImage = [f[0].item() for f in matStruct[1]]
        # 获取图像UTM定位
        utmDb = matStruct[2].T

        if is_test_cluster:
            # 获取Query图像
            qImage = [f[0].item() for f in matStruct[3]]
            # 获取Query图像UTM定位
            utmQ = matStruct[4].T

            # 整个数据集包含的图像数
            numDb = matStruct[5].item()
            # 整个数据集包含的Query图像数
            numQ = matStruct[6].item()

            posDistThr = matStruct[7].item()
            posDistSqThr = matStruct[8].item()
            nonTrivPosDistSqThr = matStruct[9].item()
        else:
            # 获取Query图像
            qImage = [f[0].item() for f in matStruct[4]]
            # 获取Query图像UTM定位
            utmQ = matStruct[5].T

            # 整个数据集包含的图像数
            numDb = matStruct[7].item()
            # 整个数据集包含的Query图像数
            numQ = matStruct[8].item()

            # 距离的阈值，超过该值的半径就认为不在范围内
            posDistThr = matStruct[9].item()
            posDistSqThr = matStruct[10].item()
            nonTrivPosDistSqThr = matStruct[11].item()

        return DBStruct(whichSet, 'Tokyo247', dbImage, utmDb, qImage, utmQ, numDb, numQ,
                        posDistThr, posDistSqThr, nonTrivPosDistSqThr)

    def image_transform(self, image):
        # 图片转化为Tensor和标准化的流程
        transform_img = transforms.Compose(
            [
                # 把图像转化为Tensor
                transforms.ToTensor(),
                # 把图像进行归一化
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )

        # 对所有图像进行转化
        img = transform_img(image)

        return img

    # 得到正例，通过拟合所有数据，然后找到Query指定范围内的图像就是正例图像
    def get_positives(self):
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)
            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.posDistThr)

        return self.positives


class Tokyo247Query(Dataset):
    def __init__(self, data_path=None, model_type='train', nNegSample=1000, nNeg=10, margin=0.1):
        self.struct_dir = os.path.join(data_path, 'Tokyo247')
        self.image_path = os.path.join(data_path, 'Tokyo247/data')
        self.query_path = os.path.join(data_path, 'Tokyo247/query')
        self.margin = margin

        if not os.path.exists(self.struct_dir):
            raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Tokyo247 dataset')

        if model_type == 'test':
            db_file = os.path.join(self.struct_dir, 'tokyo247.mat')
            if not os.path.exists(db_file):
                raise FileNotFoundError('tokyo247.mat is hardcoded, please adjust to point to tokyo247.mat')
        elif model_type == 'val':
            db_file = os.path.join(self.struct_dir, 'tokyoTM_val.mat')
            if not os.path.exists(db_file):
                raise FileNotFoundError('tokyoTM_val.mat is hardcoded, please adjust to point to tokyoTM_val.mat')
        else:
            db_file = os.path.join(self.struct_dir, 'tokyoTM_train.mat')
            if not os.path.exists(db_file):
                raise FileNotFoundError('tokyoTM_train.mat is hardcoded, please adjust to point to tokyoTM_train.mat')

        # 解析MATLAB数据结构
        self.dbStruct = self.parse_dbStruct(db_file, (model_type == 'test' or model_type == 'cluster'))

        # 随机采样的负例数
        self.nNegSample = nNegSample
        # 用于训练的负例数
        self.nNeg = nNeg

        # 在阈值半径范围内就表示是整理，并通过NN找到到他们
        knn = NearestNeighbors(n_jobs=-1)
        # 拟合所有数据的位置
        knn.fit(self.dbStruct.utmDb)

        # 找到Query周围的正例
        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ,
                                                              radius=self.dbStruct.nonTrivPosDistSqThr ** 0.5,
                                                              return_distance=False))
        # 对每个Query的所有正例进行排序
        for i, positives in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(positives)

        # 删除某些周围没有正例的Query，留下能被训练的Query
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives]) > 0)[0]

        # 找到Query某个范围内的所有对象
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                                                   radius=self.dbStruct.posDistThr,
                                                   return_distance=False)

        # 保存所有反例
        self.potential_negatives = []
        # 剔除掉正例，剩下的就反例
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb), pos, assume_unique=True))

        # 保存图像特征的HDF5文件
        self.cache = None
        #
        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, index):
        # 索引要查询的Query对象
        index = self.queries[index]
        # 读取centroids下的图像特征文件
        with h5py.File(self.cache, mode='r') as h5:
            # 获取特征
            h5feat = h5.get("features")

            qOffset = self.dbStruct.numDb
            qFeat = h5feat[index + qOffset]

            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            knn = NearestNeighbors(n_jobs=-1)  # replace with faiss?
            knn.fit(posFeat)
            dPos, posNN = knn.kneighbors(qFeat.reshape(1, -1), 1)
            dPos = dPos.item()
            posIndex = self.nontrivial_positives[index][posNN[0]].item()

            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))

            negFeat = h5feat[negSample.tolist()]
            knn.fit(negFeat)

            dNeg, negNN = knn.kneighbors(qFeat.reshape(1, -1),
                                         self.nNeg * 10)  # to quote netvlad paper code: 10x is hacky but fine
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # try to find negatives that are within margin, if there aren't any return none
            violatingNeg = dNeg < dPos + self.margin ** 0.5

            if np.sum(violatingNeg) < 1:
                # if none are violating then skip this query
                return None

            negNN = negNN[violatingNeg][:self.nNeg]
            self.negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = self.negIndices

        query = Image.open(os.path.join(self.query_path, self.dbStruct.qImage[index]))
        positive = Image.open(os.path.join(self.image_path, self.dbStruct.dbImage[posIndex]))

        query = self.image_transform(query)
        positive = self.image_transform(positive)

        negatives = []
        for negIndex in self.negIndices:
            negative = Image.open(os.path.join(self.image_path, self.dbStruct.dbImage[negIndex]))
            negative = self.image_transform(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, posIndex] + self.negIndices.tolist()

    def __len__(self):
        return len(self.queries)

    def image_transform(self, image):
        # 图片转化为Tensor和标准化的流程
        transform_img = transforms.Compose(
            [
                # 把图像转化为Tensor
                transforms.ToTensor(),
                # 把图像进行归一化
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )

        # 对所有图像进行转化
        img = transform_img(image)

        return img

    # 解析MATLAB文件
    def parse_dbStruct(self, file_path, is_test_cluster=False):
        # 载入MATLAB文件
        mat = loadmat(file_path)
        # 获取m文件中的对象
        matStruct = mat['dbStruct'].item()

        # 标记概述及类型
        whichSet = matStruct[0].item()
        # 获取图像信息
        dbImage = [f[0].item() for f in matStruct[1]]
        # 获取图像UTM定位
        utmDb = matStruct[2].T

        if is_test_cluster:
            # 获取Query图像
            qImage = [f[0].item() for f in matStruct[3]]
            # 获取Query图像UTM定位
            utmQ = matStruct[4].T

            # 整个数据集包含的图像数
            numDb = matStruct[5].item()
            # 整个数据集包含的Query图像数
            numQ = matStruct[6].item()

            posDistThr = matStruct[7].item()
            posDistSqThr = matStruct[8].item()
            nonTrivPosDistSqThr = matStruct[9].item()
        else:
            # 获取Query图像
            qImage = [f[0].item() for f in matStruct[4]]
            # 获取Query图像UTM定位
            utmQ = matStruct[5].T

            # 整个数据集包含的图像数
            numDb = matStruct[7].item()
            # 整个数据集包含的Query图像数
            numQ = matStruct[8].item()

            # 距离的阈值，超过该值的半径就认为不在范围内
            posDistThr = matStruct[9].item()
            posDistSqThr = matStruct[10].item()
            nonTrivPosDistSqThr = matStruct[11].item()

        return DBStruct(whichSet, 'Tokyo247', dbImage, utmDb, qImage, utmQ, numDb, numQ,
                        posDistThr, posDistSqThr, nonTrivPosDistSqThr)


class Pitts250k(Dataset):
    def __init__(self, struct_path=None, image_path=None, model_type='train', only_db=False):
        print('xxx')


if __name__ == '__main__':
    a = [np.empty((0,)) for _ in range(10)]

    print('xx')