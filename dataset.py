import os
import numpy as np
import torchvision.transforms as transforms

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

        if model_type == 'test' or model_type == 'cluster':
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

        self.dbStruct = self.parse_dbStruct(db_file,  model_type == 'test' or model_type == 'cluster')
        # self.dbStruct = self.parse_dbStruct(db_file,  model_type == 'test') # taowenyin
        # 获取训练图片的路径
        self.images = [os.path.join(self.image_path, qIm) for qIm in self.dbStruct.qImage]
        # self.images = [os.path.join(self.image_path, dbIm) for dbIm in self.dbStruct.dbImage] # taowenyin
        if not only_db:
            # 获取训练Query图片的路径
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
    def parse_dbStruct(self, file_path, is_test=False):
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

        if is_test:
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

    def get_positives(self):
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)
            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ, radius=self.dbStruct.posDistThr)

        return self.positives


class Tokyo247Query(Dataset):
    def __init__(self, struct_path=None, image_path=None, model_type='train', only_db=False):
        print('xxx')


class Pitts250k(Dataset):
    def __init__(self, struct_path=None, image_path=None, model_type='train', only_db=False):
        print('xxx')


if __name__ == '__main__':
    tokyo = Tokyo247('./Datasets', '/home/taowenyin/MyCode/Dataset', model_type='test')

    positives = tokyo.get_positives()

    samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
    neigh = NearestNeighbors(radius=2)
    neigh.fit(samples)
    rng = neigh.radius_neighbors([[1., 1., 1.]])
    print(np.asarray(rng[0][0]))
    print(np.asarray(rng[1][0]))

    print('xx')