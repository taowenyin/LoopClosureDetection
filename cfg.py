from enum import Enum


# 基本模型类型
class BaseModel(Enum):
    AlexNet = 1
    VGG16 = 2


BASE_MODEL_TYPE = BaseModel.VGG16

BATCH_SIZE = 1

EPOCH_NUMBER = 200

BASE_LR = 0.0001

LR_STEP = 5

LR_GAMMA = 0.5

TRIPLET_MARGIN = 0.1


class Dataset(Enum):
    Tokyo247 = 1
    Pitts250k = 2


DATASET_TYPE = Dataset.Tokyo247


Dataset_Info = [
    {
        'name': 'Tokyo247',
        'clusters': 12
    },
    {
        'name': 'Pitts250k',
        'clusters': 12
    }
]


class Pooling(Enum):
    NetVLAD = 1
    MAX = 2
    AVG = 3


POOLING_TYPE = Pooling.NetVLAD
