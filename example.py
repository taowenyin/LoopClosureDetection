import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from typing import List


if __name__ == '__main__':
    data = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
    name = np.arange(10)

    print(data.shape)

    for i in range(data.shape[0]):
        plt.plot(np.arange(data.shape[1]), data[i], label=i)

    plt.xlabel("输入数据 x")
    plt.ylabel("sin(x) 或者 cos(x)")
    plt.title("三角函数图")
    plt.legend()

    plt.savefig('results/checkpoints/Loss_Pre_{}.png'.format('True'))

    plt.show()
