import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path

from os.path import join
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from typing import List
from datetime import datetime


if __name__ == '__main__':
    recall = [
        [0.25, 0.26, 0.13, 0.78, 0.56, 0.84],
        [0.25, 0.26, 0.13, 0.78, 0.56, 0.84],
        [0.25, 0.26, 0.13, 0.78, 0.56, 0.84],
        [0.25, 0.26, 0.13, 0.78, 0.56, 0.84],
        [0.25, 0.26, 0.13, 0.78, 0.56, 0.84],
        [0.25, 0.26, 0.13, 0.78, 0.56, 0.84],
    ]

    recall = np.array(recall)

    for i in range(len(recall)):
        recall_item = recall[i]

        plt.plot(np.arange(len(recall_item)), recall_item + i, label='N={}'.format(i))

        recall_min = np.amin(recall_item, axis=0)
        recall_max = np.amax(recall_item, axis=0)

        recall_min_i = np.where(recall_item == recall_min)
        recall_max_i = np.where(recall_item == recall_max)

        plt.annotate('Max Recall = {}'.format(format(recall_max, '0.4f')),
                     xy=(recall_max_i[0], recall_max),
                     xytext=(recall_max_i[0] + 2, recall_max - 0.02),
                     arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleA=0, angleB=90'),
                     bbox=dict(boxstyle='round', fc="w"))

        plt.annotate('Min Loss = {}'.format(format(recall_min, '0.4f')),
                     xy=(recall_min_i[0], recall_min),
                     xytext=(recall_min_i[0] - 8, recall_min - 0.02),
                     arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleA=0, angleB=90'),
                     bbox=dict(boxstyle='round', fc="w"))

        plt.annotate('Last Loss = {}'.format(format(recall_item[-1], '0.4f')),
                     xy=(len(recall_item) - 1, recall_item[-1]),
                     xytext=((len(recall_item) - 1) - 6, recall_item[-1] + 0.02),
                     arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleA=0, angleB=90'),
                     bbox=dict(boxstyle='round', fc="w"))

    plt.xlim([-1, 32])
    plt.ylim([0, 7])
    plt.xlabel("EPOCH")
    plt.ylabel("验证集的Recall")
    plt.title("验证集的Recall-P{}-A{}".format(0, 1))
    plt.legend()

    val_recall_dir = join('results', 'val_recall', datetime.now().strftime('%Y-%m-%d'))

    plt.savefig(join(val_recall_dir, 'Recall_{}_P{}_A{}.png'.format('xxxx', 0, 1)))

    plt.show()
