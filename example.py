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
from semattlcd.tools.common import calculate_running_time


if __name__ == '__main__':

    start = datetime.now()

    # recall = [
    #     [0.1038961, 0.11038961, 0.12987013, 0.12987013, 0.11688312, 0.12337662, 0.11688312, 0.12337662, 0.1038961,  0.09090909, 0.0974026,  0.1038961, 0.09090909, 0.09090909],
    #     [0.21428571, 0.20779221, 0.23376623, 0.23376623, 0.19480519, 0.19480519, 0.21428571, 0.19480519, 0.19480519, 0.18181818, 0.16883117, 0.16883117, 0.16883117, 0.16233766],
    #     [0.27272727, 0.25324675, 0.25974026, 0.26623377, 0.27272727, 0.23376623, 0.23376623, 0.24675325, 0.22727273, 0.22727273, 0.24025974, 0.24675325, 0.23376623, 0.22727273],
    #     [0.35714286, 0.31818182, 0.33116883, 0.33766234, 0.32467532, 0.29220779, 0.29220779, 0.29220779, 0.2987013,  0.27272727, 0.27922078, 0.29220779, 0.27272727, 0.27272727],
    #     [0.45454545, 0.44155844, 0.42857143, 0.42857143, 0.42857143, 0.43506494, 0.43506494, 0.43506494, 0.42857143, 0.40909091, 0.41558442, 0.42207792, 0.42207792, 0.42207792],
    #     [0.52597403, 0.51948052, 0.52597403, 0.55194805, 0.52597403, 0.51948052, 0.50649351, 0.48701299, 0.48051948, 0.48701299, 0.47402597, 0.47402597, 0.47402597, 0.47402597],
    # ]
    #
    # recall = np.array(recall)
    # recall_item = recall[0]
    #
    # recall_min = np.amin(recall_item, axis=0)
    # recall_max = np.amax(recall_item, axis=0)
    #
    # recall_min_i = np.where(recall_item == recall_min)
    # recall_max_i = np.where(recall_item == recall_max)
    #
    # print('xxx')

    # plt.figure()
    #
    # for i in range(len(recall)):
    #     recall_item = recall[i]
    #
    #     plt.plot(np.arange(len(recall_item)), recall_item + i, label='N={}'.format(i))
    #
    #     recall_min = np.amin(recall_item, axis=0)
    #     recall_max = np.amax(recall_item, axis=0)
    #
    #     recall_min_i = np.where(recall_item == recall_min)
    #     recall_max_i = np.where(recall_item == recall_max)
    #
    #     a = recall_min_i[0][-1]
    #
    #     plt.annotate('Max Recall = {}'.format(format(recall_max, '0.4f')),
    #                  xy=(recall_max_i[0][-1], recall_max + i),
    #                  xytext=(recall_max_i[0][-1] + 2, recall_max - 0.02 + i),
    #                  arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleA=0, angleB=90'),
    #                  bbox=dict(boxstyle='round', fc="w"))
    #
    #     plt.annotate('Min Loss = {}'.format(format(recall_min, '0.4f')),
    #                  xy=(recall_min_i[0][-1], recall_min + i),
    #                  xytext=(recall_min_i[0][-1] - 8, recall_min - 0.02 + i),
    #                  arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleA=0, angleB=90'),
    #                  bbox=dict(boxstyle='round', fc="w"))
    #
    #     plt.annotate('Last Loss = {}'.format(format(recall_item[-1], '0.4f')),
    #                  xy=(len(recall_item) - 1, recall_item[-1] + i),
    #                  xytext=((len(recall_item) - 1) - 6, recall_item[-1] + 0.02 + i),
    #                  arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleA=0, angleB=90'),
    #                  bbox=dict(boxstyle='round', fc="w"))
    #
    # plt.xlim([-1, 32])
    # plt.ylim([0, 7])
    # plt.xlabel("EPOCH")
    # plt.ylabel("验证集的Recall")
    # plt.title("验证集的Recall-P{}-A{}".format(0, 1))
    # plt.legend()
    #
    # val_recall_dir = join('results', 'val_recall', datetime.now().strftime('%Y-%m-%d'))
    #
    # if not os.path.exists(val_recall_dir):
    #     os.makedirs(val_recall_dir)
    #
    # plt.savefig(join(val_recall_dir, 'Recall_{}_P{}_A{}.png'.format('xxxx', 0, 1)))
    #
    # plt.show()

    # =====================================================

    avg_loss = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    plt.figure()

    plt.plot(np.arange(len(avg_loss)), avg_loss, label='平均损失')

    loss_min = np.amin(avg_loss, axis=0)
    loss_max = np.amax(avg_loss, axis=0)

    loss_min_i = np.where(avg_loss == loss_min)
    loss_max_i = np.where(avg_loss == loss_max)

    plt.annotate('Max Loss = {}'.format(format(loss_max, '0.4f')),
                 xy=(loss_max_i[0][-1], loss_max),
                 xytext=(loss_max_i[0][-1] + 2, loss_max - 0.02),
                 arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleA=0, angleB=90'),
                 bbox=dict(boxstyle='round', fc="w"))

    plt.annotate('Min Loss = {}'.format(format(loss_min, '0.4f')),
                 xy=(loss_min_i[0][-1], loss_min),
                 xytext=(loss_min_i[0][-1] - 8, loss_min - 0.02),
                 arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleA=0, angleB=90'),
                 bbox=dict(boxstyle='round', fc="w"))

    plt.annotate('Last Loss = {}'.format(format(avg_loss[-1], '0.4f')),
                 xy=(len(avg_loss) - 1, avg_loss[-1]),
                 xytext=((len(avg_loss) - 1) - 6, avg_loss[-1] + 0.02),
                 arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleA=0, angleB=90'),
                 bbox=dict(boxstyle='round', fc="w"))

    # 设置X、Y坐标的最大最小值
    # plt.xlim([-1, 32])
    # plt.ylim([0.1, 0.34])
    plt.xlabel("EPOCH")
    plt.ylabel("平均损失")
    plt.title("训练损失-P{}-A{}".format(0, 1))
    plt.legend()

    plt.show()

    end = datetime.now()

    running_time = end - start

    print(calculate_running_time(running_time.microseconds))
