import numpy as np
import matplotlib.pyplot as plt
import os.path

from os.path import join


def human_bytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string"""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B/KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B/MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B/GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B/TB)


def save_checkpoint():
    print('xxx')


def save_loss(avg_loss, path, config):
    plt.plot(np.arange(len(avg_loss)), avg_loss, label='平均损失')

    loss_min = np.amin(avg_loss, axis=0)
    loss_max = np.amax(avg_loss, axis=0)

    loss_min_i = np.where(avg_loss == loss_min)
    loss_max_i = np.where(avg_loss == loss_max)

    plt.annotate('Max Loss = {}'.format(format(loss_max, '0.4f')),
                 xy=(loss_max_i[0], loss_max),
                 xytext=(loss_max_i[0] + 2, loss_max - 0.02),
                 arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleA=0, angleB=90'),
                 bbox=dict(boxstyle='round', fc="w"))

    plt.annotate('Min Loss = {}'.format(format(loss_min, '0.4f')),
                 xy=(loss_min_i[0], loss_min),
                 xytext=(loss_min_i[0] - 8, loss_min - 0.02),
                 arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleA=0, angleB=90'),
                 bbox=dict(boxstyle='round', fc="w"))

    plt.annotate('Last Loss = {}'.format(format(avg_loss[-1], '0.4f')),
                 xy=(len(avg_loss) - 1, avg_loss[-1]),
                 xytext=((len(avg_loss) - 1) - 6, avg_loss[-1] + 0.02),
                 arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleA=0, angleB=90'),
                 bbox=dict(boxstyle='round', fc="w"))

    # 设置X、Y坐标的最大最小值
    plt.xlim([-1, 32])
    plt.ylim([0.1, 0.34])
    plt.xlabel("EPOCH")
    plt.ylabel("平均损失")
    plt.title("训练损失-P{}-A{}".format(
        int(config.getboolean('pretrained')),
        int(config.getboolean('attention'))))
    plt.legend()

    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(join(path,
                     'Loss_{}_P{}_A{}.png'.format(
                         config['global_params']['arch_type'],
                         int(config.getboolean('pretrained')),
                         int(config.getboolean('attention')))))

    plt.show()