import torch
import shutil

from os.path import join, exists
from datetime import datetime
from os import makedirs


def save_checkpoint(state, opt, is_best_sofar, filename='checkpoint.pth.tar'):
    """
    保存模型参数

    :param state: 模型的训练装
    :param opt: 模型的配置信息
    :param is_best_sofar: 是否是最好的结果
    :param filename: 保存的文件名
    """

    save_path = join(opt.save_checkpoint_path, datetime.now().strftime('%Y_%m_%d'))
    # 文件夹不存在则创建文件夹
    if not exists(save_path):
        makedirs(save_path)

    # 设置模型保存的路径
    if opt.save_every_epoch:
        model_out_path = join(save_path, 'checkpoint_epoch' + str(state['epoch']) + '.pth.tar')
    else:
        model_out_path = join(save_path, filename)

    # 保存模型
    torch.save(state, model_out_path)

    # 如果是最好模型，那么就保存
    if is_best_sofar:
        shutil.copyfile(model_out_path, join(opt.save_checkpoint_path, 'model_best.pth.tar'))