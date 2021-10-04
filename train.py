import argparse
import configparser
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from os.path import join
from semattlcd.tools import LOOP_CLOSURE_ROOT_DIR
from semattlcd.tools.datasets import input_transform
from semattlcd.dataset.mapillary_sls.msls import MSLS
from semattlcd.train.train_epoch import train_epoch
from semattlcd.models.models_generic import get_model, get_backend
from tqdm import trange, tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic-Attention-LCD-train')

    parser.add_argument('--config_path', type=str, default=join(LOOP_CLOSURE_ROOT_DIR, 'configs/train.ini'),
                        help='File name (with extension) to an ini file that stores most of the configuration data for Loop Closure')
    parser.add_argument('--dataset_root_dir', type=str,
                        default='/mnt/Dataset/Mapillary_Street_Level_Sequences',
                        help='Root directory of dataset')
    parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')
    parser.add_argument('--threads', type=int, default=6, help='Number of threads for each data loader to use')
    parser.add_argument('--continuing', action='store_true',
                        help='If true, started training earlier and continuing. Else retrain.')

    opt = parser.parse_args()

    # 读取配置文件
    configfile = opt.config_path
    config = configparser.ConfigParser()
    config.read(configfile)

    # 设置GPU或CPU
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")

    # 设置随机种子
    random.seed(int(config['train']['seed']))
    np.random.seed(int(config['train']['seed']))
    torch.manual_seed(int(config['train']['seed']))
    if cuda:
        torch.cuda.manual_seed(int(config['train']['seed']))

    # 缩放图片后的大小
    resize = (int(config['train']['image_resize_h']), int(config['train']['image_resize_w']))

    optimizer = None
    scheduler = None

    print('===> Building model')
    # 构建编码器
    encoder_dim, encoder = get_backend(config['global_params'])
    # 构建Pooling
    model = get_model(encoder, encoder_dim, config['global_params'])

    # 定义优化器
    if config['train']['optim'] == 'SGD':
        optimizer = optim.SGD(filter(lambda par: par.requires_grad,
                                     model.parameters()), lr=float(config['train']['lr']),
                              momentum=float(config['train']['momentum']),
                              weight_decay=float(config['train']['weight_decay']))

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(config['train']['lr_step']),
                                              gamma=float(config['train']['lr_gamma']))
    elif config['train']['optim'] == 'ADAM':
        optimizer = optim.Adam(filter(lambda par: par.requires_grad,
                                      model.parameters()), lr=float(config['train']['lr']))  # , betas=(0, 0.9))

    # 使用三元损失函数
    criterion = nn.TripletMarginLoss(
        margin=float(config['train']['margin']) ** 0.5, p=2, reduction='sum').to(device)

    # 初始化模型对象
    model = model.to(device)

    print('===> Loading dataset(s)')
    train_dataset = MSLS(opt.dataset_root_dir,
                         cities=config['msls']['train_cities'],
                         nNeg=int(config['train']['nNeg']),
                         transform=input_transform(resize),
                         bs=int(config['train']['batch_size']),
                         mode='train',
                         threads=opt.threads,
                         margin=float(config['train']['margin']),
                         exclude_panos=config['train'].getboolean('exclude_panos'))
    validation_dataset = MSLS(opt.dataset_root_dir,
                              cities=config['msls']['validation_cities'],
                              nNeg=int(config['train']['nNeg']),
                              transform=input_transform(resize),
                              bs=int(config['train']['batch_size']),
                              mode='val',
                              threads=opt.threads,
                              margin=float(config['train']['margin']),
                              exclude_panos=config['train'].getboolean('exclude_panos'),
                              posDistThr=25)
    print('===> Training query set:', len(train_dataset.qIdx))
    print('===> Evaluating on val set, query count:', len(validation_dataset.qIdx))
    print('===> Training model')

    not_improved = 0
    best_score = 0
    avg_loss = []

    for epoch in trange(1, opt.nEpochs + 1, desc='Epoch number'.rjust(15), position=0):
        tqdm.write('===> Running Train')

        avg_loss_epoch = train_epoch(train_dataset, model, optimizer, criterion,
                                     encoder_dim, device, epoch, opt, config)
        avg_loss.append(avg_loss_epoch)

        # 更新学习率
        if scheduler is not None:
            scheduler.step(epoch)

        # 执行验证程序
        if (epoch % int(config['train']['eval_every'])) == 0:
            tqdm.write('===> Running Eval')

    # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear thes
    torch.cuda.empty_cache()

    plt.plot(np.arange(len(avg_loss)), avg_loss, label='平均损失')

    loss_min = np.amin(avg_loss, axis=0)
    loss_max = np.amax(avg_loss, axis=0)

    loss_min_i = np.where(avg_loss == loss_min)
    loss_max_i = np.where(avg_loss == loss_max)

    plt.annotate('Min Loss = {}'.format(loss_min),
                 xy=(loss_min_i[0], loss_min),
                 xytext=(loss_min_i[0] + 1, loss_min + 0.01),
                 arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleA=0, angleB=90'),
                 bbox=dict(boxstyle='round', fc="w"))

    plt.annotate('Max Loss = {}'.format(loss_max),
                 xy=(loss_max_i[0], loss_max),
                 xytext=(loss_max_i[0] + 1, loss_max + 0.01),
                 arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleA=0, angleB=90'),
                 bbox=dict(boxstyle='round', fc="w"))

    plt.annotate('Last Loss = {}'.format(avg_loss[-1]),
                 xy=(len(avg_loss) - 1, avg_loss[-1]),
                 xytext=((len(avg_loss) - 1) + 1, avg_loss[-1] + 0.01),
                 arrowprops=dict(arrowstyle='->', connectionstyle='angle3, angleA=0, angleB=90'),
                 bbox=dict(boxstyle='round', fc="w"))

    plt.text(loss_min_i[0], loss_min + 1, str(loss_min))
    plt.text(loss_max_i[0], loss_max + 1, str(loss_max))

    plt.xlabel("EPOCH")
    plt.ylabel("平均损失")
    plt.title("训练损失，预训练={}".format(config['global_params']['pretrained']))
    plt.legend()

    plt.show()

    print('Done')
