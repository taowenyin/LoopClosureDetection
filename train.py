import argparse
import configparser
import os.path
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from os.path import join
from semattlcd.tools import LOOP_CLOSURE_ROOT_DIR
from semattlcd.tools.datasets import input_transform
from semattlcd.dataset.mapillary_sls.msls import MSLS
from semattlcd.train.train_epoch import train_epoch
from semattlcd.models.models_generic import get_model, get_backend
from tqdm import trange, tqdm
from datetime import datetime
from semattlcd.tools.common import draw_train_loss, draw_validation_recall, save_checkpoint, calculate_running_time
from semattlcd.train.val import val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic-Attention-LCD-train')

    parser.add_argument('--config_path', type=str, default=join(LOOP_CLOSURE_ROOT_DIR, 'configs/train.ini'),
                        help='File name (with extension) to an ini file that stores most of the configuration data for Loop Closure')
    parser.add_argument('--dataset_root_dir', type=str,
                        default='/mnt/Dataset/Mapillary_Street_Level_Sequences',
                        help='Root directory of dataset')
    parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')
    parser.add_argument('--save_every_epoch', action='store_true',
                        help='Flag to set a separate checkpoint file for each new epoch')
    parser.add_argument('--threads', type=int, default=6, help='Number of threads for each data loader to use')
    parser.add_argument('--continuing', action='store_true',
                        help='If true, started training earlier and continuing. Else retrain.')

    # 程序开始时间
    program_start = datetime.now()

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

    # 保存Loss的路径
    train_loss_dir = join('results', 'train_loss', datetime.now().strftime('%Y-%m-%d'))
    # 保存权重的路径
    save_weights_dir = join('results', 'checkpoints', datetime.now().strftime('%Y-%m-%d'))
    # 验证集recall的路径
    val_recall_dir = join('results', 'val_recall', datetime.now().strftime('%Y-%m-%d'))

    not_improved = 0
    best_score = 0
    avg_loss = []
    val_recalls = []
    val_recalls_1 = []
    val_recalls_5 = []
    val_recalls_10 = []
    val_recalls_20 = []
    val_recalls_50 = []
    val_recalls_100 = []

    for epoch in trange(1, opt.nEpochs + 1, desc='Epoch number'.rjust(15), position=0):
        tqdm.write('===> Running Train')
        # 执行训练模型
        avg_loss_epoch = train_epoch(train_dataset, model, optimizer, criterion,
                                     encoder_dim, device, epoch, opt, config)
        avg_loss.append(avg_loss_epoch)

        # 更新学习率
        if scheduler is not None:
            scheduler.step(epoch)

        # 执行验证程序
        if (epoch % int(config['train']['eval_every'])) == 0:
            tqdm.write('===> Running Eval')
            # 执行验证模型
            recalls = val(validation_dataset, model, config.getint('global_params', 'pca_dim'),
                          device, config, pbar_position=1)

            # 保存1, 5, 10, 20, 50, 100正确率
            val_recalls_1.append(recalls[1])
            val_recalls_5.append(recalls[5])
            val_recalls_10.append(recalls[10])
            val_recalls_20.append(recalls[20])
            val_recalls_50.append(recalls[50])
            val_recalls_100.append(recalls[100])

            # 判断最佳模型
            is_best = recalls[5] > best_score
            if is_best:
                not_improved = 0
                best_score = recalls[5]
            else:
                not_improved += 1

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'recalls': recalls,
                'best_score': best_score,
                'not_improved': not_improved,
                'optimizer': optimizer.state_dict()
            }, config['global_params'], opt, save_weights_dir, is_best)

            # if int(config['train']['patience']) > 0 and \
            #         not_improved > (int(config['train']['patience']) / int(config['train']['eval_every'])):
            #     print('Performance did not improve for', config['train']['patience'], 'epochs. Stopping.')
            #     break

    val_recalls.append(val_recalls_1)
    val_recalls.append(val_recalls_5)
    val_recalls.append(val_recalls_10)
    val_recalls.append(val_recalls_20)
    val_recalls.append(val_recalls_50)
    val_recalls.append(val_recalls_100)

    torch.cuda.empty_cache()

    # 绘制训练误差
    draw_train_loss(avg_loss, train_loss_dir, config['global_params'])
    # 绘制验证的召回
    draw_validation_recall(val_recalls, val_recall_dir, config['global_params'])

    # 程序结束时间
    program_end = datetime.now()

    # 计算程序运行时间
    running_time = program_end - program_start

    print('Done, Running Time: {}'.format(calculate_running_time(running_time.microseconds)))
