import argparse
import configparser
import torch
import random
import numpy as np

from os.path import join
from loopclosure.tools import LOOP_CLOSURE_ROOT_DIR
from loopclosure.tools.datasets import input_transform
from loopclosure.dataset.mapillary_sls.msls import MSLS
from tqdm.auto import trange

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loop-Closure-Detection-train')

    parser.add_argument('--config_path', type=str, default=join(LOOP_CLOSURE_ROOT_DIR, 'configs/train.ini'),
                        help='File name (with extension) to an ini file that stores most of the configuration data for Loop Closure')
    parser.add_argument('--dataset_root_dir', type=str,
                        default='/home/taowenyin/MyCode/Dataset/Mapillary_Street_Level_Sequences',
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

    if opt.continuing:
        print('continuing')
    else:
        resize = (int(config['train']['image_resize_w']), int(config['train']['image_resize_h']))

        print('===> Loading dataset(s)')
        train_dataset = MSLS(opt.dataset_root_dir,
                             cities=config['msls']['train_cities'],
                             nNeg=int(config['train']['nNeg']),
                             transform=input_transform(resize),
                             bs=int(config['train']['batch_size']),
                             threads=opt.threads,
                             margin=float(config['train']['margin']),
                             exclude_panos=config['train'].getboolean('exclude_panos'),
                             mode='train')
        validation_dataset = MSLS(opt.dataset_root_dir,
                                  cities=config['msls']['validation_cities'],
                                  nNeg=int(config['train']['nNeg']),
                                  transform=input_transform(resize),
                                  bs=int(config['train']['batch_size']),
                                  threads=opt.threads,
                                  margin=float(config['train']['margin']),
                                  exclude_panos=config['train'].getboolean('exclude_panos'),
                                  mode='val',
                                  posDistThr=25)
        print('===> Training query set:', len(train_dataset.qIdx))
        print('===> Evaluating on val set, query count:', len(validation_dataset.qIdx))

        not_improved = 0
        best_score = 0

        for epoch in trange(1, opt.nEpochs + 1, desc='Epoch number'.rjust(15), position=0):
            print('xxx')
