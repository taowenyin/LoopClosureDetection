import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from os.path import join
from tools.parallel import setup_parallel, parallel_cleanup
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange
from training.train_epoch import train_epoch


def run_parallel(parallel_fn, world_size, model, encoding_dim, config, opt, train_dataset, validation_dataset):
    """
    多进程产生函数。不用这个的话需要在运行训练代码的时候，用'python-m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE train.py'才能启动。

    :param parallel_fn: 分布式的函数
    :param world_size: GPU数量
    :param model: 模型
    :param encoding_dim: 编码器的输出通道数
    :param config: 配置信息
    :param opt: 参数对象
    :param train_dataset: 训练数据集
    :param validation_dataset: 验证数据集
    """
    mp.spawn(parallel_fn,
             args=(world_size, model, encoding_dim, config, opt, train_dataset, validation_dataset),
             nprocs=world_size, join=True)


def main_parallel_train(rank, world_size, model, encoding_dim, config, opt, train_dataset, validation_dataset):
    """
    并行训练模型
    :param rank: 当前进程中GPU的编号
    :param world_size: 总共有多少个GPU
    :param model: 训练模型
    :param config: 配置信息
    :param opt: 参数对象
    :param train_dataset: 训练数据集
    :param validation_dataset: 验证数据集
    """
    setup_parallel(rank, world_size)

    print(f'GPU:{rank}开始运行...')

    # 使用三元损失函数，并使用欧氏距离作为距离函数
    criterion = nn.TripletMarginLoss(margin=config['train'].getfloat('margin') ** 0.5,
                                     p=2, reduction='sum')

    if config['train'].get('optim') == 'ADAM':
        optimizer = optim.Adam(filter(lambda par: par.requires_grad, model.parameters()),
                               lr=config['train'].getfloat('lr'))
    else:
        raise ValueError('未知的优化器: ' + config['train'].get('optim'))

    if rank == 0:
        # 创建TensorBoard的写入对象
        writer = SummaryWriter(log_dir=join(opt.result_dir, 'logs'))
    else:
        writer = None

    # 把模型改成并行模型
    model = DDP(model.to(rank), device_ids=[rank])

    # 开始训练，从opt.start_epoch + 1次开始，到opt.epochs_count次结束
    train_epoch_bar = trange(opt.start_epoch + 1, opt.epochs_count + 1)
    for epoch in train_epoch_bar:
        if rank == 0:
            train_epoch_bar.set_description(f'GPU:{rank},第{epoch}/{opt.epochs_count - opt.start_epoch}次训练周期')

        # 执行一个训练周期
        train_epoch(rank, world_size, train_dataset, model, optimizer,
                    criterion, encoding_dim, epoch, config, opt, writer)

    # 训练完成后在GPU:0上关闭
    if rank == 0:
        writer.close()

    parallel_cleanup()