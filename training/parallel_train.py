import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import h5py
import torch
import torch.distributed as dist

from time import sleep
from os.path import join, isfile
from tools.parallel import setup_parallel, parallel_cleanup
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange
from training.train_epoch import train_epoch
from models.models_generic import get_backbone, get_model, create_image_clusters
from tools import ROOT_DIR
from shutil import copyfile
from dataset.mapillary_sls.MSLS import MSLS
from training.val import validation
from tools.common import save_checkpoint


def run_parallel(parallel_fn, world_size, config, opt, train_dataset, validation_dataset):
    """
    多进程产生函数。不用这个的话需要在运行训练代码的时候，用'python-m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE train.py'才能启动。

    :param parallel_fn: 分布式的函数
    :param world_size: GPU数量
    :param config: 配置信息
    :param opt: 参数对象
    :param train_dataset: 训练数据集
    :param validation_dataset: 验证数据集
    """
    mp.spawn(parallel_fn,
             args=(world_size, config, opt, train_dataset, validation_dataset),
             nprocs=world_size, join=True)


def main_parallel_train(rank, world_size, config, opt, train_dataset, validation_dataset):
    """
    并行训练模型
    :param rank: 当前进程中GPU的编号
    :param world_size: 总共有多少个GPU
    :param config: 配置信息
    :param opt: 参数对象
    :param train_dataset: 训练数据集
    :param validation_dataset: 验证数据集
    """
    setup_parallel(rank, world_size)

    # 获得数据集名称
    dataset_name = config['dataset'].get('name')

    print(f'GPU:{rank} ===> 构建网络模型')
    encoding_model, encoding_dim = get_backbone(config)
    model = get_model(encoding_model, encoding_dim, config, append_pca_layer=config['train'].getboolean('wpca'))

    # 保存的图像特征
    init_cache_file = join(join(ROOT_DIR, 'desired', 'centroids'),
                           config['model'].get('backbone') + '_' +
                           dataset_name + '_' +
                           str(config[dataset_name].getint('num_clusters')) + '_desc_cen.hdf5')

    if opt.cluster_file:
        opt.cluster_file = join(join(ROOT_DIR, 'desired', 'centroids'), opt.cluster_file)

        if isfile(opt.cluster_file):
            if (opt.cluster_file != init_cache_file) and (rank == 0):
                copyfile(opt.cluster_file, init_cache_file)
        else:
            raise FileNotFoundError("=> 在'{}'中没有找到聚类数据".format(opt.cluster_file))
    else:
        if rank == 0:
            print(f'GPU:{rank} ===> 寻找聚类中心点并载入聚类数据集')
            train_dataset = MSLS(opt.dataset_root_dir, device=rank, config=config, mode='test', cities_list='train',
                                 img_resize=tuple(map(int, str.split(config['train'].get('resize'), ','))),
                                 batch_size=config['train'].getint('cache_batch_size'))
            print(f'GPU:{rank} ===> 聚类数据集中的数据数量为: {len(train_dataset.db_images_key)}')

            print(f'GPU:{rank} ===> 计算图像特征并创建聚类文件')
            model = model.to(rank)
            create_image_clusters(train_dataset, model, encoding_dim, rank, config, init_cache_file)
            # 把模型转为CPU模式，用于载入参数
            model = model.to(device='cpu')

        # 其他GPU等待GPU0完成聚类的创建
        dist.barrier()

    # 打开保存的聚类文件
    with h5py.File(init_cache_file, mode='r') as h5:
        # 获取图像聚类信息
        image_clusters = h5.get('centroids')[:]
        # 获取图像特征信息
        image_descriptors = h5.get('descriptors')[:]

        # 初始化模型参数
        model.pool.init_params(image_clusters, image_descriptors)

        del image_clusters, image_descriptors
        # 回头GPU内存
        torch.cuda.empty_cache()

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

    # 其他GPU等待GPU0完成Log文件夹的创建
    dist.barrier()

    # 把模型改成并行模型
    model = DDP(model.to(rank), device_ids=[rank])

    # 保存训练结果没有改善的次数
    not_improved = 0
    # 最好结果的分数
    best_score = 0

    # 开始训练，从opt.start_epoch + 1次开始，到opt.epochs_count次结束
    train_epoch_bar = trange(opt.start_epoch + 1, opt.epochs_count + 1)
    for epoch in train_epoch_bar:
        if rank == 0:
            train_epoch_bar.set_description(f'GPU:{rank},第{epoch}/{opt.epochs_count - opt.start_epoch}次训练周期')

        # 执行一个训练周期
        # train_epoch(rank, world_size, train_dataset, model, optimizer, criterion, encoding_dim,
        #             epoch, config, opt, writer)

        # 每训练eval_every次，进行一次验证
        if(epoch % config['train'].getint('eval_every')) == 0:
            if rank == 0:
                recalls = validation(rank, world_size, validation_dataset, model,
                                     encoding_dim, config, writer, epoch)

                # 如果在验证集上结果大于保存的结果，那么就把not_improved清零，并且保存当前最好结果，否not_improved加1
                is_best = recalls[5] > best_score
                if is_best:
                    not_improved = 0
                    best_score = recalls[5]
                else:
                    not_improved += 1

                # 保存模型参数
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'recalls': recalls,
                    'best_score': best_score,
                    'not_improved': not_improved,
                    'optimizer': optimizer.state_dict(),
                }, opt, is_best)

                # 假如patience大于0，且not_improved大于分配到每个验证周期的patience次数，那么就认为模型已经无法再改进了
                if config['train'].getint('patience') > 0 and \
                        not_improved > (config['train'].getint('patience') / config['train'].getint('eval_every')):
                    print('经过{}个训练周期，模型的性能已经不能再改进，停止训练...'.format(config['train'].getint('patience')))
                    break

        # 等待GPU0测试完成
        dist.barrier()

    # 训练完成后在GPU:0上关闭
    if rank == 0:
        # 显示最好的结果
        print('===> 最好的结果 Recalls@5: {:.4f}'.format(best_score), flush=True)
        writer.close()

    parallel_cleanup()