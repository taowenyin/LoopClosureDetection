import os
import torch.distributed as dist
import torch


def setup_parallel(rank, world_size):
    """
    分布式训练初始化

    :param rank: 当前进程中GPU的编号
    :param world_size: 总共有多少个GPU
    """
    # 确定可用的GPU，注意这句话一定要放在任何对CUDA的操作之前（和别人公用服务器时使用）
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # 设置两个环境变量，localhost是本地的ip地址，12355是用来通信的端口
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    # 实现GPU的负载均衡
    torch.cuda.set_device(rank)


def parallel_cleanup():
    """
    在所有任务行完以后消灭进程用的。
    """
    dist.destroy_process_group()


def reduce_mean(value, n_procs):
    """
    收集GPU上的数据，然后求平均
    :param value:
    :param n_procs:
    :return:
    """
    res = value.clone()
    dist.all_reduce(res, op=dist.ReduceOp.SUM)
    res /= n_procs
    return res