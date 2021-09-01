import torch

from tqdm.auto import trange, tqdm
from torch.utils.data import DataLoader
from segattlcd.dataset.mapillary_sls.msls import MSLS
from segattlcd.tools.common import human_bytes


def train_epoch(train_dataset, model, optimizer, criterion, encoder_dim, device, epoch_num, opt, config):
    if device.type == 'cuda':
        cuda = True
    else:
        cuda = False

    train_dataset.new_epoch()

    epoch_loss = 0
    start_iter = 1  # keep track of batch iter across subsets for logging

    # 计算每个epoch中batch的数量
    n_batches = (len(train_dataset.qIdx) + int(config['train']['batch_size']) - 1) // \
                int(config['train']['batch_size'])

    for sub_iter in trange(train_dataset.nCacheSubset, desc='Cache refresh'.rjust(15), position=1):
        pool_size = encoder_dim

        # 更新训练集的数据，重新计算图像的描述，并组成三元数据
        tqdm.write('====> Building Cache')
        train_dataset.update_sub_cache(model, pool_size)

        training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads,
                                          batch_size=int(config['train']['batch_size']), shuffle=True,
                                          collate_fn=MSLS.collate_fn, pin_memory=cuda)

        tqdm.write('Allocated: ' + human_bytes(torch.cuda.memory_allocated()))
        tqdm.write('Cached:    ' + human_bytes(torch.cuda.memory_cached()))

        model.train()
        for iteration, (query, positives, negatives, negCounts, indices) in \
                enumerate(tqdm(training_data_loader, position=2, leave=False, desc='Train Iter'.rjust(15)), sub_iter):
            print('xxx')

