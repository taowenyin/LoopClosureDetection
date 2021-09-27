import torch

from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from semattlcd.dataset.mapillary_sls.msls import MSLS
from semattlcd.tools.common import human_bytes


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
        # train_dataset.update_sub_cache(model, pool_size)
        train_dataset.update_sub_cache()

        training_data_loader = DataLoader(dataset=train_dataset,
                                          batch_size=int(config['train']['batch_size']), shuffle=True,
                                          collate_fn=MSLS.collate_fn)

        model.train()
        for iteration, (query, positives, negatives, negCounts, indices) in \
                enumerate(tqdm(training_data_loader, position=2, leave=False, desc='Train Iter'.rjust(15)), sub_iter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            if query is None:
                continue  # in case we get an empty batch

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            data_input = torch.cat([query, positives, negatives])

            data_input = data_input.to(device)
            image_encoding = model.encoder(data_input)
            pooling_encoding = model.pool(image_encoding)

            # 拆分结果
            pooling_Q, pooling_P, pooling_N = torch.split(pooling_encoding, [B, B, nNeg])

            optimizer.zero_grad()

            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(pooling_Q[i: i + 1], pooling_P[i: i + 1], pooling_N[negIx:negIx + 1])

            loss /= nNeg.float().to(device)  # normalise by actual number of negatives
            loss.backward()
            optimizer.step()
            del data_input, image_encoding, pooling_encoding, pooling_Q, pooling_P, pooling_N
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or n_batches <= 10:
                tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch_num, iteration, n_batches, batch_loss))

        start_iter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    avg_loss = epoch_loss / n_batches
    tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch_num, avg_loss))
