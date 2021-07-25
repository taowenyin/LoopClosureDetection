import torch

import cfg


def train_one_epoch(model, train_data_loader, optimizer, criterion, epoch, device):
    epoch_loss = 0
    model.train()

    for i, (query, positives, negatives, negCounts, indices) in enumerate(train_data_loader):
        if query is None: continue

        # 获得输入数据Batch、Channel、H、W
        B, C, H, W = query.shape
        # 反例的数量
        nNeg = torch.sum(negCounts)
        # 把数据拼接成一个
        input = (torch.cat([query, positives, negatives])).to(device)

        vlad = model(input)
        # 得到query、positives、negatives特征向量
        vladQ, vladP, vladN = torch.split(vlad, [B, B, nNeg])

        optimizer.zero_grad()

        loss = 0.0
        for i, negCount in enumerate(negCounts):
            for n in range(negCount):
                negIx = (torch.sum(negCounts[:i]) + n).item()
                loss += criterion(vladQ[i:i + 1], vladP[i:i + 1], vladN[negIx:negIx + 1])

        loss /= nNeg
        loss.backward()
        optimizer.step()

        # 计算损失
        batch_loss = loss.item()
        epoch_loss += batch_loss

    return epoch_loss / cfg.BATCH_SIZE