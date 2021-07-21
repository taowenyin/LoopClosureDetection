import torch
import torch.nn.functional as F

from Models.NetVLAD import NetVLAD, EmbedNet
from Utils.hard_triplet_loss import HardTripletLoss


if __name__ == '__main__':
    x = torch.rand(40, 3, 128, 128)
    labels = torch.randint(0, 10, (40,)).long()
    # labels = torch.LongTensor(torch.randint(0, 10, (40,)))

    labels = labels ^ 1

    output = torch.rand(40, 256)

    criterion = HardTripletLoss(margin=0.1)
    triplet_loss = criterion(output, labels)


    print('xxx')