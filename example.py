import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path

from os.path import join
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from typing import List
from datetime import datetime


if __name__ == '__main__':
    save_loss_dir = join('results', 'loss_pics', datetime.now().strftime('%b%d_%H-%M-%S'))
    if not os.path.exists(save_loss_dir):
        os.makedirs(save_loss_dir)
