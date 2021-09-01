#  Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImagesFromList(Dataset):
    def __init__(self, images, transform):
        self.images = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = [Image.open(im) for im in self.images[idx].split(",")]
        img = [self.transform(im) for im in img]

        if len(img) == 1:
            img = img[0]

        return img, idx
