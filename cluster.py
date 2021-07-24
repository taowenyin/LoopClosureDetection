import numpy as np
import cfg
import os
import h5py
import torch
import faiss as faiss

from math import ceil
from os import makedirs
from torch.utils.data import DataLoader, SubsetRandomSampler


def calculate_clusters(cluster_dataset, model, conv_output_channels, base_model_name, opt, device):
    nDescriptors = 50000
    nPerImage = 1000
    # nPerImage = 100
    nIm = ceil(nDescriptors / nPerImage)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_dataset), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True, sampler=sampler)

    if cfg.DATASET_TYPE == cfg.Dataset.Tokyo247:
        data_path = './Datasets/Tokyo247'
    elif cfg.DATASET_TYPE == cfg.Dataset.Pitts250k:
        data_path = './Datasets/Pitts250k'
    else:
        data_path = './Datasets/Tokyo247'

    # 创建保存聚类中心点文件夹
    if not os.path.exists(os.path.join(data_path, 'centroids')):
        makedirs(os.path.join(data_path, 'centroids'))

    init_cache = os.path.join(data_path, 'centroids', base_model_name + '_' + cluster_dataset.dataset +
                              '_' + str(opt.num_clusters) + '_desc_cen.hdf5')

    with h5py.File(init_cache, mode='w') as h5:
        with torch.no_grad():
            # 由于已经载入了预训练模型，并且是基于BaseMode，没有导入NetVLAD，所以不需要训练
            model.eval()
            print('====> Extracting Descriptors')
            db_descriptors = h5.create_dataset("descriptors", [nDescriptors, conv_output_channels],
                                               dtype=np.float32)

            for iteration, (image, index, file_name) in enumerate(data_loader, 1):
                input = image.to(device)
                # 提取图像特征 (B C H W)
                image_descriptors = model(input)
                # 特征形状转换 (B HW C)
                image_descriptors = image_descriptors.view(input.size(0), conv_output_channels, -1).permute(0, 2, 1)

                batchix = (iteration - 1) * cfg.BATCH_SIZE * nPerImage
                # 读取每一张图片的特征
                for ix in range(image_descriptors.size(0)):
                    # 采样每一张图片中不同位置的特征
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    # 获取采样后的图像特征 (nPerImage C)
                    each_image_descriptors = image_descriptors[ix, sample, :].detach().cpu().numpy()
                    # 保存图像特征的索引
                    startix = batchix + ix * nPerImage
                    # 保存图像特征
                    db_descriptors[startix:startix+nPerImage, :] = each_image_descriptors

        print('====> Clustering..')
        # 聚类的迭代次数
        niter = 100
        kmeans = faiss.Kmeans(conv_output_channels, opt.num_clusters, niter=niter, verbose=False)
        kmeans.train(db_descriptors[...])

    print('111')