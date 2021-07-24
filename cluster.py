import numpy as np
import cfg
import os
import h5py
import torch
import faiss

from math import ceil
from os import makedirs
from torch.utils.data import DataLoader, SubsetRandomSampler


def calculate_clusters(cluster_dataset, descriptors_model, conv_output_channels, descriptors_model_name,
                       opt, device, num_descriptors=50000, per_image_descriptors=1000, iter_count=100):
    """
    计算聚类
    :param cluster_dataset: 计算聚类的数据集
    :param descriptors_model: 计算图像特征的模型
    :param conv_output_channels: 图像特征模型输出的通道数
    :param descriptors_model_name: 计算图像特征的模型名称
    :param opt: 参数
    :param device: 在CPU还是GPU上运行
    :param num_descriptors: 一共要保存多少个图像特征
    :param per_image_descriptors: 每张图片要采样的特征数量
    :param iter_count: 聚类迭代的次数
    :return:
    """
    # 一共要保存多少个图像特征
    nDescriptors = num_descriptors
    # 每张图片要采样的特征数量
    nPerImage = per_image_descriptors
    # nPerImage = 100
    # 计算聚类质心的采样图片数量
    nIm = ceil(nDescriptors / nPerImage)

    # 随机子采样的索引列表
    sampler = SubsetRandomSampler(np.random.choice(len(cluster_dataset), nIm, replace=False))
    # 构建采样数据集
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

    # 保存聚类中心点的数据文件
    init_cache = os.path.join(data_path, 'centroids', descriptors_model_name + '_' + cluster_dataset.dataset +
                              '_' + str(opt.num_clusters) + '_desc_cen.hdf5')

    # 创建文件，并设置对象
    with h5py.File(init_cache, mode='w') as h5:
        # 关闭梯度计算
        with torch.no_grad():
            # 由于已经载入了预训练模型，并且是基于BaseMode，没有导入NetVLAD，所以不需要训练
            descriptors_model.eval()
            print('====> Extracting Descriptors')
            # 创建保存图像特征的数据
            db_descriptors = h5.create_dataset("descriptors", [nDescriptors, conv_output_channels],
                                               dtype=np.float32)

            # 迭代每一个采样后的图像
            for iteration, (image, index, file_name) in enumerate(data_loader, 1):
                input = image.to(device)
                # 提取图像特征 (B C H W)
                image_descriptors = descriptors_model(input)
                # 改变特征形状 (B C H W)->(B HW C)
                image_descriptors = image_descriptors.view(input.size(0), conv_output_channels, -1).permute(0, 2, 1)

                # 计算保存图像特征的起始偏移量
                batchix = (iteration - 1) * cfg.BATCH_SIZE * nPerImage
                # 读取每一张图片的特征
                for ix in range(image_descriptors.size(0)):
                    # 采样每一张图片特征中nPerImage个不同位置的特征
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    # 提取采样后的图像特征 (nPerImage C)
                    each_image_descriptors = image_descriptors[ix, sample, :].detach().cpu().numpy()
                    # 计算图像特征采样后的保存位置
                    startix = batchix + ix * nPerImage
                    # 保存图像特征
                    db_descriptors[startix:startix+nPerImage, :] = each_image_descriptors

        print('====> Clustering..')
        # 构建KMEANS，设置聚类的维度和聚类的个数，以及迭代次数
        kmeans = faiss.Kmeans(conv_output_channels, opt.num_clusters, niter=iter_count, verbose=False)
        # 聚类训练
        kmeans.train(db_descriptors.value)

        print('====> Storing centroids', kmeans.centroids.shape)
        # 保存聚类的质心
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')
