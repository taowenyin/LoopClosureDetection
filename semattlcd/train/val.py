import torch
import numpy as np
import faiss

from torch.utils.data import DataLoader
from semattlcd.dataset.mapillary_sls.generic_dataset import ImagesFromList
from semattlcd.tools.datasets import input_transform
from tqdm import trange, tqdm


def val(eval_set, model, pca_dim, device, opt, config, pbar_position=0):
    if device.type == 'cuda':
        cuda = True
    else:
        cuda = False

    # 缩放图片后的大小
    image_resize_h = config.getint('train', 'image_resize_h')
    image_resize_w = config.getint('train', 'image_resize_w')
    resize = (image_resize_h, image_resize_w)

    eval_set_queries = ImagesFromList(eval_set.qImages, transform=input_transform(resize))
    eval_set_dbs = ImagesFromList(eval_set.dbImages, transform=input_transform(resize))

    test_data_loader_queries = DataLoader(dataset=eval_set_queries,
                                          batch_size=config.getint('train', 'batch_size'),
                                          shuffle=False)
    test_data_loader_dbs = DataLoader(dataset=eval_set_dbs,
                                          batch_size=config.getint('train', 'batch_size'),
                                          shuffle=False)

    model.eval()

    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        pool_size = pca_dim * (image_resize_h // 4) * (image_resize_w // 4)

        qFeat = np.empty((len(eval_set_queries), pool_size), dtype=np.float32)
        dbFeat = np.empty((len(eval_set_dbs), pool_size), dtype=np.float32)

        for feat, test_data_loader in zip([qFeat, dbFeat], [test_data_loader_queries, test_data_loader_dbs]):
            for iteration, (input_data, indices) in enumerate(
                    tqdm(test_data_loader, position=pbar_position, leave=False, desc='Test Iter'.rjust(15)), 1):
                input_data = input_data.to(device)
                image_encoding = model.encoder(input_data)
                pooling_encoding = model.pool(image_encoding)

                feat[indices.detach().numpy(), :] = pooling_encoding.detach().cpu().numpy()

                del input_data, image_encoding, pooling_encoding

    del test_data_loader_queries, test_data_loader_dbs

    tqdm.write('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    # noinspection PyArgumentList
    faiss_index.add(dbFeat)

    tqdm.write('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20, 50, 100]

    # 通过欧氏距离获得前100名的相似对象的索引
    _, predictions = faiss_index.search(qFeat, max(n_values))

    # 获得每个Query的正例索引
    gt = eval_set.all_pos_indices

    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            # 计算不同Top N的预测准确的数量
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break

    # 计算每个N的正确数量比例
    recall_at_n = correct_at_n / len(eval_set.qIdx)

    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]

    return all_recalls
