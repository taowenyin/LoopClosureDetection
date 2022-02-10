import argparse
import configparser
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.models_generic import get_backbone, get_model
from os.path import join
from tools import ROOT_DIR
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize, ToTensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='移动机器人回环检测模型的CAM')

    parser.add_argument('--config_path', type=str, default=join(ROOT_DIR, 'configs'), help='模型训练的配置文件的目录')
    parser.add_argument('--save_checkpoint_path', type=str, default=join(ROOT_DIR, 'desired', 'checkpoint'),
                        help='模型checkpoint的保存目录')
    parser.add_argument('--no_cuda', action='store_true', help='如果使用该参数表示只使用CPU，否则使用GPU')
    parser.add_argument('--resume_file', type=str, help='checkpoint文件的保存路径，用于从checkpoint载入训练参数，再次恢复训练')
    parser.add_argument('--cluster_file', type=str, help='聚类数据的保存路径，恢复训练')

    opt = parser.parse_args()

    # 配置文件的地址
    config_file = join(opt.config_path, 'train.ini')
    # 读取配置文件
    config = configparser.ConfigParser()
    config.read(config_file)

    encoding_model, encoding_dim = get_backbone(config)
    model = get_model(encoding_model, encoding_dim, config, append_pca_layer=config['train'].getboolean('wpca'))
    target_layers = [model.encoder[-1]]

    # model = resnet50(pretrained=True)
    # target_layers = [model.layer4]

    rgb_img = cv2.imread('./example-1.jpg', 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    targets = None
    with GradCAM(model=model, target_layers=target_layers, use_cuda=True) as cam:
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=False,
                            eigen_smooth=False)

        grayscale_cam = grayscale_cam[0, :]

        # 把CAM编程图片
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.imshow(cam_image)

    plt.show()
