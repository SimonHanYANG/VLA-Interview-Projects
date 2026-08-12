"""
Project 3: 图像分割统一框架配置文件
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SegmentationConfig:
    """分割模型统一配置"""

    # 数据集配置
    dataset_name: str = "voc2012"
    data_root: str = "./data"
    num_classes: int = 21  # VOC 2012: 20 classes + background
    image_size: Tuple[int, int] = (520, 520)
    subset_ratio: float = 0.1  # 使用 10% 数据快速跑通

    # 训练配置
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4

    # 模型配置
    models: List[str] = None

    # 输出配置
    results_dir: str = "./results"
    save_plots: bool = True

    def __post_init__(self):
        if self.models is None:
            self.models = ["fcn", "deeplabv3", "unet"]

        # 创建结果目录
        for model_name in self.models:
            os.makedirs(os.path.join(self.results_dir, model_name), exist_ok=True)


# 模型配置字典
MODEL_CONFIGS = {
    "fcn": {
        "name": "FCN ResNet50",
        "type": "semantic",
        "backbone": "resnet50",
        "description": "全卷积网络，使用转置卷积上采样"
    },
    "deeplabv3": {
        "name": "DeepLabV3 ResNet50",
        "type": "semantic",
        "backbone": "resnet50",
        "description": "空洞卷积 + ASPP 多尺度特征融合"
    },
    "unet": {
        "name": "U-Net",
        "type": "semantic",
        "backbone": "resnet34",
        "description": "对称 Encoder-Decoder + Skip Connection"
    },
    "mask_rcnn": {
        "name": "Mask R-CNN ResNet50 FPN",
        "type": "instance",
        "backbone": "resnet50_fpn",
        "description": "实例分割，RoI Align + 并行 Mask 分支"
    }
}

# VOC 2012 类别名称
VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

# VOC 2012 颜色映射（用于可视化）
VOC_COLORMAP = [
    [0, 0, 0],        # background
    [128, 0, 0],      # aeroplane
    [0, 128, 0],      # bicycle
    [128, 128, 0],    # bird
    [0, 0, 128],      # boat
    [128, 0, 128],    # bottle
    [0, 128, 128],    # bus
    [128, 128, 128],  # car
    [64, 0, 0],       # cat
    [192, 0, 0],      # chair
    [64, 128, 0],     # cow
    [192, 128, 0],    # diningtable
    [64, 0, 128],     # dog
    [192, 0, 128],    # horse
    [64, 128, 128],   # motorbike
    [192, 128, 128],  # person
    [0, 64, 0],       # pottedplant
    [128, 64, 0],     # sheep
    [0, 192, 0],      # sofa
    [128, 192, 0],    # train
    [0, 64, 128],     # tvmonitor
]
