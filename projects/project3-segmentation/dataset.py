"""
Project 3: VOC 2012 数据集加载
支持语义分割和实例分割
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# VOC 2012 类别数
NUM_CLASSES = 21


class VOCSegmentationDataset(Dataset):
    """
    VOC 2012 语义分割数据集
    返回：(image, mask)
    - image: [3, H, W] tensor
    - mask: [H, W] long tensor，值为类别索引 (0-20)
    """

    def __init__(self, root, split="train", image_size=(520, 520), augment=False):
        """
        Args:
            root: VOC2012 数据根目录
            split: "train", "val", 或 "trainval"
            image_size: 图像尺寸 (H, W)
            augment: 是否数据增强
        """
        self.root = root
        self.split = split
        self.image_size = image_size
        self.augment = augment

        # 图像和标注路径
        self.image_dir = os.path.join(root, "JPEGImages")
        self.mask_dir = os.path.join(root, "SegmentationClass")

        # 读取分割列表
        split_file = os.path.join(root, "ImageSets", "Segmentation", f"{split}.txt")
        with open(split_file, "r") as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(self.image_ids)} {split} images")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # 加载图像
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        # 加载分割标注
        mask_path = os.path.join(self.mask_dir, f"{image_id}.png")
        mask = Image.open(mask_path)

        # 数据增强
        if self.augment:
            image, mask = self._augment(image, mask)

        # 调整大小
        image = TF.resize(image, self.image_size, interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=T.InterpolationMode.NEAREST)

        # 转换为 tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # mask 转为 long tensor
        mask = torch.from_numpy(np.array(mask)).long()

        # VOC 标注中 255 表示忽略的边界，映射到 0（背景）
        mask[mask == 255] = 0

        return image, mask

    def _augment(self, image, mask):
        """数据增强：随机水平翻转"""
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


def get_dataloaders(config):
    """
    创建训练和验证数据加载器

    Args:
        config: SegmentationConfig 配置对象

    Returns:
        train_loader, val_loader
    """
    # 创建完整数据集
    train_dataset = VOCSegmentationDataset(
        root=os.path.join(config.data_root, "VOC2012"),
        split="train",
        image_size=config.image_size,
        augment=True
    )

    val_dataset = VOCSegmentationDataset(
        root=os.path.join(config.data_root, "VOC2012"),
        split="val",
        image_size=config.image_size,
        augment=False
    )

    # 子集采样（快速跑通）
    if config.subset_ratio < 1.0:
        train_size = int(len(train_dataset) * config.subset_ratio)
        val_size = int(len(val_dataset) * config.subset_ratio)

        train_indices = random.sample(range(len(train_dataset)), train_size)
        val_indices = random.sample(range(len(val_dataset)), val_size)

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

        print(f"Using subset: {train_size} train, {val_size} val images")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def decode_segmap(mask, num_classes=21):
    """
    将分割 mask 转换为彩色图像用于可视化

    Args:
        mask: [H, W] tensor，类别索引
        num_classes: 类别数

    Returns:
        rgb_image: [H, W, 3] numpy array
    """
    # VOC 颜色映射
    voc_colors = [
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

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_idx in range(num_classes):
        rgb[mask == cls_idx] = voc_colors[cls_idx]

    return rgb
