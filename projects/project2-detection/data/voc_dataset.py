"""
VOC 2007 数据集加载
使用 torchvision 内置的 VOCDetection
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import VOCDetection
import torchvision.transforms.v2 as T


# VOC 2007 类别（按字母序）
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]

# 类别名到索引（1-20，0 为背景）
CLASS_TO_IDX = {cls: idx + 1 for idx, cls in enumerate(VOC_CLASSES)}
CLASS_TO_IDX['background'] = 0


def parse_voc_annotation(annotation):
    """
    解析 VOC XML 标注为 torchvision Faster R-CNN 需要的格式

    Args:
        annotation: VOCDetection 返回的标注字典

    Returns:
        dict with 'boxes' (Tensor [N, 4]) and 'labels' (Tensor [N])
    """
    objects = annotation['annotation']['object']

    # 处理单个目标的情况（dict vs list）
    if isinstance(objects, dict):
        objects = [objects]

    boxes = []
    labels = []

    for obj in objects:
        # 提取边界框
        bbox = obj['bndbox']
        xmin = float(bbox['xmin'])
        ymin = float(bbox['ymin'])
        xmax = float(bbox['xmax'])
        ymax = float(bbox['ymax'])

        # 跳过无效框
        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])

        # 类别索引（1-20）
        label = CLASS_TO_IDX.get(obj['name'], 0)
        labels.append(label)

    target = {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64),
    }

    return target


class VOCDetectionWrapper(VOCDetection):
    """
    VOCDetection 的包装类，返回 torchvision 检测模型所需的格式
    """

    def __init__(self, root, year='2007', image_set='train', download=False, transforms=None):
        super().__init__(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transforms=transforms,
        )

    def __getitem__(self, idx):
        img, annotation = super().__getitem__(idx)

        # 解析标注
        target = parse_voc_annotation(annotation)

        # 应用变换（需要同时变换图像和边界框）
        if self.transforms is not None:
            # torchvision 检测变换接受 (image, target) 元组
            img, target = self.transforms(img, target)

        return img, target


def get_detection_transforms(train=True):
    """
    获取检测任务的数据变换

    注意：检测任务的变换需要同时处理图像和边界框
    """
    transforms = []

    # 基础变换：ToTensor（同时处理 target 中的 boxes）
    transforms.append(T.ToTensor())

    if train:
        # 训练时的随机水平翻转
        transforms.append(T.RandomHorizontalFlip(0.5))

    # 组合变换
    return T.Compose(transforms)


def collate_fn(batch):
    """
    自定义 collate 函数，处理不同数量的目标

    目标检测中每张图的目标数量不同，不能直接 stack
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    return images, targets


def get_voc_loaders(
    data_root='./data',
    year='2007',
    batch_size=4,
    num_workers=2,
    subset_size=None,
    download=False,
):
    """
    获取 VOC 2007 训练和验证数据加载器

    Args:
        data_root: 数据存放目录
        year: VOC 数据集年份
        batch_size: 批次大小
        num_workers: 数据加载线程数
        subset_size: 子集大小（None 使用全部数据）
        download: 是否自动下载

    Returns:
        (train_loader, val_loader, test_loader)
    """
    transform_train = get_detection_transforms(train=True)
    transform_val = get_detection_transforms(train=False)

    # 创建数据集
    print(f"[数据] 加载 VOC {year} 数据集...")

    train_dataset = VOCDetectionWrapper(
        root=data_root, year=year, image_set='train',
        download=download, transforms=transform_train,
    )

    val_dataset = VOCDetectionWrapper(
        root=data_root, year=year, image_set='val',
        download=download, transforms=transform_val,
    )

    # VOC 2007 test 集（如果 test.txt 不存在，用 val 集代替）
    try:
        test_dataset = VOCDetectionWrapper(
            root=data_root, year=year, image_set='test',
            download=download, transforms=transform_val,
        )
    except (FileNotFoundError, RuntimeError):
        print(f"[数据] test 集不可用，使用 val 集作为测试集")
        test_dataset = val_dataset

    # 子集模式（用于快速测试）
    if subset_size is not None:
        train_dataset = Subset(train_dataset, range(min(subset_size, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(subset_size // 4, len(val_dataset))))
        test_dataset = Subset(test_dataset, range(min(subset_size // 4, len(test_dataset))))

    print(f"[数据] 训练集: {len(train_dataset)} 张")
    print(f"[数据] 验证集: {len(val_dataset)} 张")
    print(f"[数据] 测试集: {len(test_dataset)} 张")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
