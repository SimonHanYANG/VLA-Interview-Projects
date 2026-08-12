"""
图像分割评估指标
- mIoU (mean Intersection over Union)
- Pixel Accuracy
"""

import torch
import numpy as np


def compute_miou(preds, targets, num_classes=21):
    """
    计算 mIoU (mean Intersection over Union)

    Args:
        preds: 预测结果 [B, H, W] 或 [B, C, H, W]（logits）
        targets: 真实标签 [B, H, W]
        num_classes: 类别数

    Returns:
        miou: 所有类别的平均 IoU
        iou_per_class: 每个类别的 IoU

    IoU 计算公式：
    IoU = TP / (TP + FP + FN)

    mIoU = 所有类别 IoU 的平均值
    """
    # 如果 preds 是 logits，转换为类别索引
    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1)

    # 转换为 numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    iou_per_class = []

    for cls_idx in range(num_classes):
        pred_mask = (preds == cls_idx)
        target_mask = (targets == cls_idx)

        # 计算交集和并集
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        # 计算 IoU
        if union == 0:
            # 如果该类别不存在，IoU 为 1（完美）或 0（忽略）
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        iou_per_class.append(iou)

    # 计算 mIoU
    miou = np.mean(iou_per_class)

    return miou, iou_per_class


def compute_pixel_accuracy(preds, targets):
    """
    计算像素准确率

    Args:
        preds: 预测结果 [B, H, W] 或 [B, C, H, W]（logits）
        targets: 真实标签 [B, H, W]

    Returns:
        accuracy: 像素准确率

    像素准确率 = 正确预测的像素数 / 总像素数
    """
    # 如果 preds 是 logits，转换为类别索引
    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1)

    # 转换为 numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # 计算准确率
    correct = (preds == targets).sum()
    total = targets.size

    accuracy = correct / total

    return accuracy


def compute_mean_pixel_accuracy(preds, targets, num_classes=21):
    """
    计算平均像素准确率（每个类别的准确率的平均值）

    Args:
        preds: 预测结果 [B, H, W] 或 [B, C, H, W]（logits）
        targets: 真实标签 [B, H, W]
        num_classes: 类别数

    Returns:
        mean_accuracy: 平均像素准确率
        accuracy_per_class: 每个类别的准确率
    """
    # 如果 preds 是 logits，转换为类别索引
    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1)

    # 转换为 numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    accuracy_per_class = []

    for cls_idx in range(num_classes):
        pred_mask = (preds == cls_idx)
        target_mask = (targets == cls_idx)

        # 计算该类别的准确率
        if target_mask.sum() == 0:
            # 如果该类别不存在，准确率为 1（完美）或 0（忽略）
            accuracy = 1.0 if pred_mask.sum() == 0 else 0.0
        else:
            accuracy = np.logical_and(pred_mask, target_mask).sum() / target_mask.sum()

        accuracy_per_class.append(accuracy)

    # 计算平均准确率
    mean_accuracy = np.mean(accuracy_per_class)

    return mean_accuracy, accuracy_per_class


if __name__ == "__main__":
    # 测试指标计算
    num_classes = 21
    batch_size = 2
    height, width = 100, 100

    # 创建模拟数据
    preds = torch.randint(0, num_classes, (batch_size, height, width))
    targets = torch.randint(0, num_classes, (batch_size, height, width))

    # 计算 mIoU
    miou, iou_per_class = compute_miou(preds, targets, num_classes)
    print(f"mIoU: {miou:.4f}")
    print(f"IoU per class: {iou_per_class[:5]}...")

    # 计算像素准确率
    accuracy = compute_pixel_accuracy(preds, targets)
    print(f"Pixel Accuracy: {accuracy:.4f}")

    # 计算平均像素准确率
    mean_acc, acc_per_class = compute_mean_pixel_accuracy(preds, targets, num_classes)
    print(f"Mean Pixel Accuracy: {mean_acc:.4f}")
    print(f"Accuracy per class: {acc_per_class[:5]}...")
