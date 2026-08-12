"""
图像分割可视化工具
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch


def plot_segmentation_results(image, pred_mask, true_mask, save_path=None, title="Segmentation Result"):
    """
    绘制分割结果对比图

    Args:
        image: 原始图像 [3, H, W] 或 [H, W, 3]
        pred_mask: 预测 mask [H, W]
        true_mask: 真实 mask [H, W]
        save_path: 保存路径
        title: 图表标题
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 处理图像格式
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

    # 反归一化图像
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)

    # 处理 mask
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()

    # 绘制原始图像
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 绘制预测 mask
    axes[1].imshow(pred_mask, cmap="tab20", vmin=0, vmax=20)
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    # 绘制真实 mask
    axes[2].imshow(true_mask, cmap="tab20", vmin=0, vmax=20)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.close()


def plot_training_curves(train_losses, val_losses, val_mious, save_path=None, title="Training Curves"):
    """
    绘制训练曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        val_mious: 验证 mIoU 列表
        save_path: 保存路径
        title: 图表标题
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    # 绘制损失曲线
    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 绘制 mIoU 曲线
    ax2.plot(epochs, val_mious, "g-", label="Val mIoU")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mIoU")
    ax2.set_title("mIoU Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.close()


def plot_miou_comparison(model_mious, save_path=None):
    """
    绘制不同模型的 mIoU 对比图

    Args:
        model_mious: 字典 {model_name: miou}
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(model_mious.keys())
    mious = list(model_mious.values())

    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, mious, color=colors)

    # 添加数值标签
    for bar, miou in zip(bars, mious):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f"{miou:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Model")
    ax.set_ylabel("mIoU")
    ax.set_title("Model Comparison - mIoU")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.close()


def visualize_predictions(model, dataloader, device, save_dir, num_samples=5):
    """
    可视化模型预测结果

    Args:
        model: 分割模型
        dataloader: 数据加载器
        device: 设备
        save_dir: 保存目录
        num_samples: 可视化样本数
    """
    model.eval()

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break

            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)

            # 获取预测结果
            if isinstance(outputs, dict):
                preds = torch.argmax(outputs["out"], dim=1)
            else:
                preds = torch.argmax(outputs, dim=1)

            # 可视化每个样本
            for j in range(min(images.shape[0], 2)):  # 每个 batch 最多显示 2 个
                save_path = os.path.join(save_dir, f"sample_{i}_{j}.png")
                plot_segmentation_results(
                    images[j],
                    preds[j],
                    masks[j],
                    save_path=save_path,
                    title=f"Sample {i}-{j}"
                )


if __name__ == "__main__":
    # 测试可视化
    # 创建模拟数据
    image = torch.randn(3, 100, 100)
    pred_mask = torch.randint(0, 21, (100, 100))
    true_mask = torch.randint(0, 21, (100, 100))

    # 测试分割结果可视化
    plot_segmentation_results(
        image, pred_mask, true_mask,
        save_path="test_segmentation.png",
        title="Test Segmentation"
    )

    # 测试训练曲线可视化
    train_losses = [0.5, 0.4, 0.35, 0.3, 0.28]
    val_losses = [0.45, 0.38, 0.33, 0.31, 0.29]
    val_mious = [0.6, 0.65, 0.7, 0.72, 0.75]

    plot_training_curves(
        train_losses, val_losses, val_mious,
        save_path="test_training_curves.png",
        title="Test Training Curves"
    )

    # 测试 mIoU 对比图
    model_mious = {
        "FCN": 0.65,
        "DeepLabV3": 0.72,
        "U-Net": 0.70,
    }

    plot_miou_comparison(
        model_mious,
        save_path="test_miou_comparison.png"
    )

    print("Visualization tests completed!")
