"""
Project 3: 图像分割统一训练脚本
支持 FCN/DeepLabV3/U-Net 语义分割模型
"""

import os
import gc
import sys
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SegmentationConfig, MODEL_CONFIGS
from dataset import get_dataloaders
from models import get_model
from utils.metrics import compute_miou, compute_pixel_accuracy
from utils.visualization import plot_training_curves


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """
    训练一个 epoch

    Args:
        model: 分割模型
        dataloader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch
        writer: TensorBoard writer

    Returns:
        avg_loss: 平均损失
        miou: mIoU
        pixel_acc: 像素准确率
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        # 前向传播
        outputs = model(images)

        # 处理不同模型的输出格式
        if isinstance(outputs, dict):
            # torchvision 模型返回字典
            loss = criterion(outputs["out"], masks)
            preds = torch.argmax(outputs["out"], dim=1)
        else:
            # smp 模型返回 tensor
            loss = criterion(outputs, masks)
            preds = torch.argmax(outputs, dim=1)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

        # 释放中间变量
        del loss, preds, outputs

        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {total_loss / (batch_idx + 1):.4f}")

    # 计算指标
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    miou, _ = compute_miou(all_preds, all_targets)
    pixel_acc = compute_pixel_accuracy(all_preds, all_targets)

    avg_loss = total_loss / len(dataloader)

    # 记录到 TensorBoard
    if writer:
        writer.add_scalar("Train/Loss", avg_loss, epoch)
        writer.add_scalar("Train/mIoU", miou, epoch)
        writer.add_scalar("Train/PixelAcc", pixel_acc, epoch)

    return avg_loss, miou, pixel_acc


def validate(model, dataloader, criterion, device, epoch, writer=None):
    """
    验证模型

    Args:
        model: 分割模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        epoch: 当前 epoch
        writer: TensorBoard writer

    Returns:
        avg_loss: 平均损失
        miou: mIoU
        pixel_acc: 像素准确率
        iou_per_class: 每个类别的 IoU
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)

            # 处理不同模型的输出格式
            if isinstance(outputs, dict):
                loss = criterion(outputs["out"], masks)
                preds = torch.argmax(outputs["out"], dim=1)
            else:
                loss = criterion(outputs, masks)
                preds = torch.argmax(outputs, dim=1)

            # 统计
            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

            # 释放中间变量
            del loss, preds, outputs

    # 计算指标
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    miou, iou_per_class = compute_miou(all_preds, all_targets)
    pixel_acc = compute_pixel_accuracy(all_preds, all_targets)

    avg_loss = total_loss / len(dataloader)

    # 记录到 TensorBoard
    if writer:
        writer.add_scalar("Val/Loss", avg_loss, epoch)
        writer.add_scalar("Val/mIoU", miou, epoch)
        writer.add_scalar("Val/PixelAcc", pixel_acc, epoch)

    return avg_loss, miou, pixel_acc, iou_per_class


def train_model(model_name, config):
    """
    训练指定的分割模型

    Args:
        model_name: 模型名称 ("fcn", "deeplabv3", "unet")
        config: 配置对象
    """
    print(f"\n{'='*60}")
    print(f"Training {MODEL_CONFIGS[model_name]['name']}")
    print(f"{'='*60}")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建结果目录
    result_dir = os.path.join(config.results_dir, model_name)
    os.makedirs(result_dir, exist_ok=True)

    # 创建 TensorBoard writer
    log_dir = os.path.join("runs", f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir)

    # 加载数据
    print("\nLoading data...")
    train_loader, val_loader = get_dataloaders(config)

    # 创建模型
    print(f"\nCreating model: {model_name}")
    model = get_model(model_name, num_classes=config.num_classes, pretrained=True)
    model = model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                          weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # 训练循环
    print(f"\nStarting training for {config.num_epochs} epochs...")
    train_losses = []
    val_losses = []
    val_mious = []
    best_miou = 0.0

    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()

        # 每个 epoch 开始前清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 训练
        train_loss, train_miou, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        # 验证
        val_loss, val_miou, val_acc, iou_per_class = validate(
            model, val_loader, criterion, device, epoch, writer
        )

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 清理内存，防止 CPU 内存泄漏
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_mious.append(val_miou)

        # 计算时间
        epoch_time = time.time() - epoch_start_time

        # 打印结果
        print(f"\nEpoch [{epoch + 1}/{config.num_epochs}] ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f}, mIoU: {train_miou:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}, Acc: {val_acc:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": best_miou,
            }, os.path.join(result_dir, "best_model.pth"))
            print(f"  New best model saved! mIoU: {best_miou:.4f}")

    # 保存最终模型
    torch.save({
        "epoch": config.num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_miou": val_mious[-1],
    }, os.path.join(result_dir, "final_model.pth"))

    # 绘制训练曲线
    plot_training_curves(
        train_losses, val_losses, val_mious,
        save_path=os.path.join(result_dir, "training_curves.png"),
        title=f"{MODEL_CONFIGS[model_name]['name']} Training Curves"
    )

    # 保存训练历史
    import json
    history = {
        "model_name": model_name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_mious": val_mious,
        "best_miou": best_miou,
        "final_miou": val_mious[-1],
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

    with open(os.path.join(result_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # 关闭 TensorBoard writer
    writer.close()

    print(f"\nTraining completed!")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Final mIoU: {val_mious[-1]:.4f}")
    print(f"Results saved to: {result_dir}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train segmentation models")
    parser.add_argument("--model", type=str, default="fcn",
                       choices=["fcn", "deeplabv3", "unet"],
                       help="Model to train")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--subset_ratio", type=float, default=0.1,
                       help="Subset ratio for quick training")
    parser.add_argument("--data_root", type=str, default="./data",
                       help="Data root directory")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of data loading workers (0 for Windows)")
    parser.add_argument("--image_size", type=int, default=320,
                       help="Image size for training")

    args = parser.parse_args()

    # 创建配置
    config = SegmentationConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        subset_ratio=args.subset_ratio,
        data_root=args.data_root,
        num_workers=args.num_workers,
        image_size=(args.image_size, args.image_size),
        models=[args.model],
    )

    # 训练模型
    train_model(args.model, config)


if __name__ == "__main__":
    main()
