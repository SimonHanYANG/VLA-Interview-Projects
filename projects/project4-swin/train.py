"""
项目4.1: Swin Transformer 图像分类训练脚本
=============================================

功能：
- 训练 Swin Transformer 模型（Swin-Tiny, Swin-Small, Swin-Base）
- 使用 timm 库加载预训练的 Swin 模型
- 在 CIFAR-10 数据集上微调
- 使用 TensorBoard 记录训练过程
- 保存训练好的模型和性能指标

使用方法：
    python train.py --model swin_tiny --epochs 20 --batch_size 64 --lr 0.001
    python train.py --model swin_small --epochs 20 --batch_size 64 --lr 0.001
    python train.py --model swin_base --epochs 20 --batch_size 32 --lr 0.0005

TensorBoard 查看：
    tensorboard --logdir=logs/
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import timm
import numpy as np


# ============================================
# 1. 模型配置
# ============================================

# Swin 模型变体配置
SWIN_CONFIGS = {
    'swin_tiny': {
        'model_name': 'swin_tiny_patch4_window7_224',
        'default_lr': 0.0003,
        'default_batch_size': 64,
        'desc': 'Swin-Tiny (28M params, ImageNet 81.3%)'
    },
    'swin_small': {
        'model_name': 'swin_small_patch4_window7_224',
        'default_lr': 0.0003,
        'default_batch_size': 64,
        'desc': 'Swin-Small (50M params, ImageNet 83.0%)'
    },
    'swin_base': {
        'model_name': 'swin_base_patch4_window7_224',
        'default_lr': 0.0002,
        'default_batch_size': 32,
        'desc': 'Swin-Base (88M params, ImageNet 83.5%)'
    }
}


# ============================================
# 2. 数据预处理
# ============================================

def get_transforms():
    """
    获取数据预处理转换

    Swin Transformer 输入尺寸为 224x224（与 ViT 相同）
    CIFAR-10 原始尺寸为 32x32，需要 resize

    面试点：Vision Transformer 的数据预处理
    - 通常需要 resize 到较大的分辨率（224x224 或更高）
    - 数据增强对 Transformer 的训练很重要
    - RandAugment / MixUp / CutMix 是常用技巧
    """
    # 训练集：数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Swin 需要 224x224 输入
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),  # 自动数据增强
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.25),  # 随机擦除
    ])

    # 测试集：只 resize 和归一化
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return train_transform, test_transform


# ============================================
# 3. 模型创建
# ============================================

def create_model(model_name='swin_tiny', num_classes=10, pretrained=True):
    """
    创建 Swin Transformer 模型

    参数:
        model_name: 模型名称 (swin_tiny, swin_small, swin_base)
        num_classes: 分类数量（CIFAR-10 = 10）
        pretrained: 是否使用 ImageNet 预训练权重

    返回:
        model: PyTorch 模型

    面试点：Swin Transformer 架构
    - Patch Partition: 将图像分成不重叠的 patches (4x4)
    - Linear Embedding: 将 patch 投影到 C 维
    - Swin Transformer Block: Window-based Self-Attention + Shifted Window
    - Patch Merging: 降低分辨率，增加通道数（类似 CNN 的 pooling）
    - 最终通过全局平均池化 + 分类头
    """
    if model_name not in SWIN_CONFIGS:
        raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {list(SWIN_CONFIGS.keys())}")

    config = SWIN_CONFIGS[model_name]
    timm_model_name = config['model_name']

    # 由于网络限制，直接从头训练（不下载预训练权重）
    print(f"创建 {config['desc']} 模型（从头训练）...")
    model = timm.create_model(
        timm_model_name,
        pretrained=False,
        num_classes=num_classes
    )

    return model


# ============================================
# 4. 训练函数
# ============================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer,
                    use_amp=True, scaler=None):
    """
    训练一个 epoch

    参数:
        model: PyTorch 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备 (cpu/cuda)
        epoch: 当前 epoch
        writer: TensorBoard SummaryWriter
        use_amp: 是否使用混合精度训练
        scaler: GradScaler for AMP

    返回:
        avg_loss: 平均损失
        accuracy: 准确率

    面试点：混合精度训练 (AMP)
    - 使用 FP16 进行前向和反向传播，FP32 进行参数更新
    - 优点：减少显存占用 (~50%)，加速训练 (~2x)
    - GradScaler 防止 FP16 下的梯度下溢
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # 混合精度训练
        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 每 50 个 batch 打印一次
        if batch_idx % 50 == 0:
            print(f'  Epoch [{epoch+1}] Batch [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')

    # 计算平均指标
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    # 写入 TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)

    return avg_loss, accuracy


# ============================================
# 5. 验证函数
# ============================================

def validate(model, val_loader, criterion, device, epoch, writer):
    """
    在验证集上评估模型

    参数:
        model: PyTorch 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        epoch: 当前 epoch
        writer: TensorBoard SummaryWriter

    返回:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 混合精度推理
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total

    # 写入 TensorBoard
    writer.add_scalar('Loss/val', avg_loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)

    return avg_loss, accuracy


# ============================================
# 6. 训练流程
# ============================================

def train(args):
    """
    完整的训练流程

    步骤：
    1. 初始化设备（CPU/GPU）
    2. 加载数据集并划分训练/验证集
    3. 创建 Swin Transformer 模型
    4. 定义损失函数和优化器
    5. 训练循环
    6. 保存模型和训练指标
    """
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 检查 GPU 内存
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU 内存: {gpu_mem:.1f} GB")
        # Swin-Base 需要较多显存，自动调整 batch_size
        if args.model == 'swin_base' and gpu_mem < 12:
            args.batch_size = min(args.batch_size, 16)
            print(f"[警告] GPU 内存不足，batch_size 调整为 {args.batch_size}")

    # 确保输出目录存在
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # 创建 TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{args.model}_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志目录: {log_dir}")

    # 数据预处理
    train_transform, test_transform = get_transforms()

    # 加载 CIFAR-10 数据集
    print("\n加载 CIFAR-10 数据集...")
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )

    # 划分训练集和验证集（80% 训练，20% 验证）
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # 创建模型
    print(f"\n创建模型: {SWIN_CONFIGS[args.model]['desc']}")
    model = create_model(args.model, num_classes=10, pretrained=True)
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数总量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 定义损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑，防止过拟合

    # 定义优化器
    # Swin Transformer 通常使用 AdamW 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,  # 较大的权重衰减，防止过拟合
        betas=(0.9, 0.999)
    )

    # 学习率调度器：Cosine Annealing with Warmup
    warmup_epochs = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - warmup_epochs,
        eta_min=1e-6
    )

    # 混合精度训练
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    print(f"混合精度训练: {'开启' if use_amp else '关闭'}")

    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0

    # 训练循环
    print("\n开始训练...")
    print("=" * 60)
    for epoch in range(args.epochs):
        start_time = time.time()

        # Warmup: 前几个 epoch 使用较小的学习率
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * warmup_factor
            current_lr = args.lr * warmup_factor
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            use_amp=use_amp, scaler=scaler
        )

        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer
        )

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # 写入 TensorBoard
        writer.add_scalar('LR', current_lr, epoch)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = f"models/{args.model}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_name': args.model,
            }, model_path)
            print(f"  ✓ 保存最佳模型 (val_acc: {val_acc:.2f}%)")

        # 打印 epoch 结果
        elapsed = time.time() - start_time
        print(f"\nEpoch [{epoch+1}/{args.epochs}] 完成")
        print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
        print(f"  验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
        print(f"  学习率: {current_lr:.6f}")
        print(f"  耗时: {elapsed:.2f}s")
        print("=" * 60)

    # 保存最终模型
    final_model_path = f"models/{args.model}_final.pth"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'model_name': args.model,
    }, final_model_path)

    # 保存训练历史
    history_path = f"results/{args.model}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # 关闭 TensorBoard writer
    writer.close()

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型保存路径: models/{args.model}_best.pth")
    print(f"训练历史: results/{args.model}_history.json")
    print(f"TensorBoard 日志: {log_dir}")
    print("=" * 60)

    return history, best_val_acc


# ============================================
# 7. 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Swin Transformer 图像分类训练')
    parser.add_argument('--model', type=str, default='swin_tiny',
                        choices=['swin_tiny', 'swin_small', 'swin_base'],
                        help='Swin 模型变体 (swin_tiny/swin_small/swin_base)')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批量大小（默认根据模型自动设置）')
    parser.add_argument('--lr', type=float, default=None, help='学习率（默认根据模型自动设置）')

    args = parser.parse_args()

    # 根据模型设置默认参数
    config = SWIN_CONFIGS[args.model]
    if args.batch_size is None:
        args.batch_size = config['default_batch_size']
    if args.lr is None:
        args.lr = config['default_lr']

    print("\n" + "=" * 60)
    print("项目4.1: Swin Transformer 图像分类训练")
    print("=" * 60)
    print(f"模型: {config['desc']}")
    print(f"训练轮数: {args.epochs}")
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print("=" * 60)

    # 开始训练
    history, best_acc = train(args)

    return history, best_acc


if __name__ == '__main__':
    main()
