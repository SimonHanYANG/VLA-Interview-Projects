"""
项目1: CNN 分类训练脚本
=====================================

功能：
- 训练多种 CNN 模型（AlexNet, VGG, GoogLeNet, ResNet, EfficientNet）
- 使用 TensorBoard 记录训练过程
- 保存训练好的模型和性能指标

使用方法：
    python train.py --model alexnet --epochs 20 --batch_size 128
    python train.py --model resnet18 --epochs 20 --batch_size 128
    python train.py --model vgg16 --epochs 20 --batch_size 128
    python train.py --model googlenet --epochs 20 --batch_size 128
    python train.py --model efficientnet_b0 --epochs 20 --batch_size 128

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
import torchvision.models as models
import numpy as np


# ============================================
# 1. 数据预处理
# ============================================

def get_transforms(model_name='resnet18'):
    """
    获取数据预处理转换

    参数:
        model_name: 模型名称，用于决定输入尺寸

    AlexNet 和 VGG 设计用于 224x224 的 ImageNet 图片，
    而 CIFAR-10 只有 32x32，需要 resize 到 224x224。
    其他模型（ResNet、GoogLeNet、EfficientNet）可以处理 32x32 输入。

    面试点：数据增强的作用
    - 随机裁剪：增加位置不变性
    - 水平翻转：增加视角不变性
    - 归一化：加速收敛，稳定训练
    """
    # AlexNet 和 VGG 需要 224x224 输入
    needs_resize = model_name in ['alexnet', 'vgg16']
    input_size = 224 if needs_resize else 32

    if needs_resize:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    return train_transform, test_transform


# ============================================
# 2. 模型创建
# ============================================

def create_model(model_name, num_classes=10):
    """
    创建 CNN 模型

    参数:
        model_name: 模型名称 (alexnet, vgg16, googlenet, resnet18, efficientnet_b0)
        num_classes: 分类数量（CIFAR-10 = 10）

    返回:
        model: PyTorch 模型

    面试点：各个模型的核心思想
    - AlexNet：第一个深层 CNN，ReLU，Dropout
    - VGG：小卷积核堆叠（3x3），更深的网络
    - GoogLeNet：Inception 模块，多尺度特征提取
    - ResNet：残差连接，解决梯度消失问题
    - EfficientNet：复合缩放，效率与精度平衡
    """
    # 预训练模型权重
    weights_map = {
        'alexnet': models.AlexNet_Weights.IMAGENET1K_V1,
        'vgg16': models.VGG16_Weights.IMAGENET1K_V1,
        'googlenet': models.GoogLeNet_Weights.IMAGENET1K_V1,
        'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
        'efficientnet_b0': models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    }

    if model_name not in weights_map:
        raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {list(weights_map.keys())}")

    print(f"正在加载预训练的 {model_name} 模型...")
    weights = weights_map[model_name]

    if model_name == 'alexnet':
        model = models.alexnet(weights=weights)
        # 修改最后一层全连接层，适应 CIFAR-10 的 10 个类别
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=weights)
        # VGG 的最后一层是 classifier[6]
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'googlenet':
        model = models.googlenet(weights=weights)
        # GoogLeNet 的最后一层是 fc
        model.fc = nn.Linear(1024, num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        # ResNet 的最后一层是 fc
        model.fc = nn.Linear(512, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        # EfficientNet 的最后一层是 classifier[1]
        model.classifier[1] = nn.Linear(1280, num_classes)

    # 冻结预训练层（可选）
    # 这里我们选择微调所有层，因为 CIFAR-10 和 ImageNet 差异较大
    # 如果想冻结前面的层，可以取消注释下面的代码
    # for param in model.parameters():
    #     param.requires_grad = False
    # # 只训练最后一层
    # if model_name == 'alexnet':
    #     for param in model.classifier[6].parameters():
    #         param.requires_grad = True

    return model


# ============================================
# 3. 训练函数
# ============================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
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

    返回:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 每 100 个 batch 打印一次
        if batch_idx % 100 == 0:
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
# 4. 验证函数
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
# 5. 训练流程
# ============================================

def train(args):
    """
    完整的训练流程

    步骤：
    1. 初始化设备（CPU/GPU）
    2. 加载数据集并划分训练/验证集
    3. 创建模型
    4. 定义损失函数和优化器
    5. 训练循环
    6. 保存模型和训练指标
    """
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 创建 TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{args.model}_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志目录: {log_dir}")

    # 数据预处理（AlexNet/VGG 需要 resize 到 224x224）
    train_transform, test_transform = get_transforms(args.model)

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
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # 创建模型
    print(f"\n创建模型: {args.model}")
    model = create_model(args.model, num_classes=10)
    model = model.to(device)

    # 打印模型结构
    print("\n模型结构:")
    print(model)
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 定义损失函数
    # CrossEntropyLoss = LogSoftmax + NLLLoss
    # 面试点：交叉熵损失是分类任务的标准损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    # SGD + Momentum 是经典的优化策略
    # 面试点：为什么用 SGD 而不是 Adam？
    # - SGD + Momentum 通常在图像分类任务上泛化性更好
    # - Adam 收敛快但可能过拟合
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    # 学习率调度器
    # 面试点：学习率衰减策略
    # - StepLR：每隔固定 epoch 降低学习率
    # - CosineAnnealingLR：余弦退火，平滑衰减
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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

        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer
        )

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

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
# 6. 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description='CNN 分类训练脚本')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['alexnet', 'vgg16', 'googlenet', 'resnet18', 'efficientnet_b0'],
                        help='模型名称')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("项目1: CNN 分类训练")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"训练轮数: {args.epochs}")
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print("=" * 60)

    # 开始训练
    history, best_acc = train(args)

    return history, best_acc


if __name__ == '__main__':
    main()
