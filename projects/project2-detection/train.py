"""
目标检测训练脚本
支持 Faster R-CNN 系列模型
用法: python train.py --model fasterrcnn_resnet50 --epochs 20 --batch_size 4
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from models import create_model, get_supported_models
from data import get_voc_loaders


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    训练一个 epoch

    检测模型在训练时返回 losses 字典，不需要手动计算 loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        # 数据移到 GPU
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 前向传播（训练模式返回 losses）
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # 检查 loss 是否异常
        if torch.isnan(losses) or torch.isinf(losses):
            print(f"  [警告] Batch {batch_idx} loss 异常: {losses.item()}, 跳过")
            continue

        # 反向传播
        optimizer.zero_grad()
        losses.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

        # 打印进度
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / num_batches
            print(f"  Epoch [{epoch}] Batch [{batch_idx + 1}/{len(data_loader)}] "
                  f"Loss: {losses.item():.4f} Avg Loss: {avg_loss:.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    评估模型

    返回平均 loss（用于模型选择）
    注意：完整的 mAP 计算在 test.py 中实现
    """
    model.train()  # 切换到 train 模式以获取 loss（torchvision 检测模型的特殊要求）

    total_loss = 0.0
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if not (torch.isnan(losses) or torch.isinf(losses)):
            total_loss += losses.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def train(args):
    """主训练函数"""
    # ============================================================
    # 1. 超参数
    # ============================================================
    model_name = args.model
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    num_workers = args.num_workers
    subset_size = args.subset_size
    step_size = args.step_size
    gamma = args.gamma

    print("=" * 60)
    print("  VOC 2007 目标检测训练")
    print("=" * 60)
    print(f"  模型:       {model_name}")
    print(f"  Epochs:     {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  学习率:     {learning_rate}")
    print(f"  子集大小:   {subset_size or '全部'}")
    print("=" * 60)

    # 创建必要目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # ============================================================
    # 2. 设备选择（优先 CUDA）
    # ============================================================
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n[设备] 使用 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device('cpu')
        print(f"\n[设备] 使用 CPU（训练会很慢！）")

    # ============================================================
    # 3. 加载数据集
    # ============================================================
    print(f"\n[数据] 加载 VOC 2007 数据集...")
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    train_loader, val_loader, _ = get_voc_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        subset_size=subset_size,
        download=False,
    )

    # ============================================================
    # 4. 创建模型
    # ============================================================
    print(f"\n[模型] 创建 {model_name}...")
    model = create_model(model_name, pretrained=True)
    model.to(device)

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 总参数: {total_params / 1e6:.2f}M, 可训练: {trainable_params / 1e6:.2f}M")

    # ============================================================
    # 5. 优化器 & 学习率调度
    # ============================================================
    # 只对需要梯度的参数进行优化
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # ============================================================
    # 6. 训练循环
    # ============================================================
    print(f"\n[训练] 开始训练 {model_name}...")
    print(f"[训练] 训练集: {len(train_loader.dataset)} 张, 验证集: {len(val_loader.dataset)} 张")

    best_val_loss = float('inf')
    best_epoch = 0
    history = {
        'model': model_name,
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': None,
        'best_epoch': None,
        'training_time': None,
    }

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # --- 训练 ---
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # --- 验证 ---
        val_loss = evaluate(model, val_loader, device)

        # --- 学习率调度 ---
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        # --- 保存最佳模型 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_path = os.path.join('models', f'{model_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"  ✓ 保存最佳模型: {save_path}")

    # ============================================================
    # 7. 训练完成
    # ============================================================
    total_time = time.time() - start_time
    history['best_val_loss'] = best_val_loss
    history['best_epoch'] = best_epoch
    history['training_time'] = total_time

    # 保存训练历史
    history_path = os.path.join('results', f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # 保存最终模型
    final_path = os.path.join('models', f'{model_name}_final.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, final_path)

    print("\n" + "=" * 60)
    print(f"  训练完成！")
    print(f"  模型:        {model_name}")
    print(f"  总时间:      {total_time / 60:.1f} 分钟")
    print(f"  最佳验证Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"  最终模型:    {final_path}")
    print(f"  最佳模型:    models/{model_name}_best.pth")
    print(f"  训练历史:    {history_path}")
    print("=" * 60)

    return history


def parse_args():
    parser = argparse.ArgumentParser(description='VOC 2007 目标检测训练')

    parser.add_argument('--model', type=str, default='fasterrcnn_resnet50',
                        choices=get_supported_models(),
                        help='模型名称')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.005, help='初始学习率')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--subset_size', type=int, default=None, help='数据集子集大小（None使用全部）')
    parser.add_argument('--step_size', type=int, default=8, help='学习率衰减步长')
    parser.add_argument('--gamma', type=float, default=0.1, help='学习率衰减因子')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
