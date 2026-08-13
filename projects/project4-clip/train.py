"""
CLIP (Contrastive Language-Image Pre-training) 训练脚本
在 CIFAR-10 上训练 CLIP 模型，学习图像-类别对的联合表示

核心思想：通过对比学习让匹配的图像-类别嵌入在嵌入空间中靠近
使用可学习的类别嵌入代替文本编码器（更高效）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
import json
import argparse
from datetime import datetime

# CLIP 配置
CLIP_CONFIGS = {
    'clip_vit': {
        'image_encoder': 'vit_base_patch32_224',
        'embed_dim': 512,
        'num_classes': 10,
        'default_lr': 0.0005,
        'default_batch_size': 64,
        'desc': 'CLIP-ViT-B/32 (对比学习)'
    },
}


class ImageEncoder(nn.Module):
    """CLIP 图像编码器 - ViT 架构"""

    def __init__(self, model_name='vit_base_patch32_224', embed_dim=512):
        super().__init__()

        # 使用 timm 加载 ViT 模型
        self.vit = timm.create_model(model_name, pretrained=False, num_classes=0)

        # 获取 ViT 的特征维度
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            vit_features = self.vit(dummy).shape[-1]

        # Projection head
        self.projection = nn.Linear(vit_features, embed_dim)

    def forward(self, images):
        """
        Args:
            images: [batch_size, 3, 224, 224]
        Returns:
            image_features: [batch_size, embed_dim] - 归一化的图像嵌入
        """
        # Extract features from ViT
        x = self.vit(images)  # [B, vit_features]

        # Project to shared embedding space
        x = self.projection(x)

        # L2 normalize
        x = F.normalize(x, dim=-1)

        return x


class CLIPModel(nn.Module):
    """CLIP 模型 - 对比学习（简化版：使用可学习类别嵌入）"""

    def __init__(self, config_name='clip_vit'):
        super().__init__()

        self.config = CLIP_CONFIGS[config_name]
        embed_dim = self.config['embed_dim']
        num_classes = self.config['num_classes']

        # Image encoder
        self.image_encoder = ImageEncoder(
            model_name=self.config['image_encoder'],
            embed_dim=embed_dim
        )

        # 可学习的类别嵌入（代替文本编码器）
        self.class_embeddings = nn.Parameter(
            torch.randn(num_classes, embed_dim) * 0.02
        )

        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6593)  # log(1/0.07)

    def forward(self, images, labels=None):
        """
        Args:
            images: [batch_size, 3, 224, 224]
            labels: [batch_size] - 类别标签（训练时使用）
        Returns:
            logits: [batch_size, num_classes] - 相似度 logits
        """
        # Encode images
        image_features = self.image_encoder(images)  # [B, embed_dim]

        # Get class embeddings
        class_embeds = F.normalize(self.class_embeddings, dim=-1)  # [C, embed_dim]

        # Compute similarity
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ class_embeds.T  # [B, C]

        return logits

    def get_image_features(self, images):
        """获取图像特征（用于推理）"""
        return self.image_encoder(images)

    def get_class_features(self):
        """获取类别特征（用于推理）"""
        return F.normalize(self.class_embeddings, dim=-1)


def create_clip_model(config_name='clip_vit'):
    """创建 CLIP 模型"""
    config = CLIP_CONFIGS[config_name]
    print(f"\n创建 {config['desc']} 模型...")

    model = CLIPModel(config_name)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    image_params = sum(p.numel() for p in model.image_encoder.parameters())
    class_params = model.class_embeddings.numel()

    print(f"模型参数量: {total_params/1e6:.1f}M")
    print(f"  - 图像编码器: {image_params/1e6:.1f}M")
    print(f"  - 类别嵌入: {class_params/1e3:.1f}K")
    print(f"  - 嵌入维度: {config['embed_dim']}")
    print(f"  - 类别数量: {config['num_classes']}")

    return model


def train_clip(config_name='clip_vit', epochs=30, batch_size=64, lr=0.0005):
    """训练 CLIP 模型"""

    config = CLIP_CONFIGS[config_name]
    print(f"\n{'='*60}")
    print(f"训练 {config['desc']}")
    print(f"{'='*60}")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 创建模型
    model = create_clip_model(config_name)
    model = model.to(device)

    # 数据预处理 - CLIP 标准预处理
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

    # 加载数据集
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n加载 CIFAR-10 数据集...")
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=val_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=val_transform)

    # 划分训练集和验证集
    train_size = 45000
    val_size = 5000
    indices = torch.randperm(len(train_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"训练集: {len(train_subset)} 样本")
    print(f"验证集: {len(val_subset)} 样本")

    # 优化器 - CLIP 使用 AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.2,
        betas=(0.9, 0.98),
        eps=1e-6
    )

    # 学习率调度 - Cosine with Warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / (epochs - warmup_epochs) * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda')

    # 交叉熵损失
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 训练循环
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, f'{config_name}_best.pth')

    print(f"\n开始训练...")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_batches += 1

            # 计算准确率
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 200 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / train_batches
        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                val_batches += 1

                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / val_batches
        val_acc = 100. * val_correct / val_total

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'config': config,
            }, best_model_path)
            print(f"  ✓ 保存最佳模型 (Val Acc: {val_acc:.2f}%)")

    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"最佳模型: Epoch {best_epoch}, Val Acc: {best_val_acc:.2f}%, Val Loss: {best_val_loss:.4f}")

    # 保存训练历史
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    history_path = os.path.join(results_dir, f'{config_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)

    # 保存最终模型
    final_model_path = os.path.join(model_dir, f'{config_name}_final.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'val_acc': val_acc,
        'config': config,
    }, final_model_path)

    return best_val_acc, best_epoch, history


def main():
    parser = argparse.ArgumentParser(description='CLIP Training')
    parser.add_argument('--model', type=str, default='clip_vit', help='Model config name')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')

    args = parser.parse_args()

    if args.model not in CLIP_CONFIGS:
        print(f"错误: 未知模型 {args.model}")
        print(f"可用模型: {list(CLIP_CONFIGS.keys())}")
        return

    val_acc, best_epoch, history = train_clip(
        config_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    print(f"\n最终结果:")
    print(f"  Best Val Acc: {val_acc:.2f}%")
    print(f"  Best Epoch: {best_epoch}")


if __name__ == '__main__':
    main()
