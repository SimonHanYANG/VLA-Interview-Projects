"""
CLIP (Contrastive Language-Image Pre-training) 训练脚本
在 CIFAR-10 上训练 CLIP 模型，学习图像-文本对的联合表示

核心思想：通过对比学习让匹配的图像-文本对在嵌入空间中靠近
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from transformers import AutoTokenizer
import os
import json
import argparse
from datetime import datetime

# CLIP 配置
CLIP_CONFIGS = {
    'clip_vit': {
        'image_encoder': 'vit_base_patch32_224',
        'embed_dim': 512,
        'text_embed_dim': 768,
        'text_heads': 8,
        'text_layers': 6,
        'default_lr': 0.0005,
        'default_batch_size': 64,
        'desc': 'CLIP-ViT-B/32 (双编码器对比学习)'
    },
}


class TextEncoder(nn.Module):
    """CLIP 文本编码器 - Transformer 架构"""

    def __init__(self, vocab_size=49408, embed_dim=512, text_embed_dim=768,
                 num_heads=8, num_layers=6, max_length=77):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_length = max_length

        # Token embedding + Positional embedding
        self.token_embedding = nn.Embedding(vocab_size, text_embed_dim)
        self.positional_embedding = nn.Parameter(
            torch.randn(max_length, text_embed_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_embed_dim,
            nhead=num_heads,
            dim_feedforward=text_embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm + Projection
        self.ln_final = nn.LayerNorm(text_embed_dim)
        self.projection = nn.Linear(text_embed_dim, embed_dim)

    def forward(self, text_tokens):
        """
        Args:
            text_tokens: [batch_size, seq_len] - token ids
        Returns:
            text_features: [batch_size, embed_dim] - 归一化的文本嵌入
        """
        # Token + Position embedding
        x = self.token_embedding(text_tokens)  # [B, L, D]
        x = x + self.positional_embedding[:x.size(1), :]

        # Transformer encoding
        x = self.transformer(x)

        # Take the [EOS] token embedding (last token)
        x = self.ln_final(x[:, -1, :])

        # Project to shared embedding space
        x = self.projection(x)

        # L2 normalize
        x = F.normalize(x, dim=-1)

        return x


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
    """CLIP 模型 - 双编码器对比学习"""

    def __init__(self, config_name='clip_vit'):
        super().__init__()

        self.config = CLIP_CONFIGS[config_name]

        # Image encoder
        self.image_encoder = ImageEncoder(
            model_name=self.config['image_encoder'],
            embed_dim=self.config['embed_dim']
        )

        # Text encoder
        self.text_encoder = TextEncoder(
            embed_dim=self.config['embed_dim'],
            text_embed_dim=self.config['text_embed_dim'],
            num_heads=self.config['text_heads'],
            num_layers=self.config['text_layers']
        )

        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6593)  # log(1/0.07)

    def forward(self, images, text_tokens):
        """
        Args:
            images: [batch_size, 3, 224, 224]
            text_tokens: [batch_size, seq_len]
        Returns:
            logits_per_image: [batch_size, batch_size]
            logits_per_text: [batch_size, batch_size]
        """
        # Encode images and texts
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(text_tokens)

        # Compute similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text


def get_cifar10_text_descriptions():
    """获取 CIFAR-10 类别的文本描述"""
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # 为每个类别创建多种文本描述
    descriptions = {}
    for i, name in enumerate(class_names):
        descriptions[i] = [
            f"a photo of a {name}",
            f"a {name} in the image",
            f"this is a {name}",
            f"an image of a {name}",
            f"a picture showing a {name}",
        ]

    return class_names, descriptions


class CLIPTokenizer:
    """简化的 CLIP Tokenizer"""

    def __init__(self, vocab_size=49408, max_length=77):
        self.vocab_size = vocab_size
        self.max_length = max_length

        # 简单的字符级 tokenizer（实际 CLIP 使用 BPE）
        self.char_to_id = {chr(i): i for i in range(256)}
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, texts, padding=True, truncation=True, return_tensors='pt'):
        """
        Args:
            texts: list of strings or single string
        Returns:
            input_ids: [batch_size, seq_len]
        """
        if isinstance(texts, str):
            texts = [texts]

        batch_tokens = []
        for text in texts:
            # Convert to character-level token ids
            tokens = [self.char_to_id.get(c, 2) for c in text.lower()]

            # Truncate
            if truncation:
                tokens = tokens[:self.max_length - 2]  # Reserve space for EOS

            # Add EOS token
            tokens.append(self.eos_token_id)

            # Pad
            if padding:
                tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))

            batch_tokens.append(tokens)

        if return_tensors == 'pt':
            return {'input_ids': torch.tensor(batch_tokens, dtype=torch.long)}

        return {'input_ids': batch_tokens}


def create_clip_model(config_name='clip_vit'):
    """创建 CLIP 模型"""
    config = CLIP_CONFIGS[config_name]
    print(f"\n创建 {config['desc']} 模型...")

    model = CLIPModel(config_name)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    image_params = sum(p.numel() for p in model.image_encoder.parameters())
    text_params = sum(p.numel() for p in model.text_encoder.parameters())

    print(f"模型参数量: {total_params/1e6:.1f}M")
    print(f"  - 图像编码器: {image_params/1e6:.1f}M")
    print(f"  - 文本编码器: {text_params/1e6:.1f}M")
    print(f"  - 嵌入维度: {config['embed_dim']}")

    return model


def contrastive_loss(logits_per_image, logits_per_text):
    """对比学习损失 (InfoNCE)"""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)

    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    return (loss_i2t + loss_t2i) / 2


def create_text_prompts(labels, class_names, descriptions):
    """
    为每个样本创建文本提示
    训练时随机选择一个描述，增加多样性
    """
    prompts = []
    for label in labels:
        class_idx = label.item()
        # 随机选择一个描述
        desc_idx = torch.randint(0, len(descriptions[class_idx]), (1,)).item()
        prompts.append(descriptions[class_idx][desc_idx])
    return prompts


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

    # Tokenizer
    tokenizer = CLIPTokenizer()

    # CIFAR-10 类别信息
    class_names, descriptions = get_cifar10_text_descriptions()

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
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=val_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)

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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    print(f"训练集: {len(train_subset)} 样本")
    print(f"验证集: {len(val_subset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")

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

    # 训练循环
    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

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
        train_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)

            # 创建文本提示
            text_prompts = create_text_prompts(labels, class_names, descriptions)
            text_tokens = tokenizer(text_prompts)['input_ids'].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                logits_per_image, logits_per_text = model(images, text_tokens)
                loss = contrastive_loss(logits_per_image, logits_per_text)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / train_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)

                text_prompts = create_text_prompts(labels, class_names, descriptions)
                text_tokens = tokenizer(text_prompts)['input_ids'].to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    logits_per_image, logits_per_text = model(images, text_tokens)
                    loss = contrastive_loss(logits_per_image, logits_per_text)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config,
            }, best_model_path)
            print(f"  ✓ 保存最佳模型 (Val Loss: {avg_val_loss:.4f})")

    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"最佳模型: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")

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
        'config': config,
    }, final_model_path)

    return best_val_loss, best_epoch, history


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

    val_loss, best_epoch, history = train_clip(
        config_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    print(f"\n最终结果:")
    print(f"  Best Val Loss: {val_loss:.4f}")
    print(f"  Best Epoch: {best_epoch}")


if __name__ == '__main__':
    main()
