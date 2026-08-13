# 项目 4.2: CLIP - Contrastive Language-Image Pre-training

## 项目概述

实现 CLIP (Contrastive Language-Image Pre-training) 模型在 CIFAR-10 数据集上的训练和测试。

**论文**: Learning Transferable Visual Models From Natural Language Supervision (OpenAI, 2021)

## CLIP 核心思想

CLIP 通过对比学习连接图像和文本：
1. **双编码器架构**: 图像编码器 + 文本编码器
2. **对比损失**: 让匹配的图像-文本对在嵌入空间中靠近，不匹配的远离
3. **零样本迁移**: 通过文本描述识别新类别，无需微调

```
图像 → [Image Encoder] → 图像嵌向量
                              ↓
                         对比损失 (InfoNCE)
                              ↑
文本 → [Text Encoder]  → 文本嵌向量
```

## 模型变体

| 模型 | 图像编码器 | 文本编码器 | 参数量 | ImageNet 零样本 |
|------|-----------|-----------|--------|----------------|
| CLIP-RN50 | ResNet-50 | Transformer | ~77M | 59.6% |
| CLIP-ViT-B/16 | ViT-B/16 | Transformer | ~149M | 68.3% |
| CLIP-ViT-B/32 | ViT-B/32 | Transformer | ~151M | 63.2% |
| CLIP-ViT-L/14 | ViT-L/14 | Transformer | ~427M | 75.2% |

## 本项目实现

由于网络限制无法下载预训练权重，本项目从头训练 CLIP 模型：

### 架构选择
- **图像编码器**: ViT-B/32 (Vision Transformer)
- **文本编码器**: 6 层 Transformer
- **嵌入维度**: 512

### 数据处理
- CIFAR-10 类别标签作为文本输入
- 图像 resize 到 224x224
- 文本 tokenize 后输入文本编码器

## 快速开始

```bash
conda activate vla

# 训练 CLIP-ViT
python -u train.py --model clip_vit --epochs 30 --batch_size 64

# 测试模型
python test.py --model clip_vit --weights models/clip_vit_best.pth

# 批量训练所有变体
python run_all.py
```

## 目录结构

```
project4-clip/
├── README.md           # 本文件
├── train.py            # 训练脚本
├── test.py             # 测试脚本
├── run_all.py          # 批量训练脚本
├── models/             # 模型权重
├── data/               # 数据集（自动下载）
├── results/            # 测试结果
└── logs/               # TensorBoard 日志
```

## 关键技术

### 1. CLIP 架构
```
┌─────────────────────────────────────────┐
│              CLIP Model                 │
├─────────────────┬───────────────────────┤
│  Image Encoder  │    Text Encoder       │
│    (ViT-B/32)   │   (Transformer)       │
├─────────────────┴───────────────────────┤
│         Projection Heads                │
│      (Linear → L2 Normalize)            │
├─────────────────────────────────────────┤
│      Contrastive Loss (InfoNCE)         │
└─────────────────────────────────────────┘
```

### 2. 对比学习损失
```python
# InfoNCE Loss
logits = image_embeds @ text_embeds.T * temperature
labels = torch.arange(batch_size)
loss = (F.cross_entropy(logits, labels) + 
        F.cross_entropy(logits.T, labels)) / 2
```

### 3. 温度参数
- 可学习的温度参数 τ (temperature)
- 控制相似度分布的锐度
- 初始化为 0.07 (log scale)

## 训练配置

| 配置项 | 值 |
|--------|-----|
| 优化器 | AdamW |
| 学习率 | 0.0005 |
| 权重衰减 | 0.2 |
| 批大小 | 64 |
| 训练轮数 | 30 |
| 学习率调度 | Cosine Annealing |
| 预热轮数 | 5 |

## 参考资源

- [CLIP 论文](https://arxiv.org/abs/2103.00020)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
