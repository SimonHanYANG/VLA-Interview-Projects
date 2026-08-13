# 项目 4.1: Swin Transformer 图像分类

## 项目目标
使用 Swin Transformer 在 CIFAR-10 上进行图像分类，体验 Vision Transformer 架构在视觉任务中的应用。

## 模型介绍

### Swin Transformer (2021, Microsoft Research)
- **论文**: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021 Best Paper)
- **核心创新**: Shifted Window-based Self-Attention
- **关键特点**:
  - 层次化特征图（像 CNN 一样逐步降低分辨率）
  - 窗口内自注意力（降低计算复杂度从 O(n²) 到 O(n)）
  - Shifted Window 跨窗口信息交互

### 面试点：Swin vs ViT
| 特性 | ViT | Swin Transformer |
|------|-----|------------------|
| 注意力范围 | 全局 | 局部窗口 |
| 计算复杂度 | O(n²) | O(n) |
| 多尺度特征 | 单尺度 | 多尺度层次化 |
| 位置编码 | 绝对位置 | 相对位置偏置 |
| 适用任务 | 分类 | 分类+检测+分割 |

## 使用方法

```bash
# 激活环境
conda activate vla

# 训练 Swin-Tiny
python train.py --model swin_tiny --epochs 20 --batch_size 64 --lr 0.001

# 训练 Swin-Small
python train.py --model swin_small --epochs 20 --batch_size 64 --lr 0.001

# 训练 Swin-Base
python train.py --model swin_base --epochs 20 --batch_size 32 --lr 0.0005

# 测试
python test.py --model swin_tiny --model_path models/swin_tiny_best.pth

# 一键运行所有模型
python run_all.py
```

## 模型变体

| 模型 | 参数量 | ImageNet Top-1 | 说明 |
|------|--------|----------------|------|
| Swin-T | 28M | 81.3% | Tiny，适合快速实验 |
| Swin-S | 50M | 83.0% | Small，平衡性能 |
| Swin-B | 88M | 83.5% | Base，更高精度 |

## 目录结构
```
project4-swin/
├── README.md          # 本文件
├── train.py           # 训练脚本
├── test.py            # 测试脚本
├── run_all.py         # 一键训练所有模型
├── models/            # 保存训练好的模型权重
├── results/           # 保存测试结果和指标
├── logs/              # TensorBoard 日志
└── data/              # CIFAR-10 数据集（自动下载）
```
