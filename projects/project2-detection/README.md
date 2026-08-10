# Project 2: 目标检测 (Object Detection)

## 概述

在 VOC 2007 数据集上训练和对比主流目标检测模型。

## 支持的模型

| 模型 | 来源 | 特点 |
|------|------|------|
| Faster R-CNN (ResNet50) | torchvision | 两阶段检测器，精度高 |
| Faster R-CNN (ResNet50 V2) | torchvision | 改进版，更好的特征提取 |
| Faster R-CNN (MobileNet V3) | torchvision | 轻量级，速度快 |

## 数据集

**PASCAL VOC 2007**
- 20 个目标类别 + 背景
- 训练集: ~5011 张图像
- 验证集: ~5136 张图像
- 测试集: ~4952 张图像
- 自动下载到 `data/` 目录

## 使用方法

### 环境准备

```bash
conda activate vla
```

### 单模型训练

```bash
# 训练 Faster R-CNN
python train.py --model fasterrcnn_resnet50 --epochs 20 --batch_size 4

# 使用数据子集快速测试
python train.py --model fasterrcnn_resnet50 --epochs 3 --batch_size 4 --subset_size 100
```

### 单模型测试

```bash
python test.py --model fasterrcnn_resnet50
```

### 批量训练 & 测试

```bash
# 训练所有模型
python run_all.py --epochs 20 --batch_size 4

# 使用子集快速验证
python run_all.py --epochs 3 --batch_size 4 --subset_size 100

# 只训练不测试
python run_all.py --epochs 20 --batch_size 4 --skip_testing

# 只测试不训练
python run_all.py --skip_training
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | fasterrcnn_resnet50 | 模型名称 |
| `--epochs` | 20 | 训练轮数 |
| `--batch_size` | 4 | 批次大小 |
| `--lr` | 0.005 | 初始学习率 |
| `--subset_size` | None | 数据集子集大小 |
| `--num_workers` | 2 | 数据加载线程数 |
| `--cpu` | False | 强制使用 CPU |

## 评估指标

- **mAP@0.5**: IoU 阈值为 0.5 时的 mean Average Precision
- **Per-class AP**: 每个类别的 AP
- **训练/验证 Loss**: 用于监控训练过程

## 目录结构

```
project2-detection/
├── models/              # 模型定义
│   ├── __init__.py
│   └── factory.py       # 模型工厂
├── data/                # 数据集（自动下载）
│   └── voc_dataset.py   # 数据加载
├── train.py             # 训练脚本
├── test.py              # 测试脚本
├── run_all.py           # 批量运行
├── README.md            # 本文档
├── models/              # 保存的模型权重（.gitignore）
├── results/             # 训练/测试结果（.gitignore）
└── logs/                # TensorBoard 日志（.gitignore）
```

## 面试要点

### 3 分钟陈述

1. **项目背景**: 在 VOC 2007 上训练目标检测模型，对比不同架构的性能
2. **技术选型**: 使用 torchvision 内置的 Faster R-CNN，开箱即用，代码质量高
3. **关键实现**:
   - 数据加载：处理 VOC XML 标注，转换为模型需要的格式
   - 训练循环：检测模型训练时返回 losses 字典，自动反向传播
   - 评估指标：实现 mAP 计算，包括 IoU 计算和 AP 计算
4. **实验结果**: 对比不同模型的 mAP 和训练效率
5. **踩坑经验**:
   - 检测模型的 collate_fn 需要自定义（不同数量的目标）
   - torchvision 检测模型在 eval 模式下不返回 loss，需要切回 train 模式
   - VOC 数据集标注格式的解析需要注意单目标 vs 多目标的情况

### 常见问题

**Q: 为什么用 Faster R-CNN 而不是 YOLO？**
A: torchvision 内置的 Faster R-CNN 开箱即用，代码质量高，适合快速上手。YOLO 需要额外依赖（ultralytics），可以作为后续扩展。

**Q: mAP 是怎么计算的？**
A: 对每个类别，按置信度排序预测框，计算 precision-recall 曲线，然后用 11 点插值法计算 AP，最后对所有类别取平均得到 mAP。

**Q: 为什么 batch_size 只有 4？**
A: 目标检测模型显存占用大（输入图像较大 + 区域提议网络），batch_size=4 是常见的设置。

## 参考资料

- [torchvision 目标检测教程](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [Faster R-CNN 论文](https://arxiv.org/abs/1506.01497)
- [PASCAL VOC 数据集](http://host.robots.ox.ac.uk/pascal/VOC/)
