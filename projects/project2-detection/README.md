# Project 2: 目标检测 (Object Detection)

## 概述

在 VOC 2007 数据集上训练和对比主流目标检测模型，包括两阶段检测器（Faster R-CNN）、端到端检测器（DETR）和单阶段检测器（YOLOv5）。

## 支持的模型

| 模型 | 来源 | 特点 |
|------|------|------|
| Faster R-CNN (ResNet50) | torchvision | 两阶段检测器，精度高 |
| Faster R-CNN (ResNet50 V2) | torchvision | 改进版，更好的特征提取 |
| Faster R-CNN (MobileNet V3) | torchvision | 轻量级，速度快 |
| DETR (ResNet50) | transformers | 端到端检测器，无需 NMS |
| YOLOv5s | ultralytics | 单阶段检测器，速度快 |
| YOLOv5m | ultralytics | 平衡版本 |
| YOLOv5l | ultralytics | 大模型，精度高 |

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

# 训练 DETR
python train.py --model detr_resnet50 --epochs 20 --batch_size 4

# 训练 YOLOv5（使用单独脚本）
python train_yolo.py --model s --epochs 20 --batch_size 16

# 使用数据子集快速测试
python train.py --model fasterrcnn_resnet50 --epochs 3 --batch_size 4 --subset_size 100
```

### 单模型测试

```bash
# 测试 Faster R-CNN / DETR
python test.py --model fasterrcnn_resnet50

# 测试 YOLOv5（使用单独脚本）
python test_yolo.py --model s
```

### 批量训练 & 测试

```bash
# 训练所有模型（Faster R-CNN + DETR + YOLOv5）
python run_all.py --epochs 20 --batch_size 4

# 只训练特定模型
python run_all.py --models fasterrcnn_resnet50 detr_resnet50 --epochs 20

# 使用子集快速验证
python run_all.py --epochs 3 --batch_size 4 --subset_size 100

# 只训练不测试
python run_all.py --epochs 20 --batch_size 4 --skip_testing

# 只测试不训练
python run_all.py --skip_training
```

### 命令行参数

**train.py / test.py (Faster R-CNN & DETR)**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | fasterrcnn_resnet50 | 模型名称 |
| `--epochs` | 20 | 训练轮数 |
| `--batch_size` | 4 | 批次大小 |
| `--lr` | 0.005 | 初始学习率 |
| `--subset_size` | None | 数据集子集大小 |
| `--num_workers` | 2 | 数据加载线程数 |
| `--cpu` | False | 强制使用 CPU |

**train_yolo.py / test_yolo.py (YOLOv5)**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | s | 模型大小 (s/m/l/x) |
| `--epochs` | 20 | 训练轮数 |
| `--batch_size` | 16 | 批次大小 |
| `--img_size` | 640 | 输入图像尺寸 |
| `--lr` | 0.01 | 初始学习率 |
| `--num_workers` | 8 | 数据加载线程数 |
| `--patience` | 100 | 早停耐心值 |

## 评估指标

- **mAP@0.5**: IoU 阈值为 0.5 时的 mean Average Precision
- **mAP@0.5:0.95**: IoU 阈值从 0.5 到 0.95 的平均 mAP（YOLOv5 特有）
- **Per-class AP**: 每个类别的 AP
- **Precision / Recall**: 精确率和召回率
- **训练/验证 Loss**: 用于监控训练过程

## 目录结构

```
project2-detection/
├── models/                  # 模型定义
│   ├── __init__.py
│   └── factory.py           # 模型工厂（支持 Faster R-CNN, DETR, YOLOv5）
├── data/                    # 数据集（自动下载）
│   └── voc_dataset.py       # 数据加载
├── train.py                 # Faster R-CNN & DETR 训练脚本
├── test.py                  # Faster R-CNN & DETR 测试脚本
├── train_yolo.py            # YOLOv5 专用训练脚本
├── test_yolo.py             # YOLOv5 专用测试脚本
├── run_all.py               # 批量运行所有模型
├── README.md                # 本文档
├── models/                  # 保存的模型权重（.gitignore）
├── results/                 # 训练/测试结果（.gitignore）
└── logs/                    # TensorBoard 日志（.gitignore）
```

## 面试要点

### 3 分钟陈述

1. **项目背景**: 在 VOC 2007 上训练目标检测模型，对比不同架构的性能
2. **技术选型**:
   - Faster R-CNN: torchvision 内置，开箱即用
   - DETR: Facebook 的端到端检测器，使用 transformers 库
   - YOLOv5: 工业界常用的单阶段检测器，速度快
3. **关键实现**:
   - 数据加载：处理 VOC XML 标注，转换为模型需要的格式
   - 训练循环：检测模型训练时返回 losses 字典，自动反向传播
   - 模型包装：为 DETR 和 YOLOv5 实现统一接口的包装器
   - 评估指标：实现 mAP 计算，包括 IoU 计算和 AP 计算
4. **实验结果**: 对比不同模型的 mAP 和训练效率
5. **踩坑经验**:
   - 检测模型的 collate_fn 需要自定义（不同数量的目标）
   - torchvision 检测模型在 eval 模式下不返回 loss，需要切回 train 模式
   - DETR 的标签格式与 Faster R-CNN 不同（cxcywh vs xyxy）
   - YOLOv5 的训练接口与 torchvision 模型不同，需要特殊处理
   - VOC 数据集标注格式的解析需要注意单目标 vs 多目标的情况

### 常见问题

**Q: 为什么同时使用 Faster R-CNN、DETR 和 YOLOv5？**
A: 这三种代表了目标检测的三大范式：
- **Faster R-CNN**: 两阶段检测器，先生成候选区域再分类，精度高但速度较慢
- **DETR**: 端到端检测器，使用 Transformer 去除 NMS，架构新颖
- **YOLOv5**: 单阶段检测器，直接预测边界框，速度快适合部署

**Q: DETR 相比 Faster R-CNN 有什么优势？**
A: DETR 使用 Transformer 的注意力机制，不需要 NMS 后处理，是真正的端-to-end 检测器。但训练收敛较慢，需要更多 epoch。

**Q: YOLOv5 的训练为什么需要特殊处理？**
A: YOLOv5 使用 ultralytics 库，其训练接口与 torchvision 模型不同。当前版本主要支持推理，完整训练需要使用 ultralytics 的训练 API。

**Q: mAP 是怎么计算的？**
A: 对每个类别，按置信度排序预测框，计算 precision-recall 曲线，然后用 11 点插值法计算 AP，最后对所有类别取平均得到 mAP。

**Q: 为什么 batch_size 只有 4？**
A: 目标检测模型显存占用大（输入图像较大 + 区域提议网络），batch_size=4 是常见的设置。

## 参考资料

- [torchvision 目标检测教程](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [Faster R-CNN 论文](https://arxiv.org/abs/1506.01497)
- [DETR 论文](https://arxiv.org/abs/2005.12872)
- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [PASCAL VOC 数据集](http://host.robots.ox.ac.uk/pascal/VOC/)
