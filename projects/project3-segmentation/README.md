# Project 3: 图像分割统一框架

## 项目目标

跑通经典图像分割模型，理解语义分割、实例分割的区别。

## 支持的模型

| 模型 | 类型 | 核心关注点 | 面试考点 |
|------|------|-----------|----------|
| FCN | 语义分割 | 全卷积、转置卷积、Skip Connection | 为什么需要全卷积？ |
| DeepLabV3 | 语义分割 | 空洞卷积、ASPP 多尺度 | 空洞卷积的原理？ |
| U-Net | 语义分割 | 对称 Encoder-Decoder、Concat Skip | Skip Connection 的作用？ |
| Mask R-CNN | 实例分割 | RoI Align、并行 Mask 分支 | 语义分割和实例分割的区别？ |

## 项目结构

```
project3-segmentation/
├── config.py                # 配置文件
├── dataset.py               # VOC 2012 数据加载
├── train.py                 # 统一训练脚本
├── evaluate.py              # 统一评估脚本
├── download_data.py         # 数据下载脚本
├── models/
│   ├── __init__.py
│   ├── fcn.py               # FCN 模型
│   ├── deeplabv3.py         # DeepLabV3 模型
│   ├── unet.py              # U-Net 模型
│   └── mask_rcnn.py         # Mask R-CNN 模型
├── utils/
│   ├── __init__.py
│   ├── metrics.py           # 评估指标
│   └── visualization.py     # 可视化工具
└── results/                 # 训练结果
```

## 快速开始

### 1. 安装依赖

```bash
conda activate vla
pip install torch torchvision
pip install segmentation-models-pytorch  # 可选，用于 U-Net
pip install tensorboard matplotlib pillow
```

### 2. 下载数据

```bash
# 下载 VOC 2012 数据集
python download_data.py --root ./data

# 创建子集（快速实验）
python download_data.py --root ./data --subset --subset_ratio 0.1
```

### 3. 训练模型

```bash
# 训练 FCN
python train.py --model fcn --epochs 10 --batch_size 8

# 训练 DeepLabV3
python train.py --model deeplabv3 --epochs 10 --batch_size 8

# 训练 U-Net
python train.py --model unet --epochs 10 --batch_size 8
```

### 4. 评估模型

```bash
python evaluate.py --model fcn
python evaluate.py --model deeplabv3
python evaluate.py --model unet
```

## 评估指标

- **mIoU (mean Intersection over Union)**: 主要指标
- **Pixel Accuracy**: 像素准确率
- **Mean Pixel Accuracy**: 平均像素准确率

## 面试能讲的点

1. **语义分割 vs 实例分割**
   - 语义分割：对每个像素分类，不区分个体
   - 实例分割：检测物体并分割每个实例

2. **FCN 的核心思想**
   - 全卷积：支持任意输入尺寸
   - 转置卷积：上采样恢复分辨率
   - Skip Connection：融合多尺度特征

3. **DeepLabV3 的创新**
   - 空洞卷积：扩大感受野不增加参数
   - ASPP：多尺度特征融合

4. **U-Net 的结构**
   - 对称 Encoder-Decoder
   - Skip Connection 保留细节

5. **Mask R-CNN 的工作流程**
   - 先检测物体（Faster R-CNN）
   - 再对每个物体分割（RoI Align + Mask Head）

## 参考资料

- [FCN Paper](https://arxiv.org/abs/1411.4038)
- [DeepLabV3 Paper](https://arxiv.org/abs/1706.05587)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [torchvision segmentation models](https://pytorch.org/vision/stable/models.html#semantic-segmentation)
