# 项目1: CNN 分类

## 项目概述

使用 PyTorch 和 torchvision 实现多种经典 CNN 模型在 CIFAR-10 数据集上的分类任务。

## 支持的模型

| 模型 | 特点 | 面试要点 |
|------|------|----------|
| AlexNet | 第一个深层 CNN | ReLU、Dropout、数据增强 |
| VGG | 小卷积核堆叠 | 3x3 卷积核、更深网络 |
| GoogLeNet | Inception 模块 | 多尺度特征、1x1 卷积 |
| ResNet | 残差连接 | 解决梯度消失、跳跃连接 |
| EfficientNet | 复合缩放 | 效率与精度平衡 |

## 环境要求

```bash
conda activate vla
pip install torch torchvision tensorboard scikit-learn matplotlib seaborn
```

## 项目结构

```
project1-cnn-classification/
├── train.py          # 训练脚本（支持 TensorBoard）
├── test.py           # 测试脚本（计算 F1 Score 等指标）
├── run_all.py        # 批量训练和对比脚本
├── models/           # 保存训练好的模型
├── data/             # 数据集目录
├── logs/             # TensorBoard 日志
└── results/          # 测试结果和可视化
```

## 使用方法

### 1. 训练单个模型

```bash
# 训练 ResNet-18
python train.py --model resnet18 --epochs 20 --batch_size 128

# 训练 VGG-16
python train.py --model vgg16 --epochs 20 --batch_size 128

# 训练所有模型
python train.py --model alexnet --epochs 20
python train.py --model googlenet --epochs 20
python train.py --model efficientnet_b0 --epochs 20
```

### 2. 测试模型

```bash
# 测试单个模型
python test.py --model resnet18 --model_path models/resnet18_best.pth

# 测试其他模型
python test.py --model vgg16 --model_path models/vgg16_best.pth
```

### 3. 批量训练和对比

```bash
# 训练和测试所有模型，生成对比报告
python run_all.py --epochs 20 --batch_size 128

# 只训练部分模型
python run_all.py --models resnet18 vgg16 efficientnet_b0 --epochs 10
```

## 性能指标

测试脚本会计算以下指标：

- **Accuracy（准确率）**: 整体分类正确率
- **Precision（精确率）**: 预测为正的样本中真正正样本的比例
- **Recall（召回率）**: 正样本中被正确预测的比例
- **F1 Score**: 精确率和召回率的调和平均数
- **Confusion Matrix（混淆矩阵）**: 各类别的预测详情

## TensorBoard 可视化

```bash
# 启动 TensorBoard
tensorboard --logdir=logs/

# 访问 http://localhost:6006
```

TensorBoard 包含：
- 训练/验证损失曲线
- 训练/验证准确率曲线
- 学习率变化
- 测试指标

## 输出文件

### 训练输出
- `models/{model}_best.pth` - 最佳模型权重
- `models/{model}_final.pth` - 最终模型权重
- `results/{model}_history.json` - 训练历史

### 测试输出
- `results/{model}_{timestamp}/metrics.json` - 完整性能指标
- `results/{model}_{timestamp}/confusion_matrix.png` - 混淆矩阵
- `results/{model}_{timestamp}/metrics_per_class.png` - 类别指标图
- `results/{model}_{timestamp}/top_misclassifications.png` - 最易混淆类别
- `results/{model}_{timestamp}/classification_report.txt` - 分类报告

### 对比输出（run_all.py）
- `results/comparison_{timestamp}/comparison_report.txt` - 对比报告
- `results/comparison_{timestamp}/comparison_radar.png` - 雷达图
- `results/comparison_{timestamp}/comparison_bars.png` - 柱状图
- `results/comparison_{timestamp}/training_curves.png` - 训练曲线

## 面试要点

### 1. 模型演进
```
AlexNet (2012) → VGG (2014) → GoogLeNet (2014) → ResNet (2015) → EfficientNet (2019)
```

### 2. 核心创新
- **AlexNet**: ReLU 激活函数、Dropout 正则化、数据增强
- **VGG**: 3x3 小卷积核堆叠、更深网络
- **GoogLeNet**: Inception 模块、多尺度特征、1x1 卷积降维
- **ResNet**: 残差连接、解决梯度消失、恒等映射
- **EfficientNet**: 复合缩放（宽度/深度/分辨率）

### 3. 关键概念
- 梯度消失/爆炸问题及解决方案
- 过拟合与正则化（Dropout、数据增强、权重衰减）
- 学习率调度策略
- 迁移学习与预训练模型

## 参考资料

- [PyTorch torchvision Models](https://pytorch.org/vision/stable/models.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorBoard Documentation](https://pytorch.org/docs/stable/tensorboard.html)
