"""
项目1: CNN 分类测试脚本
=====================================

功能：
- 加载训练好的模型
- 在测试集上评估模型
- 计算并保存性能指标：
  - Accuracy（准确率）
  - Precision（精确率）
  - Recall（召回率）
  - F1 Score（F1 分数）
  - Confusion Matrix（混淆矩阵）
- 生成可视化报告

使用方法：
    python test.py --model resnet18 --model_path models/resnet18_best.pth
"""

import os
import sys
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


# CIFAR-10 类别名称
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# ============================================
# 1. 模型创建（与 train.py 保持一致）
# ============================================

def create_model(model_name, num_classes=10):
    """
    创建 CNN 模型（与 train.py 中的实现保持一致）
    """
    weights_map = {
        'alexnet': models.AlexNet_Weights.IMAGENET1K_V1,
        'vgg16': models.VGG16_Weights.IMAGENET1K_V1,
        'googlenet': models.GoogLeNet_Weights.IMAGENET1K_V1,
        'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
        'efficientnet_b0': models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    }

    if model_name not in weights_map:
        raise ValueError(f"不支持的模型: {model_name}")

    weights = weights_map[model_name]

    if model_name == 'alexnet':
        model = models.alexnet(weights=weights)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'googlenet':
        model = models.googlenet(weights=weights)
        model.fc = nn.Linear(1024, num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(512, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(1280, num_classes)

    return model


# ============================================
# 2. 测试函数
# ============================================

def test(model, test_loader, device):
    """
    在测试集上评估模型

    参数:
        model: PyTorch 模型
        test_loader: 测试数据加载器
        device: 设备

    返回:
        all_predictions: 所有预测结果
        all_targets: 所有真实标签
        all_probabilities: 所有预测概率
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # 获取预测结果
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            # 收集结果
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)


# ============================================
# 3. 性能指标计算
# ============================================

def calculate_metrics(predictions, targets, probabilities):
    """
    计算各种分类性能指标

    参数:
        predictions: 预测结果
        targets: 真实标签
        probabilities: 预测概率

    返回:
        metrics: 包含所有指标的字典

    面试点：各指标的含义
    - Accuracy：整体准确率，适合平衡数据集
    - Precision：精确率 = TP / (TP + FP)，关注"预测为正的样本中有多少是真正的正样本"
    - Recall：召回率 = TP / (TP + FN)，关注"正样本中有多少被正确预测"
    - F1 Score：精确率和召回率的调和平均数，平衡两者
    - Confusion Matrix：混淆矩阵，展示各类别的预测情况
    """
    # 计算各项指标
    accuracy = accuracy_score(targets, predictions)
    precision_macro = precision_score(targets, predictions, average='macro')
    recall_macro = recall_score(targets, predictions, average='macro')
    f1_macro = f1_score(targets, predictions, average='macro')

    # 计算每个类别的指标
    precision_per_class = precision_score(targets, predictions, average=None)
    recall_per_class = recall_score(targets, predictions, average=None)
    f1_per_class = f1_score(targets, predictions, average=None)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(targets, predictions)

    # 生成分类报告
    class_report = classification_report(
        targets, predictions,
        target_names=CIFAR10_CLASSES,
        output_dict=True
    )

    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_per_class': {cls: float(p) for cls, p in zip(CIFAR10_CLASSES, precision_per_class)},
        'recall_per_class': {cls: float(r) for cls, r in zip(CIFAR10_CLASSES, recall_per_class)},
        'f1_per_class': {cls: float(f) for cls, f in zip(CIFAR10_CLASSES, f1_per_class)},
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    }

    return metrics


# ============================================
# 4. 可视化函数
# ============================================

def plot_confusion_matrix(conf_matrix, classes, save_path):
    """
    绘制混淆矩阵热力图

    面试点：混淆矩阵的解读
    - 对角线：正确分类的样本数
    - 非对角线：错误分类的样本数
    - 可以看到模型在哪些类别上容易混淆
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存到: {save_path}")


def plot_metrics_per_class(metrics, save_path):
    """
    绘制每个类别的性能指标柱状图
    """
    classes = CIFAR10_CLASSES
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    precision = [metrics['precision_per_class'][cls] for cls in classes]
    recall = [metrics['recall_per_class'][cls] for cls in classes]
    f1 = [metrics['f1_per_class'][cls] for cls in classes]

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1, width, label='F1 Score', color='#e74c3c')

    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics per Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"类别指标图已保存到: {save_path}")


def plot_top_misclassifications(conf_matrix, classes, save_path, top_n=5):
    """
    绘制最容易混淆的类别对
    """
    # 获取非对角线元素（错误分类）
    errors = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and conf_matrix[i][j] > 0:
                errors.append({
                    'true': classes[i],
                    'pred': classes[j],
                    'count': conf_matrix[i][j]
                })

    # 按错误数量排序
    errors.sort(key=lambda x: x['count'], reverse=True)
    top_errors = errors[:top_n]

    # 绘制柱状图
    labels = [f"{e['true']}→{e['pred']}" for e in top_errors]
    counts = [e['count'] for e in top_errors]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(labels)), counts, color='#e74c3c')
    plt.xlabel('Misclassification')
    plt.ylabel('Count')
    plt.title(f'Top {top_n} Misclassifications')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')

    # 添加数值标签
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"最容易混淆的类别已保存到: {save_path}")


# ============================================
# 5. 主测试流程
# ============================================

def run_test(args):
    """
    完整的测试流程

    步骤：
    1. 初始化设备
    2. 加载测试数据集
    3. 创建并加载模型
    4. 运行测试
    5. 计算性能指标
    6. 生成可视化报告
    7. 保存结果
    """
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 数据预处理（与训练时保持一致）
    # AlexNet 和 VGG 需要 resize 到 224x224
    needs_resize = args.model in ['alexnet', 'vgg16']
    if needs_resize:
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    # 加载测试数据集
    print("\n加载 CIFAR-10 测试数据集...")
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    print(f"测试集大小: {len(test_dataset)}")

    # 创建模型
    print(f"\n创建模型: {args.model}")
    model = create_model(args.model, num_classes=10)

    # 加载训练好的权重
    print(f"加载模型权重: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # 创建 TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{args.model}_test_{timestamp}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志目录: {log_dir}")

    # 运行测试
    print("\n开始测试...")
    predictions, targets, probabilities = test(model, test_loader, device)

    # 计算性能指标
    print("\n计算性能指标...")
    metrics = calculate_metrics(predictions, targets, probabilities)

    # 打印主要指标
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision_macro']:.4f}")
    print(f"召回率 (Recall): {metrics['recall_macro']:.4f}")
    print(f"F1 分数 (F1 Score): {metrics['f1_macro']:.4f}")
    print("=" * 60)

    # 打印每个类别的指标
    print("\n各类别性能指标:")
    print("-" * 60)
    print(f"{'类别':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 60)
    for cls in CIFAR10_CLASSES:
        print(f"{cls:<12} {metrics['precision_per_class'][cls]:<12.4f} "
              f"{metrics['recall_per_class'][cls]:<12.4f} "
              f"{metrics['f1_per_class'][cls]:<12.4f}")
    print("-" * 60)

    # 写入 TensorBoard
    writer.add_scalar('Test/Accuracy', metrics['accuracy'], 0)
    writer.add_scalar('Test/Precision', metrics['precision_macro'], 0)
    writer.add_scalar('Test/Recall', metrics['recall_macro'], 0)
    writer.add_scalar('Test/F1_Score', metrics['f1_macro'], 0)

    # 每个类别的指标写入 TensorBoard
    for cls in CIFAR10_CLASSES:
        writer.add_scalar(f'Test_Precision/{cls}', metrics['precision_per_class'][cls], 0)
        writer.add_scalar(f'Test_Recall/{cls}', metrics['recall_per_class'][cls], 0)
        writer.add_scalar(f'Test_F1/{cls}', metrics['f1_per_class'][cls], 0)

    # 生成可视化报告
    print("\n生成可视化报告...")
    results_dir = f"results/{args.model}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # 1. 混淆矩阵
    conf_matrix = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(
        conf_matrix, CIFAR10_CLASSES,
        os.path.join(results_dir, 'confusion_matrix.png')
    )

    # 2. 类别指标图
    plot_metrics_per_class(
        metrics,
        os.path.join(results_dir, 'metrics_per_class.png')
    )

    # 3. 最容易混淆的类别
    plot_top_misclassifications(
        conf_matrix, CIFAR10_CLASSES,
        os.path.join(results_dir, 'top_misclassifications.png')
    )

    # 保存完整指标
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n完整指标已保存到: {metrics_path}")

    # 保存分类报告文本
    report_path = os.path.join(results_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("分类报告\n")
        f.write("=" * 60 + "\n\n")
        report = metrics['classification_report']
        for cls in CIFAR10_CLASSES:
            if cls in report:
                f.write(f"\n{cls}:\n")
                f.write(f"  Precision: {report[cls]['precision']:.4f}\n")
                f.write(f"  Recall: {report[cls]['recall']:.4f}\n")
                f.write(f"  F1 Score: {report[cls]['f1-score']:.4f}\n")
                f.write(f"  Support: {report[cls]['support']}\n")

        f.write(f"\n\n整体指标:\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  Macro Avg Precision: {metrics['precision_macro']:.4f}\n")
        f.write(f"  Macro Avg Recall: {metrics['recall_macro']:.4f}\n")
        f.write(f"  Macro Avg F1: {metrics['f1_macro']:.4f}\n")
    print(f"分类报告已保存到: {report_path}")

    # 关闭 TensorBoard writer
    writer.close()

    print("\n" + "=" * 60)
    print("测试完成！")
    print(f"结果保存目录: {results_dir}")
    print("=" * 60)

    return metrics


# ============================================
# 6. 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description='CNN 分类测试脚本')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['alexnet', 'vgg16', 'googlenet', 'resnet18', 'efficientnet_b0'],
                        help='模型名称')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("项目1: CNN 分类测试")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"模型路径: {args.model_path}")
    print(f"批量大小: {args.batch_size}")
    print("=" * 60)

    # 开始测试
    metrics = run_test(args)

    return metrics


if __name__ == '__main__':
    main()
