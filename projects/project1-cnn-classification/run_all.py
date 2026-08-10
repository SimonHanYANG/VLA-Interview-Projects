"""
项目1: 批量训练和测试所有 CNN 模型
=====================================

功能：
- 训练所有支持的 CNN 模型（AlexNet, VGG, GoogLeNet, ResNet, EfficientNet）
- 在测试集上评估所有模型
- 生成对比报告和可视化

使用方法：
    python run_all.py --epochs 20 --batch_size 128
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

# 导入训练和测试函数
from train import train
from test import run_test


# ============================================
# 1. 模型对比可视化
# ============================================

def plot_comparison(results, save_dir):
    """
    绘制所有模型的对比图

    参数:
        results: 所有模型的结果字典
        save_dir: 保存目录
    """
    models = list(results.keys())
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    # 1. 雷达图（Radar Chart）
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # 设置雷达图的角度
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    # 绘制每个模型
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    for idx, model in enumerate(models):
        values = [results[model]['metrics'][m] for m in metrics]
        values += values[:1]  # 闭合

        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title('Model Comparison - Radar Chart', size=15, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_radar.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 柱状对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        values = [results[model]['metrics'][metric] for model in models]

        bars = ax.bar(models, values, color=colors)
        ax.set_xlabel('Model')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Comparison')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_bars.png'), dpi=150)
    plt.close()

    # 3. 训练曲线对比（如果有训练历史）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model in models:
        if 'history' in results[model]:
            history = results[model]['history']
            epochs = range(1, len(history['train_loss']) + 1)

            axes[0].plot(epochs, history['train_loss'], label=f'{model} (train)')
            axes[0].plot(epochs, history['val_loss'], '--', label=f'{model} (val)')

            axes[1].plot(epochs, history['train_acc'], label=f'{model} (train)')
            axes[1].plot(epochs, history['val_acc'], '--', label=f'{model} (val)')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()

    print(f"\n对比图已保存到: {save_dir}")


def generate_comparison_report(results, save_dir):
    """
    生成模型对比报告

    参数:
        results: 所有模型的结果字典
        save_dir: 保存目录
    """
    report_path = os.path.join(save_dir, 'comparison_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CNN 分类模型对比报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # 按 F1 Score 排序
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1]['metrics']['f1_macro'],
            reverse=True
        )

        # 汇总表
        f.write("模型性能排名（按 F1 Score）:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'排名':<6} {'模型':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}\n")
        f.write("-" * 80 + "\n")

        for rank, (model, data) in enumerate(sorted_models, 1):
            metrics = data['metrics']
            f.write(f"{rank:<6} {model:<15} {metrics['accuracy']:<12.4f} "
                   f"{metrics['precision_macro']:<12.4f} {metrics['recall_macro']:<12.4f} "
                   f"{metrics['f1_macro']:<12.4f}\n")

        f.write("-" * 80 + "\n\n")

        # 每个模型的详细信息
        for model, data in sorted_models:
            metrics = data['metrics']
            f.write(f"\n{'='*80}\n")
            f.write(f"模型: {model}\n")
            f.write(f"{'='*80}\n\n")

            f.write("整体指标:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision_macro']:.4f}\n")
            f.write(f"  Recall: {metrics['recall_macro']:.4f}\n")
            f.write(f"  F1 Score: {metrics['f1_macro']:.4f}\n\n")

            f.write("各类别 F1 Score:\n")
            for cls in ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']:
                f1 = metrics['f1_per_class'].get(cls, 0)
                f.write(f"  {cls:<12}: {f1:.4f}\n")

            f.write("\n")

        # 分析和建议
        f.write("\n" + "=" * 80 + "\n")
        f.write("分析和建议\n")
        f.write("=" * 80 + "\n\n")

        best_model = sorted_models[0][0]
        worst_model = sorted_models[-1][0]

        f.write(f"1. 最佳模型: {best_model}\n")
        f.write(f"   - F1 Score: {sorted_models[0][1]['metrics']['f1_macro']:.4f}\n")
        f.write(f"   - 建议：如果追求最高精度，选择此模型\n\n")

        f.write(f"2. 最弱模型: {worst_model}\n")
        f.write(f"   - F1 Score: {sorted_models[-1][1]['metrics']['f1_macro']:.4f}\n")
        f.write(f"   - 建议：可能需要更多训练数据或调整超参数\n\n")

        f.write("3. 模型选择建议:\n")
        f.write("   - 资源充足：选择最佳模型\n")
        f.write("   - 资源受限：考虑 EfficientNet（效率高）\n")
        f.write("   - 实时应用：考虑轻量级模型\n")

    print(f"对比报告已保存到: {report_path}")


# ============================================
# 2. 主流程
# ============================================

def main():
    parser = argparse.ArgumentParser(description='批量训练和测试所有 CNN 模型')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--models', nargs='+', default=['alexnet', 'vgg16', 'googlenet', 'resnet18', 'efficientnet_b0'],
                        help='要训练的模型列表')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("项目1: 批量训练和测试所有 CNN 模型")
    print("=" * 60)
    print(f"模型列表: {args.models}")
    print(f"训练轮数: {args.epochs}")
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print("=" * 60)

    # 存储所有结果
    all_results = {}

    # 训练和测试每个模型
    for model_name in args.models:
        print("\n" + "=" * 60)
        print(f"开始处理模型: {model_name}")
        print("=" * 60)

        # 创建命名空间参数
        train_args = argparse.Namespace(
            model=model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )

        # 训练
        print(f"\n[1/2] 训练 {model_name}...")
        start_time = time.time()
        history, best_acc = train(train_args)
        train_time = time.time() - start_time

        # 测试
        print(f"\n[2/2] 测试 {model_name}...")
        test_args = argparse.Namespace(
            model=model_name,
            model_path=f"models/{model_name}_best.pth",
            batch_size=args.batch_size
        )

        metrics = run_test(test_args)

        # 保存结果
        all_results[model_name] = {
            'metrics': metrics,
            'history': history,
            'best_val_acc': best_acc,
            'train_time': train_time
        }

        print(f"\n{model_name} 完成！")
        print(f"  训练时间: {train_time:.2f}s")
        print(f"  最佳验证准确率: {best_acc:.2f}%")
        print(f"  测试 F1 Score: {metrics['f1_macro']:.4f}")

    # 生成对比报告
    print("\n" + "=" * 60)
    print("生成对比报告...")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = f"results/comparison_{timestamp}"
    os.makedirs(comparison_dir, exist_ok=True)

    # 绘制对比图
    plot_comparison(all_results, comparison_dir)

    # 生成报告
    generate_comparison_report(all_results, comparison_dir)

    # 保存完整结果
    results_path = os.path.join(comparison_dir, 'all_results.json')
    # 需要将不能序列化的对象转换
    serializable_results = {}
    for model, data in all_results.items():
        serializable_results[model] = {
            'metrics': data['metrics'],
            'best_val_acc': data['best_val_acc'],
            'train_time': data['train_time']
        }

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print("\n" + "=" * 60)
    print("所有模型训练和测试完成！")
    print("=" * 60)
    print(f"\n对比结果保存在: {comparison_dir}")
    print("\n主要文件:")
    print(f"  - 对比报告: {comparison_dir}/comparison_report.txt")
    print(f"  - 雷达图: {comparison_dir}/comparison_radar.png")
    print(f"  - 柱状图: {comparison_dir}/comparison_bars.png")
    print(f"  - 训练曲线: {comparison_dir}/training_curves.png")
    print(f"  - 完整结果: {comparison_dir}/all_results.json")

    return all_results


if __name__ == '__main__':
    main()
