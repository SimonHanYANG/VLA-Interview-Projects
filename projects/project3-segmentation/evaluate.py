"""
Project 3: 图像分割统一评估脚本
"""

import os
import sys
import argparse
import json

import torch
import torch.nn as nn

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SegmentationConfig, MODEL_CONFIGS
from dataset import get_dataloaders, decode_segmap
from models import get_model
from utils.metrics import compute_miou, compute_pixel_accuracy, compute_mean_pixel_accuracy
from utils.visualization import plot_miou_comparison, visualize_predictions


def evaluate_model(model_name, config, checkpoint_path=None):
    """
    评估指定的分割模型

    Args:
        model_name: 模型名称 ("fcn", "deeplabv3", "unet")
        config: 配置对象
        checkpoint_path: 模型检查点路径（可选）

    Returns:
        results: 评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {MODEL_CONFIGS[model_name]['name']}")
    print(f"{'='*60}")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    print("\nLoading data...")
    _, val_loader = get_dataloaders(config)

    # 创建模型
    print(f"\nCreating model: {model_name}")
    model = get_model(model_name, num_classes=config.num_classes, pretrained=False)
    model = model.to(device)

    # 加载检查点
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.results_dir, model_name, "best_model.pth")

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, using random weights")

    # 评估模型
    model.eval()
    all_preds = []
    all_targets = []

    print("\nEvaluating...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)

            # 处理不同模型的输出格式
            if isinstance(outputs, dict):
                preds = torch.argmax(outputs["out"], dim=1)
            else:
                preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(val_loader)}]")

    # 合并所有预测和目标
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 计算指标
    miou, iou_per_class = compute_miou(all_preds, all_targets, config.num_classes)
    pixel_acc = compute_pixel_accuracy(all_preds, all_targets)
    mean_pixel_acc, acc_per_class = compute_mean_pixel_accuracy(all_preds, all_targets, config.num_classes)

    # 打印结果
    print(f"\n{'='*60}")
    print(f"Evaluation Results for {MODEL_CONFIGS[model_name]['name']}")
    print(f"{'='*60}")
    print(f"mIoU: {miou:.4f}")
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Mean Pixel Accuracy: {mean_pixel_acc:.4f}")

    # 打印每个类别的 IoU
    print(f"\nIoU per class:")
    from config import VOC_CLASSES
    for i, (cls_name, iou) in enumerate(zip(VOC_CLASSES, iou_per_class)):
        print(f"  {cls_name:15s}: {iou:.4f}")

    # 保存结果
    results = {
        "model_name": model_name,
        "model_full_name": MODEL_CONFIGS[model_name]["name"],
        "miou": float(miou),
        "pixel_accuracy": float(pixel_acc),
        "mean_pixel_accuracy": float(mean_pixel_acc),
        "iou_per_class": [float(iou) for iou in iou_per_class],
        "accuracy_per_class": [float(acc) for acc in acc_per_class],
        "num_samples": len(all_preds),
    }

    # 保存到文件
    result_dir = os.path.join(config.results_dir, model_name)
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {result_dir}")

    # 可视化预测结果
    print("\nGenerating visualizations...")
    visualize_predictions(
        model, val_loader, device,
        save_dir=os.path.join(result_dir, "predictions"),
        num_samples=3
    )

    return results


def compare_models(config):
    """
    对比所有模型的性能

    Args:
        config: 配置对象
    """
    print("\n" + "="*60)
    print("Comparing all models")
    print("="*60)

    all_results = {}

    # 评估每个模型
    for model_name in config.models:
        result = evaluate_model(model_name, config)
        all_results[model_name] = result

    # 绘制对比图
    model_mious = {name: result["miou"] for name, result in all_results.items()}
    plot_miou_comparison(
        model_mious,
        save_path=os.path.join(config.results_dir, "model_comparison.png")
    )

    # 打印对比表
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    print(f"{'Model':<20} {'mIoU':<10} {'Pixel Acc':<12} {'Mean Acc':<10}")
    print("-"*60)

    for name, result in all_results.items():
        print(f"{name:<20} {result['miou']:<10.4f} {result['pixel_accuracy']:<12.4f} {result['mean_pixel_accuracy']:<10.4f}")

    # 保存对比结果
    comparison = {
        "models": all_results,
        "best_model": max(all_results.items(), key=lambda x: x[1]["miou"])[0],
        "best_miou": max(result["miou"] for result in all_results.values()),
    }

    with open(os.path.join(config.results_dir, "model_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nBest model: {comparison['best_model']} (mIoU: {comparison['best_miou']:.4f})")
    print(f"\nComparison results saved to: {config.results_dir}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation models")
    parser.add_argument("--model", type=str, default=None,
                       choices=["fcn", "deeplabv3", "unet", "all"],
                       help="Model to evaluate (or 'all' for comparison)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--subset_ratio", type=float, default=0.1,
                       help="Subset ratio for quick evaluation")
    parser.add_argument("--data_root", type=str, default="./data",
                       help="Data root directory")

    args = parser.parse_args()

    # 创建配置
    config = SegmentationConfig(
        subset_ratio=args.subset_ratio,
        data_root=args.data_root,
    )

    if args.model == "all" or args.model is None:
        # 对比所有模型
        compare_models(config)
    else:
        # 评估单个模型
        evaluate_model(args.model, config, args.checkpoint)


if __name__ == "__main__":
    main()
