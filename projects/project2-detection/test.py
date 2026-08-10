"""
目标检测测试脚本
计算 mAP (mean Average Precision) 评估指标
用法: python test.py --model fasterrcnn_resnet50
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import torch
import numpy as np

from models import create_model, get_supported_models
from data import get_voc_loaders, VOC_CLASSES


def compute_iou(box1, box2):
    """
    计算两个边界框的 IoU (Intersection over Union)

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        iou: float
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    iou = intersection / union if union > 0 else 0.0
    return iou


def compute_ap(recall, precision):
    """
    计算 Average Precision (11-point interpolation)

    Args:
        recall: recall 值列表
        precision: precision 值列表

    Returns:
        ap: float
    """
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        precisions_at_recall = precision[recall >= t]
        if len(precisions_at_recall) > 0:
            ap += np.max(precisions_at_recall)
    ap /= 11.0
    return ap


@torch.no_grad()
def evaluate_detection(model, data_loader, device, iou_threshold=0.5):
    """
    评估目标检测模型的 mAP

    Args:
        model: 检测模型
        data_loader: 测试数据加载器
        device: 设备
        iou_threshold: IoU 阈值

    Returns:
        results: dict with mAP and per-class AP
    """
    model.eval()

    # 存储每个类别的预测和真值
    # predictions[class_id] = list of (confidence, is_tp, is_fp)
    all_predictions = defaultdict(list)
    # ground_truths[class_id] = total count
    all_ground_truths = defaultdict(int)

    num_images = 0

    for images, targets in data_loader:
        # 数据移到 GPU
        images = [img.to(device) for img in images]

        # 推理模式返回预测结果
        outputs = model(images)

        # 处理每张图的预测
        for i, (output, target) in enumerate(zip(outputs, targets)):
            num_images += 1

            pred_boxes = output['boxes'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()

            gt_boxes = target['boxes'].numpy()
            gt_labels = target['labels'].numpy()

            # 记录每个类别的真值数量
            for label in gt_labels:
                all_ground_truths[int(label)] += 1

            # 按置信度排序预测
            sorted_indices = np.argsort(-pred_scores)

            # 标记已匹配的真值框
            gt_matched = [False] * len(gt_boxes)

            for idx in sorted_indices:
                pred_box = pred_boxes[idx]
                pred_label = int(pred_labels[idx])
                pred_score = float(pred_scores[idx])

                # 跳过背景类
                if pred_label == 0:
                    continue

                # 寻找最佳匹配的真值框
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if int(gt_label) != pred_label:
                        continue
                    if gt_matched[gt_idx]:
                        continue

                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                # 判断 TP 或 FP
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    all_predictions[pred_label].append((pred_score, True, False))
                    gt_matched[best_gt_idx] = True
                else:
                    all_predictions[pred_label].append((pred_score, False, True))

            if num_images % 100 == 0:
                print(f"  已处理 {num_images} 张图像...")

    # 计算每个类别的 AP
    per_class_ap = {}
    per_class_metrics = {}

    for class_idx in range(1, len(VOC_CLASSES) + 1):
        class_name = VOC_CLASSES[class_idx - 1]
        gt_count = all_ground_truths.get(class_idx, 0)

        if gt_count == 0:
            per_class_ap[class_name] = 0.0
            continue

        preds = all_predictions.get(class_idx, [])
        if len(preds) == 0:
            per_class_ap[class_name] = 0.0
            continue

        # 按置信度排序
        preds.sort(key=lambda x: -x[0])

        # 计算累积 TP/FP
        tp_cumsum = 0
        fp_cumsum = 0
        recalls = []
        precisions = []

        for score, is_tp, is_fp in preds:
            if is_tp:
                tp_cumsum += 1
            if is_fp:
                fp_cumsum += 1

            recall = tp_cumsum / gt_count
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)

            recalls.append(recall)
            precisions.append(precision)

        # 计算 AP
        recalls = np.array(recalls)
        precisions = np.array(precisions)
        ap = compute_ap(recalls, precisions)

        per_class_ap[class_name] = ap
        per_class_metrics[class_name] = {
            'ap': float(ap),
            'gt_count': gt_count,
            'pred_count': len(preds),
            'tp': tp_cumsum,
            'fp': fp_cumsum,
        }

    # 计算 mAP
    mAP = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0

    results = {
        'mAP': float(mAP),
        'per_class_ap': per_class_ap,
        'per_class_metrics': per_class_metrics,
        'num_images': num_images,
        'iou_threshold': iou_threshold,
    }

    return results


def print_results(results):
    """打印评估结果"""
    print("\n" + "=" * 70)
    print("  VOC 2007 目标检测评估结果")
    print("=" * 70)

    print(f"\n  mAP@0.5: {results['mAP']:.4f}")
    print(f"  评估图像数: {results['num_images']}")
    print(f"  IoU 阈值: {results['iou_threshold']}")

    print(f"\n  {'类别':<15} {'AP':>8} {'GT数':>8} {'Pred数':>8} {'TP':>8} {'FP':>8}")
    print("  " + "-" * 60)

    for cls_name in VOC_CLASSES:
        if cls_name in results['per_class_metrics']:
            m = results['per_class_metrics'][cls_name]
            print(f"  {cls_name:<15} {m['ap']:>8.4f} {m['gt_count']:>8} "
                  f"{m['pred_count']:>8} {m['tp']:>8} {m['fp']:>8}")
        else:
            print(f"  {cls_name:<15} {'N/A':>8}")

    print("=" * 70)


def test(args):
    """主测试函数"""
    model_name = args.model
    print(f"\n[测试] 评估 {model_name}...")

    # 设备
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
        print(f"[设备] 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"[设备] 使用 CPU")

    # 加载数据
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    _, _, test_loader = get_voc_loaders(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
        download=False,  # 测试时不重复下载
    )

    # 创建模型（需要 pretrained=True 以获取正确的类别数）
    model = create_model(model_name, pretrained=True)

    # 加载权重
    checkpoint_path = os.path.join('models', f'{model_name}_best.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[模型] 已加载权重: {checkpoint_path}")
        print(f"[模型] 来自 Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
    else:
        print(f"[警告] 未找到最佳模型权重，使用默认权重")

    model.to(device)

    # 评估
    results = evaluate_detection(
        model, test_loader, device,
        iou_threshold=args.iou_threshold,
    )

    # 打印结果
    print_results(results)

    # 保存结果
    result_path = os.path.join('results', f'{model_name}_test_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[保存] 测试结果已保存至: {result_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='VOC 2007 目标检测测试')

    parser.add_argument('--model', type=str, default='fasterrcnn_resnet50',
                        choices=get_supported_models(),
                        help='模型名称')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--subset_size', type=int, default=None, help='数据集子集大小')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU 阈值')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    test(args)
