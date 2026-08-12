"""
YOLOv5 专用测试脚本
用法: python test_yolo.py --model yolov5s
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 设置中国镜像
os.environ['ULTRALYTICS_HUB'] = 'https://mirror.ghproxy.com'

from ultralytics import YOLO


def test(args):
    """YOLOv5 测试主函数"""
    print("=" * 60)
    print("  YOLOv5 目标检测测试")
    print("=" * 60)

    # 加载训练好的模型 - 搜索多个可能路径
    possible_paths = [
        f'runs/train/yolov5{args.model}_voc/weights/best.pt',
        f'runs/detect/runs/train/yolov5{args.model}_voc/weights/best.pt',
        f'runs/detect/train/yolov5{args.model}_voc/weights/best.pt',
    ]
    model_path = None
    for p in possible_paths:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        print(f"[错误] 未找到训练好的模型，尝试过的路径:")
        for p in possible_paths:
            print(f"  - {p}")
        print(f"[提示] 请先运行训练: python train_yolo.py --model {args.model} --epochs 20")
        return None

    print(f"[模型] 加载权重: {model_path}")
    model = YOLO(model_path)

    # 数据集配置
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    data_config = os.path.join(data_root, 'voc.yaml')

    # 运行测试
    print(f"\n[测试] 评估模型...")
    results = model.val(
        data=data_config,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.num_workers,
        project='runs/val',
        name=f'yolov5{args.model}_voc',
        exist_ok=True,
    )

    # 解析结果
    test_results = {
        'model': f'yolov5{args.model}',
        'mAP50': float(results.box.map50),
        'mAP50_95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
        'per_class_ap50': {},
        'per_class_ap50_95': {},
    }

    # 每个类别的 AP
    # ultralytics >= 8.x: box.ap50 和 box.ap 是 per-class 列表
    # box.ap_class_index 给出对应的类别索引
    try:
        ap50_values = results.box.ap50  # list of AP@0.5 per class
        ap_values = results.box.ap      # list of AP@0.5:0.95 per class
        class_indices = results.box.ap_class_index  # class indices

        for i, cls_idx in enumerate(class_indices):
            class_name = results.names[int(cls_idx)] if int(cls_idx) in results.names else f'class_{cls_idx}'
            if i < len(ap50_values):
                test_results['per_class_ap50'][class_name] = float(ap50_values[i])
            if i < len(ap_values):
                test_results['per_class_ap50_95'][class_name] = float(ap_values[i])
    except Exception as e:
        print(f"[警告] 获取 per-class AP 失败: {e}")
        # fallback: 尝试旧 API
        for attr_name in ['ap50_per_class', 'ap_per_class']:
            if hasattr(results.box, attr_name):
                for i, ap in enumerate(getattr(results.box, attr_name)):
                    class_name = results.names[i] if i in results.names else f'class_{i}'
                    key = 'per_class_ap50' if '50' in attr_name else 'per_class_ap50_95'
                    test_results[key][class_name] = float(ap)

    # 打印结果
    print("\n" + "=" * 70)
    print("  YOLOv5 VOC 2007 测试结果")
    print("=" * 70)
    print(f"\n  mAP@0.5:      {test_results['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {test_results['mAP50_95']:.4f}")
    print(f"  Precision:    {test_results['precision']:.4f}")
    print(f"  Recall:       {test_results['recall']:.4f}")

    if test_results['per_class_ap50']:
        print(f"\n  {'类别':<15} {'AP@0.5':>10} {'AP@0.5:0.95':>12}")
        print("  " + "-" * 40)
        for class_name in sorted(test_results['per_class_ap50'].keys()):
            ap50 = test_results['per_class_ap50'].get(class_name, 0)
            ap50_95 = test_results['per_class_ap50_95'].get(class_name, 0)
            print(f"  {class_name:<15} {ap50:>10.4f} {ap50_95:>12.4f}")

    print("=" * 70)

    # 保存结果
    os.makedirs('results', exist_ok=True)
    result_path = f'results/yolov5{args.model}_test_results.json'
    with open(result_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n[保存] 测试结果已保存至: {result_path}")

    return test_results


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv5 目标检测测试')

    parser.add_argument('--model', type=str, default='s',
                        choices=['s', 'm', 'l', 'x'],
                        help='YOLOv5 模型大小 (s/m/l/x)')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--img_size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--device', type=str, default='', help='设备 (cpu/0/0,1/...)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    test(args)
