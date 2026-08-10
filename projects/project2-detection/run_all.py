"""
批量训练 & 测试所有目标检测模型
用法: python run_all.py --epochs 20 --batch_size 4
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import torch
from models import get_supported_models


def run_training(model_name, epochs, batch_size, lr, num_workers, subset_size):
    """运行单个模型的训练"""
    print(f"\n{'='*60}")
    print(f"  开始训练: {model_name}")
    print(f"{'='*60}")

    cmd = (
        f"python train.py --model {model_name} "
        f"--epochs {epochs} --batch_size {batch_size} "
        f"--lr {lr} --num_workers {num_workers}"
    )
    if subset_size:
        cmd += f" --subset_size {subset_size}"

    exit_code = os.system(cmd)
    return exit_code == 0


def run_testing(model_name, batch_size, num_workers, subset_size):
    """运行单个模型的测试"""
    print(f"\n{'='*60}")
    print(f"  开始测试: {model_name}")
    print(f"{'='*60}")

    cmd = (
        f"python test.py --model {model_name} "
        f"--batch_size {batch_size} --num_workers {num_workers}"
    )
    if subset_size:
        cmd += f" --subset_size {subset_size}"

    exit_code = os.system(cmd)
    return exit_code == 0


def generate_comparison_report():
    """生成对比报告"""
    results = {}
    results_dir = 'results'

    for model_name in get_supported_models():
        result_file = os.path.join(results_dir, f'{model_name}_test_results.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results[model_name] = json.load(f)

    if not results:
        print("\n[警告] 未找到测试结果，跳过报告生成")
        return

    # 生成对比报告
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': {},
    }

    for model_name, result in results.items():
        history_file = os.path.join(results_dir, f'{model_name}_history.json')
        history = {}
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)

        report['models'][model_name] = {
            'mAP': result.get('mAP', 0),
            'best_val_loss': history.get('best_val_loss', None),
            'training_time': history.get('training_time', None),
            'per_class_ap': result.get('per_class_ap', {}),
        }

    # 保存报告
    report_path = os.path.join(results_dir, 'comparison_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # 打印对比表
    print("\n" + "=" * 80)
    print("  模型对比报告")
    print("=" * 80)

    # 按 mAP 排序
    sorted_models = sorted(
        report['models'].items(),
        key=lambda x: x[1].get('mAP', 0),
        reverse=True
    )

    print(f"\n  {'模型':<30} {'mAP@0.5':>10} {'Best Val Loss':>15} {'训练时间(min)':>15}")
    print("  " + "-" * 75)

    for model_name, info in sorted_models:
        mAP = info.get('mAP', 0)
        val_loss = info.get('best_val_loss', 0)
        train_time = info.get('training_time', 0)

        val_loss_str = f"{val_loss:.4f}" if val_loss else "N/A"
        time_str = f"{train_time / 60:.1f}" if train_time else "N/A"

        print(f"  {model_name:<30} {mAP:>10.4f} {val_loss_str:>15} {time_str:>15}")

    print("=" * 80)
    print(f"\n[报告] 对比报告已保存至: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='批量训练目标检测模型')
    parser.add_argument('--models', nargs='+', default=None,
                        help='要训练的模型列表（默认全部）')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.005, help='初始学习率')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--subset_size', type=int, default=None, help='数据集子集大小')
    parser.add_argument('--skip_training', action='store_true', help='跳过训练，只测试')
    parser.add_argument('--skip_testing', action='store_true', help='跳过测试，只训练')

    args = parser.parse_args()

    # 确定要训练的模型
    if args.models:
        models_to_train = args.models
        # 验证模型名称
        supported = get_supported_models()
        for m in models_to_train:
            if m not in supported:
                print(f"[错误] 不支持的模型: {m}")
                print(f"[错误] 可选模型: {supported}")
                sys.exit(1)
    else:
        models_to_train = get_supported_models()

    print("=" * 60)
    print("  VOC 2007 目标检测 - 批量训练 & 测试")
    print("=" * 60)
    print(f"  模型列表: {models_to_train}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch_size}")
    print(f"  子集大小: {args.subset_size or '全部'}")
    print("=" * 60)

    start_time = time.time()

    # 训练所有模型
    if not args.skip_training:
        for i, model_name in enumerate(models_to_train, 1):
            print(f"\n[{i}/{len(models_to_train)}] 训练 {model_name}...")
            success = run_training(
                model_name, args.epochs, args.batch_size,
                args.lr, args.num_workers, args.subset_size
            )
            if not success:
                print(f"[错误] {model_name} 训练失败!")

    # 测试所有模型
    if not args.skip_testing:
        for i, model_name in enumerate(models_to_train, 1):
            print(f"\n[{i}/{len(models_to_train)}] 测试 {model_name}...")
            success = run_testing(
                model_name, args.batch_size,
                args.num_workers, args.subset_size
            )
            if not success:
                print(f"[错误] {model_name} 测试失败!")

    # 生成对比报告
    if not args.skip_testing:
        generate_comparison_report()

    total_time = time.time() - start_time
    print(f"\n[完成] 总耗时: {total_time / 60:.1f} 分钟")


if __name__ == '__main__':
    main()
