"""
项目4.1: Swin Transformer 一键训练测试脚本
=============================================

功能：
- 依次训练所有 Swin Transformer 变体（Swin-Tiny, Swin-Small, Swin-Base）
- 训练完成后自动测试
- 生成汇总对比报告

使用方法：
    python run_all.py
    python run_all.py --epochs 20 --gpu 0
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime
import subprocess

import torch


def run_command(cmd, desc=""):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"[运行] {desc}")
    print(f"命令: {cmd}")
    print('='*60)

    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"[错误] {desc} 失败，返回码: {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Swin Transformer 一键训练测试')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=None, help='批量大小（覆盖默认值）')
    args = parser.parse_args()

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 检查 GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n使用 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("\n[警告] 未检测到 GPU，将使用 CPU 训练（非常慢）")

    # 记录开始时间
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 模型列表
    models = ['swin_tiny', 'swin_small', 'swin_base']

    # 训练结果
    results = {}

    # 依次训练和测试
    for model_name in models:
        print(f"\n\n{'#'*60}")
        print(f"# 开始训练: {model_name}")
        print(f"{'#'*60}")

        model_start_time = time.time()

        # 训练命令
        batch_size = args.batch_size if args.batch_size else ('32' if model_name == 'swin_base' else '64')
        train_cmd = (
            f"conda run -n vla --no-capture-output "
            f"python train.py --model {model_name} --epochs {args.epochs} --batch_size {batch_size}"
        )

        if not run_command(train_cmd, f"训练 {model_name}"):
            print(f"[错误] {model_name} 训练失败，跳过...")
            results[model_name] = {'status': '训练失败'}
            continue

        # 测试命令
        model_path = f"models/{model_name}_best.pth"
        if not os.path.exists(model_path):
            print(f"[错误] 未找到模型文件: {model_path}")
            results[model_name] = {'status': '未找到模型'}
            continue

        test_cmd = (
            f"conda run -n vla --no-capture-output "
            f"python test.py --model {model_name} --model_path {model_path}"
        )

        if not run_command(test_cmd, f"测试 {model_name}"):
            print(f"[错误] {model_name} 测试失败")
            results[model_name] = {'status': '测试失败'}
            continue

        model_elapsed = time.time() - model_start_time

        # 读取最新的测试结果
        results_dir = f"results"
        metrics_files = [f for f in os.listdir(results_dir) if f.startswith(f"{model_name}_") and f.endswith('.json')]
        if metrics_files:
            latest_metrics = sorted(metrics_files)[-1]
            with open(os.path.join(results_dir, latest_metrics), 'r') as f:
                metrics = json.load(f)
            results[model_name] = {
                'status': '成功',
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision_macro', 0),
                'recall': metrics.get('recall_macro', 0),
                'f1': metrics.get('f1_macro', 0),
                'training_time': f"{model_elapsed/60:.1f} min"
            }
        else:
            results[model_name] = {
                'status': '成功（未找到详细指标）',
                'training_time': f"{model_elapsed/60:.1f} min"
            }

    # 计算总耗时
    total_elapsed = time.time() - start_time

    # 打印汇总报告
    print("\n\n" + "="*60)
    print("Swin Transformer 训练测试汇总报告")
    print("="*60)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_elapsed/60:.1f} 分钟")
    print("="*60)

    # 结果对比表格
    print(f"\n{'模型':<15} {'状态':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'耗时':<12}")
    print("-"*87)
    for model_name in models:
        if model_name in results:
            r = results[model_name]
            acc = f"{r.get('accuracy', 0):.4f}" if 'accuracy' in r else 'N/A'
            prec = f"{r.get('precision', 0):.4f}" if 'precision' in r else 'N/A'
            rec = f"{r.get('recall', 0):.4f}" if 'recall' in r else 'N/A'
            f1 = f"{r.get('f1', 0):.4f}" if 'f1' in r else 'N/A'
            elapsed = r.get('training_time', 'N/A')
            print(f"{model_name:<15} {r['status']:<12} {acc:<12} {prec:<12} {rec:<12} {f1:<12} {elapsed:<12}")
    print("-"*87)

    # 保存汇总报告
    report = {
        'timestamp': timestamp,
        'total_time_minutes': total_elapsed / 60,
        'results': results
    }
    report_path = f"results/summary_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n汇总报告已保存到: {report_path}")

    print("\n" + "="*60)
    print("全部完成！")
    print("="*60)

    return results


if __name__ == '__main__':
    main()
