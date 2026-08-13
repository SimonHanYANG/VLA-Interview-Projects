"""
CLIP 批量训练脚本
训练所有 CLIP 变体并对比结果
"""

import torch
import subprocess
import sys
import os
import json
from datetime import datetime

from train import CLIP_CONFIGS


def check_gpu_memory():
    """检查 GPU 显存"""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        total_memory = gpu.total_memory / 1024**3
        print(f"GPU: {gpu.name}")
        print(f"显存: {total_memory:.1f} GB")
        return total_memory
    return 0


def train_model(config_name, epochs=30, batch_size=64, lr=0.0005):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"训练 {config_name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, '-u', 'train.py',
        '--model', config_name,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
    ]

    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{config_name}_train.log')

    print(f"日志文件: {log_file}")

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(__file__)
        )
        process.wait()

    if process.returncode == 0:
        print(f"✓ {config_name} 训练完成")
        return True
    else:
        print(f"✗ {config_name} 训练失败")
        return False


def test_model(config_name):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"测试 {config_name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, '-u', 'test.py',
        '--model', config_name,
    ]

    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    log_file = os.path.join(log_dir, f'{config_name}_test.log')

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(__file__)
        )
        process.wait()

    if process.returncode == 0:
        print(f"✓ {config_name} 测试完成")
        return True
    else:
        print(f"✗ {config_name} 测试失败")
        return False


def generate_report():
    """生成对比报告"""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')

    all_results = {}
    for config_name in CLIP_CONFIGS:
        results_file = os.path.join(results_dir, f'{config_name}_results.json')
        if os.path.exists(results_file):
            with open(results_file) as f:
                all_results[config_name] = json.load(f)

    if not all_results:
        print("\n没有找到测试结果")
        return

    # 打印对比表格
    print("\n" + "="*80)
    print("CLIP 模型对比")
    print("="*80)
    print(f"{'模型':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*80)

    for name, result in all_results.items():
        print(f"{name:<15} "
              f"{result['accuracy']*100:>8.2f}%   "
              f"{result['precision']*100:>8.2f}%   "
              f"{result['recall']*100:>8.2f}%   "
              f"{result['f1']*100:>8.2f}%")

    print("="*80)

    # 保存汇总报告
    report_path = os.path.join(results_dir, 'comparison_report.json')
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n汇总报告已保存到: {report_path}")


def main():
    print("="*60)
    print("CLIP 批量训练脚本")
    print("="*60)

    # 检查 GPU
    memory = check_gpu_memory()

    # 配置
    configs = [
        ('clip_vit', 30, 64, 0.0005),   # config, epochs, batch_size, lr
    ]

    # 根据显存调整 batch_size
    if memory < 8:
        print(f"\n警告: 显存不足 8GB，自动降低 batch_size")
        configs = [(c[0], c[1], 32, c[3]) for c in configs]

    # 训练所有模型
    print(f"\n计划训练 {len(configs)} 个模型")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    trained_models = []
    for config_name, epochs, batch_size, lr in configs:
        if train_model(config_name, epochs, batch_size, lr):
            trained_models.append(config_name)

    # 测试所有训练好的模型
    print(f"\n开始测试 {len(trained_models)} 个模型")
    for config_name in trained_models:
        test_model(config_name)

    # 生成对比报告
    generate_report()

    print(f"\n{'='*60}")
    print("批量训练完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
