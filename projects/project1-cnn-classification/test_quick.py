"""
项目1: 快速测试脚本
=====================================

功能：
- 使用小数据集快速验证代码是否可以正常运行
- 测试训练和测试流程
- 不会花费太多时间

使用方法：
    python test_quick.py
"""

import os
import sys
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import create_model, get_transforms, train_one_epoch, validate
from test import calculate_metrics, CIFAR10_CLASSES


def quick_test():
    """
    快速测试整个流程

    测试内容：
    1. 模型创建
    2. 数据加载
    3. 训练一个 epoch
    4. 测试评估
    5. 性能指标计算
    """
    print("\n" + "=" * 60)
    print("项目1: CNN 分类 - 快速测试")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 1. 测试模型创建
    print("\n[1/5] 测试模型创建...")
    models_to_test = ['alexnet', 'vgg16', 'googlenet', 'resnet18', 'efficientnet_b0']

    for model_name in models_to_test:
        try:
            model = create_model(model_name, num_classes=10)
            model = model.to(device)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  ✓ {model_name}: 参数数量 = {param_count:,}")
        except Exception as e:
            print(f"  ✗ {model_name}: 错误 - {e}")
            return False

    # 2. 测试数据加载
    print("\n[2/5] 测试数据加载...")
    train_transform, test_transform = get_transforms()

    try:
        # 只下载一次
        full_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        print(f"  ✓ 训练数据集大小: {len(full_dataset)}")

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform
        )
        print(f"  ✓ 测试数据集大小: {len(test_dataset)}")

        # 使用小子集进行快速测试
        train_subset = Subset(full_dataset, range(1000))
        test_subset = Subset(test_dataset, range(200))

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=0)

        print(f"  ✓ 训练子集大小: {len(train_subset)}")
        print(f"  ✓ 测试子集大小: {len(test_subset)}")
    except Exception as e:
        print(f"  ✗ 数据加载失败: {e}")
        return False

    # 3. 测试训练
    print("\n[3/5] 测试训练流程...")
    model = create_model('resnet18', num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 创建 TensorBoard writer（可选，测试时不写入）
    class DummyWriter:
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass

    dummy_writer = DummyWriter()

    try:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 0, dummy_writer
        )
        print(f"  ✓ 训练成功: loss={train_loss:.4f}, acc={train_acc:.2f}%")
    except Exception as e:
        print(f"  ✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 测试验证
    print("\n[4/5] 测试验证流程...")
    try:
        val_loss, val_acc = validate(
            model, test_loader, criterion, device, 0, dummy_writer
        )
        print(f"  ✓ 验证成功: loss={val_loss:.4f}, acc={val_acc:.2f}%")
    except Exception as e:
        print(f"  ✗ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. 测试性能指标计算
    print("\n[5/5] 测试性能指标计算...")
    try:
        # 模拟一些预测结果
        predictions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 20
        targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 20
        # 添加一些错误预测
        predictions[5] = 0  # 错误预测
        predictions[15] = 2  # 错误预测

        import numpy as np
        predictions = np.array(predictions)
        targets = np.array(targets)

        # 创建模拟概率
        probabilities = np.zeros((len(predictions), 10))
        for i, p in enumerate(predictions):
            probabilities[i][p] = 0.9
            for j in range(10):
                if j != p:
                    probabilities[i][j] = 0.01

        metrics = calculate_metrics(predictions, targets, probabilities)

        print(f"  ✓ 指标计算成功:")
        print(f"    - Accuracy: {metrics['accuracy']:.4f}")
        print(f"    - Precision: {metrics['precision_macro']:.4f}")
        print(f"    - Recall: {metrics['recall_macro']:.4f}")
        print(f"    - F1 Score: {metrics['f1_macro']:.4f}")
        print(f"    - 混淆矩阵形状: {len(metrics['confusion_matrix'])}x{len(metrics['confusion_matrix'][0])}")
    except Exception as e:
        print(f"  ✗ 指标计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. 测试模型保存和加载
    print("\n[6/6] 测试模型保存和加载...")
    try:
        # 保存模型
        os.makedirs('models', exist_ok=True)
        model_path = 'models/test_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)

        # 加载模型
        checkpoint = torch.load(model_path)
        model2 = create_model('resnet18', num_classes=10)
        model2.load_state_dict(checkpoint['model_state_dict'])

        print(f"  ✓ 模型保存成功: {model_path}")
        print(f"  ✓ 模型加载成功")

        # 清理测试文件
        os.remove(model_path)
    except Exception as e:
        print(f"  ✗ 模型保存/加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 完成
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    print("\n项目代码可以正常运行。")
    print("可以开始正式训练了：")
    print("  python train.py --model resnet18 --epochs 20")
    print("=" * 60)

    return True


if __name__ == '__main__':
    success = quick_test()
    sys.exit(0 if success else 1)
