"""
CLIP 模型测试脚本
使用零样本分类评估 CLIP 模型在 CIFAR-10 上的性能

评估方式：零样本分类 (Zero-shot Classification)
1. 为每个类别创建文本嵌入
2. 计算图像嵌入与所有类别文本嵌入的相似度
3. 选择相似度最高的类别作为预测
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import os
import json
import argparse
from tqdm import tqdm

from train import create_clip_model, CLIP_CONFIGS, CLIPTokenizer, get_cifar10_text_descriptions


def load_clip_model(config_name, weights_path):
    """加载训练好的 CLIP 模型"""
    print(f"\n加载模型: {weights_path}")

    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

    model = create_clip_model(config_name)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"模型来自 Epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Val Loss: {checkpoint.get('val_loss', 'unknown'):.4f}")

    return model


def create_class_text_embeddings(model, tokenizer, class_names, descriptions, device):
    """
    为每个类别创建文本嵌入
    使用每个类别的多个描述，取平均嵌入
    """
    model.eval()
    class_embeddings = {}

    with torch.no_grad():
        for class_idx, name in enumerate(class_names):
            # 获取该类别的所有描述
            class_descriptions = descriptions[class_idx]

            # 编码所有描述
            text_tokens = tokenizer(class_descriptions)['input_ids'].to(device)
            text_features = model.text_encoder(text_tokens)  # [num_descriptions, embed_dim]

            # 取平均嵌入
            class_embeddings[class_idx] = text_features.mean(dim=0, keepdim=True)  # [1, embed_dim]

    # Stack all class embeddings
    all_embeddings = torch.cat([class_embeddings[i] for i in range(len(class_names))], dim=0)
    all_embeddings = F.normalize(all_embeddings, dim=-1)

    return all_embeddings


def zero_shot_classify(model, images, class_text_embeddings):
    """
    零样本分类
    Args:
        model: CLIP 模型
        images: [batch_size, 3, 224, 224]
        class_text_embeddings: [num_classes, embed_dim]
    Returns:
        predictions: [batch_size]
        similarities: [batch_size, num_classes]
    """
    model.eval()
    with torch.no_grad():
        # 编码图像
        image_features = model.image_encoder(images)  # [B, embed_dim]

        # 计算相似度
        logit_scale = model.logit_scale.exp()
        similarities = logit_scale * image_features @ class_text_embeddings.T  # [B, num_classes]

        # 获取预测
        predictions = similarities.argmax(dim=-1)

    return predictions, similarities


def evaluate_clip(config_name='clip_vit', weights_path=None, batch_size=64):
    """评估 CLIP 模型"""

    config = CLIP_CONFIGS[config_name]
    print(f"\n{'='*60}")
    print(f"评估 {config['desc']}")
    print(f"{'='*60}")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载模型
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(__file__), 'models', f'{config_name}_best.pth')

    if not os.path.exists(weights_path):
        print(f"错误: 模型文件不存在 {weights_path}")
        print("请先运行训练: python train.py --model clip_vit")
        return

    model = load_clip_model(config_name, weights_path)
    model = model.to(device)
    model.eval()

    # Tokenizer
    tokenizer = CLIPTokenizer()

    # CIFAR-10 类别信息
    class_names, descriptions = get_cifar10_text_descriptions()

    # 创建类别文本嵌入
    print("\n创建类别文本嵌入...")
    class_text_embeddings = create_class_text_embeddings(
        model, tokenizer, class_names, descriptions, device
    )
    print(f"类别嵌入形状: {class_text_embeddings.shape}")

    # 数据预处理
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

    # 加载测试集
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    print(f"\n测试集: {len(test_dataset)} 样本")

    # 零样本分类评估
    print("\n进行零样本分类...")
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="评估"):
            images = images.to(device, non_blocking=True)

            predictions, _ = zero_shot_classify(model, images, class_text_embeddings)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )

    # 详细分类报告
    report = classification_report(all_labels, all_predictions, target_names=class_names)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)

    print(f"\n{'='*60}")
    print(f"测试结果")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print(f"\n分类报告:")
    print(report)

    # 保存结果
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'CLIP Zero-shot Classification\nAccuracy: {accuracy*100:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{config_name}_confusion_matrix.png'), dpi=150)
    plt.close()

    # 绘制每个类别的准确率
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), class_accuracies * 100)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-class Accuracy - CLIP Zero-shot')
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.ylim(0, 100)

    # 添加数值标签
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{config_name}_per_class_accuracy.png'), dpi=150)
    plt.close()

    # 保存结果到 JSON
    results = {
        'model': config_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'per_class_accuracy': {name: float(acc) for name, acc in zip(class_names, class_accuracies)},
    }

    results_path = os.path.join(results_dir, f'{config_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {results_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description='CLIP Evaluation')
    parser.add_argument('--model', type=str, default='clip_vit', help='Model config name')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    args = parser.parse_args()

    if args.model not in CLIP_CONFIGS:
        print(f"错误: 未知模型 {args.model}")
        print(f"可用模型: {list(CLIP_CONFIGS.keys())}")
        return

    results = evaluate_clip(
        config_name=args.model,
        weights_path=args.weights,
        batch_size=args.batch_size
    )

    if results:
        print(f"\n最终结果:")
        print(f"  Accuracy: {results['accuracy']*100:.2f}%")
        print(f"  F1 Score: {results['f1']*100:.2f}%")


if __name__ == '__main__':
    main()
