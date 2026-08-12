"""
Mask R-CNN 模型

核心思想：
1. 两阶段实例分割：先检测物体，再对每个物体分割
2. RoI Align：精确的特征对齐，避免量化误差
3. 并行 Mask 分支：与分类和回归分支并行预测 mask

面试考点：
- RoI Align 和 RoI Pooling 的区别？
- Mask R-CNN 如何实现实例分割？
- 语义分割和实例分割的区别？
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_mask_rcnn_model(num_classes=21, pretrained=True):
    """
    获取 Mask R-CNN ResNet50 FPN 模型

    Args:
        num_classes: 类别数（VOC 2012: 21，包含背景）
        pretrained: 是否使用预训练权重

    Returns:
        model: torchvision.models.detection.maskrcnn_resnet50_fpn

    模型结构：
    - Backbone: ResNet50 + FPN（特征金字塔网络）
    - RPN: 区域提议网络
    - RoI Align: 特征对齐
    - Box Head: 分类 + 边界框回归
    - Mask Head: 实例 mask 预测

    输入格式：
    - 训练：[{"image": [3, H, W], "boxes": [N, 4], "labels": [N], "masks": [N, H, W]}]
    - 推理：[{"image": [3, H, W]}]

    输出格式：
    - 训练：{"loss_classifier": ..., "loss_box_reg": ..., "loss_mask": ..., "loss_objectness": ...}
    - 推理：[{"boxes": [N, 4], "labels": [N], "scores": [N], "masks": [N, 1, H, W]}]
    """
    weights = models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = models.detection.maskrcnn_resnet50_fpn(weights=weights)

    # 修改类别数（包含背景）
    if num_classes != 91:  # COCO 默认 91 类
        # 获取原始 box predictor 的参数
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # 替换 box predictor
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )

        # 替换 mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

    return model


if __name__ == "__main__":
    # 测试模型
    model = get_mask_rcnn_model(num_classes=21, pretrained=False)
    print(f"Mask R-CNN 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试前向传播（训练模式）
    model.train()
    images = [torch.randn(3, 520, 520)]
    targets = [{
        "boxes": torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "masks": torch.randint(0, 2, (1, 520, 520), dtype=torch.uint8)
    }]

    loss_dict = model(images, targets)
    print(f"训练损失: {loss_dict}")

    # 测试前向传播（推理模式）
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    print(f"预测结果: {predictions[0].keys()}")
    print(f"预测框形状: {predictions[0]['boxes'].shape}")
    print(f"预测 mask 形状: {predictions[0]['masks'].shape}")
