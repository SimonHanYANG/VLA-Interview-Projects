"""
模型工厂 - 统一创建目标检测模型
支持: Faster R-CNN (torchvision)
"""

import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# VOC 2007: 20 类 + 1 背景
NUM_CLASSES = 21  # 20 object classes + background


def get_supported_models():
    """返回支持的模型列表"""
    return [
        'fasterrcnn_resnet50',
        'fasterrcnn_resnet50_v2',
        'fasterrcnn_mobilenet_v3',
    ]


def create_model(model_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    创建目标检测模型

    Args:
        model_name: 模型名称
        num_classes: 类别数（含背景）
        pretrained: 是否使用预训练权重

    Returns:
        PyTorch 检测模型
    """
    if model_name == 'fasterrcnn_resnet50':
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn(weights=weights)

    elif model_name == 'fasterrcnn_resnet50_v2':
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    elif model_name == 'fasterrcnn_mobilenet_v3':
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    else:
        raise ValueError(
            f"不支持的模型: {model_name}\n"
            f"可选模型: {get_supported_models()}"
        )

    # 替换分类头以适配 VOC 类别数
    if pretrained:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    print(f"[模型] 已创建 {model_name}, 类别数={num_classes}, 预训练={pretrained}")
    return model
