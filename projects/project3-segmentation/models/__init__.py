"""
Project 3: 图像分割模型
"""

from .fcn import get_fcn_model
from .deeplabv3 import get_deeplabv3_model
from .unet import get_unet_model
from .mask_rcnn import get_mask_rcnn_model


def get_model(model_name, num_classes=21, pretrained=True):
    """
    统一的模型获取接口

    Args:
        model_name: 模型名称 ("fcn", "deeplabv3", "unet", "mask_rcnn")
        num_classes: 类别数
        pretrained: 是否使用预训练权重

    Returns:
        model: PyTorch 模型
    """
    model_getters = {
        "fcn": get_fcn_model,
        "deeplabv3": get_deeplabv3_model,
        "unet": get_unet_model,
        "mask_rcnn": get_mask_rcnn_model,
    }

    if model_name not in model_getters:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_getters.keys())}")

    return model_getters[model_name](num_classes=num_classes, pretrained=pretrained)


# 导出所有模型获取函数
__all__ = [
    "get_model",
    "get_fcn_model",
    "get_deeplabv3_model",
    "get_unet_model",
    "get_mask_rcnn_model",
]
