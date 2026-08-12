"""
DeepLabV3 模型

核心思想：
1. 空洞卷积 (Atrous/Dilated Convolution)：在不增加参数的情况下扩大感受野
2. ASPP (Atrous Spatial Pyramid Pooling)：多尺度特征融合
3. 多尺度上下文：捕获不同尺度的语义信息

面试考点：
- 空洞卷积的原理和优势？
- ASPP 模块的作用？
- DeepLabV3 和 V3+ 的区别？
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_deeplabv3_model(num_classes=21, pretrained=True):
    """
    获取 DeepLabV3 ResNet50 模型

    Args:
        num_classes: 类别数（VOC 2012: 21）
        pretrained: 是否使用预训练权重

    Returns:
        model: torchvision.models.segmentation.deeplabv3_resnet50

    模型结构：
    - Backbone: ResNet50（使用空洞卷积）
    - ASPP: 多尺度特征融合
    - 分类头: 1x1 卷积

    输出格式：
    - 训练模式：{"out": [B, num_classes, H, W], "aux": [B, num_classes, H, W]}
    - 推理模式：{"out": [B, num_classes, H, W]}
    """
    weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = models.segmentation.deeplabv3_resnet50(weights=weights)

    # 修改分类头以匹配类别数
    if num_classes != 21:
        # 修改主分类头
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

        # 修改辅助分类头
        if model.aux_classifier is not None:
            model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    return model


class ASPPConv(nn.Module):
    """
    ASPP 空洞卷积模块
    使用不同扩张率的空洞卷积捕获多尺度信息
    """

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                     dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ASPPPooling(nn.Module):
    """
    ASPP 全局池化模块
    使用全局平均池化捕获全局上下文
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        x = self.pool(x)
        return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)


if __name__ == "__main__":
    # 测试模型
    model = get_deeplabv3_model(num_classes=21, pretrained=False)
    print(f"DeepLabV3 ResNet50 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 520, 520)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output['out'].shape}")
