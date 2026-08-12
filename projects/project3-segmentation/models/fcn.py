"""
FCN (Fully Convolutional Network) 模型

核心思想：
1. 全卷积：将分类网络的全连接层替换为 1x1 卷积
2. 转置卷积：通过转置卷积（反卷积）进行上采样
3. Skip Connection：融合不同层的特征，保留细节信息

面试考点：
- 为什么需要全卷积？（支持任意输入尺寸）
- 转置卷积和上采样的区别？
- Skip Connection 的作用？
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_fcn_model(num_classes=21, pretrained=True):
    """
    获取 FCN ResNet50 模型

    Args:
        num_classes: 类别数（VOC 2012: 21）
        pretrained: 是否使用预训练权重

    Returns:
        model: torchvision.models.segmentation.fcn_resnet50

    模型结构：
    - Backbone: ResNet50（特征提取）
    - Head: FCN 分类头（1x1 卷积 + 转置卷积上采样）

    输出格式：
    - 训练模式：{"out": [B, num_classes, H, W], "aux": [B, num_classes, H, W]}
    - 推理模式：{"out": [B, num_classes, H, W]}
    """
    weights = models.segmentation.FCN_ResNet50_Weights.DEFAULT if pretrained else None
    model = models.segmentation.fcn_resnet50(weights=weights)

    # 修改分类头以匹配类别数
    if num_classes != 21:
        # 修改主分类头
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

        # 修改辅助分类头
        if model.aux_classifier is not None:
            model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    return model


class FCNHead(nn.Module):
    """
    FCN 分类头
    用于将 backbone 的特征映射到类别数

    结构：
    - Conv 3x3 -> BN -> ReLU -> Dropout -> Conv 1x1
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.head(x)


if __name__ == "__main__":
    # 测试模型
    model = get_fcn_model(num_classes=21, pretrained=False)
    print(f"FCN ResNet50 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 520, 520)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output['out'].shape}")
