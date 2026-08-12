"""
U-Net 模型

核心思想：
1. 对称 Encoder-Decoder 结构：编码器提取特征，解码器恢复分辨率
2. Skip Connection：将编码器的特征拼接到解码器，保留细节信息
3. 多尺度特征融合：不同分辨率的特征图融合

面试考点：
- U-Net 的 Skip Connection 和 ResNet 的区别？
- 为什么 U-Net 在医学图像分割中效果好？
- Encoder-Decoder 结构的优缺点？
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("segmentation_models_pytorch not installed. Using custom U-Net.")


def get_unet_model(num_classes=21, pretrained=True):
    """
    获取 U-Net 模型

    Args:
        num_classes: 类别数（VOC 2012: 21）
        pretrained: 是否使用预训练编码器

    Returns:
        model: U-Net 模型

    模型结构：
    - Encoder: ResNet34（特征提取）
    - Decoder: 转置卷积 + Skip Connection
    - 分类头: 1x1 卷积

    输出格式：
    - [B, num_classes, H, W]
    """
    if SMP_AVAILABLE:
        # 使用 segmentation_models_pytorch 的 U-Net
        encoder_name = "resnet34"
        encoder_weights = "imagenet" if pretrained else None

        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
    else:
        # 使用自定义 U-Net
        model = UNet(in_channels=3, out_channels=num_classes)

    return model


class DoubleConv(nn.Module):
    """
    U-Net 双层卷积模块
    结构：Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    自定义 U-Net 实现

    结构：
    - Encoder: 4 次下采样（MaxPool + DoubleConv）
    - Bottleneck: 最底层特征提取
    - Decoder: 4 次上采样（转置卷积 + Skip Connection + DoubleConv）
    - 输出层: 1x1 卷积
    """

    def __init__(self, in_channels=3, out_channels=21):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # 输出层
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # [B, 512, H/8, W/8]

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # [B, 1024, H/16, W/16]

        # Decoder with Skip Connections
        d4 = self.upconv4(b)  # [B, 512, H/8, W/8]
        # 处理尺寸不匹配（输入尺寸不是 16 的倍数时）
        if d4.shape != e4.shape:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode="bilinear", align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)  # [B, 1024, H/8, W/8]
        d4 = self.dec4(d4)    # [B, 512, H/8, W/8]

        d3 = self.upconv3(d4)  # [B, 256, H/4, W/4]
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)  # [B, 512, H/4, W/4]
        d3 = self.dec3(d3)    # [B, 256, H/4, W/4]

        d2 = self.upconv2(d3)  # [B, 128, H/2, W/2]
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)  # [B, 256, H/2, W/2]
        d2 = self.dec2(d2)    # [B, 128, H/2, W/2]

        d1 = self.upconv1(d2)  # [B, 64, H, W]
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)  # [B, 128, H, W]
        d1 = self.dec1(d1)    # [B, 64, H, W]

        # 输出
        return self.out_conv(d1)  # [B, num_classes, H, W]


if __name__ == "__main__":
    # 测试模型
    model = get_unet_model(num_classes=21, pretrained=False)
    print(f"U-Net 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 520, 520)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
