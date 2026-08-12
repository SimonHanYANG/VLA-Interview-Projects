"""
模型工厂 - 统一创建目标检测模型
支持: Faster R-CNN (torchvision), DETR (transformers), YOLOv5 (ultralytics)
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# VOC 2007: 20 类 + 1 背景
NUM_CLASSES = 21  # 20 object classes + background

# VOC 类别名
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]


def get_supported_models():
    """返回支持的模型列表"""
    return [
        'fasterrcnn_resnet50',
        'fasterrcnn_resnet50_v2',
        'fasterrcnn_mobilenet_v3',
        'detr_resnet50',
        'yolov5s',
        'yolov5m',
        'yolov5l',
    ]


class DETRWrapper(nn.Module):
    """
    DETR 模型包装器
    统一接口以兼容 Faster R-CNN 的训练流程
    """

    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        import os

        # 设置 HuggingFace 中国镜像
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        from transformers import DetrForObjectDetection, DetrImageProcessor

        # 尝试加载预训练模型
        try:
            self.model = DetrForObjectDetection.from_pretrained(
                'facebook/detr-resnet-50' if pretrained else None,
                num_labels=num_classes - 1,  # DETR 不包含背景类
                ignore_mismatched_sizes=True,
            )
            self.processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
            print("[模型] DETR 预训练权重加载成功")
        except Exception as e:
            print(f"[警告] 无法加载 DETR 预训练权重: {e}")
            print("[模型] 使用随机初始化的 DETR 模型")
            # 使用随机初始化
            from transformers import DetrConfig
            config = DetrConfig(num_labels=num_classes - 1)
            self.model = DetrForObjectDetection(config)
            self.processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

        # 用于存储设备信息
        self._device = None

    def to(self, device):
        self._device = device
        self.model = self.model.to(device)
        return self

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, images, targets=None):
        """
        前向传播

        Args:
            images: list of tensors [C, H, W] (已归一化到 [0, 1])
            targets: list of dicts with 'boxes' [N, 4] (xyxy) and 'labels' [N]

        Returns:
            训练时: dict with losses
            推理时: list of dicts with 'boxes', 'labels', 'scores'
        """
        if self.training and targets is not None:
            # 训练模式 - 返回 loss
            return self._forward_train(images, targets)
        else:
            # 推理模式 - 返回预测结果
            return self._forward_eval(images)

    def _forward_train(self, images, targets):
        """训练时的前向传播"""
        # 将图像和目标转换为 DETR 格式
        # DETR 需要 pixel_values 和 labels
        batch_size = len(images)

        # 准备标签 - DETR 需要 class_labels 和 boxes (cxcywh 格式，归一化)
        detr_targets = []
        for target in targets:
            boxes = target['boxes']  # [N, 4] xyxy format
            labels = target['labels']  # [N]

            # 转换为 cxcywh 格式
            boxes_cxcywh = self._xyxy_to_cxcywh(boxes)

            # 获取图像尺寸进行归一化
            # 假设所有图像尺寸相同（已 resize）
            img_h, img_w = images[0].shape[-2:]

            # 归一化到 [0, 1]
            boxes_cxcywh[:, 0] /= img_w  # cx
            boxes_cxcywh[:, 1] /= img_h  # cy
            boxes_cxcywh[:, 2] /= img_w  # w
            boxes_cxcywh[:, 3] /= img_h  # h

            # DETR 使用 0-indexed 标签 (0-19)，VOC 使用 1-indexed (1-20)
            detr_targets.append({
                'class_labels': (labels - 1).to(self._device),
                'boxes': boxes_cxcywh.to(self._device),
            })

        # 堆叠图像
        pixel_values = torch.stack(images).to(self._device)

        # 前向传播
        outputs = self.model(
            pixel_values=pixel_values,
            labels=detr_targets,
        )

        return {'loss': outputs.loss}

    def _forward_eval(self, images):
        """推理时的前向传播"""
        self.model.eval()
        results = []

        with torch.no_grad():
            for image in images:
                # 单张图像处理
                pixel_values = image.unsqueeze(0).to(self._device)

                outputs = self.model(pixel_values=pixel_values)

                # 后处理 - 转换为标准格式
                target_sizes = torch.tensor([image.shape[-2:]]).to(self._device)
                processed = self.processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=0.5,  # 标准阈值
                )[0]

                # 转换为 xyxy 格式
                boxes = processed['boxes'].cpu()
                scores = processed['scores'].cpu()
                labels = processed['labels'].cpu()

                results.append({
                    'boxes': boxes,
                    'labels': labels + 1,  # DETR 标签从 0 开始，加 1 对齐 VOC
                    'scores': scores,
                })

        return results

    def _xyxy_to_cxcywh(self, boxes):
        """将 xyxy 格式转换为 cxcywh 格式"""
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)


class YOLOv5Wrapper(nn.Module):
    """
    YOLOv5 模型包装器
    统一接口以兼容 Faster R-CNN 的训练流程
    注意：YOLOv5 的完整训练建议使用单独的 train_yolo.py 脚本
    """

    def __init__(self, model_size='s', num_classes=21, pretrained=True):
        super().__init__()
        import os

        # 设置 ultralytics 使用中国镜像
        os.environ['ULTRALYTICS_HUB'] = 'https://mirror.ghproxy.com'

        from ultralytics import YOLO

        # 加载预训练模型
        model_name = f'yolov5{model_size}.pt'
        try:
            self.model = YOLO(model_name)
            print(f"[模型] YOLOv5{model_size} 预训练权重加载成功")
        except Exception as e:
            print(f"[警告] 无法加载 YOLOv5 预训练权重: {e}")
            print(f"[模型] 使用 YOLOv5{model_size}u.yaml 构建模型")
            self.model = YOLO(f'yolov5{model_size}u.yaml')

        self.num_classes = num_classes
        self._device = None

    def to(self, device):
        self._device = device
        self.model.to(device)
        return self

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, images, targets=None):
        """
        前向传播

        Args:
            images: list of tensors [C, H, W]
            targets: list of dicts with 'boxes' and 'labels'

        Returns:
            训练时: dict with losses
            推理时: list of dicts with 'boxes', 'labels', 'scores'
        """
        if self.training and targets is not None:
            return self._forward_train(images, targets)
        else:
            return self._forward_eval(images)

    def _forward_train(self, images, targets):
        """训练时的前向传播"""
        # YOLOv5 训练需要特殊处理
        # 这里简化处理，实际使用时可能需要调整
        raise NotImplementedError("YOLOv5 训练需要使用 ultralytics 的训练接口")

    def _forward_eval(self, images):
        """推理时的前向传播"""
        results = []

        for image in images:
            # YOLOv5 推理
            # 需要将 tensor 转换为 numpy 或 PIL
            img_np = image.permute(1, 2, 0).cpu().numpy() * 255
            img_np = img_np.astype('uint8')

            # 推理
            outputs = self.model(img_np, verbose=False)

            # 解析结果
            for output in outputs:
                boxes = output.boxes.xyxy.cpu()
                scores = output.boxes.conf.cpu()
                labels = output.boxes.cls.cpu().int() + 1  # 转换为 1-indexed

                results.append({
                    'boxes': boxes,
                    'labels': labels,
                    'scores': scores,
                })

        return results


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
    # ====== Faster R-CNN 系列 ======
    if model_name == 'fasterrcnn_resnet50':
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn(weights=weights)

    elif model_name == 'fasterrcnn_resnet50_v2':
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    elif model_name == 'fasterrcnn_mobilenet_v3':
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    # ====== DETR 系列 ======
    elif model_name == 'detr_resnet50':
        model = DETRWrapper(num_classes=num_classes, pretrained=pretrained)

    # ====== YOLOv5 系列 ======
    elif model_name in ['yolov5s', 'yolov5m', 'yolov5l']:
        size = model_name[-1]  # s, m, l
        model = YOLOv5Wrapper(model_size=size, num_classes=num_classes, pretrained=pretrained)

    else:
        raise ValueError(
            f"不支持的模型: {model_name}\n"
            f"可选模型: {get_supported_models()}"
        )

    # 替换 Faster R-CNN 的分类头
    if 'fasterrcnn' in model_name and pretrained:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    print(f"[模型] 已创建 {model_name}, 类别数={num_classes}, 预训练={pretrained}")
    return model


def is_torchvision_model(model_name):
    """判断是否是 torchvision 模型（接口兼容）"""
    return 'fasterrcnn' in model_name


def is_detr_model(model_name):
    """判断是否是 DETR 模型"""
    return 'detr' in model_name


def is_yolo_model(model_name):
    """判断是否是 YOLOv5 模型"""
    return 'yolov5' in model_name
