"""
YOLOv5 专用训练脚本
使用 ultralytics 库进行训练
用法: python train_yolo.py --model yolov5s --epochs 20 --batch_size 16
"""

import os
import sys
import argparse
from pathlib import Path

# 设置中国镜像
os.environ['ULTRALYTICS_HUB'] = 'https://mirror.ghproxy.com'

from ultralytics import YOLO


def prepare_voc_dataset(data_root='./data'):
    """
    准备 VOC 数据集配置文件（YOLO 格式）

    YOLOv5 需要特定的数据集格式，这里创建配置文件
    """
    # VOC 类别
    voc_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
    ]

    voc_root = os.path.join(os.path.abspath(data_root), 'VOCdevkit', 'VOC2007')

    # 创建数据集配置文件
    config_content = f"""# VOC 2007 Dataset Configuration for YOLOv5
# 用法: python train_yolo.py --data voc.yaml

path: {voc_root}  # 数据集根目录
train: train.txt  # 训练集文件列表
val: val.txt  # 验证集文件列表
test: test.txt  # 测试集文件列表（可选）

# 类别数量
nc: {len(voc_classes)}

# 类别名称
names: {voc_classes}
"""

    config_path = os.path.join(data_root, 'voc.yaml')
    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"[数据] 数据集配置已保存至: {config_path}")
    return config_path


def convert_voc_to_yolo(data_root='./data'):
    """
    将 VOC 格式转换为 YOLO 格式

    VOC 格式: XML 文件 (xmin, ymin, xmax, ymax)
    YOLO 格式: TXT 文件 (class_id, x_center, y_center, width, height) 归一化
    """
    import xml.etree.ElementTree as ET
    from pathlib import Path

    voc_root = Path(data_root) / 'VOCdevkit' / 'VOC2007'
    labels_dir = voc_root / 'labels'
    labels_dir.mkdir(exist_ok=True)

    # VOC 类别
    voc_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
    ]
    class_to_idx = {cls: idx for idx, cls in enumerate(voc_classes)}

    # 获取所有标注文件
    annotations_dir = voc_root / 'Annotations'
    image_files = list((voc_root / 'JPEGImages').glob('*.jpg'))

    print(f"[数据] 转换 VOC 标注为 YOLO 格式...")
    print(f"[数据] 找到 {len(image_files)} 张图像")

    converted = 0
    for img_path in image_files:
        # 对应的标注文件
        xml_path = annotations_dir / f'{img_path.stem}.xml'

        if not xml_path.exists():
            continue

        # 解析 XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        # 转换标注
        yolo_lines = []
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in class_to_idx:
                continue

            cls_id = class_to_idx[cls_name]
            bbox = obj.find('bndbox')

            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # 转换为 YOLO 格式 (归一化的中心点坐标和宽高)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 保存 YOLO 格式标注
        label_path = labels_dir / f'{img_path.stem}.txt'
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        converted += 1

    print(f"[数据] 转换完成: {converted} 个标注文件")

    # 创建 YOLO 格式的数据集文件（包含完整路径）
    for split in ['train', 'val', 'test']:
        src_file = voc_root / 'ImageSets' / 'Main' / f'{split}.txt'
        if src_file.exists():
            # 创建 YOLO 格式的数据集文件
            dst_file = voc_root / f'{split}.txt'
            with open(src_file, 'r') as f:
                image_ids = f.read().strip().split('\n')

            with open(dst_file, 'w') as f:
                for img_id in image_ids:
                    img_path = voc_root / 'images' / f'{img_id}.jpg'
                    if (voc_root / 'JPEGImages' / f'{img_id}.jpg').exists():
                        # 创建 images 目录的符号链接
                        images_dir = voc_root / 'images'
                        images_dir.mkdir(exist_ok=True)
                        src_img = voc_root / 'JPEGImages' / f'{img_id}.jpg'
                        dst_img = images_dir / f'{img_id}.jpg'
                        if not dst_img.exists():
                            dst_img.symlink_to(src_img.absolute())
                        f.write(f'{dst_img.absolute()}\n')

            print(f"[数据] {split} 集: {len(image_ids)} 张图像")


def train(args):
    """YOLOv5 训练主函数"""
    print("=" * 60)
    print("  YOLOv5 目标检测训练")
    print("=" * 60)
    print(f"  模型:       yolov5{args.model}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  图像尺寸:   {args.img_size}")
    print("=" * 60)

    # 准备数据集
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    # 转换 VOC 标注为 YOLO 格式
    convert_voc_to_yolo(data_root)

    # 创建数据集配置文件
    data_config = prepare_voc_dataset(data_root)

    # 加载预训练模型
    model_name = f'yolov5{args.model}.pt'
    print(f"\n[模型] 加载 {model_name}...")

    try:
        model = YOLO(model_name)
        print(f"[模型] 预训练权重加载成功")
    except Exception as e:
        print(f"[警告] 无法加载预训练权重: {e}")
        print(f"[模型] 使用 yolov5{args.model}u.yaml 构建模型")
        model = YOLO(f'yolov5{args.model}u.yaml')

    # 开始训练
    print(f"\n[训练] 开始训练...")
    results = model.train(
        data=data_config,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        name=f'yolov5{args.model}_voc',
        patience=args.patience,
        device=args.device,
        workers=args.num_workers,
        project='runs/train',
        exist_ok=True,
        pretrained=True,
        optimizer='SGD',
        lr0=args.lr,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=0.05,
        cls=0.5,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )

    print("\n" + "=" * 60)
    print(f"  训练完成！")
    print(f"  模型: yolov5{args.model}")
    print(f"  结果保存在: runs/train/yolov5{args.model}_voc/")
    print("=" * 60)

    return results


def test(args):
    """YOLOv5 测试主函数"""
    print("=" * 60)
    print("  YOLOv5 目标检测测试")
    print("=" * 60)

    # 加载训练好的模型
    model_path = f'runs/train/yolov5{args.model}_voc/weights/best.pt'

    if not os.path.exists(model_path):
        print(f"[错误] 未找到训练好的模型: {model_path}")
        print(f"[提示] 请先运行训练: python train_yolo.py --model {args.model} --epochs 20")
        return None

    model = YOLO(model_path)

    # 数据集配置
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    data_config = os.path.join(data_root, 'voc.yaml')

    # 运行测试
    print(f"\n[测试] 评估模型...")
    results = model.val(
        data=data_config,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.num_workers,
        project='runs/val',
        name=f'yolov5{args.model}_voc',
        exist_ok=True,
    )

    # 打印结果
    print("\n" + "=" * 60)
    print(f"  测试结果")
    print("=" * 60)
    print(f"  mAP@0.5:     {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"  Precision:   {results.box.mp:.4f}")
    print(f"  Recall:      {results.box.mr:.4f}")
    print("=" * 60)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv5 目标检测训练/测试')

    parser.add_argument('--model', type=str, default='s',
                        choices=['s', 'm', 'l', 'x'],
                        help='YOLOv5 模型大小 (s/m/l/x)')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--img_size', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--lr', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--patience', type=int, default=100, help='早停耐心值')
    parser.add_argument('--device', type=str, default='', help='设备 (cpu/0/0,1/...)')
    parser.add_argument('--test', action='store_true', help='测试模式')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.test:
        test(args)
    else:
        train(args)
