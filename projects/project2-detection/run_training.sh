#!/bin/bash
# 批量训练脚本 - 3 个模型: fasterrcnn_resnet50_v2, detr_resnet50, yolov5m
# 使用 GPU 0

set -e

cd /root/VLA-Inter/projects/project2-detection
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "  正式训练开始 - $(date)"
echo "=========================================="

# 1. Faster R-CNN ResNet50 V2
echo ""
echo "[1/3] 训练 Faster R-CNN ResNet50 V2..."
echo "开始时间: $(date)"
conda run -n vla python train.py --model fasterrcnn_resnet50_v2 --epochs 20 --batch_size 4 --lr 0.005 --num_workers 2 2>&1 | tee logs/fasterrcnn_resnet50_v2_train.log
echo "[1/3] Faster R-CNN V2 训练完成 - $(date)"

# 测试
echo "[1/3] 测试 Faster R-CNN ResNet50 V2..."
conda run -n vla python test.py --model fasterrcnn_resnet50_v2 --batch_size 4 --num_workers 2 2>&1 | tee logs/fasterrcnn_resnet50_v2_test.log
echo "[1/3] Faster R-CNN V2 测试完成 - $(date)"

# 2. DETR ResNet50
echo ""
echo "[2/3] 训练 DETR ResNet50..."
echo "开始时间: $(date)"
conda run -n vla python train.py --model detr_resnet50 --epochs 20 --batch_size 4 --lr 0.0001 --num_workers 2 2>&1 | tee logs/detr_resnet50_train.log
echo "[2/3] DETR 训练完成 - $(date)"

# 测试
echo "[2/3] 测试 DETR ResNet50..."
conda run -n vla python test.py --model detr_resnet50 --batch_size 4 --num_workers 2 2>&1 | tee logs/detr_resnet50_test.log
echo "[2/3] DETR 测试完成 - $(date)"

# 3. YOLOv5m
echo ""
echo "[3/3] 训练 YOLOv5m..."
echo "开始时间: $(date)"
conda run -n vla python train_yolo.py --model m --epochs 20 --batch_size 16 --device 0 --num_workers 2 2>&1 | tee logs/yolov5m_train.log
echo "[3/3] YOLOv5m 训练完成 - $(date)"

# 测试
echo "[3/3] 测试 YOLOv5m..."
conda run -n vla python test_yolo.py --model m --batch_size 16 --device 0 --num_workers 2 2>&1 | tee logs/yolov5m_test.log
echo "[3/3] YOLOv5m 测试完成 - $(date)"

# 生成对比报告
echo ""
echo "生成对比报告..."
conda run -n vla python -c "
from run_all import generate_comparison_report
generate_comparison_report()
"

echo ""
echo "=========================================="
echo "  全部训练完成 - $(date)"
echo "=========================================="
