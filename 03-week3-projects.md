# 第三周：动手实践项目设计

## 目标
1. 跑通核心模型的训练-测试流程，深入理解每个模型的核心代码
2. 在统一框架中对比不同模型，建立直觉
3. 完成 VLA/IL/RL 的端到端仿真项目，面试时能说"我跑过这个实验"
4. 所有项目使用开源代码和数据集，不从头手写模型

---

## 项目总览

| # | 项目 | 模型 | 框架 | 数据集 | 预计时间 |
|---|------|------|------|--------|----------|
| 1 | CNN 分类 | AlexNet/VGG/GoogLeNet/ResNet/EfficientNet | torchvision | CIFAR-10 子集 | 1 天 |
| 2 | 目标检测 | Faster R-CNN/YOLO/DETR | torchvision + 官方代码 | VOC 2007 子集 | 1.5 天 |
| 3 | 图像分割 | FCN/DeepLab/U-Net/Mask R-CNN/SAM | torchvision + 官方代码 | VOC 2012 子集 | 1.5 天 |
| 4 | 独立模型 | Swin/CLIP/DINO/DDPM/DDIM/MVSNet/Mip-NeRF/Mip-Splatting | 各自官方代码 | 各自数据集 | 2 天 |
| 5 | Diffusion Policy | Diffusion Policy | LeRobot | ALOHA 仿真 | 1 天 |
| 6 | ACT | ACT | LeRobot | ALOHA 仿真 | 1 天 |
| 7 | RL 对比 | DQN/PPO/SAC/TD3 | robosuite + SB3 | Lift 任务 | 1 天 |
| 8 | OpenVLA 微调 | OpenVLA-7B | 官方代码 | LIBERO 仿真 | 1.5 天 |
| 9 | Octo 微调 | Octo | 官方代码 | LIBERO 仿真 | 1.5 天 |

---

## 项目 1：CNN 图像分类统一框架

### 目标
用同一个 PyTorch 框架，跑通所有经典 CNN 分类模型（1.1 节），对比理解每个模型的核心代码。

### 技术方案

**框架**：PyTorch + torchvision.models
**数据集**：CIFAR-10（10 类，50K 训练 + 10K 测试，用子集快速跑通）

### 项目结构

```
project1-cnn-classification/
├── config.py                # 统一配置（学习率、batch size、epoch 等）
├── dataset.py               # CIFAR-10 数据加载（支持子集采样）
├── train.py                 # 统一训练脚本
├── evaluate.py              # 统一评估脚本
├── models/
│   ├── __init__.py
│   ├── alexnet.py           # AlexNet（torchvision.models.alexnet）
│   ├── vgg.py               # VGG-16（torchvision.models.vgg16）
│   ├── googlenet.py         # GoogLeNet（torchvision.models.googlenet）
│   ├── resnet.py            # ResNet-18/50（torchvision.models.resnet18/50）
│   └── efficientnet.py      # EfficientNet-B0（torchvision.models.efficientnet_b0）
├── utils/
│   ├── metrics.py           # accuracy 计算、混淆矩阵
│   └── visualization.py     # loss 曲线、accuracy 曲线
├── results/                 # 每个模型的训练结果
│   ├── alexnet/
│   ├── vgg/
│   ├── googlenet/
│   ├── resnet/
│   └── efficientnet/
└── compare.py               # 所有模型的对比脚本（参数量、accuracy、训练时间）
```

### 每个模型的核心关注点

| 模型 | 核心代码关注点 | 面试考点 |
|------|--------------|----------|
| AlexNet | ReLU、Dropout、GPU 并行 | 为什么 ReLU 比 sigmoid 好 |
| VGG-16 | 3×3 小卷积核堆叠、Block 设计 | 为什么小卷积核更好 |
| GoogLeNet | Inception 模块、1×1 卷积、GAP | 1×1 卷积的作用 |
| ResNet | 残差连接、Bottleneck Block | 残差连接为什么能训练深网络 |
| EfficientNet | MBConv、SE 模块、复合缩放 | 复合缩放的核心思想 |

### 统一训练脚本逻辑

```python
# train.py 核心逻辑（伪代码）
def train(model_name, config):
    # 1. 加载模型（从 torchvision.models）
    model = get_model(model_name, num_classes=10)

    # 2. 加载 CIFAR-10 数据（支持子集）
    train_loader, val_loader = get_dataloaders(
        config.batch_size, subset_ratio=config.subset_ratio
    )

    # 3. 训练循环
    optimizer = Adam(model.parameters(), lr=config.lr)
    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)
        log_metrics(epoch, train_loss, train_acc, val_loss, val_acc)

    # 4. 保存结果
    save_results(model_name, train_history, model)
```

### 输出物
- 每个模型的 train/val loss 和 accuracy 曲线
- 模型参数量和 FLOPs 对比表
- 训练时间对比
- 核心模块的代码注释（在 models/ 目录中）

### 面试时能讲的点
- "我用统一框架跑通了 5 个经典 CNN 模型，对比了它们的参数量、accuracy 和训练效率"
- "我理解了每个模型的核心创新：ResNet 的残差连接、GoogLeNet 的 Inception 模块、EfficientNet 的复合缩放"
- "我可以手写 ResNet 的残差块和 GoogLeNet 的 Inception 模块"

---

## 项目 2：目标检测统一框架

### 目标
用统一的评估流程，跑通经典目标检测模型（1.2 节），理解每个模型的核心代码。

### 技术方案

**框架**：torchvision + ultralytics（YOLO）+ 官方 DETR repo
**数据集**：Pascal VOC 2007（20 类，用子集快速跑通）
**评估指标**：mAP@0.5（统一评估标准）

### 项目结构

```
project2-object-detection/
├── config.py
├── dataset.py               # VOC 数据加载（支持子集）
├── evaluate.py              # 统一 mAP 评估脚本
├── models/
│   ├── faster_rcnn.py       # torchvision.models.detection.fasterrcnn_resnet50_fpn
│   ├── yolov8.py            # ultralytics.YOLO('yolov8n.pt')
│   └── detr.py              # facebook/detr 官方代码
├── train_faster_rcnn.py     # Faster R-CNN 训练脚本
├── train_yolov8.py          # YOLOv8 训练脚本
├── train_detr.py            # DETR 训练脚本
├── compare.py               # 所有模型的 mAP 对比
└── results/
```

### 每个模型的核心关注点

| 模型 | 代码来源 | 核心关注点 |
|------|---------|-----------|
| Faster R-CNN | torchvision.models.detection | RPN、Anchor、RoI Pooling |
| YOLOv8 | ultralytics 官方 | 单阶段检测、Anchor-Free、多尺度 |
| DETR | facebook/detr 官方 | Object Query、匈牙利匹配、Transformer |

### 统一评估脚本

```python
# evaluate.py 核心逻辑
def evaluate_detection(model, val_loader, iou_threshold=0.5):
    """
    统一评估所有检测模型的 mAP
    - Faster R-CNN: 直接用 torchvision 的评估
    - YOLO: 转换为统一格式后评估
    - DETR: 转换为统一格式后评估
    """
    all_predictions = []
    all_ground_truths = []

    for images, targets in val_loader:
        preds = model(images)
        all_predictions.extend(preds)
        all_ground_truths.extend(targets)

    mAP = compute_map(all_predictions, all_ground_truths, iou_threshold)
    return mAP
```

### 输出物
- 每个模型的 mAP@0.5 对比
- 推理速度（FPS）对比
- 检测结果可视化（在测试图像上画框）
- 核心模块代码注释

### 面试时能讲的点
- "我用统一评估流程对比了 Faster R-CNN、YOLOv8 和 DETR"
- "我理解了两阶段和单阶段检测的核心区别"
- "我理解了 DETR 的匈牙利匹配是怎么工作的"

---

## 项目 3：图像分割统一框架

### 目标
跑通经典图像分割模型（1.3 节），理解语义分割、实例分割的区别。

### 技术方案

**框架**：torchvision + segmentation_models_pytorch + 官方 SAM
**数据集**：Pascal VOC 2012（语义分割，21 类，用子集）

### 项目结构

```
project3-segmentation/
├── config.py
├── dataset.py               # VOC 分割数据加载
├── evaluate.py              # 统一 mIoU 评估
├── models/
│   ├── fcn.py               # torchvision.models.segmentation.fcn_resnet50
│   ├── deeplabv3.py         # torchvision.models.segmentation.deeplabv3_resnet50
│   ├── unet.py              # segmentation_models_pytorch.Unet
│   ├── mask_rcnn.py         # torchvision.models.detection.maskrcnn_resnet50_fpn
│   └── sam.py               # segment-anything 官方代码（单独处理）
├── train_semantic.py        # 语义分割训练（FCN/DeepLab/U-Net）
├── train_instance.py        # 实例分割训练（Mask R-CNN）
├── eval_sam.py              # SAM 评估（prompt-based）
├── compare.py
└── results/
```

### 每个模型的核心关注点

| 模型 | 类型 | 核心关注点 |
|------|------|-----------|
| FCN | 语义分割 | 全卷积、转置卷积、Skip Connection |
| DeepLab v3 | 语义分割 | 空洞卷积、ASPP 多尺度 |
| U-Net | 语义分割 | 对称 Encoder-Decoder、Concat Skip |
| Mask R-CNN | 实例分割 | RoI Align、并行 Mask 分支 |
| SAM | 可提示分割 | Prompt Encoder、ViT-H、零样本迁移 |

### SAM 的特殊处理

SAM 是 prompt-based 模型，和传统分割框架不同，需要单独处理：

```python
# eval_sam.py 核心逻辑
def evaluate_sam(image, prompt_points=None, prompt_boxes=None):
    """
    SAM 评估方式：
    1. 用目标检测器找到物体的 bounding box
    2. 用 box 作为 prompt 给 SAM
    3. SAM 输出 mask
    4. 和 ground truth mask 比较 mIoU
    """
    # 加载 SAM 模型
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
    predictor = SamPredictor(sam)

    # 设置图像
    predictor.set_image(image)

    # 用 bounding box 作为 prompt
    masks, scores, _ = predictor.predict(box=prompt_boxes)
    return masks
```

### 输出物
- 语义分割：mIoU 对比（FCN vs DeepLab vs U-Net）
- 实例分割：Mask AP 对比（Mask R-CNN）
- 可提示分割：SAM 的 mask 质量可视化
- 分割结果可视化

### 面试时能讲的点
- "我理解语义分割、实例分割、全景分割的区别"
- "我对比了 FCN、DeepLab 和 U-Net 的架构差异"
- "我跑过 SAM，理解 prompt-based 分割的工作原理"

---

## 项目 4：独立模型项目（8 个）

### 4.1 Swin Transformer

**代码**：`microsoft/Swin-Transformer`（GitHub 官方 repo）
**数据集**：CIFAR-100（子集，快速跑通）
**任务**：图像分类

```bash
# 安装
git clone https://github.com/microsoft/Swin-Transformer.git
cd Swin-Transformer
pip install -r requirements.txt

# 训练（CIFAR-100）
python main.py --cfg configs/swin/swin_tiny_cifar100.yaml \
    --data-path ./data/cifar100 \
    --batch-size 64 \
    --epochs 100
```

**核心代码关注点**：
- `models/swin_transformer.py`：窗口注意力、Shifted Window、Patch Merging
- 重点理解 Window Attention 和 Shifted Window 的实现
- Relative Position Bias 的实现

**面试能讲的点**：
- "我跑过 Swin Transformer，理解了窗口注意力和移位窗口的实现"
- "Swin 的层次化结构让它可以直接替代 CNN 做检测和分割"

---

### 4.2 CLIP

**代码**：`openai/CLIP`（GitHub 官方 repo）
**数据集**：Flickr30K（子集）
**任务**：图文对比学习 + 零样本分类

```bash
# 安装
pip install git+https://github.com/openai/CLIP.git

# 零样本分类评估
python -c "
import clip
import torch
from torchvision.datasets import CIFAR100

model, preprocess = clip.load('ViT-B/32')
dataset = CIFAR100(root='./data', download=True, transform=preprocess)

# 零样本分类
text_inputs = clip.tokenize([f'a photo of a {c}' for c in dataset.classes])
# ... 评估代码
"
```

**核心代码关注点**：
- 对比学习的 loss 计算（双向 Cross-Entropy）
- Image Encoder（ViT）和 Text Encoder（Transformer）的结构
- 零样本分类的 prompt engineering

**面试能讲的点**：
- "我理解 CLIP 的对比学习框架和零样本分类原理"
- "CLIP 是 VLA 的视觉编码器基座，OpenVLA 用的 Prismatic 就是 CLIP 的变体"

---

### 4.3 DINO / DINOv2

**代码**：`facebookresearch/dino` + `facebookresearch/dinov2`
**数据集**：CIFAR-100（子集）
**任务**：自监督视觉特征学习

```bash
# DINO
git clone https://github.com/facebookresearch/dino.git
cd dino
python main_dino.py --arch vit_small --data_path ../data/cifar100 \
    --batch_size 64 --epochs 100

# DINOv2（用官方 API）
pip install dinov2
python -c "
import dinov2
model = dinov2.dinov2_vits14(pretrained=True)
# 提取特征
features = model(image)
"
```

**核心代码关注点**：
- 自蒸馏框架（Teacher-Student + EMA）
- Global crops 和 Local crops 的设计
- Centering 和 Sharpening 机制

**面试能讲的点**：
- "我理解 DINO 的自蒸馏框架：Teacher 用 EMA 更新，Student 从局部预测全局"
- "DINOv2 的特征可以直接用于下游任务，不需要微调"

---

### 4.4 DDPM

**代码**：`lucidrains/denoising-diffusion-pytorch`（PyPI 包）
**数据集**：CIFAR-10
**任务**：图像生成

```bash
# 安装
pip install denoising-diffusion-pytorch

# 训练
python -c "
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(dim=64, dim_mults=(1, 2, 4, 8))
diffusion = GaussianDiffusion(model, image_size=32, timesteps=1000)

trainer = Trainer(
    diffusion,
    './data/cifar10',
    train_batch_size=32,
    train_lr=2e-4,
    train_num_steps=50000,
)
trainer.train()
"
```

**核心代码关注点**：
- 前向加噪过程：`q_sample()` 方法
- 训练目标：预测噪声 ε
- 采样过程：从纯噪声逐步去噪
- 噪声调度（beta schedule）

**面试能讲的点**：
- "我跑过 DDPM，理解了前向加噪和逆向去噪的完整流程"
- "训练目标是 L = ||ε - ε_θ(x_t, t)||²，非常简洁"

---

### 4.5 DDIM

**代码**：同 DDPM（修改采样部分）
**数据集**：CIFAR-10
**任务**：加速采样

```python
# DDIM 采样（修改 trainer 的 sample 方法）
def ddim_sample(model, shape, ddim_timesteps=50):
    """
    DDIM 采样：跳步采样，从 1000 步减少到 50 步
    """
    # 定义子序列
    timestep_seq = list(range(0, 1000, 1000 // ddim_timesteps))

    x = torch.randn(shape)
    for t in reversed(timestep_seq):
        # 预测噪声
        pred_noise = model(x, t)
        # DDIM 更新公式
        x0_pred = predict_x0(x, pred_noise, t)
        x = ddim_step(x, x0_pred, pred_noise, t, t_prev)
    return x
```

**核心代码关注点**：
- DDIM 采样公式：`x_{t-1} = √ᾱ_{t-1} · predicted_x_0 + √(1-ᾱ_{t-1}) · ε_θ`
- 跳步采样的实现
- σ 参数控制随机性

**面试能讲的点**：
- "我理解 DDIM 和 DDPM 的核心区别：DDIM 是确定性采样，可以跳步"
- "DDIM 把 1000 步减少到 50 步，速度快 20 倍，质量几乎不变"

---

### 4.6 MVSNet

**代码**：`YoYo000/MVSNet`（GitHub 官方 repo）
**数据集**：DTU dataset（官方多视图立体匹配 benchmark）
**任务**：多视图深度估计

```bash
# 安装
git clone https://github.com/YoYo000/MVSNet.git
cd MVSNet
pip install -r requirements.txt

# 准备 DTU 数据集
python dtu_preprocess.py --dtu_dir ./data/DTU

# 训练
python train.py --batch_size 4 --epochs 16 --lr 1e-3
```

**核心代码关注点**：
- 代价体构建：`build_cost_volume()` — 多视图特征 warp + 方差计算
- 3D CNN：对代价体做 3D 卷积
- Soft Argmin：可微的深度回归

**面试能讲的点**：
- "我跑过 MVSNet，理解了代价体的构建过程"
- "MVSNet 的核心是：对每个候选深度，warp 多视图特征，用方差衡量一致性"

---

### 4.7 Mip-NeRF

**代码**：`google/mipnerf`（GitHub 官方 repo）
**数据集**：NeRF Synthetic（官方 8 场景：chair, drums, ficus, hotdog, lego, materials, mic, ship）
**任务**：新视角合成

```bash
# 安装
git clone https://github.com/google/mipnerf.git
cd mipnerf

# 下载 NeRF Synthetic 数据集
bash scripts/download_nerf_synthetic.sh

# 训练
python train.py --config configs/nerf_synthetic.txt \
    --data_dir ./data/nerf_synthetic/lego \
    --exp_dir ./exp/lego
```

**核心代码关注点**：
- 积分位置编码：`integrated_pos_enc()` — 对高斯区域积分 sin/cos
- 锥形区域参数化：3D 高斯表示锥形区域
- 从"点查询"到"区域查询"的转变

**面试能讲的点**：
- "我跑过 Mip-NeRF，理解了积分位置编码的原理"
- "Mip-NeRF 用 3D 高斯表示锥形区域，解决了 NeRF 的混叠问题"

---

### 4.8 Mip-Splatting

**代码**：`autonomousvision/mip-splatting`（GitHub 官方 repo）
**数据集**：NeRF Synthetic（同上）
**任务**：抗混叠 3D 高斯渲染

```bash
# 安装
git clone https://github.com/autonomousvision/mip-splatting.git
cd mip-splatting
pip install -r requirements.txt

# 训练
python train.py -s data/nerf_synthetic/lego \
    --eval --mip_filter
```

**核心代码关注点**：
- 3D 平滑：`Σ_smooth = Σ + σ²_smooth · I`
- 2D Mip Filter：在图像空间做低通滤波
- 和原始 3DGS 的对比

**面试能讲的点**：
- "我跑过 Mip-Splatting，理解了 3DGS 的混叠问题和解决方案"
- "Mip-Splatting 通过 3D 平滑和 2D 滤波，使得不同分辨率渲染结果一致"

---

## 项目 5：Diffusion Policy（机械臂抓取）

### 目标
用 LeRobot 框架，在 ALOHA 仿真环境中训练 Diffusion Policy，完成双臂物体转移任务。

### 技术方案

**框架**：LeRobot（HuggingFace）
**环境**：ALOHA 仿真（`lerobot/aloha_sim_transfer_cube_human`）
**任务**：双臂协作转移方块
**可视化**：LeRobot 自带评估可视化

### 环境搭建

```bash
# 创建环境
conda create -n vla-dp python=3.10
conda activate vla-dp

# 安装 LeRobot
pip install lerobot

# 或从源码安装（推荐，方便看代码）
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

### 训练流程

```bash
# Step 1: 训练 Diffusion Policy
python lerobot/scripts/train.py \
    policy=diffusion \
    env=aloha \
    dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
    training.num_epochs=100 \
    training.batch_size=64

# Step 2: 评估
python lerobot/scripts/eval.py \
    policy=diffusion \
    env=aloha \
    hydra.run.dir=outputs/eval

# Step 3: 可视化（渲染机器人执行过程）
python lerobot/scripts/eval.py \
    policy=diffusion \
    env=aloha \
    hydra.run.dir=outputs/vis \
    eval.n_episodes=5 \
    eval.render=true
```

### 核心代码阅读

```
lerobot/common/policies/diffusion/
├── modeling_diffusion.py    # 核心模型
│   ├── forward()            # 训练时的前向传播
│   ├── select_action()      # 推理时的动作选择
│   └── diffusion.py         # 扩散过程（加噪/去噪）
├── configuration_diffusion.py  # 超参数配置
└── scheduler.py             # 噪声调度器
```

**重点理解**：
1. `forward()` 中的训练流程：观测 → 加噪动作 → 预测噪声 → loss
2. `select_action()` 中的推理流程：从噪声开始 → 多步去噪 → 动作
3. Action chunking：一次预测多步动作，减少决策频率
4. 条件注入：观测信息如何注入到去噪过程中

### 面试时能讲的点
- "我用 LeRobot 跑通了 Diffusion Policy，训练了 ALOHA 双臂做物体转移任务"
- "我理解了去噪过程：从 N(0,I) 噪声开始，经过 T 步去噪，得到动作序列"
- "Action chunking 的好处：一次预测多步动作，减少 compounding error"
- "我读过 LeRobot 的核心代码，理解了 condition injection 的实现"

---

## 项目 6：ACT（机械臂抓取）

### 目标
用 LeRobot 框架，在 ALOHA 仿真环境中训练 ACT，理解 CVAE 架构。

### 技术方案

**框架**：LeRobot（和项目 5 同框架）
**环境**：ALOHA 仿真（同上）
**任务**：同上（方便对比 Diffusion Policy 和 ACT）

### 训练流程

```bash
# 训练 ACT
python lerobot/scripts/train.py \
    policy=act \
    env=aloha \
    dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
    training.num_epochs=100 \
    training.batch_size=64

# 评估和可视化（同项目 5）
python lerobot/scripts/eval.py \
    policy=act \
    env=aloha \
    hydra.run.dir=outputs/eval_act
```

### 核心代码阅读

```
lerobot/common/policies/act/
├── modeling_act.py          # 核心模型
│   ├── forward()            # CVAE 训练
│   ├── select_action()      # 推理
│   └── encoder() / decoder() # CVAE 的编码器和解码器
└── configuration_act.py     # 超参数配置
```

**重点理解**：
1. CVAE 框架：Encoder 把 (图像, 动作序列) 编码为 latent z，Decoder 从 (图像, z) 预测动作
2. KL 散度 loss：正则化 latent space
3. Action chunking：和 Diffusion Policy 类似
4. 推理时从 N(0,I) 采样 z

### 对比 Diffusion Policy 和 ACT

| 特性 | Diffusion Policy | ACT |
|------|-----------------|-----|
| 多模态建模 | 扩散模型（去噪） | CVAE（latent z） |
| 训练目标 | 预测噪声 ε | 重建 + KL 散度 |
| 推理过程 | 多步去噪 | 一次前向传播 |
| Action Chunking | 支持 | 支持 |
| 训练稳定性 | 较好 | 需要调 KL 权重 |

### 面试时能讲的点
- "我用 LeRobot 跑通了 ACT，理解了 CVAE 的架构"
- "ACT 的 Encoder 学习 latent space，推理时从先验采样"
- "我对比了 Diffusion Policy 和 ACT：前者用去噪建模多模态，后者用 VAE"

---

## 项目 7：RL 算法对比（机械臂抓取）

### 目标
在 robosuite 的 Lift 任务中，对比 DQN、PPO、SAC、TD3 四个强化学习算法。

### 技术方案

**框架**：robosuite + Stable-Baselines3（SB3）
**环境**：robosuite `Lift` 任务（Panda 机械臂抓取方块）
**算法**：DQN、PPO、SAC、TD3（SB3 实现）
**可视化**：robosuite 的 MuJoCo 渲染器

### 环境搭建

```bash
# 安装
pip install robosuite stable-baselines3

# 验证环境
python -c "
import robosuite as suite
env = suite.make(
    'Lift',
    robots='Panda',
    has_renderer=True,       # 开启可视化
    has_offscreen_renderer=False,
    use_camera_obs=False,    # 用低维观测（状态向量）
)
obs = env.reset()
for i in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
env.close()
"
```

### 训练脚本

```python
# train_rl.py
import robosuite as suite
from stable_baselines3 import PPO, SAC, TD3, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    env = suite.make(
        'Lift',
        robots='Panda',
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,  # 低维观测
        reward_shaping=True,   # 稠密 reward，加速训练
    )
    return env

def train(algorithm, total_timesteps=100_000):
    env = DummyVecEnv([make_env])

    if algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1)
    elif algorithm == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1)
    elif algorithm == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1)
    elif algorithm == 'DQN':
        model = DQN('MlpPolicy', env, verbose=1)

    model.learn(total_timesteps=total_timesteps)
    model.save(f"models/{algorithm}_lift")
    return model
```

### 可视化评估

```python
# visualize.py
def visualize(algorithm, model_path):
    env = suite.make(
        'Lift',
        robots='Panda',
        has_renderer=True,  # 开启可视化
        use_camera_obs=False,
    )
    model = load_model(algorithm, model_path)

    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()
```

### 对比指标

| 指标 | DQN | PPO | SAC | TD3 |
|------|-----|-----|-----|-----|
| 类型 | Off-policy | On-policy | Off-policy | Off-policy |
| 动作空间 | 离散 | 连续 | 连续 | 连续 |
| 训练稳定性 | 中 | 高 | 高 | 中 |
| 样本效率 | 中 | 低 | 高 | 高 |
| 最终成功率 | ? | ? | ? | ? |
| 训练时间 | ? | ? | ? | ? |

**注意**：DQN 是离散动作算法，需要把连续动作空间离散化，或者只用 PPO/SAC/TD3 做对比。

### 面试时能讲的点
- "我在 robosuite 的 Lift 任务上对比了 4 个 RL 算法"
- "PPO 实现简单训练稳定，SAC 样本效率高适合真实机器人"
- "在 VLA 中，通常用 IL 做预训练，RL 做 fine-tuning"

---

## 项目 8：OpenVLA 微调（机械臂抓取）

### 目标
用 OpenVLA 官方代码，在 LIBERO 仿真数据上微调 VLA 模型。

### 技术方案

**框架**：OpenVLA 官方代码
**数据集**：LIBERO 仿真数据集（`openvla/libero_spatial_no_noops`）
**微调方法**：LoRA（低秩适配）
**可视化**：LIBERO 自带评估环境

### 环境搭建

```bash
# 克隆 OpenVLA
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# 下载 LIBERO 数据集（HuggingFace）
# 数据集: openvla/libero_spatial_no_noops
```

### 微调流程

```bash
# LoRA 微调 OpenVLA-7B
python vla-scripts/finetune.py \
    --vla_path openvla-7b \
    --data_root_dir ./data/libero \
    --dataset_name libero_spatial_no_noops \
    --run_root_dir ./runs \
    --adapter_tmp_dir ./adapters \
    --lora_rank 32 \
    --batch_size 16 \
    --grad_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --image_aug True \
    --wandb_project openvla-finetune \
    --run_name my_run
```

### 评估和可视化

```python
# eval_openvla.py
from libero.libero import benchmark
from openvla import OpenVLA

# 加载微调后的模型
vla = OpenVLA.from_pretrained("openvla-7b")
vla.load_adapter("./adapters/my_run")

# 在 LIBERO 环境中评估
benchmark = benchmark.get_benchmark("spatial")()
task = benchmark.get_task(0)
env = benchmark.make_env(task)

obs = env.reset()
for step in range(300):
    # VLA 推理：图像 + 语言指令 → 动作
    action = vla.predict_action(
        image=obs["agentview_image"],
        instruction=task.language,
        unnorm_key="libero_spatial_no_noops",
    )
    obs, reward, done, info = env.step(action)
    env.render()  # 可视化
    if done:
        break
```

### 核心理解点

1. **Visual Encoder**：Prismatic（SigLIP + DinoV2 双编码器）
2. **连接方式**：MLP projection 把视觉特征映射到 LLM token space
3. **Action Tokenization**：7DoF 动作 → 7 个 token（256 bins）
4. **LoRA 微调**：只调 adapter 参数，冻结 LLM backbone
5. **微调数据量**：几百到几千个 demonstration 就够

### 面试时能讲的点
- "我微调过 OpenVLA，用 LoRA 在 LIBERO 仿真任务上"
- "OpenVLA 用 Prismatic 双视觉编码器，结合了语义（SigLIP）和空间（DinoV2）特征"
- "Action tokenization 把 7DoF 动作每个维度量化为 256 bins，共享 LLM 词表"

---

## 项目 9：Octo 微调（替代 π0）

### 目标
用 Octo（Berkeley 开源 VLA）在 LIBERO 仿真数据上微调，体验 VLA 微调流程。

### 技术方案

**框架**：Octo 官方代码（`octo-model`）
**数据集**：LIBERO 或 robosuite
**微调方法**：LoRA 或 full fine-tune
**可视化**：robosuite 渲染器

### 环境搭建

```bash
# 安装 Octo
pip install octo-model

# 或从源码
git clone https://github.com/octo-models/octo.git
cd octo
pip install -e .
```

### 微调流程

```python
# finetune_octo.py
from octo.model.octo_model import OctoModel
from octo.utils.gym_utils import make_env

# 加载预训练模型
model = OctoModel.load_pretrained("octo-base")

# 微调
model.finetune(
    dataset=finetune_dataset,
    task="pick_up_the_red_cube",
    num_epochs=50,
    learning_rate=3e-4,
    use_lora=True,
)
```

### 评估和可视化

```python
# eval_octo.py
import robosuite as suite

env = suite.make(
    'Lift',
    robots='Panda',
    has_renderer=True,
    use_camera_obs=True,
)

obs = env.reset()
for step in range(300):
    action = model.predict(
        observation=obs,
        task_instruction="pick up the red cube",
    )
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
```

### 核心理解点

1. Octo 的架构：Transformer-based policy + diffusion action head
2. 多任务支持：同一个模型可以做不同的操作任务
3. 跨 embodiment：支持不同机器人形态
4. 和 OpenVLA 的对比：Octo 更侧重 diffusion policy，OpenVLA 更侧重 action tokenization

### 面试时能讲的点
- "我微调过 Octo，它是 Berkeley 开源的通用机器人策略"
- "Octo 用 diffusion action head，支持多任务和跨 embodiment"
- "Octo 和 OpenVLA 的区别：Octo 用扩散模型做 action prediction，OpenVLA 用 action tokenization"

---

## 项目笔记模板

```markdown
## 项目名称

### 做了什么
一句话描述。

### 环境搭建
- 框架版本：
- 依赖包：
- 数据集下载：

### 关键步骤
1. ...
2. ...

### 核心代码理解
用自己的话解释关键代码模块的作用。

### 遇到的问题及解决
- 问题：...
  解决：...

### 实验结果
- 训练曲线截图
- 最终指标

### 面试版本（3分钟）
准备一个 3 分钟的口头陈述，用于面试。

### 踩坑记录
记录遇到的坑和解决方法。
```

---

## 面试中讲项目的框架（STAR 法）

```
S (Situation): 我在准备 VLA 算法工程师面试，需要深入理解核心算法
T (Task):      跑通核心模型的训练-评估 pipeline，建立工程实践感
A (Action):    用官方代码和开源数据集，完成了 9 类项目的训练和评估
               - CNN/检测/分割：统一框架对比经典模型
               - Diffusion Policy/ACT：在 ALOHA 仿真中训练机械臂
               - RL 对比：在 robosuite 中对比 4 个算法
               - OpenVLA/Octo：微调 VLA 模型做抓取任务
R (Result):    建立了对 VLA 技术栈的系统性理解
               面试中能清晰讲解每个算法的核心设计和工程实现
```

---

## 每日计划（第三周）

### Day 1（周一）：项目 1 — CNN 分类
- 上午：搭建环境，下载 CIFAR-10
- 下午：跑通所有 CNN 模型的训练
- 晚上：对比结果，理解核心代码

### Day 2（周二）：项目 2 — 目标检测
- 上午：搭建环境，下载 VOC 数据
- 下午：训练 Faster R-CNN 和 YOLO
- 晚上：训练 DETR，对比 mAP

### Day 3（周三）：项目 3 — 图像分割
- 上午：训练 FCN 和 DeepLab
- 下午：训练 U-Net 和 Mask R-CNN
- 晚上：评估 SAM，对比结果

### Day 4（周四）：项目 4 — 独立模型（上）
- 上午：Swin Transformer + CLIP
- 下午：DINO/DINOv2 + DDPM
- 晚上：DDIM，整理笔记

### Day 5（周五）：项目 4 — 独立模型（下）
- 上午：MVSNet
- 下午：Mip-NeRF
- 晚上：Mip-Splatting

### Day 6（周六）：项目 5-6 — Diffusion Policy + ACT
- 上午：搭建 LeRobot 环境
- 下午：训练 Diffusion Policy
- 晚上：训练 ACT，对比两者

### Day 7（周日）：项目 7 — RL 对比
- 上午：搭建 robosuite 环境
- 下午：训练 PPO 和 SAC
- 晚上：训练 TD3，对比结果

### Day 8-9：项目 8-9 — OpenVLA + Octo 微调
- 搭建环境，下载数据
- 微调模型
- 评估和可视化

### Day 10：复盘 + 准备面试
- 整理所有项目笔记
- 准备 3 分钟项目陈述
- 模拟面试
