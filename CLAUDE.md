# VLA-Inter 第三周实践项目

## ⚠️ 必须遵守的规则（每次回答前检查）

1. **每次回答开头必须叫"涵哥"** — 证明已读过 CLAUDE.md
2. **禁止让模型自己看图片、视频等模态内容** — 只处理文本，不读取/分析图片、视频、音频等非文本文件
3. **每次开发完成后主动提交到 GitHub** — 严格按 Git Flow 规范操作
4. **使用 conda 环境 `vla`** — 所有项目开发和训练都在这个环境中执行

## GitHub 仓库

```
远程仓库: git@github.com:SimonHanYANG/VLA-Interview-Projects.git
```

### 提交流程（每次开发完成后执行）

```bash
# 0. 激活 conda 环境
conda activate vla

# 1. 确保在正确的 feature 分支上
git status

# 2. 添加修改
git add .

# 3. 按规范提交
git commit -m "feat(projectX): 描述完成了什么"

# 4. 推送到远程
git push origin feature/projectX
```

### 自动提交时机

- 每个项目的核心代码修改完成后
- 训练脚本跑通后
- 项目笔记写完后
- 不要攒到最后一次性提交，小步快跑

## 项目目标
跑通 VLA 技术栈核心模型的训练-测试流程，在统一框架中对比不同模型，建立工程实践感，面试时能说"我跑过这个实验"。所有项目使用开源代码和数据集。

## 项目总览

| # | 项目 | 模型 | 框架 | 数据集 |
|---|------|------|------|--------|
| 1 | CNN 分类 | AlexNet/VGG/GoogLeNet/ResNet/EfficientNet | torchvision | CIFAR-10 子集 |
| 2 | 目标检测 | Faster R-CNN/YOLO/DETR | torchvision + 官方代码 | VOC 2007 子集 |
| 3 | 图像分割 | FCN/DeepLab/U-Net/Mask R-CNN/SAM | torchvision + 官方代码 | VOC 2012 子集 |
| 4 | 独立模型(8个) | Swin/CLIP/DINO/DDPM/DDIM/MVSNet/Mip-NeRF/Mip-Splatting | 各自官方代码 | 各自数据集 |
| 5 | Diffusion Policy | Diffusion Policy | LeRobot | ALOHA 仿真 |
| 6 | ACT | ACT | LeRobot | ALOHA 仿真 |
| 7 | RL 对比 | DQN/PPO/SAC/TD3 | robosuite + SB3 | Lift 任务 |
| 8 | OpenVLA 微调 | OpenVLA-7B | 官方代码 | LIBERO 仿真 |
| 9 | Octo 微调 | Octo | 官方代码 | LIBERO 仿真 |

## Git Flow 分支规范

### 分支结构

```
main          ← 生产就绪代码，只接受 release/hotfix 合并
├── develop   ← 开发主线，所有 feature 合入此分支
│   ├── feature/project1-cnn         ← 项目1: CNN 分类
│   ├── feature/project2-detection   ← 项目2: 目标检测
│   ├── feature/project3-segmentation← 项目3: 图像分割
│   ├── feature/project4-swin        ← 项目4.1: Swin Transformer
│   ├── feature/project4-clip        ← 项目4.2: CLIP
│   ├── feature/project4-dino        ← 项目4.3: DINO/DINOv2
│   ├── feature/project4-ddpm        ← 项目4.4: DDPM
│   ├── feature/project4-ddim        ← 项目4.5: DDIM
│   ├── feature/project4-mvsnet      ← 项目4.6: MVSNet
│   ├── feature/project4-mipnerf     ← 项目4.7: Mip-NeRF
│   ├── feature/project4-mipsplatting← 项目4.8: Mip-Splatting
│   ├── feature/project5-diffusion   ← 项目5: Diffusion Policy
│   ├── feature/project6-act         ← 项目6: ACT
│   ├── feature/project7-rl          ← 项目7: RL 对比
│   ├── feature/project8-openvla     ← 项目8: OpenVLA 微调
│   └── feature/project9-octo        ← 项目9: Octo 微调
├── release/v1.0                      ← 发布准备（可选）
└── hotfix/*                          ← 紧急修复
```

### 工作流程

1. **从 develop 创建 feature 分支**：`git checkout -b feature/projectX develop`
2. **在 feature 分支上开发**：小步提交，commit message 清晰
3. **每次开发完成主动推送**：`git push origin feature/projectX`
4. **完成后合并回 develop**：通过 PR 合并，删除 feature 分支
5. **阶段性成果打 tag**：如 `v0.1-project1-done`
6. **全部完成后**：从 develop 创建 release 分支，合并到 main

### Commit Message 规范

```
<type>(<scope>): <subject>

类型：
- feat:     新功能（新模型跑通）
- fix:      修复 bug
- refactor: 重构代码
- docs:     文档/注释
- chore:    环境搭建、依赖更新

示例：
feat(project1): 跑通 ResNet-18 在 CIFAR-10 上的训练
feat(project5): 完成 Diffusion Policy 在 ALOHA 仿真中的训练
fix(project2): 修复 DETR 评估时的 mAP 计算错误
docs(project4): 添加 CLIP 对比学习 loss 的代码注释
```

### 每个项目的产出物

- 模型训练代码（基于官方代码修改）
- 训练日志和结果（loss 曲线、指标对比）
- 核心代码注释（在 models/ 目录中）
- 项目笔记（面试 3 分钟陈述）

## 注意事项

- 所有命令和训练都在 conda 环境 `vla` 中执行，先 `conda activate vla`
- 所有项目使用官方开源代码，不从头手写模型
- 每个项目完成后整理面试能讲的点
- DQN 是离散动作算法，项目 7 中可能需要离散化或只对比 PPO/SAC/TD3
- SAM 是 prompt-based 模型，项目 3 中需要单独处理评估方式
- 禁止让模型自己看图片、视频等模态内容，只处理文本
