"""
下载 VOC 2012 数据集
"""

import os
import sys
import zipfile
import tarfile
import urllib.request
from pathlib import Path

def download_voc2012(root="./data"):
    """
    下载 VOC 2012 数据集

    Args:
        root: 数据保存根目录
    """
    # 创建数据目录
    os.makedirs(root, exist_ok=True)

    # VOC 2012 数据集 URL
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    filename = "VOCtrainval_11-May-2012.tar"
    filepath = os.path.join(root, filename)

    # 检查是否已下载
    if os.path.exists(os.path.join(root, "VOC2012")):
        print("VOC2012 dataset already exists!")
        return

    # 下载数据集
    print(f"Downloading VOC 2012 dataset...")
    print(f"URL: {url}")
    print(f"Save to: {filepath}")

    try:
        # 显示下载进度
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / 1024 / 1024
                mb_total = total_size / 1024 / 1024
                sys.stdout.write(f"\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, filepath, reporthook)
        print("\nDownload completed!")

    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nAlternative: Download manually from:")
        print("  https://pjreddie.com/projects/pascal-voc-dataset-mirror/")
        print("  or")
        print("  http://host.robots.ox.ac.uk/pascal/VOC/voc2012/")
        return

    # 解压数据集
    print("Extracting dataset...")
    try:
        with tarfile.open(filepath, "r") as tar:
            tar.extractall(path=root)
        print("Extraction completed!")

        # 删除压缩包
        os.remove(filepath)
        print(f"Removed {filepath}")

    except Exception as e:
        print(f"Extraction failed: {e}")
        return

    # 验证数据集
    voc_dir = os.path.join(root, "VOC2012")
    if os.path.exists(voc_dir):
        # 统计图像数量
        image_dir = os.path.join(voc_dir, "JPEGImages")
        mask_dir = os.path.join(voc_dir, "SegmentationClass")

        num_images = len([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        num_masks = len([f for f in os.listdir(mask_dir) if f.endswith(".png")])

        print(f"\nDataset Statistics:")
        print(f"  Total images: {num_images}")
        print(f"  Segmentation masks: {num_masks}")

        # 检查分割列表
        split_dir = os.path.join(voc_dir, "ImageSets", "Segmentation")
        if os.path.exists(split_dir):
            for split in ["train", "val", "trainval"]:
                split_file = os.path.join(split_dir, f"{split}.txt")
                if os.path.exists(split_file):
                    with open(split_file, "r") as f:
                        num_samples = len([line for line in f if line.strip()])
                    print(f"  {split}: {num_samples} samples")

        print("\nDataset ready!")
    else:
        print("Dataset extraction failed!")


def create_subset(root="./data", subset_ratio=0.1):
    """
    创建数据子集用于快速实验

    Args:
        root: 数据根目录
        subset_ratio: 子集比例
    """
    voc_dir = os.path.join(root, "VOC2012")
    split_dir = os.path.join(voc_dir, "ImageSets", "Segmentation")

    # 读取训练集
    train_file = os.path.join(split_dir, "train.txt")
    with open(train_file, "r") as f:
        train_ids = [line.strip() for line in f if line.strip()]

    # 创建子集
    subset_size = int(len(train_ids) * subset_ratio)
    subset_ids = train_ids[:subset_size]

    # 保存子集列表
    subset_file = os.path.join(split_dir, "train_subset.txt")
    with open(subset_file, "w") as f:
        for image_id in subset_ids:
            f.write(f"{image_id}\n")

    print(f"Created subset: {subset_size} training samples")
    print(f"Saved to: {subset_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download VOC 2012 dataset")
    parser.add_argument("--root", type=str, default="./data",
                       help="Data root directory")
    parser.add_argument("--subset", action="store_true",
                       help="Create subset for quick experiments")
    parser.add_argument("--subset_ratio", type=float, default=0.1,
                       help="Subset ratio")

    args = parser.parse_args()

    # 下载数据集
    download_voc2012(args.root)

    # 创建子集
    if args.subset:
        create_subset(args.root, args.subset_ratio)
