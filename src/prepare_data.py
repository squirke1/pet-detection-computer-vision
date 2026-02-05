#!/usr/bin/env python3
"""
Data preparation utilities for YOLOv8 training.

Utilities for:
- Converting datasets to YOLO format
- Splitting datasets (train/val/test)
- Dataset validation and statistics
- Annotation format conversion (COCO, Pascal VOC, etc.)

Usage:
    python prepare_data.py --input path/to/dataset --output path/to/yolo_dataset
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random

import cv2
import numpy as np
from tqdm import tqdm


def split_dataset(
    image_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[Path]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        image_dir: Directory containing images
        output_dir: Output directory for split dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with paths for each split
    """
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0")
    
    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = [f for f in image_dir.glob('*') if f.suffix.lower() in image_extensions]
    
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"📁 Found {len(images)} images")
    
    # Shuffle images
    random.seed(seed)
    random.shuffle(images)
    
    # Calculate split indices
    n_train = int(len(images) * train_ratio)
    n_val = int(len(images) * val_ratio)
    
    # Split dataset
    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]
    
    print(f"✂️  Split:")
    print(f"   Train: {len(train_images)} ({len(train_images)/len(images)*100:.1f}%)")
    print(f"   Val: {len(val_images)} ({len(val_images)/len(images)*100:.1f}%)")
    print(f"   Test: {len(test_images)} ({len(test_images)/len(images)*100:.1f}%)")
    
    # Create output directories
    splits = {'train': train_images, 'val': val_images, 'test': test_images}
    
    for split_name, split_images in splits.items():
        if not split_images:
            continue
        
        img_dir = output_dir / split_name / 'images'
        lbl_dir = output_dir / split_name / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images and labels
        for img_path in tqdm(split_images, desc=f"Copying {split_name}"):
            # Copy image
            shutil.copy2(img_path, img_dir / img_path.name)
            
            # Copy label if exists
            label_path = img_path.parent.parent / 'labels' / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, lbl_dir / label_path.name)
    
    print(f"✅ Dataset split saved to {output_dir}")
    
    return splits


def validate_yolo_dataset(data_dir: Path) -> Dict:
    """
    Validate YOLO format dataset and compute statistics.
    
    Args:
        data_dir: Root directory of YOLO dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_images': 0,
        'total_labels': 0,
        'images_without_labels': 0,
        'labels_without_images': 0,
        'class_distribution': {},
        'bbox_sizes': [],
        'splits': {}
    }
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        img_dir = split_dir / 'images'
        lbl_dir = split_dir / 'labels'
        
        if not img_dir.exists():
            print(f"⚠️  Warning: {img_dir} does not exist")
            continue
        
        # Get images and labels
        images = list(img_dir.glob('*.[jp][pn][g]')) + list(img_dir.glob('*.jpeg'))
        labels = list(lbl_dir.glob('*.txt')) if lbl_dir.exists() else []
        
        split_stats = {
            'images': len(images),
            'labels': len(labels),
            'class_counts': {}
        }
        
        # Check for missing labels
        image_stems = {img.stem for img in images}
        label_stems = {lbl.stem for lbl in labels}
        
        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems
        
        if missing_labels:
            stats['images_without_labels'] += len(missing_labels)
            print(f"⚠️  {split}: {len(missing_labels)} images without labels")
        
        if missing_images:
            stats['labels_without_images'] += len(missing_images)
            print(f"⚠️  {split}: {len(missing_images)} labels without images")
        
        # Analyze labels
        for label_path in labels:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    
                    # Update class distribution
                    if class_id not in stats['class_distribution']:
                        stats['class_distribution'][class_id] = 0
                    stats['class_distribution'][class_id] += 1
                    
                    if class_id not in split_stats['class_counts']:
                        split_stats['class_counts'][class_id] = 0
                    split_stats['class_counts'][class_id] += 1
                    
                    # Track bbox sizes
                    stats['bbox_sizes'].append((w, h))
        
        stats['total_images'] += split_stats['images']
        stats['total_labels'] += split_stats['labels']
        stats['splits'][split] = split_stats
    
    # Compute bbox statistics
    if stats['bbox_sizes']:
        bbox_widths = [w for w, h in stats['bbox_sizes']]
        bbox_heights = [h for w, h in stats['bbox_sizes']]
        
        stats['bbox_stats'] = {
            'mean_width': np.mean(bbox_widths),
            'mean_height': np.mean(bbox_heights),
            'median_width': np.median(bbox_widths),
            'median_height': np.median(bbox_heights),
            'min_width': np.min(bbox_widths),
            'min_height': np.min(bbox_heights),
            'max_width': np.max(bbox_widths),
            'max_height': np.max(bbox_heights)
        }
    
    return stats


def print_dataset_stats(stats: Dict, class_names: Optional[List[str]] = None):
    """
    Print dataset statistics in a formatted way.
    
    Args:
        stats: Statistics dictionary from validate_yolo_dataset
        class_names: List of class names (optional)
    """
    print("\n" + "=" * 60)
    print("📊 Dataset Statistics")
    print("=" * 60)
    
    print(f"\n📁 Overall:")
    print(f"   Total images: {stats['total_images']}")
    print(f"   Total labels: {stats['total_labels']}")
    
    if stats['images_without_labels'] > 0:
        print(f"   ⚠️  Images without labels: {stats['images_without_labels']}")
    if stats['labels_without_images'] > 0:
        print(f"   ⚠️  Labels without images: {stats['labels_without_images']}")
    
    print(f"\n📂 Splits:")
    for split_name, split_stats in stats['splits'].items():
        print(f"   {split_name.capitalize()}:")
        print(f"      Images: {split_stats['images']}")
        print(f"      Labels: {split_stats['labels']}")
        if split_stats['class_counts']:
            print(f"      Annotations: {sum(split_stats['class_counts'].values())}")
    
    print(f"\n🏷️  Class Distribution:")
    for class_id, count in sorted(stats['class_distribution'].items()):
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
        percentage = count / sum(stats['class_distribution'].values()) * 100
        print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    if 'bbox_stats' in stats:
        print(f"\n📏 Bounding Box Statistics (normalized):")
        print(f"   Mean size: {stats['bbox_stats']['mean_width']:.3f} x {stats['bbox_stats']['mean_height']:.3f}")
        print(f"   Median size: {stats['bbox_stats']['median_width']:.3f} x {stats['bbox_stats']['median_height']:.3f}")
        print(f"   Min size: {stats['bbox_stats']['min_width']:.3f} x {stats['bbox_stats']['min_height']:.3f}")
        print(f"   Max size: {stats['bbox_stats']['max_width']:.3f} x {stats['bbox_stats']['max_height']:.3f}")
    
    print("=" * 60 + "\n")


def visualize_sample(
    image_path: Path,
    label_path: Path,
    class_names: List[str],
    output_path: Optional[Path] = None
):
    """
    Visualize a sample with its annotations.
    
    Args:
        image_path: Path to image
        label_path: Path to label file
        class_names: List of class names
        output_path: Path to save visualization (optional)
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read {image_path}")
        return
    
    h, w = image.shape[:2]
    
    # Read labels
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # Convert to pixel coordinates
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                
                # Draw box
                color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                cv2.putText(image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save or display
    if output_path:
        cv2.imwrite(str(output_path), image)
        print(f"✅ Visualization saved: {output_path}")
    else:
        cv2.imshow('Sample', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main data preparation entry point."""
    parser = argparse.ArgumentParser(
        description='🛠️  Pet Detection Data Preparation Utilities',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Split dataset command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument('--input', type=str, required=True, help='Input directory with images')
    split_parser.add_argument('--output', type=str, required=True, help='Output directory')
    split_parser.add_argument('--train', type=float, default=0.8, help='Train ratio (default: 0.8)')
    split_parser.add_argument('--val', type=float, default=0.1, help='Val ratio (default: 0.1)')
    split_parser.add_argument('--test', type=float, default=0.1, help='Test ratio (default: 0.1)')
    split_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Validate dataset command
    validate_parser = subparsers.add_parser('validate', help='Validate YOLO dataset')
    validate_parser.add_argument('--data-dir', type=str, required=True, help='Dataset directory')
    validate_parser.add_argument('--class-names', type=str, nargs='+', help='Class names')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize dataset samples')
    viz_parser.add_argument('--image', type=str, required=True, help='Image path')
    viz_parser.add_argument('--label', type=str, required=True, help='Label path')
    viz_parser.add_argument('--class-names', type=str, nargs='+', required=True, help='Class names')
    viz_parser.add_argument('--output', type=str, help='Output path (optional)')
    
    args = parser.parse_args()
    
    if args.command == 'split':
        split_dataset(
            Path(args.input),
            Path(args.output),
            args.train,
            args.val,
            args.test,
            args.seed
        )
    
    elif args.command == 'validate':
        stats = validate_yolo_dataset(Path(args.data_dir))
        print_dataset_stats(stats, args.class_names)
    
    elif args.command == 'visualize':
        visualize_sample(
            Path(args.image),
            Path(args.label),
            args.class_names,
            Path(args.output) if args.output else None
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
