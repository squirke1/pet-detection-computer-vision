#!/usr/bin/env python3
"""
Advanced Visualization Utilities for Model Evaluation

This module provides additional visualization tools for YOLOv8 evaluation:
- Precision-Recall curves
- Per-class performance charts
- Training history plots
- Detection examples with annotations

Example Usage:
    from visualize_metrics import plot_pr_curve, plot_training_history
    
    # Plot PR curve
    plot_pr_curve(precisions, recalls, output_path='outputs/pr_curve.png')
    
    # Visualize training history
    plot_training_history('runs/train/exp/results.csv', output_path='outputs/training.png')
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_pr_curve(
    precisions: np.ndarray,
    recalls: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot Precision-Recall curve.
    
    Args:
        precisions: Array of precision values for each class
        recalls: Array of recall values for each class
        class_names: List of class names
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Plot curves for each class
    if precisions.ndim == 2:  # Multiple classes
        for i in range(precisions.shape[0]):
            label = class_names[i] if class_names else f'Class {i}'
            plt.plot(recalls[i], precisions[i], linewidth=2, label=label)
    else:  # Single class
        label = class_names[0] if class_names else 'All Classes'
        plt.plot(recalls, precisions, linewidth=2, label=label)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ PR curve saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(
    results_csv: Path,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> None:
    """
    Plot training history from YOLOv8 results CSV.
    
    Args:
        results_csv: Path to results.csv from training
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")
    
    # Load training results
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()  # Remove whitespace
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Extract epoch numbers
    epochs = df['epoch'].values if 'epoch' in df.columns else np.arange(len(df))
    
    # Plot 1: Losses
    ax = axes[0, 0]
    metrics_to_plot = []
    labels = []
    
    if 'train/box_loss' in df.columns:
        metrics_to_plot.append(df['train/box_loss'])
        labels.append('Box Loss')
    if 'train/cls_loss' in df.columns:
        metrics_to_plot.append(df['train/cls_loss'])
        labels.append('Class Loss')
    if 'train/dfl_loss' in df.columns:
        metrics_to_plot.append(df['train/dfl_loss'])
        labels.append('DFL Loss')
    
    for metric, label in zip(metrics_to_plot, labels):
        ax.plot(epochs, metric, linewidth=2, label=label)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: mAP
    ax = axes[0, 1]
    if 'metrics/mAP50(B)' in df.columns:
        ax.plot(epochs, df['metrics/mAP50(B)'], linewidth=2, label='mAP@0.5', marker='o')
    if 'metrics/mAP50-95(B)' in df.columns:
        ax.plot(epochs, df['metrics/mAP50-95(B)'], linewidth=2, label='mAP@0.5:0.95', marker='s')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('Mean Average Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 3: Precision & Recall
    ax = axes[0, 2]
    if 'metrics/precision(B)' in df.columns:
        ax.plot(epochs, df['metrics/precision(B)'], linewidth=2, label='Precision', marker='o')
    if 'metrics/recall(B)' in df.columns:
        ax.plot(epochs, df['metrics/recall(B)'], linewidth=2, label='Recall', marker='s')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 4: Validation Loss
    ax = axes[1, 0]
    metrics_to_plot = []
    labels = []
    
    if 'val/box_loss' in df.columns:
        metrics_to_plot.append(df['val/box_loss'])
        labels.append('Box Loss')
    if 'val/cls_loss' in df.columns:
        metrics_to_plot.append(df['val/cls_loss'])
        labels.append('Class Loss')
    if 'val/dfl_loss' in df.columns:
        metrics_to_plot.append(df['val/dfl_loss'])
        labels.append('DFL Loss')
    
    for metric, label in zip(metrics_to_plot, labels):
        ax.plot(epochs, metric, linewidth=2, label=label)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Learning Rate
    ax = axes[1, 1]
    lr_cols = [col for col in df.columns if 'lr/' in col]
    for col in lr_cols:
        label = col.replace('lr/', '').replace('pg', 'param_group_')
        ax.plot(epochs, df[col], linewidth=2, label=label)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    if lr_cols:
        ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 6: F1 Score (if available or calculated)
    ax = axes[1, 2]
    if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
        precision = df['metrics/precision(B)'].values
        recall = df['metrics/recall(B)'].values
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # type: ignore
        ax.plot(epochs, f1, linewidth=2, label='F1-Score', marker='o', color='green')
        ax.set_ylim((0, 1))
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_class_distribution(
    class_counts: Dict[str, int],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot class distribution bar chart.
    
    Args:
        class_counts: Dictionary mapping class names to counts
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Create bar chart
    bars = plt.bar(classes, counts, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Instances', fontsize=12)
    plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Class distribution plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_class_performance(
    class_metrics: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot per-class performance metrics.
    
    Args:
        class_metrics: Dictionary mapping class names to their metrics
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    classes = list(class_metrics.keys())
    
    # Extract metrics
    ap50 = [class_metrics[c].get('ap50', 0) for c in classes]
    ap50_95 = [class_metrics[c].get('ap50_95', 0) for c in classes]
    
    # Create grouped bar chart
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width/2, ap50, width, label='AP@0.5', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, ap50_95, width, label='AP@0.5:0.95', alpha=0.8, color='darkorange')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Average Precision', fontsize=12)
    ax.set_title('Per-Class Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim((0, 1))
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-class performance plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_predictions_grid(
    image_paths: List[Path],
    predictions: List[List[Dict]],
    class_names: List[str],
    output_path: Optional[Path] = None,
    grid_size: Tuple[int, int] = (3, 3),
    conf_threshold: float = 0.25
) -> None:
    """
    Create a grid of sample predictions.
    
    Args:
        image_paths: List of image file paths
        predictions: List of predictions for each image
        class_names: List of class names
        output_path: Path to save the grid
        grid_size: Grid dimensions (rows, cols)
        conf_threshold: Confidence threshold for display
    """
    rows, cols = grid_size
    num_images = min(len(image_paths), rows * cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    for idx in range(num_images):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Load and display image
        img = cv2.imread(str(image_paths[idx]))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw predictions
        for pred in predictions[idx]:
            if pred['confidence'] < conf_threshold:
                continue
            
            x1, y1, x2, y2 = map(int, pred['bbox'])
            class_id = pred['class_id']
            conf = pred['confidence']
            
            # Draw box
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_names[class_id]}: {conf:.2f}"
            cv2.putText(img_rgb, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        ax.imshow(img_rgb)
        ax.axis('off')
        ax.set_title(f"Image {idx + 1}", fontsize=10)
    
    # Hide unused subplots
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Prediction grid saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confidence_histogram(
    confidences: List[float],
    output_path: Optional[Path] = None,
    bins: int = 50,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot histogram of detection confidences.
    
    Args:
        confidences: List of confidence scores
        output_path: Path to save the plot
        bins: Number of histogram bins
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    mean_conf = float(np.mean(confidences))
    median_conf = float(np.median(confidences))
    
    plt.hist(confidences, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    plt.axvline(mean_conf, color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {mean_conf:.3f}')
    plt.axvline(median_conf, color='green', linestyle='--',
                linewidth=2, label=f'Median: {median_conf:.3f}')
    
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Detection Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confidence histogram saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main execution for standalone usage."""
    parser = argparse.ArgumentParser(
        description='Visualize model evaluation metrics'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Visualization command')
    
    # Training history visualization
    train_parser = subparsers.add_parser('training', help='Plot training history')
    train_parser.add_argument('--results', required=True, help='Path to results.csv')
    train_parser.add_argument('--output', default='outputs/training_history.png',
                             help='Output path')
    
    # Per-class performance
    class_parser = subparsers.add_parser('class-performance',
                                         help='Plot per-class performance')
    class_parser.add_argument('--metrics', required=True,
                             help='Path to metrics JSON file')
    class_parser.add_argument('--output', default='outputs/class_performance.png',
                             help='Output path')
    
    args = parser.parse_args()
    
    if args.command == 'training':
        plot_training_history(Path(args.results), Path(args.output))
    elif args.command == 'class-performance':
        import json
        with open(args.metrics) as f:
            data = json.load(f)
        class_metrics = data['models'][0]['class_metrics']
        plot_per_class_performance(class_metrics, Path(args.output))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
