#!/usr/bin/env python3
"""
Model Evaluation Script for YOLOv8 Pet Detection

This script provides comprehensive model evaluation capabilities including:
- mAP (mean Average Precision) at various IoU thresholds
- Precision, Recall, F1-Score calculations
- Confusion matrix generation and visualization
- Per-class performance metrics
- Model comparison across multiple checkpoints
- Inference speed benchmarking

Example Usage:
    # Basic evaluation
    python src/evaluate.py --model models/yolov8n-pets.pt --data config/data.yaml
    
    # With visualizations
    python src/evaluate.py --model models/yolov8n-pets.pt --data config/data.yaml \\
        --save-plots --save-txt
    
    # Compare multiple models
    python src/evaluate.py --compare models/model1.pt models/model2.pt models/model3.pt \\
        --data config/data.yaml
    
    # Speed benchmark
    python src/evaluate.py --model models/yolov8n-pets.pt --benchmark \\
        --benchmark-samples 100
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics.models.yolo import YOLO


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    model_name: str
    map50: float  # mAP at IoU=0.5
    map50_95: float  # mAP at IoU=0.5:0.95
    precision: float
    recall: float
    f1_score: float
    
    # Per-class metrics
    class_metrics: Dict[str, Dict[str, float]]
    
    # Speed metrics
    inference_time_ms: Optional[float] = None
    fps: Optional[float] = None
    
    # Additional statistics
    total_images: Optional[int] = None
    total_detections: Optional[int] = None


@dataclass
class ConfusionMatrixData:
    """Container for confusion matrix data."""
    
    matrix: np.ndarray
    class_names: List[str]
    normalized: bool = False


def load_data_config(data_yaml: Path) -> Dict:
    """
    Load and validate dataset configuration.
    
    Args:
        data_yaml: Path to data.yaml configuration file
        
    Returns:
        Dictionary containing dataset configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")
    
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['path', 'val', 'names']
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Missing required fields in data.yaml: {missing}")
    
    return config


def evaluate_model(
    model_path: Path,
    data_yaml: Path,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.6,
    verbose: bool = True
) -> Tuple[EvaluationMetrics, Optional[ConfusionMatrixData]]:
    """
    Evaluate a YOLOv8 model on validation dataset.
    
    Args:
        model_path: Path to the model weights
        data_yaml: Path to dataset configuration
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for NMS
        verbose: Print detailed progress
        
    Returns:
        Tuple of (EvaluationMetrics, ConfusionMatrixData)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATING MODEL: {model_path.name}")
        print(f"{'='*70}\n")
    
    # Load model
    model = YOLO(str(model_path))
    data_config = load_data_config(data_yaml)
    
    # Run validation
    if verbose:
        print("Running validation on test/val dataset...")
    
    results = model.val(
        data=str(data_yaml),
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=verbose,
        plots=False  # We'll create custom plots
    )
    
    # Extract metrics
    metrics = EvaluationMetrics(
        model_name=model_path.stem,
        map50=float(results.results_dict.get('metrics/mAP50(B)', 0.0)),
        map50_95=float(results.results_dict.get('metrics/mAP50-95(B)', 0.0)),
        precision=float(results.results_dict.get('metrics/precision(B)', 0.0)),
        recall=float(results.results_dict.get('metrics/recall(B)', 0.0)),
        f1_score=0.0,  # Will calculate below
        class_metrics={},
        total_images=len(results.speed) if hasattr(results, 'speed') else None
    )
    
    # Calculate F1 score
    if metrics.precision > 0 or metrics.recall > 0:
        metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall + 1e-6)
    
    # Extract per-class metrics if available
    class_names = data_config['names']
    if hasattr(results, 'ap_class_index') and hasattr(results, 'ap50'):
        for idx, class_idx in enumerate(results.ap_class_index):
            class_name = class_names[int(class_idx)]
            metrics.class_metrics[class_name] = {
                'ap50': float(results.ap50[idx]) if len(results.ap50) > idx else 0.0,
                'ap50_95': float(results.ap[idx]) if len(results.ap) > idx else 0.0,
            }
    
    # Extract confusion matrix if available
    confusion_matrix_data = None
    if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
        cm = results.confusion_matrix.matrix
        confusion_matrix_data = ConfusionMatrixData(
            matrix=cm,
            class_names=class_names,
            normalized=False
        )
    
    return metrics, confusion_matrix_data


def benchmark_speed(
    model_path: Path,
    num_samples: int = 100,
    img_size: int = 640,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model_path: Path to model weights
        num_samples: Number of inference runs
        img_size: Input image size
        verbose: Print progress
        
    Returns:
        Tuple of (average_inference_time_ms, fps)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"BENCHMARKING SPEED: {model_path.name}")
        print(f"{'='*70}\n")
        print(f"Running {num_samples} inference iterations...")
    
    model = YOLO(str(model_path))
    
    # Create dummy image
    dummy_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    
    # Warmup
    if verbose:
        print("Warming up model...")
    for _ in range(10):
        _ = model(dummy_img, verbose=False)
    
    # Benchmark
    if verbose:
        print("Benchmarking...")
    
    times = []
    for i in range(num_samples):
        start = time.perf_counter()
        _ = model(dummy_img, verbose=False)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
        
        if verbose and (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{num_samples}")
    
    avg_time_ms = float(np.mean(times))
    fps = 1000.0 / avg_time_ms
    
    if verbose:
        print(f"\n✓ Average inference time: {avg_time_ms:.2f} ms")
        print(f"✓ FPS: {fps:.2f}")
        print(f"✓ Min time: {float(np.min(times)):.2f} ms")
        print(f"✓ Max time: {float(np.max(times)):.2f} ms")
        print(f"✓ Std dev: {float(np.std(times)):.2f} ms")
    
    return avg_time_ms, fps


def visualize_confusion_matrix(
    cm_data: ConfusionMatrixData,
    output_path: Path,
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Visualize confusion matrix and save to file.
    
    Args:
        cm_data: Confusion matrix data
        output_path: Path to save the plot
        normalize: Normalize the matrix
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    matrix = cm_data.matrix.copy()
    
    # Normalize if requested
    if normalize and not cm_data.normalized:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, where=row_sums != 0)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=cm_data.class_names + ['Background'],
        yticklabels=cm_data.class_names + ['Background'],
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {output_path}")


def plot_metrics_comparison(
    metrics_list: List[EvaluationMetrics],
    output_path: Path,
    figsize: Tuple[int, int] = (14, 6)
) -> None:
    """
    Create comparison plots for multiple models.
    
    Args:
        metrics_list: List of evaluation metrics
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    
    model_names = [m.model_name for m in metrics_list]
    
    # Prepare data
    map50 = [m.map50 for m in metrics_list]
    map50_95 = [m.map50_95 for m in metrics_list]
    precision = [m.precision for m in metrics_list]
    recall = [m.recall for m in metrics_list]
    f1 = [m.f1_score for m in metrics_list]
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: mAP scores
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0].bar(x - width/2, map50, width, label='mAP@0.5', alpha=0.8)
    axes[0].bar(x + width/2, map50_95, width, label='mAP@0.5:0.95', alpha=0.8)
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('mAP Score')
    axes[0].set_title('Mean Average Precision Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Plot 2: Precision, Recall, F1
    width = 0.25
    axes[1].bar(x - width, precision, width, label='Precision', alpha=0.8)
    axes[1].bar(x, recall, width, label='Recall', alpha=0.8)
    axes[1].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Precision, Recall, F1-Score Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plot saved to: {output_path}")


def print_metrics_table(metrics_list: List[EvaluationMetrics]) -> None:
    """
    Print evaluation metrics in a formatted table.
    
    Args:
        metrics_list: List of evaluation metrics
    """
    print(f"\n{'='*90}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'='*90}\n")
    
    # Header
    print(f"{'Model':<20} {'mAP@0.5':<10} {'mAP@0.5:0.95':<15} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
    print(f"{'-'*20} {'-'*10} {'-'*15} {'-'*12} {'-'*10} {'-'*10}")
    
    # Rows
    for m in metrics_list:
        print(f"{m.model_name:<20} {m.map50:<10.4f} {m.map50_95:<15.4f} {m.precision:<12.4f} {m.recall:<10.4f} {m.f1_score:<10.4f}")
    
    # Speed metrics if available
    if any(m.inference_time_ms is not None for m in metrics_list):
        print(f"\n{'Model':<20} {'Inference Time':<18} {'FPS':<10}")
        print(f"{'-'*20} {'-'*18} {'-'*10}")
        for m in metrics_list:
            if m.inference_time_ms is not None:
                print(f"{m.model_name:<20} {m.inference_time_ms:<17.2f}ms {m.fps:<10.2f}")
    
    print(f"\n{'='*90}\n")


def print_per_class_metrics(metrics: EvaluationMetrics) -> None:
    """
    Print per-class performance metrics.
    
    Args:
        metrics: Evaluation metrics containing class-level data
    """
    if not metrics.class_metrics:
        return
    
    print(f"\n{'='*70}")
    print(f"PER-CLASS METRICS: {metrics.model_name}")
    print(f"{'='*70}\n")
    
    print(f"{'Class':<15} {'AP@0.5':<12} {'AP@0.5:0.95':<15}")
    print(f"{'-'*15} {'-'*12} {'-'*15}")
    
    for class_name, class_metrics in metrics.class_metrics.items():
        ap50 = class_metrics.get('ap50', 0.0)
        ap50_95 = class_metrics.get('ap50_95', 0.0)
        print(f"{class_name:<15} {ap50:<12.4f} {ap50_95:<15.4f}")
    
    print(f"\n{'='*70}\n")


def save_metrics_json(metrics_list: List[EvaluationMetrics], output_path: Path) -> None:
    """
    Save evaluation metrics to JSON file.
    
    Args:
        metrics_list: List of evaluation metrics
        output_path: Path to save JSON file
    """
    data = {
        'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models': [asdict(m) for m in metrics_list]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"✓ Metrics saved to: {output_path}")


def save_metrics_txt(metrics: EvaluationMetrics, output_path: Path) -> None:
    """
    Save evaluation metrics to text file.
    
    Args:
        metrics: Evaluation metrics
        output_path: Path to save text file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"Model Evaluation Report: {metrics.model_name}\n")
        f.write(f"{'='*70}\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  mAP@0.5:        {metrics.map50:.4f}\n")
        f.write(f"  mAP@0.5:0.95:   {metrics.map50_95:.4f}\n")
        f.write(f"  Precision:      {metrics.precision:.4f}\n")
        f.write(f"  Recall:         {metrics.recall:.4f}\n")
        f.write(f"  F1-Score:       {metrics.f1_score:.4f}\n")
        
        if metrics.inference_time_ms is not None:
            f.write(f"\nSpeed Metrics:\n")
            f.write(f"  Inference Time: {metrics.inference_time_ms:.2f} ms\n")
            f.write(f"  FPS:            {metrics.fps:.2f}\n")
        
        if metrics.class_metrics:
            f.write(f"\nPer-Class Metrics:\n")
            for class_name, class_metrics in metrics.class_metrics.items():
                f.write(f"\n  {class_name}:\n")
                f.write(f"    AP@0.5:       {class_metrics.get('ap50', 0.0):.4f}\n")
                f.write(f"    AP@0.5:0.95:  {class_metrics.get('ap50_95', 0.0):.4f}\n")
    
    print(f"✓ Report saved to: {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Evaluate YOLOv8 model performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model evaluation
  python src/evaluate.py --model models/yolov8n-pets.pt --data config/data.yaml
  
  # With visualizations and reports
  python src/evaluate.py --model models/yolov8n-pets.pt --data config/data.yaml \\
      --save-plots --save-txt --save-json
  
  # Compare multiple models
  python src/evaluate.py --compare models/model1.pt models/model2.pt models/model3.pt \\
      --data config/data.yaml --save-plots
  
  # Speed benchmark
  python src/evaluate.py --model models/yolov8n-pets.pt --benchmark \\
      --benchmark-samples 200
        """
    )
    
    # Model arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model', type=str, help='Path to model weights')
    model_group.add_argument('--compare', nargs='+', help='Compare multiple models')
    
    # Data arguments
    parser.add_argument('--data', type=str, default='config/data.yaml',
                        help='Path to data.yaml (default: config/data.yaml)')
    
    # Evaluation parameters
    parser.add_argument('--conf', type=float, default=0.001,
                        help='Confidence threshold (default: 0.001)')
    parser.add_argument('--iou', type=float, default=0.6,
                        help='IoU threshold for NMS (default: 0.6)')
    
    # Output options
    parser.add_argument('--output', type=str, default='outputs/evaluation',
                        help='Output directory (default: outputs/evaluation)')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save visualization plots')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save metrics to text file')
    parser.add_argument('--save-json', action='store_true',
                        help='Save metrics to JSON file')
    
    # Benchmarking
    parser.add_argument('--benchmark', action='store_true',
                        help='Run speed benchmark')
    parser.add_argument('--benchmark-samples', type=int, default=100,
                        help='Number of benchmark samples (default: 100)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for benchmark (default: 640)')
    
    # Other options
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    # Set verbosity
    verbose = args.verbose and not args.quiet
    
    # Setup paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = Path(args.data)
    
    # Get model paths
    if args.model:
        model_paths = [Path(args.model)]
    else:
        model_paths = [Path(m) for m in args.compare]
    
    # Validate models exist
    for model_path in model_paths:
        if not model_path.exists():
            print(f"❌ Error: Model not found: {model_path}")
            return 1
    
    # Evaluate models
    all_metrics = []
    all_confusion_matrices = []
    
    for model_path in model_paths:
        # Run evaluation
        metrics, cm_data = evaluate_model(
            model_path,
            data_yaml,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            verbose=verbose
        )
        
        # Run benchmark if requested
        if args.benchmark:
            avg_time, fps = benchmark_speed(
                model_path,
                num_samples=args.benchmark_samples,
                img_size=args.img_size,
                verbose=verbose
            )
            metrics.inference_time_ms = avg_time
            metrics.fps = fps
        
        all_metrics.append(metrics)
        if cm_data:
            all_confusion_matrices.append((metrics.model_name, cm_data))
        
        # Print per-class metrics for single model
        if len(model_paths) == 1:
            print_per_class_metrics(metrics)
    
    # Print results table
    print_metrics_table(all_metrics)
    
    # Save outputs
    if args.save_json:
        json_path = output_dir / 'metrics.json'
        save_metrics_json(all_metrics, json_path)
    
    if args.save_txt:
        for metrics in all_metrics:
            txt_path = output_dir / f'{metrics.model_name}_report.txt'
            save_metrics_txt(metrics, txt_path)
    
    if args.save_plots:
        # Save confusion matrices
        for model_name, cm_data in all_confusion_matrices:
            cm_path = output_dir / f'{model_name}_confusion_matrix.png'
            visualize_confusion_matrix(cm_data, cm_path, normalize=True)
        
        # Save comparison plot if multiple models
        if len(all_metrics) > 1:
            comparison_path = output_dir / 'model_comparison.png'
            plot_metrics_comparison(all_metrics, comparison_path)
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())
